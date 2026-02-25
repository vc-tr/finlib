#!/usr/bin/env python3
"""
Production daily pipeline: generate tomorrow's target portfolio + staged orders.

Dry-run by default; use --apply to update data/state/current_portfolio.json.

Usage:
    python scripts/daily_run.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M
    python scripts/daily_run.py --factor combo --combo "momentum_12_1,reversal_5d" --combo-method auto_robust --asof 2024-06-30
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.factors import get_universe, estimate_beta
from src.factors.factors import get_prices_wide
from src.factors.portfolio import rebalance_dates
from src.paper.strategy_adapter import get_factor_target_weights
from src.backtest.cost_models import FixedBpsCostModel, LiquidityAwareCostModel
from src.utils.jsonable import to_jsonable

STATE_PATH = Path("data/state/current_portfolio.json")
MARKET_SYMBOL = "SPY"


def _load_current_portfolio(path: Path) -> dict:
    """Load current portfolio from JSON. Returns {cash, positions, asof} or all-cash default."""
    if not path.exists():
        return {"cash": 0.0, "positions": {}, "asof": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "cash": float(data.get("cash", 0)),
            "positions": dict(data.get("positions", {})),
            "asof": data.get("asof"),
        }
    except Exception:
        return {"cash": 0.0, "positions": {}, "asof": None}


def _save_current_portfolio(path: Path, cash: float, positions: dict, asof: str) -> None:
    """Save portfolio state to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cash": cash, "positions": positions, "asof": asof}, indent=2),
        encoding="utf-8",
    )


def _portfolio_value(cash: float, positions: dict, prices: dict[str, float]) -> float:
    """Total portfolio value = cash + sum(positions * price)."""
    pv = cash
    for sym, qty in positions.items():
        pv += qty * prices.get(sym, 0)
    return pv


def _current_weights(cash: float, positions: dict, prices: dict[str, float]) -> dict[str, float]:
    """Current weights by symbol (fraction of portfolio)."""
    pv = _portfolio_value(cash, positions, prices)
    if pv <= 0:
        return {}
    w = {}
    for sym, qty in positions.items():
        pr = prices.get(sym, 0)
        if pr > 0 and qty != 0:
            w[sym] = (qty * pr) / pv
    return w


def _run_daily(
    df_by_symbol: dict[str, pd.DataFrame],
    strategy: str,
    factor: str,
    combo_list: list[str] | None,
    combo_method: str,
    asof: pd.Timestamp,
    rebalance: str,
    initial_cash: float,
    cost_model: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    top_k: int,
    bottom_k: int,
    max_gross: float | None,
    max_net: float | None,
    max_position_weight: float | None,
    beta_threshold: float,
    state_path: Path,
    output_dir: Path,
    apply: bool,
) -> dict:
    """Run daily pipeline. Returns summary dict."""
    prices = get_prices_wide(df_by_symbol)
    if prices.empty:
        return {"error": "No price data"}

    idx = prices.index[prices.index <= asof]
    if len(idx) < 30:
        return {"error": "Insufficient history"}

    # 1) Target weights for asof (or next rebalance)
    target_weights_df = get_factor_target_weights(
        df_by_symbol,
        factor=factor,
        combo_list=combo_list,
        combo_method=combo_method,
        start=idx[0],
        end=asof,
        rebalance=rebalance,
        top_k=top_k,
        bottom_k=bottom_k,
    )
    if target_weights_df.empty:
        return {"error": "No target weights"}

    # Use last available target (ffill semantics)
    avail = target_weights_df.loc[target_weights_df.index <= asof]
    target_row = avail.iloc[-1] if len(avail) > 0 else target_weights_df.iloc[-1]
    target_w = {s: float(target_row[s]) for s in target_row.index if pd.notna(target_row[s]) and abs(target_row[s]) > 1e-10}

    # 2) Current portfolio
    portfolio = _load_current_portfolio(state_path)
    cash = portfolio["cash"]
    positions = portfolio["positions"]
    if cash == 0 and not positions:
        cash = initial_cash
    prices_ts = {s: float(prices.loc[asof, s]) for s in prices.columns if asof in prices.index and pd.notna(prices.loc[asof, s])}
    pv = _portfolio_value(cash, positions, prices_ts)
    if pv <= 0:
        pv = initial_cash
        cash = initial_cash
        positions = {}

    current_w = _current_weights(cash, positions, prices_ts)

    # 3) Delta orders
    all_syms = set(current_w.keys()) | set(target_w.keys())
    orders = []
    for sym in all_syms:
        tw = target_w.get(sym, 0.0)
        cw = current_w.get(sym, 0.0)
        delta = tw - cw
        if abs(delta) < 1e-6:
            continue
        pr = prices_ts.get(sym, 0)
        if pr <= 0:
            continue
        notional = delta * pv
        qty = notional / pr
        if abs(qty) < 1e-6:
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append({
            "symbol": sym,
            "side": side,
            "quantity": abs(qty),
            "weight_delta": delta,
            "notional": abs(notional),
            "fill_price": pr,
        })

    # 4) Simulate post-trade state for risk checks
    new_positions = dict(positions)
    new_cash = cash
    for o in orders:
        sym, side, qty, pr = o["symbol"], o["side"], o["quantity"], o["fill_price"]
        if side == "buy":
            new_positions[sym] = new_positions.get(sym, 0) + qty
            new_cash -= qty * pr
        else:
            new_positions[sym] = new_positions.get(sym, 0) - qty
            new_cash += qty * pr
    new_positions = {s: q for s, q in new_positions.items() if abs(q) > 1e-6}

    # 5) Risk checks
    new_pv = _portfolio_value(new_cash, new_positions, prices_ts)
    gross = sum(abs(new_positions.get(s, 0) * prices_ts.get(s, 0)) for s in new_positions) if new_pv > 0 else 0
    net = sum(new_positions.get(s, 0) * prices_ts.get(s, 0) for s in new_positions)
    gross_w = gross / new_pv if new_pv > 0 else 0
    net_w = net / new_pv if new_pv > 0 else 0
    max_single_w = 0.0
    for s, q in new_positions.items():
        w = abs(q * prices_ts.get(s, 0)) / new_pv if new_pv > 0 else 0
        max_single_w = max(max_single_w, w)

    returns = prices.pct_change()
    market_ret = returns[MARKET_SYMBOL] if MARKET_SYMBOL in returns.columns else pd.Series(dtype=float)
    beta_p = 0.0
    if len(market_ret) > 20 and MARKET_SYMBOL in df_by_symbol:
        betas = estimate_beta(returns, market_ret, window=min(252, len(returns)))
        if asof in betas.index:
            for s, q in new_positions.items():
                if s in betas.columns:
                    beta_p += (q * prices_ts.get(s, 0) / new_pv) * betas.loc[asof, s] if new_pv > 0 else 0

    risk_checks = {
        "gross_weight": gross_w,
        "net_weight": net_w,
        "max_single_weight": max_single_w,
        "portfolio_beta": beta_p,
        "max_gross_breach": max_gross is not None and gross_w > max_gross,
        "max_net_breach": max_net is not None and abs(net_w) > max_net,
        "max_position_breach": max_position_weight is not None and max_single_w > max_position_weight,
        "beta_breach": abs(beta_p) > beta_threshold,
    }

    # 6) Expected costs (build trades from orders)
    if orders:
        rows = []
        for o in orders:
            trade_weight = abs(o["weight_delta"])
            rows.append({
                "timestamp": asof,
                "symbol": o["symbol"],
                "trade_weight": trade_weight,
                "side": o["side"],
                "fill_price": o["fill_price"],
                "trade_notional": o["notional"],
            })
        trades_df = pd.DataFrame(rows)
        if cost_model == "liquidity":
            model = LiquidityAwareCostModel()
        else:
            model = FixedBpsCostModel()
        config = {"fee_bps": fee_bps, "slippage_bps": slippage_bps, "spread_bps": spread_bps}
        trades_df = model.estimate_costs(trades_df, df_by_symbol, config)
        expected_costs = float(trades_df["total_cost"].sum())
    else:
        expected_costs = 0.0

    turnover = sum(o["notional"] for o in orders) / pv if pv > 0 else 0

    # 7) Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    orders_df = pd.DataFrame(orders)
    if not orders_df.empty:
        orders_df.to_csv(output_dir / "orders_to_place.csv", index=False)
    else:
        (output_dir / "orders_to_place.csv").write_text("symbol,side,quantity,weight_delta,notional,fill_price\n", encoding="utf-8")

    # Write daily_report.md first (even if risk_checks serialization fails later)
    report_lines = [
        "# Daily Report",
        "",
        f"**As-of date:** {asof.date()}",
        "",
        "## Target Weights Summary",
        "",
        "| Symbol | Weight |",
        "|--------|--------|",
    ]
    for s in sorted(target_w.keys(), key=lambda x: -abs(target_w[x])):
        report_lines.append(f"| {s} | {target_w[s]:.4f} |")
    report_lines.extend([
        "",
        "## Order Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Orders | {len(orders)} |",
        f"| Turnover | {turnover:.2%} |",
        f"| Expected Costs | ${expected_costs:,.2f} |",
        "",
        "## Risk Checks",
        "",
        f"| Check | Value | Threshold | Pass |",
        f"|-------|-------|-----------|------|",
        f"| Gross | {gross_w:.2%} | {max_gross or 'N/A'} | {'✓' if not risk_checks['max_gross_breach'] else '✗'} |",
        f"| Net | {net_w:.2%} | {max_net or 'N/A'} | {'✓' if not risk_checks['max_net_breach'] else '✗'} |",
        f"| Max Single | {max_single_w:.2%} | {max_position_weight or 'N/A'} | {'✓' if not risk_checks['max_position_breach'] else '✗'} |",
        f"| Beta | {beta_p:.2f} | ±{beta_threshold} | {'✓' if not risk_checks['beta_breach'] else '✗'} |",
        "",
    ])
    (output_dir / "daily_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # risk_checks.json with JSON-serializable types (numpy/pandas scalars)
    (output_dir / "risk_checks.json").write_text(json.dumps(to_jsonable(risk_checks), indent=2), encoding="utf-8")

    # 8) Apply if requested
    if apply and not any([risk_checks["max_gross_breach"], risk_checks["max_net_breach"], risk_checks["max_position_breach"]]):
        _save_current_portfolio(state_path, new_cash, new_positions, str(asof.date()))

    summary = {
        "asof": str(asof.date()),
        "n_orders": len(orders),
        "turnover": turnover,
        "expected_costs": expected_costs,
        "portfolio_value": pv,
        "portfolio_beta": beta_p,
        "risk_checks": risk_checks,
        "applied": apply,
    }
    (output_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily pipeline: target portfolio + staged orders")
    parser.add_argument("--strategy", default="factors", choices=["factors"])
    parser.add_argument("--universe", default="liquid_etfs")
    parser.add_argument("--factor", default="momentum_12_1", choices=["momentum_12_1", "reversal_5d", "lowvol_20d", "combo"])
    parser.add_argument("--combo", default=None)
    parser.add_argument("--combo-method", default="equal", choices=["equal", "ic_weighted", "ridge", "sharpe_opt", "auto", "auto_robust"])
    parser.add_argument("--rebalance", default="M", choices=["D", "W", "M"])
    parser.add_argument("--asof", default=None, help="YYYY-MM-DD (default: latest)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--cost-model", default="fixed", choices=["fixed", "liquidity"])
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10)
    parser.add_argument("--max-gross", type=float, default=None)
    parser.add_argument("--max-net", type=float, default=None)
    parser.add_argument("--max-position-weight", type=float, default=0.2)
    parser.add_argument("--beta-threshold", type=float, default=0.5)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--apply", action="store_true", help="Update current_portfolio.json")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--lock-timeout", type=float, default=0)
    args = parser.parse_args()

    if args.factor == "combo" and not args.combo:
        print("[ERROR] --factor combo requires --combo")
        sys.exit(1)
    combo_list = [f.strip() for f in args.combo.split(",")] if args.combo else None

    state_path = Path(args.state_path) if args.state_path else Path(__file__).resolve().parent.parent / STATE_PATH
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"{args.factor}_{args.rebalance}"
        output_dir = Path("output") / "runs" / f"{ts}_daily_{suffix}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = get_universe(args.universe, n=30)
    if MARKET_SYMBOL not in symbols:
        symbols = [MARKET_SYMBOL] + [s for s in symbols if s != MARKET_SYMBOL]
    print(f"[1/3] Fetching {len(symbols)} symbols ({args.period})...")
    from src.pipeline.data_fetcher_yahoo import YahooDataFetcher

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df_by_symbol = {}
    for sym in symbols:
        try:
            df = fetcher.fetch_ohlcv(sym, "1d", period=args.period)
            df = df.dropna()
            if len(df) >= 30:
                df_by_symbol[sym] = df
        except Exception as e:
            print(f"  [WARN] Skip {sym}: {e}")
    if len(df_by_symbol) < args.top_k + args.bottom_k:
        print(f"[ERROR] Need at least {args.top_k + args.bottom_k} symbols")
        sys.exit(1)
    print(f"      Loaded {len(df_by_symbol)} symbols")

    prices = get_prices_wide(df_by_symbol)
    asof = pd.Timestamp(args.asof) if args.asof else prices.index.max()
    if asof not in prices.index:
        asof = prices.index[prices.index <= asof].max() if len(prices.index[prices.index <= asof]) > 0 else prices.index.max()

    print(f"[2/3] Running daily pipeline (asof={asof.date()})...")
    result = _run_daily(
        df_by_symbol,
        strategy=args.strategy,
        factor=args.factor,
        combo_list=combo_list,
        combo_method=args.combo_method,
        asof=asof,
        rebalance=args.rebalance,
        initial_cash=args.initial_cash,
        cost_model=args.cost_model,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        max_gross=args.max_gross,
        max_net=args.max_net,
        max_position_weight=args.max_position_weight,
        beta_threshold=args.beta_threshold,
        state_path=state_path,
        output_dir=output_dir,
        apply=args.apply,
    )

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        sys.exit(1)

    print("[3/3] Done:")
    print("-" * 40)
    print(f"  Orders:     {result['n_orders']}")
    print(f"  Turnover:   {result['turnover']:.2%}")
    print(f"  Beta:       {result['risk_checks']['portfolio_beta']:.2f}")
    print(f"  Applied:    {result['applied']}")
    print("-" * 40)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--no-lock", action="store_true")
    _p.add_argument("--lock-timeout", type=float, default=0)
    _pre, _ = _p.parse_known_args()
    if _pre.no_lock:
        main()
    else:
        from src.utils.runlock import RunLock

        with RunLock(lock_path=str(Path(__file__).resolve().parent.parent / ".runlock"), timeout_s=_pre.lock_timeout):
            main()
