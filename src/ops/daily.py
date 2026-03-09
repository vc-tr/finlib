"""
Daily pipeline: generate target portfolio + staged orders.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.cost_models import FixedBpsCostModel, LiquidityAwareCostModel
from src.factors import estimate_beta
from src.factors.factors import get_prices_wide
from src.factors.portfolio import rebalance_dates
from src.paper.strategy_adapter import get_factor_target_weights
from src.utils.jsonable import to_jsonable

MARKET_SYMBOL = "SPY"
DEFAULT_STATE_PATH = Path("data/state/current_portfolio.json")


def load_current_portfolio(path: Path) -> dict:
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


def save_current_portfolio(path: Path, cash: float, positions: dict, asof: str) -> None:
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


def _compute_turnover(target_w: dict[str, float], current_w: dict[str, float]) -> float:
    """Turnover = 0.5 * sum_i |w_target_i - w_current_i| over all tradable symbols."""
    all_syms = set(target_w.keys()) | set(current_w.keys())
    return 0.5 * sum(abs(target_w.get(s, 0.0) - current_w.get(s, 0.0)) for s in all_syms)


def run_daily(
    df_by_symbol: dict[str, pd.DataFrame],
    *,
    strategy: str = "factors",
    factor: str = "momentum_12_1",
    combo_list: list[str] | None = None,
    combo_method: str = "equal",
    asof: pd.Timestamp,
    rebalance: str = "M",
    initial_cash: float = 100_000.0,
    cost_model: str = "fixed",
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    spread_bps: float = 1.0,
    top_k: int = 10,
    bottom_k: int = 10,
    max_gross: float | None = None,
    max_net: float | None = None,
    max_position_weight: float | None = 0.2,
    beta_threshold: float = 0.5,
    state_path: Path,
    output_dir: Path,
    apply: bool = False,
    force_rebalance: bool = False,
) -> dict[str, Any]:
    """Run daily pipeline. Returns summary dict."""
    prices = get_prices_wide(df_by_symbol)
    if prices.empty:
        return {"error": "No price data"}

    idx = prices.index[prices.index <= asof]
    if len(idx) < 30:
        return {"error": "Insufficient history"}

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

    avail = target_weights_df.loc[target_weights_df.index <= asof]
    target_row = avail.iloc[-1] if len(avail) > 0 else target_weights_df.iloc[-1]
    target_w = {
        s: float(target_row[s])
        for s in target_row.index
        if pd.notna(target_row[s]) and abs(target_row[s]) > 1e-10
    }

    state_path_resolved = state_path.resolve()
    portfolio = load_current_portfolio(state_path_resolved)
    cash = portfolio["cash"]
    positions = portfolio["positions"]
    if portfolio["asof"] is not None and (cash != 0 or positions):
        print(f"Loaded state: {state_path_resolved} (positions={len(positions)}, cash=${cash:,.0f})")
    else:
        print("No prior state, starting from cash.")
    if cash == 0 and not positions:
        cash = initial_cash
    prices_ts = {
        s: float(prices.loc[asof, s])
        for s in prices.columns
        if asof in prices.index and pd.notna(prices.loc[asof, s])
    }
    pv = _portfolio_value(cash, positions, prices_ts)
    if pv <= 0:
        pv = initial_cash
        cash = initial_cash
        positions = {}

    current_w = _current_weights(cash, positions, prices_ts)
    rb_dates = rebalance_dates(prices.index, rebalance)
    is_rebalance_day = asof in rb_dates

    if not is_rebalance_day and not force_rebalance:
        print("Not a rebalance day; no orders generated.")
        orders = []
        turnover = 0.0
        expected_costs = 0.0
        new_positions = dict(positions)
        new_cash = cash
        gross_w = sum(abs(positions.get(s, 0) * prices_ts.get(s, 0)) for s in positions) / pv if pv > 0 else 0
        net_w = sum(positions.get(s, 0) * prices_ts.get(s, 0) for s in positions) / pv if pv > 0 else 0
        max_single_w = (
            max((abs(positions.get(s, 0) * prices_ts.get(s, 0)) / pv for s in positions), default=0.0)
            if pv > 0
            else 0.0
        )
        beta_p = 0.0
        state_bootstrap = portfolio["asof"] is None
        risk_checks = {
            "gross_weight": gross_w,
            "net_weight": net_w,
            "max_single_weight": max_single_w,
            "portfolio_beta": beta_p,
            "max_gross_breach": False,
            "max_net_breach": False,
            "max_position_breach": False,
            "beta_breach": False,
            "state_bootstrap": state_bootstrap,
        }
    else:
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

        state_bootstrap = portfolio["asof"] is None
        risk_checks = {
            "gross_weight": gross_w,
            "net_weight": net_w,
            "max_single_weight": max_single_w,
            "portfolio_beta": beta_p,
            "max_gross_breach": max_gross is not None and gross_w > max_gross,
            "max_net_breach": max_net is not None and abs(net_w) > max_net,
            "max_position_breach": max_position_weight is not None and max_single_w > max_position_weight,
            "beta_breach": abs(beta_p) > beta_threshold,
            "state_bootstrap": state_bootstrap,
        }

        if orders:
            rows = []
            for o in orders:
                rows.append({
                    "timestamp": asof,
                    "symbol": o["symbol"],
                    "trade_weight": abs(o["weight_delta"]),
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
        turnover = _compute_turnover(target_w, current_w)

    output_dir.mkdir(parents=True, exist_ok=True)
    orders_df = pd.DataFrame(orders)
    if not orders_df.empty:
        orders_df.to_csv(output_dir / "orders_to_place.csv", index=False)
    else:
        (output_dir / "orders_to_place.csv").write_text(
            "symbol,side,quantity,weight_delta,notional,fill_price\n", encoding="utf-8"
        )

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
        "| Metric | Value |",
        "|--------|-------|",
        f"| Orders | {len(orders)} |",
        f"| Turnover | {turnover:.2%} |",
        f"| Expected Costs | ${expected_costs:,.2f} |",
        "",
        "## Risk Checks",
        "",
        "| Check | Value | Threshold | Pass |",
        "|-------|-------|-----------|------|",
        f"| Gross | {gross_w:.2%} | {max_gross or 'N/A'} | {'✓' if not risk_checks['max_gross_breach'] else '✗'} |",
        f"| Net | {net_w:.2%} | {max_net or 'N/A'} | {'✓' if not risk_checks['max_net_breach'] else '✗'} |",
        f"| Max Single | {max_single_w:.2%} | {max_position_weight or 'N/A'} | {'✓' if not risk_checks['max_position_breach'] else '✗'} |",  # noqa: E501
        f"| Beta | {beta_p:.2f} | ±{beta_threshold} | {'✓' if not risk_checks['beta_breach'] else '✗'} |",
        "",
    ])
    (output_dir / "daily_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_dir / "risk_checks.json").write_text(json.dumps(to_jsonable(risk_checks), indent=2), encoding="utf-8")

    if (
        apply
        and (is_rebalance_day or force_rebalance)
        and not any([risk_checks["max_gross_breach"], risk_checks["max_net_breach"], risk_checks["max_position_breach"]])  # noqa: E501
    ):
        state_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        save_current_portfolio(state_path_resolved, new_cash, new_positions, str(asof.date()))

    summary = {
        "asof": str(asof.date()),
        "n_orders": len(orders),
        "turnover": turnover,
        "expected_costs": expected_costs,
        "portfolio_value": pv,
        "portfolio_beta": beta_p,
        "risk_checks": risk_checks,
        "applied": apply,
        "rebalance_day": is_rebalance_day or force_rebalance,
    }
    (output_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    return summary


def build_run_meta(
    *,
    output_dir: Path,
    state_path: Path,
    state_loaded: bool,
    asof_requested: str | None,
    asof_trading: str,
    rebalance: str,
    universe: str,
    factor: str,
    combo: str | None,
    combo_method: str | None,
    cost_model: str,
    apply: bool,
    result: dict | None = None,
    status: str = "ok",
    error: str | None = None,
) -> dict:
    """Build run_meta dict for run_meta.json."""
    meta: dict = {
        "run_type": "daily",
        "asof_requested": asof_requested or "",
        "asof_trading": asof_trading,
        "state_path": str(state_path.resolve()),
        "state_loaded": state_loaded,
        "rebalance": rebalance,
        "interval": "1d",
        "universe": universe,
        "factor": factor,
        "combo": combo,
        "combo_method": combo_method,
        "cost_model": cost_model,
        "fill_mode": "next_close",
        "applied": apply,
        "output_dir": str(output_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if status == "error":
        meta["status"] = "error"
        meta["error"] = error or ""
        meta["rebalance_day"] = False
        meta["orders_count"] = 0
        meta["turnover"] = 0.0
        meta["beta"] = None
    else:
        meta["rebalance_day"] = result.get("rebalance_day", False)
        meta["orders_count"] = result.get("n_orders", 0)
        meta["turnover"] = float(result.get("turnover", 0.0))
        meta["beta"] = result.get("risk_checks", {}).get("portfolio_beta")
    return meta


def write_run_meta(output_dir: Path, run_meta: dict) -> Path:
    """Write run_meta.json. Returns path to file."""
    path = output_dir / "run_meta.json"
    path.write_text(json.dumps(to_jsonable(run_meta), indent=2), encoding="utf-8")
    return path
