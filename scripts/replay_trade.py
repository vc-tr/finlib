#!/usr/bin/env python3
"""
Event-driven paper trading replay engine.

Replays historical bars chronologically, generates target weights on rebalance dates,
creates orders, simulates fills with costs, updates portfolio state.

Usage:
    python scripts/replay_trade.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M --start 2024-01-01 --end 2025-12-31
    python scripts/replay_trade.py --strategy factors --factor combo --combo "momentum_12_1,reversal_5d,lowvol_20d" --combo-method auto_robust --rebalance M
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.factors import get_universe
from src.factors.factors import get_prices_wide
from src.factors.portfolio import rebalance_dates
from src.paper import PaperBroker, PaperExchange, RiskManager
from src.paper.orders import Order, OrderSide, OrderType
from src.paper.strategy_adapter import get_factor_target_weights
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher


def _fetch_universe(
    symbols: list[str],
    interval: str,
    period: str,
    fetcher: YahooDataFetcher,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for each symbol."""
    df_by_symbol = {}
    for sym in symbols:
        try:
            df = fetcher.fetch_ohlcv(sym, interval, period=period)
            df = df.dropna()
            if len(df) >= 10:
                df_by_symbol[sym] = df
        except Exception as e:
            print(f"  [WARN] Skip {sym}: {e}")
    return df_by_symbol


def _run_replay(
    df_by_symbol: dict[str, pd.DataFrame],
    strategy: str,
    factor: str,
    combo_list: list[str] | None,
    combo_method: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    rebalance: str,
    initial_cash: float,
    fill_mode: str,
    cost_model: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    top_k: int,
    bottom_k: int,
    output_dir: Path,
) -> dict:
    """Run paper trading replay."""
    prices = get_prices_wide(df_by_symbol)
    if prices.empty:
        return {"error": "No price data"}

    # Align index
    all_idx = prices.index.sort_values()
    if start is not None:
        all_idx = all_idx[all_idx >= start]
    if end is not None:
        all_idx = all_idx[all_idx <= end]
    if len(all_idx) < 5:
        return {"error": "Insufficient data in date range"}

    # Precompute target weights
    target_weights = get_factor_target_weights(
        df_by_symbol,
        factor=factor,
        combo_list=combo_list,
        combo_method=combo_method,
        start=all_idx[0],
        end=all_idx[-1],
        rebalance=rebalance,
        top_k=top_k,
        bottom_k=bottom_k,
    )
    if target_weights.empty:
        return {"error": "No target weights"}

    rb_dates = rebalance_dates(all_idx, rebalance)
    rb_set = set(rb_dates)

    # Build bars for exchange (ensure OHLCV columns)
    bars_by_symbol = {}
    for sym, df in df_by_symbol.items():
        d = df.copy()
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in d.columns and c.capitalize() in d.columns:
                d[c] = d[c.capitalize()]
        if "volume" not in d.columns:
            d["volume"] = 0.0
        bars_by_symbol[sym] = d

    exchange = PaperExchange(
        bars_by_symbol=bars_by_symbol,
        cost_model=cost_model,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        fill_mode=fill_mode,
    )
    risk = RiskManager(max_position_weight=0.2)
    broker = PaperBroker(
        exchange=exchange,
        initial_cash=initial_cash,
        risk_manager=risk,
    )

    orders_log = []
    positions_snapshots = []

    for ts in all_idx:
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        prices_ts = {s: float(prices.loc[ts, s]) for s in prices.columns if ts in prices.index and pd.notna(prices.loc[ts, s])}

        # 1) Replay bar: fill pending orders
        fills = exchange.replay_bar(ts_dt)
        if fills:
            broker.process_fills(fills)
            for f in fills:
                orders_log.append({
                    "timestamp": ts,
                    "order_id": f.order_id,
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                    "cost_bps": f.cost_bps,
                    "status": "filled",
                })

        # 2) Record equity
        broker.record_equity(ts_dt, prices_ts)

        # 3) On rebalance: submit new orders
        if ts in rb_set:
            pv = broker.portfolio_value(prices_ts)
            if pv <= 0:
                continue

            # Current weights
            current_w = {}
            for sym, pos in broker.positions.items():
                pr = prices_ts.get(sym, 0)
                if pr > 0:
                    current_w[sym] = (pos * pr) / pv
                else:
                    current_w[sym] = 0.0

            # Target weights at this date
            tw_row = target_weights.loc[ts] if ts in target_weights.index else target_weights.iloc[-1]
            target_w = tw_row.to_dict()

            all_syms = set(current_w.keys()) | set(target_w.keys()) | set(prices_ts.keys())
            for sym in all_syms:
                tw = target_w.get(sym, 0.0)
                if pd.isna(tw):
                    tw = 0.0
                cw = current_w.get(sym, 0.0)
                delta = tw - cw
                if abs(delta) < 1e-6:
                    continue

                pr = prices_ts.get(sym, 0)
                if pr <= 0:
                    continue

                notional = delta * pv
                qty = abs(notional) / pr
                if qty < 1e-6:
                    continue

                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                order = Order(symbol=sym, side=side, quantity=qty, order_type=OrderType.MARKET)
                result = broker.place_order(order, submit_ts=ts_dt, prices=prices_ts)
                if result and result.status.value != "rejected":
                    orders_log.append({
                        "timestamp": ts,
                        "order_id": result.order_id,
                        "symbol": sym,
                        "side": side.value,
                        "quantity": qty,
                        "price": None,
                        "cost_bps": None,
                        "status": "submitted",
                    })

            # Snapshot positions
            w_held = {s: (broker.positions.get(s, 0) * prices_ts.get(s, 0)) / pv if pv > 0 else 0 for s in prices_ts}
            positions_snapshots.append({"timestamp": ts, **w_held})

    # Write outputs
    orders_df = pd.DataFrame(orders_log)
    if not orders_df.empty:
        orders_df.to_csv(output_dir / "orders.csv", index=False)

    blotter_df = broker.blotter_df()
    if not blotter_df.empty:
        blotter_df.to_csv(output_dir / "blotter.csv", index=False)

    equity_df = broker.equity_curve_df()
    if not equity_df.empty:
        equity_df.to_csv(output_dir / "equity_curve.csv", index=False)

    if positions_snapshots:
        pos_df = pd.DataFrame(positions_snapshots).set_index("timestamp")
        pos_df.to_csv(output_dir / "positions_snapshot.csv")

    # Compute metrics
    if len(equity_df) >= 2:
        eq = equity_df.set_index("timestamp")["equity"]
        ret = eq.pct_change().dropna()
        total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if eq.iloc[0] != 0 else 0
        sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 1e-12 else 0.0
        cum = (1 + ret).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0
        turnover = 0.0
        if not blotter_df.empty and "quantity" in blotter_df.columns and "price" in blotter_df.columns:
            turnover = float((blotter_df["quantity"].abs() * blotter_df["price"]).sum() / initial_cash) if initial_cash > 0 else 0
        costs_total = 0.0
        if not blotter_df.empty and "cost_bps" in blotter_df.columns and "quantity" in blotter_df.columns and "price" in blotter_df.columns:
            costs_total = float((blotter_df["cost_bps"] / 10_000 * blotter_df["quantity"] * blotter_df["price"]).sum())
    else:
        total_return = 0.0
        sharpe = 0.0
        max_dd = 0.0
        turnover = 0.0
        costs_total = 0.0

    report_lines = [
        "# Replay Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Return | {total_return:.2%} |",
        f"| Sharpe Ratio | {sharpe:.2f} |",
        f"| Max Drawdown | {max_dd:.2%} |",
        f"| Turnover | {turnover:.2f} |",
        f"| Total Costs | ${costs_total:,.2f} |",
        "",
        "## Outputs",
        "",
        "- orders.csv — all submitted orders",
        "- blotter.csv — filled orders only",
        "- equity_curve.csv — timestamp, equity",
        "- positions_snapshot.csv — weights at rebalance timestamps",
        "",
    ]
    (output_dir / "replay_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover": turnover,
        "costs_total": costs_total,
        "n_orders": len(orders_log),
        "n_fills": len(blotter_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading replay engine")
    parser.add_argument("--strategy", default="factors", choices=["factors"])
    parser.add_argument("--universe", default="liquid_etfs")
    parser.add_argument("--factor", default="momentum_12_1", choices=["momentum_12_1", "reversal_5d", "lowvol_20d", "combo"])
    parser.add_argument("--combo", default=None, help='Comma-separated factors (e.g. "momentum_12_1,reversal_5d,lowvol_20d")')
    parser.add_argument("--combo-method", default="equal", choices=["equal", "ic_weighted", "ridge", "sharpe_opt", "auto", "auto_robust"])
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--rebalance", default="M", choices=["D", "W", "M"])
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--fill-mode", default="next_close", choices=["next_open", "next_close"])
    parser.add_argument("--cost-model", default="fixed", choices=["fixed", "liquidity"])
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--lock-timeout", type=float, default=0)
    args = parser.parse_args()

    if args.factor == "combo" and not args.combo:
        print("[ERROR] --factor combo requires --combo")
        sys.exit(1)
    combo_list = [f.strip() for f in args.combo.split(",")] if args.combo else None

    start_ts = pd.Timestamp(args.start) if args.start else None
    end_ts = pd.Timestamp(args.end) if args.end else None

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"{args.factor}_{args.rebalance}"
        output_dir = Path("output") / "runs" / f"{ts}_replay_{suffix}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = get_universe(args.universe, n=30)
    print(f"[1/3] Fetching {len(symbols)} symbols ({args.period})...")
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df_by_symbol = _fetch_universe(symbols, "1d", args.period, fetcher)
    if len(df_by_symbol) < args.top_k + args.bottom_k:
        print(f"[ERROR] Need at least {args.top_k + args.bottom_k} symbols")
        sys.exit(1)
    print(f"      Loaded {len(df_by_symbol)} symbols")

    print("[2/3] Running replay...")
    result = _run_replay(
        df_by_symbol,
        strategy=args.strategy,
        factor=args.factor,
        combo_list=combo_list,
        combo_method=args.combo_method,
        start=start_ts,
        end=end_ts,
        rebalance=args.rebalance,
        initial_cash=args.initial_cash,
        fill_mode=args.fill_mode,
        cost_model=args.cost_model,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        output_dir=output_dir,
    )

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        sys.exit(1)

    print("[3/3] Results:")
    print("-" * 40)
    print(f"  Total Return:  {result['total_return']:.2%}")
    print(f"  Sharpe:        {result['sharpe']:.2f}")
    print(f"  Max Drawdown:  {result['max_drawdown']:.2%}")
    print(f"  Orders:        {result['n_orders']}")
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
