"""
Paper trading replay runner.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.factors.factors import get_prices_wide
from src.factors.portfolio import rebalance_dates
from src.paper import PaperBroker, PaperExchange, RiskManager
from src.paper.orders import Order, OrderSide, OrderType
from src.paper.strategy_adapter import get_factor_target_weights


def run_replay(
    df_by_symbol: dict[str, pd.DataFrame],
    *,
    strategy: str = "factors",
    factor: str = "momentum_12_1",
    combo_list: list[str] | None = None,
    combo_method: str = "equal",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    rebalance: str = "M",
    initial_cash: float = 100_000.0,
    fill_mode: str = "next_close",
    cost_model: str = "fixed",
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    spread_bps: float = 1.0,
    top_k: int = 10,
    bottom_k: int = 10,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run paper trading replay. Returns summary dict with total_return, sharpe, max_drawdown, etc."""
    prices = get_prices_wide(df_by_symbol)
    if prices.empty:
        return {"error": "No price data"}

    all_idx = prices.index.sort_values()
    if start is not None:
        all_idx = all_idx[all_idx >= start]
    if end is not None:
        all_idx = all_idx[all_idx <= end]
    if len(all_idx) < 5:
        return {"error": "Insufficient data in date range"}

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
    broker = PaperBroker(exchange=exchange, initial_cash=initial_cash, risk_manager=risk)

    orders_log = []
    positions_snapshots = []

    for ts in all_idx:
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        prices_ts = {
            s: float(prices.loc[ts, s])
            for s in prices.columns
            if ts in prices.index and pd.notna(prices.loc[ts, s])
        }

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

        broker.record_equity(ts_dt, prices_ts)

        if ts in rb_set:
            pv = broker.portfolio_value(prices_ts)
            if pv <= 0:
                continue

            current_w = {}
            for sym, pos in broker.positions.items():
                pr = prices_ts.get(sym, 0)
                current_w[sym] = (pos * pr) / pv if pr > 0 else 0.0

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

            w_held = {
                s: (broker.positions.get(s, 0) * prices_ts.get(s, 0)) / pv if pv > 0 else 0
                for s in prices_ts
            }
            positions_snapshots.append({"timestamp": ts, **w_held})

    if output_dir is None:
        output_dir = Path("output") / "runs" / "replay_out"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "output_dir": output_dir,
    }
