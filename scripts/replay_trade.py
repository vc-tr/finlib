#!/usr/bin/env python3
"""
Event-driven replay paper trading (deterministic historical, no broker).

Usage:
    python scripts/replay_trade.py --strategy momentum --universe liquid_etfs --start 2023-01-01 --end 2024-01-01
    python scripts/replay_trade.py --strategy factors --interval 1d --cost-model liquidity
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from src.factors import get_universe, compute_factor, UniverseRegistry, UniverseRegistry, UniverseRegistry
from src.paper import PaperExchange, PaperBroker, RiskManager
from src.paper.orders import OrderSide


def _fetch_bars(
    symbols: list[str],
    start: str,
    end: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for symbols with start/end dates."""
    bars = {}
    for sym in symbols:
        try:
            df = yf.download(
                sym,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if df.empty or len(df) < 5:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={
                "Open": "open", "High": "high",
                "Low": "low", "Close": "close",
                "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            bars[sym] = df
        except Exception as e:
            print(f"  [WARN] Skip {sym}: {e}")
    return bars


def _strategy_momentum(
    bars: dict[str, pd.DataFrame],
    timestamp: pd.Timestamp,
    broker: PaperBroker,
    exchange: PaperExchange,
    prices: dict[str, float],
) -> None:
    """Simple momentum: long top 1, short bottom 1 by 20d return."""
    closes = pd.DataFrame({s: df["close"] for s, df in bars.items()})
    ret_20 = closes.pct_change(20)
    mask = ret_20.index <= timestamp
    if mask.sum() < 21:
        return
    row = ret_20.loc[mask].iloc[-1]
    valid = row.dropna()
    if len(valid) < 2:
        return
    long_sym = valid.idxmax()
    short_sym = valid.idxmin()
    pv = broker.portfolio_value(prices)
    if pv <= 0:
        return
    target_val = pv * 0.25
    px_long = prices.get(long_sym, 0)
    px_short = prices.get(short_sym, 0)
    if px_long > 0:
        target_long = target_val / px_long
        pos = broker.positions.get(long_sym, 0)
        if target_long - pos > 0.01:
            broker.submit_order(long_sym, OrderSide.BUY, target_long - pos, prices=prices)
        elif pos - target_long > 0.01:
            broker.submit_order(long_sym, OrderSide.SELL, pos - target_long, prices=prices)
    if px_short > 0:
        target_short = -target_val / px_short
        pos = broker.positions.get(short_sym, 0)
        if target_short - pos < -0.01:
            broker.submit_order(short_sym, OrderSide.SELL, pos - target_short, prices=prices)
        elif target_short - pos > 0.01:
            broker.submit_order(short_sym, OrderSide.BUY, target_short - pos, prices=prices)


def _strategy_factors(
    df_by_symbol: dict[str, pd.DataFrame],
    timestamp: pd.Timestamp,
    broker: PaperBroker,
    exchange: PaperExchange,
    prices: dict[str, float],
    factor_df: pd.DataFrame,
    top_k: int = 2,
    bottom_k: int = 2,
) -> None:
    """Factor strategy: rank by factor, long top_k, short bottom_k."""
    if factor_df.empty or timestamp not in factor_df.index:
        return
    row = factor_df.loc[timestamp].dropna()
    if len(row) < top_k + bottom_k:
        return
    ranked = row.rank(ascending=False)
    longs = ranked.nsmallest(top_k).index.tolist()
    shorts = ranked.nlargest(bottom_k).index.tolist()
    pv = broker.portfolio_value(prices)
    if pv <= 0:
        return
    target_per_leg = pv * 0.5 / (top_k + bottom_k)  # 50% gross
    for sym in longs:
        px = prices.get(sym, 0)
        if px <= 0:
            continue
        target = target_per_leg / px
        pos = broker.positions.get(sym, 0)
        delta = target - pos
        if delta > 0.01:
            broker.submit_order(sym, OrderSide.BUY, delta, prices=prices)
    for sym in shorts:
        px = prices.get(sym, 0)
        if px <= 0:
            continue
        target = -target_per_leg / px
        pos = broker.positions.get(sym, 0)
        delta = target - pos
        if delta < -0.01:
            broker.submit_order(sym, OrderSide.SELL, abs(delta), prices=prices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-driven replay paper trading")
    parser.add_argument("--strategy", default="momentum", choices=["momentum", "factors"])
    parser.add_argument(
        "--universe",
        default="liquid_etfs",
        help=f"Universe name (choices: {', '.join(UniverseRegistry.list_names())})",
    )
    parser.add_argument("--list-universes", action="store_true", help="List available universes and exit")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (required unless --list-universes)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (required unless --list-universes)")
    parser.add_argument("--interval", default="1d", choices=["1d", "1m"])
    parser.add_argument("--cost-model", default="fixed", choices=["fixed", "liquidity"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-symbols", type=int, default=10)
    args = parser.parse_args()

    if args.list_universes:
        print("Available universes:")
        for name in UniverseRegistry.list_names():
            meta = UniverseRegistry.get_meta(name)
            symbols, _ = UniverseRegistry.get(name, n=5)
            preview = ", ".join(symbols) + ("..." if len(UniverseRegistry.get_symbols(name)) > 5 else "")
            print(f"  {name}: {meta.description} ({meta.category})")
            print(f"    Preview: {preview}")
        return

    if not args.start or not args.end:
        parser.error("--start and --end are required (or use --list-universes)")

    symbols = get_universe(args.universe, n=args.n_symbols)
    print(f"[1/4] Fetching {len(symbols)} symbols ({args.start} to {args.end}, {args.interval})...")
    bars = _fetch_bars(symbols, args.start, args.end, args.interval)
    if len(bars) < 2:
        print("[ERROR] Need at least 2 symbols")
        sys.exit(1)
    print(f"      Loaded {len(bars)} symbols")

    # Build df_by_symbol for factors
    df_by_symbol = {}
    for sym, df in bars.items():
        df_by_symbol[sym] = df.copy()

    # Cost model params
    fee_bps = 10.0
    slippage_bps = 5.0
    if args.cost_model == "liquidity":
        slippage_bps = 8.0

    exchange = PaperExchange(
        bars,
        cost_model=args.cost_model,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    risk = RiskManager(max_gross=2e6, max_net=1e6, max_single_weight=0.2)
    broker = PaperBroker(exchange, initial_cash=1_000_000.0, risk_manager=risk)

    # Precompute factor for factors strategy
    factor_df = pd.DataFrame()
    if args.strategy == "factors":
        print("[2/4] Computing factors...")
        factor_df = compute_factor(df_by_symbol, "momentum_12_1")

    print("[3/4] Replaying...")
    all_ts = exchange._all_timestamps()
    # Skip first 30 bars for warmup
    warmup = 30
    for i, ts in enumerate(all_ts):
        if i < warmup:
            continue
        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        prices = {s: exchange.get_close(s, ts_dt) or 0.0 for s in bars}
        prices = {s: p for s, p in prices.items() if p > 0}

        if args.strategy == "momentum":
            _strategy_momentum(bars, ts, broker, exchange, prices)
        else:
            _strategy_factors(df_by_symbol, ts, broker, exchange, prices, factor_df)

        fills = exchange.replay_bar(ts_dt)
        broker.process_fills(fills)
        broker.record_equity(ts_dt, prices)

    print("[4/4] Writing outputs...")
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("output") / "runs" / f"{ts}_replay_{args.strategy}"
    out.mkdir(parents=True, exist_ok=True)

    blotter = broker.blotter_df()
    blotter.to_csv(out / "blotter.csv", index=False)

    equity = broker.equity_curve_df()
    equity.to_csv(out / "equity_curve.csv", index=False)

    # Replay report
    eq = equity.set_index("timestamp")["equity"]
    ret = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 1e-10 else 0.0
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    max_dd = dd.min() if len(dd) > 0 else 0.0

    last_ts = all_ts[-1].to_pydatetime() if hasattr(all_ts[-1], "to_pydatetime") else all_ts[-1]
    gross = sum(abs(broker.positions.get(s, 0)) * (exchange.get_close(s, last_ts) or 0.0) for s in bars)
    pv = broker.portfolio_value({s: exchange.get_close(s, all_ts[-1].to_pydatetime()) or 0 for s in bars})

    report = f"""# Replay Report

## Summary

| Metric | Value |
|--------|-------|
| Total Return | {total_ret:.2%} |
| Sharpe (ann.) | {sharpe:.2f} |
| Max Drawdown | {max_dd:.2%} |
| Final Equity | {eq.iloc[-1]:,.0f} |
| Trades | {len(blotter)} |

## Exposures (final)

| Metric | Value |
|--------|-------|
| Gross | {gross:,.0f} |
| Net | {sum(broker.positions.get(s,0) * (exchange.get_close(s, all_ts[-1].to_pydatetime()) or 0) for s in bars):,.0f} |
| Cash | {broker.cash:,.0f} |

## Outputs

- [blotter.csv](blotter.csv) — trade log
- [equity_curve.csv](equity_curve.csv) — equity over time
"""
    (out / "replay_report.md").write_text(report, encoding="utf-8")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
