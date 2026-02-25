#!/usr/bin/env python3
"""
One-command demo: data download → strategy → backtest → tear-sheet.

Usage:
    python scripts/run_demo.py --symbol SPY --period 2y
    python scripts/run_demo.py --symbol SPY --period 30d --interval 1m
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.pipeline import reindex_and_backfill
from src.backtest import Backtester
from src.backtest.execution import ExecutionConfig, compute_turnover
from src.strategies import MomentumStrategy


# Period caps for minute data (Yahoo: 1m ~7d, 5m ~60d)
INTERVAL_PERIOD_CAP = {"1m": "7d", "5m": "60d", "1h": "730d"}


def _throttle_positions(positions: pd.Series, decision_interval_bars: int) -> pd.Series:
    """
    Only allow position changes at bars where index % K == 0.
    At other bars, hold the last decision. Still applies delay_bars downstream.
    """
    if decision_interval_bars <= 1:
        return positions
    arr = positions.values.copy()
    last = float(arr[0]) if len(arr) > 0 and not np.isnan(arr[0]) else 0.0
    for i in range(len(arr)):
        if i % decision_interval_bars == 0:
            last = arr[i]
        else:
            arr[i] = last
    return pd.Series(arr, index=positions.index)


def _parse_period_days(period: str) -> int:
    """Convert period string to approximate days (yfinance format)."""
    period = period.lower().strip()
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("wk"):
        return int(period[:-2]) * 5
    if period.endswith("mo"):
        return int(period[:-2]) * 21  # ~21 trading days/month
    if period.endswith("y"):
        return int(period[:-1]) * 252
    return 252  # unknown: assume long


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Forecast demo: data → strategy → backtest"
    )
    parser.add_argument("--config", help="Path to JSON config (overrides CLI)")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--period", default="2y", help="Data period (e.g. 30d, 2y)")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "1m", "5m"], help="Bar interval")
    parser.add_argument("--lookback", type=int, default=20, help="Momentum lookback")
    parser.add_argument("--min-hold-bars", type=int, default=None, help="Min bars to hold (default: 1 for 1d, 5 for 1m)")
    parser.add_argument("--decision-interval-bars", type=int, default=None, help="Only allow position changes every K bars (default: 1 for 1d, 15 for 1m)")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Fee in bps per trade")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage in bps")
    parser.add_argument("--spread-bps", type=float, default=1.0, help="Spread proxy in bps")
    parser.add_argument("--no-execution", action="store_true", help="Disable fees/slippage")
    parser.add_argument("--cost-sensitivity", action="store_true", default=True, help="Run with/without costs and report deltas (default True)")
    parser.add_argument("--no-cost-sensitivity", action="store_false", dest="cost_sensitivity", help="Skip cost sensitivity comparison")
    parser.add_argument("--use-cached-data", action="store_true", default=True, help="Use cached data if available (default True)")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache (always fetch)")
    parser.add_argument("--period-override", action="store_true", help="Allow period > cap for minute data")
    parser.add_argument("--force", action="store_true", help="Alias for --period-override (bypass period cap)")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        args.symbol = cfg.get("symbol", args.symbol)
        args.period = cfg.get("period", args.period)
        args.interval = cfg.get("interval", args.interval)
        sp = cfg.get("strategy_params", {})
        args.lookback = sp.get("lookback", args.lookback)
        args.min_hold_bars = sp.get("min_hold_bars", args.min_hold_bars)
        if "decision_interval_bars" in cfg:
            args.decision_interval_bars = cfg["decision_interval_bars"]
        ex = cfg.get("execution", {})
        if ex and not args.no_execution:
            args.fee_bps = ex.get("fee_bps", args.fee_bps)
            args.slippage_bps = ex.get("slippage_bps", args.slippage_bps)
            args.spread_bps = ex.get("spread_bps", args.spread_bps)

    args.period_override = args.period_override or args.force
    # Guardrail: cap period for minute bars
    if args.interval in INTERVAL_PERIOD_CAP and not args.period_override:
        cap = INTERVAL_PERIOD_CAP[args.interval]
        cap_days = _parse_period_days(cap)
        req_days = _parse_period_days(args.period)
        if req_days > cap_days:
            print(f"[WARN] {args.interval} data limited to ~{cap}; capping period from {args.period} to {cap}")
            args.period = cap

    # min_hold_bars default: 1 for daily, 5 for minute
    if args.min_hold_bars is None:
        args.min_hold_bars = 5 if args.interval in ("1m", "5m") else 1

    # decision_interval_bars default: 1 for daily, 15 for minute
    if args.decision_interval_bars is None:
        args.decision_interval_bars = 15 if args.interval in ("1m", "5m") else 1

    # 1) Data download
    print(f"[1/5] Fetching {args.symbol} ({args.period}, {args.interval})...")
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, args.interval, period=args.period)
    df = df.dropna()
    if df.empty or len(df) < 2:
        print(f"[ERROR] No data returned for {args.symbol} {args.interval} {args.period}. Try shorter period (e.g. 7d for 1m).")
        sys.exit(1)

    if args.interval in ("1m", "5m"):
        freq = "1min" if args.interval == "1m" else "5min"
        df = reindex_and_backfill(df, freq=freq)

    close = df["close"]
    print(f"      Loaded {len(close)} bars from {close.index[0]} to {close.index[-1]}")

    # 2) Strategy
    print(f"[2/5] Momentum (lookback={args.lookback}, min_hold={args.min_hold_bars})...")
    strategy = MomentumStrategy(lookback=args.lookback, min_hold_bars=args.min_hold_bars)
    positions, _ = strategy.backtest_returns(close)
    positions = _throttle_positions(positions, args.decision_interval_bars)

    # 3) Backtest
    print("[3/5] Backtesting...")
    # 1d: 252, 1h: 252*6.5, 1m/5m: 252*6.5*60
    annualization = (
        252 if args.interval == "1d"
        else 252 * 6.5 if args.interval == "1h"
        else 252 * 6.5 * 60
    )
    backtester = Backtester(annualization_factor=annualization)
    exec_config = None if args.no_execution else ExecutionConfig(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        execution_delay_bars=1,
    )
    exec_zero = ExecutionConfig(
        fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0, execution_delay_bars=1
    )

    result = backtester.run_from_signals(close, positions, execution_config=exec_config)
    result_zero = None
    if args.cost_sensitivity and exec_config is not None:
        result_zero = backtester.run_from_signals(close, positions, execution_config=exec_zero)

    # 4) Turnover
    pos_held = positions.shift(1).fillna(0)
    turnover = compute_turnover(pos_held)
    turnover_ann = float(turnover.mean() * annualization)

    # 5) Top 5 worst drawdowns
    cum = result.cumulative_returns.reindex(close.index).ffill().bfill().fillna(1.0)
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    dd_series = dd[dd < 0].copy()
    dd_series = dd_series.sort_values()
    top5_dd = dd_series.head(5)

    # 6) Recruiter-friendly output
    print("[4/5] Results:")
    print("-" * 55)
    print(f"  Interval:       {args.interval}")
    print(f"  Decision int.:  {args.decision_interval_bars} bars")
    if exec_config:
        print(f"  Execution:      fee={args.fee_bps}bps slip={args.slippage_bps}bps spread={args.spread_bps}bps delay_bars=1")
    else:
        print(f"  Execution:      (disabled)")
    print(f"  Sharpe (ann.):  {result.sharpe_ratio:.2f}")
    print(f"  Total Return:   {result.total_return:.2%}")
    if result_zero is not None:
        sharpe_delta = result.sharpe_ratio - result_zero.sharpe_ratio
        ret_delta = result.total_return - result_zero.total_return
        print(f"  [Cost sensitivity] Zero-cost: Sharpe={result_zero.sharpe_ratio:.2f} Return={result_zero.total_return:.2%}")
        print(f"  [Cost sensitivity] Delta:     Sharpe={sharpe_delta:+.2f} Return={ret_delta:+.2%}")
    print(f"  Max Drawdown:   {result.max_drawdown:.2%}")
    print(f"  Trades:         {result.n_trades}")
    print(f"  Turnover (ann): {turnover_ann:.2f}")
    print(f"  Win Rate:       {result.win_rate:.1%}")
    if len(top5_dd) > 0:
        print("  Top 5 worst drawdowns:")
        for i, (idx, v) in enumerate(top5_dd.items(), 1):
            print(f"    {i}. {v:.2%} @ {idx}")
    print("-" * 55)

    # 7) Tear-sheet
    print("[5/5] Writing tear-sheet...")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    from src.reporting.tearsheet import generate_tearsheet
    generate_tearsheet(
        result, close, positions, out_dir,
        annualization=annualization,
        config={
            "symbol": args.symbol,
            "period": args.period,
            "interval": args.interval,
            "lookback": args.lookback,
            "min_hold_bars": args.min_hold_bars,
            "decision_interval_bars": args.decision_interval_bars,
            "fee_bps": args.fee_bps,
            "slippage_bps": args.slippage_bps,
            "spread_bps": args.spread_bps,
        },
    )
    if result_zero is not None:
        summary_path = out_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["with_costs"] = {"sharpe": result.sharpe_ratio, "total_return": result.total_return}
        summary["zero_costs"] = {"sharpe": result_zero.sharpe_ratio, "total_return": result_zero.total_return}
        summary["delta"] = {
            "sharpe": result.sharpe_ratio - result_zero.sharpe_ratio,
            "total_return": result.total_return - result_zero.total_return,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved to {out_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
