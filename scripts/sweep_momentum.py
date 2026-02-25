#!/usr/bin/env python3
"""
Momentum parameter sweep: run backtests for each combo of lookback, min_hold, decision_interval.

Usage:
    python scripts/sweep_momentum.py --symbol SPY --period 2y
    python scripts/sweep_momentum.py --symbol SPY --period 7d --interval 1m --lookbacks "10,20,50" --min_holds "1,5,15" --decision_intervals "1,15"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import Backtester
from src.backtest.execution import ExecutionConfig, compute_turnover, throttle_positions
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.pipeline import reindex_and_backfill
from src.strategies import MomentumStrategy


INTERVAL_PERIOD_CAP = {"1m": "7d", "5m": "60d", "1h": "730d"}


def _parse_period_days(period: str) -> int:
    period = period.lower().strip()
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("wk"):
        return int(period[:-2]) * 5
    if period.endswith("mo"):
        return int(period[:-2]) * 21
    if period.endswith("y"):
        return int(period[:-1]) * 252
    return 252


def main() -> None:
    parser = argparse.ArgumentParser(description="Momentum parameter sweep")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--period", default="2y", help="Data period")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "1m", "5m"], help="Bar interval")
    parser.add_argument("--lookbacks", default="10,20,50", help="Comma-separated lookbacks")
    parser.add_argument("--min_holds", default="1,5,15", help="Comma-separated min_hold_bars")
    parser.add_argument("--decision_intervals", default="1,15", help="Comma-separated decision_interval_bars")
    parser.add_argument("--force", action="store_true", help="Allow period > cap for 1m (e.g. >7d)")
    args = parser.parse_args()

    lookbacks = [int(x.strip()) for x in args.lookbacks.split(",")]
    min_holds = [int(x.strip()) for x in args.min_holds.split(",")]
    decision_intervals = [int(x.strip()) for x in args.decision_intervals.split(",")]

    # Cap period for 1m unless --force
    if args.interval == "1m" and not args.force:
        cap_days = _parse_period_days(INTERVAL_PERIOD_CAP["1m"])
        req_days = _parse_period_days(args.period)
        if req_days > cap_days:
            args.period = INTERVAL_PERIOD_CAP["1m"]
            print(f"[WARN] 1m capped to {args.period} (use --force to override)")

    # Fetch data once
    print(f"Fetching {args.symbol} ({args.period}, {args.interval})...")
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, args.interval, period=args.period)
    df = df.dropna()
    if df.empty or len(df) < 2:
        print(f"[ERROR] No data for {args.symbol} {args.interval} {args.period}")
        sys.exit(1)

    if args.interval in ("1m", "5m"):
        freq = "1min" if args.interval == "1m" else "5min"
        df = reindex_and_backfill(df, freq=freq)

    close = df["close"]
    print(f"Loaded {len(close)} bars")

    annualization = (
        252 if args.interval == "1d"
        else 252 * 6.5 if args.interval == "1h"
        else 252 * 6.5 * 60
    )
    backtester = Backtester(annualization_factor=annualization)
    exec_config = ExecutionConfig(fee_bps=1.0, slippage_bps=2.0, spread_bps=1.0, execution_delay_bars=1)

    rows = []
    combos = [
        (lb, mh, di)
        for lb in lookbacks
        for mh in min_holds
        for di in decision_intervals
    ]
    for i, (lookback, min_hold, decision_interval) in enumerate(combos):
        strategy = MomentumStrategy(lookback=lookback, min_hold_bars=min_hold)
        positions, _ = strategy.backtest_returns(close)
        positions = throttle_positions(positions, decision_interval)
        result = backtester.run_from_signals(close, positions, execution_config=exec_config)
        pos_held = positions.shift(1).fillna(0)
        turnover_ann = float(compute_turnover(pos_held).mean() * annualization)
        rows.append({
            "lookback": lookback,
            "min_hold": min_hold,
            "decision_interval": decision_interval,
            "sharpe": result.sharpe_ratio,
            "total_return": result.total_return,
            "max_dd": result.max_drawdown,
            "trades": result.n_trades,
            "turnover": turnover_ann,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(combos):
            print(f"  {i + 1}/{len(combos)} combos done")

    # Save CSV
    out_dir = Path("output/sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"momentum_{ts}.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    # Top 5 by Sharpe and by total return
    by_sharpe = df_out.sort_values("sharpe", ascending=False).head(5)
    by_return = df_out.sort_values("total_return", ascending=False).head(5)
    print("\nTop 5 by Sharpe:")
    for _, r in by_sharpe.iterrows():
        print(f"  lookback={r['lookback']} min_hold={r['min_hold']} dec_int={r['decision_interval']}  Sharpe={r['sharpe']:.2f} Return={r['total_return']:.2%}")
    print("\nTop 5 by Total Return:")
    for _, r in by_return.iterrows():
        print(f"  lookback={r['lookback']} min_hold={r['min_hold']} dec_int={r['decision_interval']}  Sharpe={r['sharpe']:.2f} Return={r['total_return']:.2%}")


if __name__ == "__main__":
    _no_lock = "--no-lock" in sys.argv
    if _no_lock:
        sys.argv.remove("--no-lock")
    if _no_lock:
        main()
    else:
        from src.utils.runlock import RunLock
        with RunLock():
            main()
