#!/usr/bin/env python3
"""
Walk-forward evaluation: rolling OOS backtest with per-fold and aggregated metrics.

Usage:
    python scripts/walkforward_demo.py --symbol SPY --interval 1d --folds 6 --train-days 90 --test-days 30
    python scripts/walkforward_demo.py --symbol SPY --interval 1m --folds 4 --train-days 2 --test-days 1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import Backtester
from src.backtest.walkforward import run_walkforward
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.pipeline import reindex_and_backfill
from src.strategies import MomentumStrategy


from src.utils.io import cap_period_for_interval


def _strategy_factory(config: dict):
    return MomentumStrategy(
        lookback=config.get("lookback", 20),
        min_hold_bars=config.get("min_hold_bars", 1),
    )


def _backtest_factory(config: dict):
    return Backtester(annualization_factor=config.get("annualization", 252))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation: rolling OOS backtest"
    )
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h", "1m", "5m"])
    parser.add_argument("--period", default="2y", help="Data period (e.g. 30d, 2y)")
    parser.add_argument("--folds", type=int, default=6, help="Number of folds")
    parser.add_argument(
        "--train-days",
        type=int,
        default=None,
        help="Train window (default: 60 for 1d, 2 for 1m)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=None,
        help="Test window (default: 30 for 1d, 1 for 1m)",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=None,
        help="Step between folds (default: test-days)",
    )
    parser.add_argument(
        "--strategy",
        default="momentum",
        choices=["momentum"],
        help="Strategy (future-proof)",
    )
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--min-hold-bars", type=int, default=None)
    parser.add_argument("--decision-interval-bars", type=int, default=None)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--delay-bars", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: output/runs/<timestamp>_walkforward_<symbol>_<interval>/)",
    )
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--lock-timeout", type=float, default=0, metavar="SEC")
    parser.add_argument("--use-cached-data", action="store_true", default=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force", action="store_true", help="Bypass minute period cap")
    args = parser.parse_args()

    # Defaults by interval
    if args.interval == "1d":
        train_default, test_default = 60, 30
    elif args.interval == "1m":
        train_default, test_default = 2, 1
    else:
        train_default, test_default = 60, 30

    args.train_days = args.train_days or train_default
    args.test_days = args.test_days or test_default
    args.step_days = args.step_days or args.test_days

    if args.interval == "1m":
        args.lookback = args.lookback or 50
        args.min_hold_bars = args.min_hold_bars or 30
        args.decision_interval_bars = args.decision_interval_bars or 30
    else:
        args.lookback = args.lookback or 20
        args.min_hold_bars = args.min_hold_bars or 1
        args.decision_interval_bars = args.decision_interval_bars or 1

    # Period cap for minute
    new_period = cap_period_for_interval(args.interval, args.period, period_override=args.force)
    if new_period != args.period:
        args.period = new_period
        print(f"[WARN] {args.interval} capped to ~{args.period}; use --force to override")

    annualization = (
        252 if args.interval == "1d"
        else 252 * 6.5 if args.interval == "1h"
        else 252 * 6.5 * 60
    )

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / "runs" / f"{ts}_walkforward_{args.symbol}_{args.interval}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fetch data
    print(f"[1/4] Fetching {args.symbol} ({args.period}, {args.interval})...")
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, args.interval, period=args.period)
    df = df.dropna()
    if df.empty or len(df) < 2:
        print(f"[ERROR] No data for {args.symbol} {args.interval} {args.period}")
        sys.exit(1)

    if args.interval in ("1m", "5m"):
        freq = "1min" if args.interval == "1m" else "5min"
        df = reindex_and_backfill(df, freq=freq)

    print(f"      Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # 2) Config
    config = {
        "lookback": args.lookback,
        "min_hold_bars": args.min_hold_bars,
        "decision_interval_bars": args.decision_interval_bars,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "spread_bps": args.spread_bps,
        "delay_bars": args.delay_bars,
        "annualization": annualization,
    }

    # 3) Run walk-forward
    print(f"[2/4] Walk-forward ({args.folds} folds, train={args.train_days}d test={args.test_days}d)...")
    result = run_walkforward(
        df,
        _strategy_factory,
        _backtest_factory,
        folds=args.folds,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        config=config,
    )

    per_fold = result["per_fold"]
    agg = result["aggregated"]

    # 4) Write outputs
    print("[3/4] Writing outputs...")

    folds_df = pd.DataFrame(per_fold)
    folds_df.to_csv(output_dir / "walkforward_folds.csv", index=False)

    summary = {
        "symbol": args.symbol,
        "interval": args.interval,
        "period": args.period,
        "folds": args.folds,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "aggregated": agg,
        "config": config,
    }
    (output_dir / "walkforward_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # WALKFORWARD_REPORT.md
    report_lines = [
        "# Walk-Forward Report",
        "",
        f"**Symbol:** {args.symbol}  **Interval:** {args.interval}  **Period:** {args.period}",
        "",
        "## Anti-Lookahead",
        "- Train window ends before test window (no future data in calibration)",
        "- Signal at close t → fill at t+`delay_bars` (execution delay)",
        f"- `delay_bars={args.delay_bars}`",
        "",
        "## Folds",
        "",
    ]
    if not folds_df.empty:
        report_lines.append("| fold_idx | train_start | train_end | test_start | test_end | sharpe | total_return | max_drawdown | n_trades |")
        report_lines.append("|----------|-------------|-----------|------------|----------|--------|--------------|--------------|----------|")
        for _, r in folds_df.iterrows():
            report_lines.append(
                f"| {r['fold_idx']} | {r['train_start']} | {r['train_end']} | "
                f"{r['test_start']} | {r['test_end']} | {r['sharpe']:.2f} | "
                f"{r['total_return']:.2%} | {r['max_drawdown']:.2%} | {r['n_trades']} |"
            )
    report_lines.extend([
        "",
        "## Aggregated Metrics",
        "",
        f"- Mean Sharpe: {agg['mean_sharpe']:.2f}",
        f"- Median Sharpe: {agg['median_sharpe']:.2f}",
        f"- Mean Return: {agg['mean_return']:.2%}",
        f"- Worst Drawdown (across folds): {agg['worst_drawdown']:.2%}",
        f"- Aggregate OOS Sharpe: {agg['agg_sharpe']:.2f}",
        f"- Aggregate OOS Return: {agg['agg_total_return']:.2%}",
        f"- Aggregate Max DD: {agg['agg_max_drawdown']:.2%}",
        f"- Folds completed: {agg['n_folds']}",
        "",
    ])
    (output_dir / "WALKFORWARD_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    # 5) Console summary
    print("[4/4] Done.")
    print("-" * 55)
    print("Aggregated OOS:")
    print(f"  Mean Sharpe:    {agg['mean_sharpe']:.2f}")
    print(f"  Median Sharpe: {agg['median_sharpe']:.2f}")
    print(f"  Mean Return:   {agg['mean_return']:.2%}")
    print(f"  Worst DD:      {agg['worst_drawdown']:.2%}")
    print(f"  Agg Sharpe:    {agg['agg_sharpe']:.2f}")
    print(f"  Agg Return:    {agg['agg_total_return']:.2%}")
    print(f"  Folds:         {agg['n_folds']}")
    print("-" * 55)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--no-lock", action="store_true")
    _parser.add_argument("--lock-timeout", type=float, default=0)
    _pre, _ = _parser.parse_known_args()
    if _pre.no_lock:
        main()
    else:
        from src.utils.runlock import RunLock

        _lock_path = Path(__file__).resolve().parent.parent / ".runlock"
        with RunLock(lock_path=str(_lock_path), timeout_s=_pre.lock_timeout):
            main()
