#!/usr/bin/env python3
"""
Walk-forward evaluation: rolling OOS backtest.

Usage:
    python scripts/walkforward.py --symbol SPY --period 2y
    python scripts/walkforward.py --symbol SPY --period 2y --train-days 252 --test-days 63
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.backtest import Backtester
from src.backtest.walkforward import run_walkforward
from src.strategies import MomentumStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--period", default="2y", help="Data period")
    parser.add_argument("--train-days", type=int, default=252, help="Train window")
    parser.add_argument("--test-days", type=int, default=63, help="Test window")
    parser.add_argument("--step-days", type=int, default=None, help="Step between folds")
    parser.add_argument("--lookback", type=int, default=20, help="Momentum lookback")
    args = parser.parse_args()

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, "1d", period=args.period)
    prices = df["close"].dropna()

    strategy = MomentumStrategy(lookback=args.lookback)

    def strategy_fn(p: pd.Series) -> tuple:
        return strategy.backtest_returns(p)

    folds, agg_result = run_walkforward(
        prices,
        strategy_fn,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )

    print(f"\n=== Walk-Forward: {args.symbol} ({args.period}) ===\n")
    print(f"Folds: {len(folds)}")
    for i, f in enumerate(folds):
        if f.result:
            print(f"  Fold {i+1}: {f.test_start.date()} - {f.test_end.date()}  "
                  f"Sharpe={f.result.sharpe_ratio:.2f}  Return={f.result.total_return:.2%}")

    print(f"\nAggregate OOS:")
    print(f"  Sharpe: {agg_result.sharpe_ratio:.2f}")
    print(f"  Return: {agg_result.total_return:.2%}")
    print(f"  MaxDD:  {agg_result.max_drawdown:.2%}")


if __name__ == "__main__":
    main()
