#!/usr/bin/env python3
"""
Backtest quantitative strategies on historical data.

Usage:
    python scripts/backtest_strategies.py [--symbol SPY] [--period 365d]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.pipeline import reindex_and_backfill
from src.backtest import Backtester
from src.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    RenaissanceSignalEnsemble,
)


def main():
    parser = argparse.ArgumentParser(description="Backtest quant strategies")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--period", default="365d", help="Data period (e.g. 365d, 2y)")
    parser.add_argument("--interval", default="1d", help="Data interval (1d, 1h, 1m)")
    args = parser.parse_args()

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, args.interval, period=args.period)
    if args.interval == "1m":
        df = reindex_and_backfill(df)
    close = df["close"].dropna()

    backtester = Backtester(annualization_factor=252 if args.interval == "1d" else 252 * 6.5 * 60)

    strategies = {
        "Mean Reversion (z=2)": MeanReversionStrategy(lookback=20, entry_z=2.0, exit_z=0.5),
        "Momentum (20d)": MomentumStrategy(lookback=20),
        "Renaissance Ensemble": RenaissanceSignalEnsemble(min_signal_agreement=0.3),
    }

    print(f"\n=== Backtest: {args.symbol} ({args.period}, {args.interval}) ===\n")
    print(f"{'Strategy':<25} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Trades':>8}")
    print("-" * 65)

    for name, strategy in strategies.items():
        signals, strat_returns = strategy.backtest_returns(close)
        result = backtester.run_from_signals(close, signals)
        print(f"{name:<25} {result.sharpe_ratio:>8.2f} {result.total_return:>10.2%} {result.max_drawdown:>8.2%} {result.n_trades:>8}")


if __name__ == "__main__":
    main()
