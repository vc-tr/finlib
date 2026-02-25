#!/usr/bin/env python3
"""
Backtest quantitative strategies on historical data.

Usage:
    python scripts/backtest_strategies.py [--symbol SPY] [--period 365d]
    python scripts/backtest_strategies.py --symbol SPY --period 365d --all  # run all strategies
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
    VolumeSentimentStrategy,
    ScalpingStrategy,
    OpeningRangeBreakoutStrategy,
    EmaStochasticStrategy,
    VWAPReversionStrategy,
    ATRBreakoutStrategy,
)


def main():
    parser = argparse.ArgumentParser(description="Backtest quant strategies")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--period", default="365d", help="Data period (e.g. 365d, 2y)")
    parser.add_argument("--interval", default="1d", help="Data interval (1d, 1h, 1m)")
    parser.add_argument("--all", action="store_true", help="Run all strategies including OHLCV-based")
    args = parser.parse_args()

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, args.interval, period=args.period)
    if args.interval in ("1m", "5m"):
        freq = "1min" if args.interval == "1m" else "5min"
        df = reindex_and_backfill(df, freq=freq)
    df = df.dropna()
    close = df["close"]

    backtester = Backtester(annualization_factor=252 if args.interval == "1d" else 252 * 6.5 * 60)

    # Price-only strategies
    strategies = {
        "Mean Reversion (z=2)": (
            MeanReversionStrategy(lookback=20, entry_z=2.0, exit_z=0.5),
            lambda s: s.backtest_returns(close),
        ),
        "Momentum (20d)": (
            MomentumStrategy(lookback=20),
            lambda s: s.backtest_returns(close),
        ),
        "Renaissance Ensemble": (
            RenaissanceSignalEnsemble(min_signal_agreement=0.3),
            lambda s: s.backtest_returns(close),
        ),
    }

    if args.all:
        strategies.update({
            "Volume Sentiment": (
                VolumeSentimentStrategy(volume_multiplier=2.0),
                lambda s: s.backtest_returns(close, df["volume"]),
            ),
            "Scalping (EMA)": (
                ScalpingStrategy(fast_ema=9, slow_ema=21),
                lambda s: s.backtest_returns(close),
            ),
            "ORB (15 bars)": (
                OpeningRangeBreakoutStrategy(orb_bars=15),
                lambda s: s.backtest_returns(df),
            ),
            "EMA+Stochastic": (
                EmaStochasticStrategy(),
                lambda s: s.backtest_returns(df),
            ),
            "VWAP Reversion": (
                VWAPReversionStrategy(entry_z=2.0),
                lambda s: s.backtest_returns(df),
            ),
            "ATR Breakout": (
                ATRBreakoutStrategy(atr_period=14, atr_multiplier=2.0),
                lambda s: s.backtest_returns(df),
            ),
        })

    print(f"\n=== Backtest: {args.symbol} ({args.period}, {args.interval}) ===\n")
    print(f"{'Strategy':<25} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Trades':>8}")
    print("-" * 65)

    for name, (strategy, run_fn) in strategies.items():
        try:
            signals, strat_returns = run_fn(strategy)
            result = backtester.run_from_signals(close, signals)
            print(f"{name:<25} {result.sharpe_ratio:>8.2f} {result.total_return:>10.2%} {result.max_drawdown:>8.2%} {result.n_trades:>8}")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")


if __name__ == "__main__":
    main()
