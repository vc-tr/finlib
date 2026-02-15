#!/usr/bin/env python3
"""
Backtest pairs trading strategies (OLS and Kalman filter).

Usage:
    python scripts/backtest_pairs.py --symbol1 SPY --symbol2 IVV --period 365d
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.backtest import Backtester
from src.strategies import PairsTradingStrategy, KalmanPairsStrategy


def main():
    parser = argparse.ArgumentParser(description="Backtest pairs strategies")
    parser.add_argument("--symbol1", default="SPY", help="First symbol")
    parser.add_argument("--symbol2", default="IVV", help="Second symbol (e.g. IVV, QQQ)")
    parser.add_argument("--period", default="365d")
    args = parser.parse_args()

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df1 = fetcher.fetch_ohlcv(args.symbol1, "1d", period=args.period)
    df2 = fetcher.fetch_ohlcv(args.symbol2, "1d", period=args.period)
    a, b = df1["close"].align(df2["close"], join="inner")
    a, b = a.dropna(), b.dropna()

    backtester = Backtester(annualization_factor=252)

    print(f"\n=== Pairs Backtest: {args.symbol1} / {args.symbol2} ({args.period}) ===\n")

    # OLS pairs
    pairs = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, lookback=60)
    is_coint, pval = pairs.test_cointegration(a, b)
    print(f"Cointegration (Engle-Granger): p={pval:.4f}, cointegrated={is_coint}\n")
    signals, spread, hr = pairs.generate_signals(a, b)
    scale = a.rolling(20).mean().abs().replace(0, np.nan).ffill().bfill()
    spread_ret = (signals.shift(1) * spread.diff()) / scale
    spread_ret = spread_ret.fillna(0).clip(-0.1, 0.1)
    result = backtester.run(spread_ret)
    print(f"{'OLS Pairs':<20} Sharpe: {result.sharpe_ratio:.2f}  Return: {result.total_return:.2%}  MaxDD: {result.max_drawdown:.2%}")

    # Kalman pairs
    try:
        kalman = KalmanPairsStrategy(entry_z=2.0, exit_z=0.5)
        signals_k, returns_k = kalman.backtest_returns(a, b)
        result_k = backtester.run(returns_k)
        print(f"{'Kalman Pairs':<20} Sharpe: {result_k.sharpe_ratio:.2f}  Return: {result_k.total_return:.2%}  MaxDD: {result_k.max_drawdown:.2%}")
    except Exception as e:
        print(f"Kalman Pairs: {e} (install pykalman for full support)")


if __name__ == "__main__":
    main()
