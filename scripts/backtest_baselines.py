#!/usr/bin/env python3
"""
Backtest baseline forecasting models (persistence, MA, ARIMA).

Usage:
    python scripts/backtest_baselines.py [--symbol SPY] [--period 365d]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.baselines import (
    persistence_forecast,
    moving_average_forecast,
    arima_forecast,
)


def evaluate(true: pd.Series, pred: pd.Series, name: str):
    true, pred = true.align(pred, join="inner")
    mask = ~(true.isna() | pred.isna())
    true, pred = true[mask], pred[mask]
    if len(true) == 0:
        print(f"{name:20s} | No valid data")
        return
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    print(f"{name:20s} | MAE: {mae:.4f}  RMSE: {rmse:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--period", default="365d")
    args = parser.parse_args()

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, "1d", period=args.period)
    close = df["close"]

    print(f"\n=== Baseline Forecasts: {args.symbol} ({args.period}) ===\n")
    evaluate(close.iloc[1:], persistence_forecast(close), "Persistence")
    evaluate(close.iloc[5:], moving_average_forecast(close, window=5), "MA(5)")
    ar = arima_forecast(close, order=(1, 0, 0))
    evaluate(close.iloc[-len(ar) :], ar, "ARIMA(1,0,0)")


if __name__ == "__main__":
    main()
