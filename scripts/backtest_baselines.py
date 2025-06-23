# scripts/backtest_baselines.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.baselines import (
    persistence_forecast,
    moving_average_forecast,
    arima_forecast,
)

def evaluate(true: pd.Series, pred: pd.Series, name: str):
    # align indices
    true, pred = true.align(pred, join="inner")
    
    # Check for NaN values and drop them
    if pred.isna().any():
        print(f"Warning: {name} contains {pred.isna().sum()} NaN values, dropping them")
        mask = ~(true.isna() | pred.isna())
        true, pred = true[mask], pred[mask]
    
    if len(true) == 0:
        print(f"{name:20s} | No valid data points available")
        return true, pred
    
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"{name:20s} | MAE: {mae:.4f}  RMSE: {rmse:.4f}  MAPE: {mape:.2f}%")
    return true, pred

def main():
    fetcher = YahooDataFetcher(max_retries=1, retry_delay=0)
    # 1 year of daily closes (use SPY as proxy)
    df = fetcher.fetch_ohlcv("SPY", "1d", period="365d")
    close = df["close"]

    print("=== Baseline Backtest on SPY daily close ===")
    # Persistence
    p_true, p_pred = evaluate(close.iloc[1:], persistence_forecast(close), "Persistence")
    # Moving avg (5-day)
    ma_true, ma_pred = evaluate(close.iloc[5:], moving_average_forecast(close, window=5), "MA (5)")
    # ARIMA (1,0,0) - use last 30 points for evaluation
    ar_forecast = arima_forecast(close, order=(1,0,0))
    ar_true, ar_pred = evaluate(close.iloc[-len(ar_forecast):], ar_forecast, "ARIMA(1,0,0)")

    # Plot last 100 days
    plt.figure(figsize=(10,4))
    plt.plot(close[-100:], label="Actual", lw=2)
    plt.plot(p_pred[-100:], "--", label="Persistence")
    plt.plot(ma_pred[-100:], "-.", label="MA(5)")
    # ARIMA only one point; skip plotting on timeseries
    plt.title("SPY Close & Baseline Forecasts (last 100 days)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/baseline_backtest.png")
    plt.show()

if __name__ == "__main__":
    main()