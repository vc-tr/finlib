# tests/test_data_fetcher.py
import pandas as pd
import pytest
import yfinance as yf
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.data_fetcher_yahoo import YahooDataFetcher

@pytest.fixture
def yahoo_fetcher():
    return YahooDataFetcher(max_retries=2, retry_delay=0)

def test_yahoo_fetch_returns_dataframe(yahoo_fetcher):
    df = yahoo_fetcher.fetch_ohlcv("SPY", "1m", period="1d")
    assert isinstance(df, pd.DataFrame)
    # Must have correct columns
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    # Index must be datetime
    assert pd.api.types.is_datetime64_any_dtype(df.index)

def test_yahoo_retry_logic(monkeypatch, yahoo_fetcher):
    calls = {"n": 0}
    def fake_download(symbol, interval, period, progress, auto_adjust=True, threads=True):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("Rate limit simulated")
        # Return minimal DataFrame
        return pd.DataFrame({
            "Open": [1], "High": [2], "Low": [0.5],
            "Close": [1.5], "Volume": [100]
        }, index=pd.to_datetime(["2025-01-01 00:00"]))
    monkeypatch.setattr(yf, "download", fake_download)

    df = yahoo_fetcher.fetch_ohlcv("SPY", "1m", period="1d")
    assert calls["n"] == 2
    # Columns should be renamed
    assert "open" in df.columns and df["open"].iloc[0] == 1

