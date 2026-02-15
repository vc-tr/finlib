"""Tests for influencer, daytrader, and institutional strategies."""

import numpy as np
import pandas as pd
import pytest
from src.strategies import (
    SentimentStrategy,
    VolumeSentimentStrategy,
    ScalpingStrategy,
    OpeningRangeBreakoutStrategy,
    EmaStochasticStrategy,
    VWAPReversionStrategy,
    ATRBreakoutStrategy,
)
from src.backtest import Backtester


@pytest.fixture
def ohlcv_df():
    """Sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 200
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    volume = np.random.randint(1000000, 5000000, n)
    return pd.DataFrame({
        "open": close * 0.99,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_sentiment_strategy():
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
    sentiment = pd.Series(np.random.randn(100) * 0.5, index=prices.index)
    strat = SentimentStrategy(bullish_threshold=0.5, bearish_threshold=-0.5)
    signals, returns = strat.backtest_returns(sentiment, prices)
    assert len(signals) == len(prices)


def test_volume_sentiment_strategy(ohlcv_df):
    strat = VolumeSentimentStrategy(volume_multiplier=2.5)
    signals, returns = strat.backtest_returns(ohlcv_df["close"], ohlcv_df["volume"])
    assert len(signals) == len(ohlcv_df)


def test_scalping_strategy(ohlcv_df):
    strat = ScalpingStrategy(fast_ema=5, slow_ema=15)
    signals, returns = strat.backtest_returns(ohlcv_df["close"])
    assert len(signals) == len(ohlcv_df)


def test_orb_strategy(ohlcv_df):
    strat = OpeningRangeBreakoutStrategy(orb_bars=20)
    signals, returns = strat.backtest_returns(ohlcv_df)
    assert len(signals) == len(ohlcv_df)


def test_ema_stochastic_strategy(ohlcv_df):
    strat = EmaStochasticStrategy(ema_period=10, stoch_k=10)
    signals, returns = strat.backtest_returns(ohlcv_df)
    assert len(signals) == len(ohlcv_df)


def test_vwap_reversion_strategy(ohlcv_df):
    strat = VWAPReversionStrategy(entry_z=2.0, lookback=20)
    signals, returns = strat.backtest_returns(ohlcv_df)
    assert len(signals) == len(ohlcv_df)


def test_atr_breakout_strategy(ohlcv_df):
    strat = ATRBreakoutStrategy(atr_period=14, atr_multiplier=1.5)
    signals, returns = strat.backtest_returns(ohlcv_df)
    assert len(signals) == len(ohlcv_df)
