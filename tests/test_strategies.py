"""Tests for quantitative strategies."""

import numpy as np
import pandas as pd
import pytest
from src.strategies import MeanReversionStrategy, MomentumStrategy
from src.backtest import Backtester


def test_mean_reversion_signals():
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5))
    mr = MeanReversionStrategy(lookback=20, entry_z=2.0)
    signals, returns = mr.backtest_returns(prices)
    assert len(signals) == len(prices)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})


def test_momentum_signals():
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.02)))
    mom = MomentumStrategy(lookback=10)
    signals, returns = mom.backtest_returns(prices)
    assert len(signals) == len(prices)


def test_backtester():
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100) * 0.01)
    bt = Backtester()
    result = bt.run(returns)
    assert result.sharpe_ratio is not None
    assert -1 <= result.max_drawdown <= 1
