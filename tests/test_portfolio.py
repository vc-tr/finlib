"""Tests for portfolio module."""

import numpy as np
import pandas as pd
import pytest
from src.portfolio import PortfolioAllocator, AllocationMethod, MultiStrategyPortfolio
from src.strategies import MeanReversionStrategy, MomentumStrategy
from src.backtest import Backtester


@pytest.fixture
def strategy_returns():
    """Simulated strategy returns."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "A": np.random.randn(n) * 0.01,
        "B": np.random.randn(n) * 0.015,
        "C": np.random.randn(n) * 0.008,
    })


def test_equal_allocation(strategy_returns):
    alloc = PortfolioAllocator(method=AllocationMethod.EQUAL, rebalance_freq=0)
    weights = alloc.allocate(strategy_returns)
    assert weights.shape == strategy_returns.shape
    np.testing.assert_array_almost_equal(weights.iloc[0].values, [1/3, 1/3, 1/3])


def test_inverse_vol_allocation(strategy_returns):
    alloc = PortfolioAllocator(method=AllocationMethod.INVERSE_VOLATILITY, lookback=50, rebalance_freq=0)
    weights = alloc.allocate(strategy_returns)
    # Weights should sum to 1 (after sufficient lookback)
    assert abs(weights.iloc[-1].sum() - 1.0) < 1e-6


def test_portfolio_returns(strategy_returns):
    alloc = PortfolioAllocator(method=AllocationMethod.EQUAL)
    port_ret = alloc.portfolio_returns(strategy_returns)
    assert len(port_ret) == len(strategy_returns)
    expected = strategy_returns.mean(axis=1)
    np.testing.assert_array_almost_equal(port_ret.values, expected.values)


def test_multi_strategy_portfolio():
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(200) * 0.02)))
    portfolio = MultiStrategyPortfolio()
    portfolio.add_strategy("MR", MeanReversionStrategy(lookback=20), lambda s: s.backtest_returns(prices))
    portfolio.add_strategy("Mom", MomentumStrategy(lookback=20), lambda s: s.backtest_returns(prices))
    returns_df, weights_df, port_returns = portfolio.run()
    assert len(port_returns) == len(prices)
    assert returns_df.shape[1] == 2
