"""Tests for quantitative strategies."""

import numpy as np
import pandas as pd
import pytest
from src.strategies import (
    BlackScholes,
    MonteCarloPricer,
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    MeanReversionStrategy,
    MomentumStrategy,
)
from src.backtest import Backtester


def test_black_scholes():
    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    call = bs.call_price()
    put = bs.put_price()
    assert call > 0 and put > 0
    assert bs.delta("call") > 0 and bs.delta("call") < 1
    assert bs.gamma() > 0


def test_monte_carlo():
    mc = MonteCarloPricer(S=100, K=100, T=1, r=0.05, sigma=0.2, n_paths=1000, seed=42)
    price = mc.price("european", "call")
    assert 5 < price < 20


def test_gbm_simulation():
    gbm = GeometricBrownianMotion(S0=100, mu=0.05, sigma=0.2, seed=42)
    paths = gbm.simulate(T=1, n_steps=252, n_paths=10)
    assert paths.shape == (10, 253)
    assert np.all(paths > 0)


def test_ou_simulation():
    ou = OrnsteinUhlenbeck(theta=1, mu=0, sigma=0.5, x0=1, seed=42)
    paths = ou.simulate(T=1, n_steps=100, n_paths=5)
    assert paths.shape == (5, 101)


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
