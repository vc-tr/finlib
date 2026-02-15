"""Tests for academic paper strategies."""

import numpy as np
import pandas as pd
import pytest
from src.strategies import (
    MoskowitzTimeSeriesMomentum,
    JegadeeshTitmanMomentum,
    GatevGoetzmannRouwenhorstPairs,
    DeBondtThalerReversal,
)


@pytest.fixture
def long_prices():
    """~1000 days for De Bondt-Thaler."""
    np.random.seed(42)
    return pd.Series(100 * np.exp(np.cumsum(np.random.randn(1000) * 0.015)))


def test_moskowitz_tsmom(long_prices):
    strat = MoskowitzTimeSeriesMomentum(formation_period=252, holding_period=21)
    signals, returns = strat.backtest_returns(long_prices)
    assert len(signals) == len(long_prices)
    assert set(signals.dropna().unique()).issubset({-1, 0, 1})


def test_de_bondt_thaler(long_prices):
    strat = DeBondtThalerReversal(formation_period=252, holding_period=252)
    signals, returns = strat.backtest_returns(long_prices)
    assert len(signals) == len(long_prices)


def test_gatev_pairs():
    np.random.seed(42)
    n = 300
    base = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    a = pd.Series(base)
    b = pd.Series(base * 0.99 + np.random.randn(n) * 0.5)
    strat = GatevGoetzmannRouwenhorstPairs(formation_period=60, entry_std=2.0)
    signals, returns = strat.backtest_returns(a, b)
    assert len(signals) == n


def test_jegadeesh_titman():
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(300, 5) * 0.01,
        columns=["A", "B", "C", "D", "E"],
    )
    strat = JegadeeshTitmanMomentum(formation_period=60, holding_period=21, n_long=1, n_short=1)
    port_ret, _ = strat.backtest_returns(returns)
    assert len(port_ret) == len(returns)
