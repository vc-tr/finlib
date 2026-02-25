"""Tests for cross-sectional factor engine. Uses synthetic df_by_symbol."""

import numpy as np
import pandas as pd

from src.factors import compute_factor, cross_sectional_rank, build_portfolio, get_universe
from src.factors.factors import get_prices_wide
from src.factors.portfolio import apply_rebalance_costs


def _synthetic_df_by_symbol(n_dates: int = 300, n_symbols: int = 20, seed: int = 42) -> dict:
    """Create synthetic df_by_symbol for testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    df_by = {}
    for i in range(n_symbols):
        sym = f"S{i}"
        ret = rng.randn(n_dates) * 0.01
        close = 100 * (1 + pd.Series(ret, index=idx)).cumprod()
        df_by[sym] = pd.DataFrame({"close": close})
    return df_by


def test_get_universe() -> None:
    """Universe returns list of symbols."""
    syms = get_universe("liquid_etfs", n=10)
    assert len(syms) == 10
    assert "SPY" in syms
    assert "QQQ" in syms


def test_compute_factor_momentum() -> None:
    """Momentum factor has correct shape."""
    df_by = _synthetic_df_by_symbol(300, 15)
    factor_df = compute_factor(df_by, "momentum_12_1")
    assert factor_df.shape[0] <= 300
    assert factor_df.shape[1] == 15
    assert factor_df.notna().any().any()


def test_compute_factor_reversal() -> None:
    """Reversal factor has correct shape."""
    df_by = _synthetic_df_by_symbol(100, 10)
    factor_df = compute_factor(df_by, "reversal_5d")
    assert factor_df.shape[1] == 10


def test_compute_factor_lowvol() -> None:
    """Low-vol factor has correct shape."""
    df_by = _synthetic_df_by_symbol(100, 10)
    factor_df = compute_factor(df_by, "lowvol_20d")
    assert factor_df.shape[1] == 10


def test_cross_sectional_rank() -> None:
    """Ranking produces weights with correct structure."""
    df_by = _synthetic_df_by_symbol(100, 15)
    factor_df = compute_factor(df_by, "reversal_5d")
    weights = cross_sectional_rank(factor_df, top_k=3, bottom_k=3)
    assert weights.shape == factor_df.shape
    long_sum = (weights > 0).sum(axis=1)
    short_sum = (weights < 0).sum(axis=1)
    assert (long_sum <= 3).all() or (long_sum == 0).all()
    assert (short_sum <= 3).all() or (short_sum == 0).all()


def test_build_portfolio() -> None:
    """Portfolio returns have correct index."""
    df_by = _synthetic_df_by_symbol(100, 10)
    factor_df = compute_factor(df_by, "reversal_5d")
    weights = cross_sectional_rank(factor_df, top_k=2, bottom_k=2)
    prices = get_prices_wide(df_by)
    port_ret = build_portfolio(weights, prices, rebalance="M")
    assert len(port_ret) > 0
    assert port_ret.index.equals(prices.index)


def test_apply_rebalance_costs() -> None:
    """Costs reduce returns when turnover > 0."""
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    port_ret = pd.Series(0.001, index=idx)
    w = np.zeros((50, 3))
    w[:25, 0] = 0.5
    w[25:, 1] = 0.5
    weights = pd.DataFrame(w, index=idx, columns=["A", "B", "C"])
    after = apply_rebalance_costs(port_ret, weights, cost_bps=10)
    assert after.iloc[25] < port_ret.iloc[25]
