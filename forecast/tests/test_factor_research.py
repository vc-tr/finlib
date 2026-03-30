"""Tests for src/factors/research.py (IC/IR, forward returns). No network."""

import numpy as np
import pandas as pd

from src.factors.research import (
    forward_returns,
    cross_sectional_ic,
    summarize_ic,
)


def _synthetic_prices(n_dates: int = 100, n_symbols: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create prices_wide (date x symbol) for testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]
    ret = rng.randn(n_dates, n_symbols) * 0.01
    close = 100 * (1 + pd.DataFrame(ret, index=idx, columns=cols)).cumprod()
    return close


def test_forward_returns_shape():
    """forward_returns returns dict of (date x symbol) DataFrames."""
    prices = _synthetic_prices(50, 10)
    fwd = forward_returns(prices, horizons=[1, 5, 21])
    assert 1 in fwd and 5 in fwd and 21 in fwd
    for h, df in fwd.items():
        assert df.shape == prices.shape
        assert df.index.equals(prices.index)
        assert list(df.columns) == list(prices.columns)


def test_forward_returns_h1():
    """h=1 forward return: (price[t+1]/price[t]) - 1."""
    prices = _synthetic_prices(20, 3)
    fwd = forward_returns(prices, horizons=[1])
    expected = prices.shift(-1) / prices - 1
    pd.testing.assert_frame_equal(fwd[1], expected)


def test_factor_predicts_returns_positive_ic():
    """factor = future_return + noise => positive mean IC."""
    n_dates, n_symbols = 80, 10
    rng = np.random.RandomState(42)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]

    # Future 5d return (known at t)
    ret = rng.randn(n_dates, n_symbols) * 0.01
    close = 100 * (1 + pd.DataFrame(ret, index=idx, columns=cols)).cumprod()
    fwd_5 = close.shift(-5) / close - 1

    # Factor = future return + small noise (predictive)
    noise = rng.randn(n_dates, n_symbols) * 0.001
    factor_df = fwd_5 + noise

    ic_s = cross_sectional_ic(factor_df, fwd_5, method="spearman")
    summary = summarize_ic(ic_s)
    assert summary["n"] > 0
    assert summary["mean_ic"] > 0.3  # Strong predictive relationship
    assert summary["ir"] > 0


def test_constant_factor_ic_all_nan():
    """Constant factor => IC all NaN, no crash."""
    prices = _synthetic_prices(50, 10)
    fwd = forward_returns(prices, horizons=[1])[1]
    factor_df = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    ic_s = cross_sectional_ic(factor_df, fwd, method="spearman")
    assert ic_s.isna().all()
    summary = summarize_ic(ic_s)
    assert summary["n"] == 0
    assert np.isnan(summary["mean_ic"])
    assert np.isnan(summary["ir"])
    assert np.isnan(summary["t_stat"])


def test_summarize_ic_n0():
    """summarize_ic handles n==0 (empty or all NaN)."""
    s = pd.Series(dtype=float)
    out = summarize_ic(s)
    assert out["n"] == 0
    assert np.isnan(out["mean_ic"])
    assert np.isnan(out["t_stat"])

    s2 = pd.Series([np.nan, np.nan])
    out2 = summarize_ic(s2)
    assert out2["n"] == 0


def test_cross_sectional_ic_pearson():
    """Pearson method produces expected correlation."""
    n_dates, n_symbols = 30, 8
    rng = np.random.RandomState(7)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]
    factor_df = pd.DataFrame(rng.randn(n_dates, n_symbols), index=idx, columns=cols)
    fwd_df = factor_df * 0.5 + rng.randn(n_dates, n_symbols) * 0.1  # Correlated
    ic_s = cross_sectional_ic(factor_df, fwd_df, method="pearson")
    assert ic_s.notna().sum() > 0
    summary = summarize_ic(ic_s)
    assert summary["mean_ic"] > 0


def test_cross_sectional_ic_few_symbols():
    """< 5 symbols => IC NaN for that date."""
    prices = _synthetic_prices(20, 4)  # Only 4 symbols
    fwd = forward_returns(prices, horizons=[1])[1]
    factor_df = prices.pct_change(5)  # Momentum-like
    ic_s = cross_sectional_ic(factor_df, fwd, method="spearman")
    assert ic_s.isna().all()  # Never enough symbols
    summary = summarize_ic(ic_s)
    assert summary["n"] == 0
