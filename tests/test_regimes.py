"""
Tests for src/research/regimes.py
"""

import numpy as np
import pandas as pd

from src.research.regimes import (
    volatility_regimes,
    hurst_regime,
    _hurst_rs,
    conditional_performance,
    top_drawdowns,
    DrawdownEvent,
    RegimeStats,
    format_regime_report,
    regime_report_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prices(n: int = 500, seed: int = 0, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    prices = pd.Series(
        100 * np.exp(np.cumsum(rets)),
        index=pd.bdate_range("2018-01-01", periods=n),
    )
    return prices


def _make_returns(
    n: int = 500, seed: int = 0, mean: float = 0.0002
) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mean, 0.01, n)
    return pd.Series(rets, index=pd.bdate_range("2018-01-01", periods=n))


# ---------------------------------------------------------------------------
# volatility_regimes
# ---------------------------------------------------------------------------

def test_vol_regimes_returns_series():
    prices = _make_prices()
    labels = volatility_regimes(prices)
    assert isinstance(labels, pd.Series)
    assert labels.name == "vol_regime"


def test_vol_regimes_index_alignment():
    prices = _make_prices(300)
    labels = volatility_regimes(prices, window=21)
    assert labels.index.equals(prices.index)


def test_vol_regimes_valid_values():
    prices = _make_prices(400)
    labels = volatility_regimes(prices, n_regimes=5)
    valid = labels.dropna()
    assert set(valid.unique()).issubset({1.0, 2.0, 3.0, 4.0, 5.0})


def test_vol_regimes_has_nans_at_start():
    """Early bars should be NaN (not enough history)."""
    prices = _make_prices(300)
    labels = volatility_regimes(prices, window=21)
    # First few should be NaN
    assert labels.iloc[:30].isna().all()


def test_vol_regimes_custom_n_regimes():
    prices = _make_prices(400)
    labels = volatility_regimes(prices, n_regimes=3)
    valid = labels.dropna()
    assert set(valid.unique()).issubset({1.0, 2.0, 3.0})


def test_vol_regimes_higher_vol_higher_regime():
    """Append a high-vol period; it should end up in regime 5."""
    rng = np.random.default_rng(42)
    # Low-vol first half
    low_rets = rng.normal(0, 0.005, 300)
    # High-vol second half
    high_rets = rng.normal(0, 0.05, 200)
    rets = np.concatenate([low_rets, high_rets])
    prices = pd.Series(
        100 * np.exp(np.cumsum(rets)),
        index=pd.bdate_range("2018-01-01", periods=500),
    )
    labels = volatility_regimes(prices, window=21, n_regimes=5)
    last_regime = labels.iloc[-10:].dropna().mean()
    first_half_mean = labels.iloc[100:300].dropna().mean()
    assert last_regime > first_half_mean


# ---------------------------------------------------------------------------
# _hurst_rs
# ---------------------------------------------------------------------------

def test_hurst_rs_random_walk_near_half():
    """Brownian motion should produce H ≈ 0.5."""
    rng = np.random.default_rng(99)
    log_p = np.cumsum(rng.normal(0, 1, 500))
    h = _hurst_rs(log_p)
    assert 0.3 < h < 0.7


def test_hurst_rs_output_in_range():
    rng = np.random.default_rng(1)
    log_p = np.cumsum(rng.normal(0, 1, 200))
    h = _hurst_rs(log_p)
    assert 0.0 <= h <= 1.0


def test_hurst_rs_too_short_returns_half():
    log_p = np.array([0.0, 0.1, 0.2])
    h = _hurst_rs(log_p, min_n=10)
    assert h == 0.5


def test_hurst_rs_multiple_series_output_in_range():
    """R/S should output valid H for various series types."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        log_p = np.cumsum(rng.normal(0, 1, 300))
        h = _hurst_rs(log_p)
        assert 0.0 <= h <= 1.0


# ---------------------------------------------------------------------------
# hurst_regime
# ---------------------------------------------------------------------------

def test_hurst_regime_returns_series():
    prices = _make_prices(300)
    labels = hurst_regime(prices, window=100)
    assert isinstance(labels, pd.Series)
    assert labels.name == "hurst_regime"


def test_hurst_regime_index_aligned():
    prices = _make_prices(300)
    labels = hurst_regime(prices, window=100)
    assert labels.index.equals(prices.index)


def test_hurst_regime_valid_values():
    prices = _make_prices(400)
    labels = hurst_regime(prices, window=126)
    valid = labels.dropna()
    assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})


def test_hurst_regime_nans_at_start():
    prices = _make_prices(300)
    labels = hurst_regime(prices, window=126)
    assert labels.iloc[:126].isna().all()


# ---------------------------------------------------------------------------
# conditional_performance
# ---------------------------------------------------------------------------

def test_conditional_performance_returns_dict():
    ret = _make_returns(500)
    prices = _make_prices(500)
    labels = volatility_regimes(prices, window=21, n_regimes=3)
    result = conditional_performance(ret, labels)
    assert isinstance(result, dict)


def test_conditional_performance_regime_stats_type():
    ret = _make_returns(500)
    prices = _make_prices(500)
    labels = volatility_regimes(prices, window=21, n_regimes=3)
    result = conditional_performance(ret, labels)
    for v in result.values():
        assert isinstance(v, RegimeStats)


def test_conditional_performance_n_days_positive():
    ret = _make_returns(500)
    prices = _make_prices(500)
    labels = volatility_regimes(prices, window=21, n_regimes=3)
    result = conditional_performance(ret, labels)
    for v in result.values():
        assert v.n_days > 0


def test_conditional_performance_sharpe_finite():
    ret = _make_returns(500)
    prices = _make_prices(500)
    labels = volatility_regimes(prices, window=21, n_regimes=3)
    result = conditional_performance(ret, labels)
    for v in result.values():
        assert np.isfinite(v.sharpe)


def test_conditional_performance_hit_rate_in_range():
    ret = _make_returns(500)
    prices = _make_prices(500)
    labels = volatility_regimes(prices, window=21, n_regimes=3)
    result = conditional_performance(ret, labels)
    for v in result.values():
        assert 0.0 <= v.hit_rate <= 1.0


def test_conditional_performance_min_obs_respected():
    """With very high min_obs, all regimes should be excluded."""
    ret = _make_returns(300)
    prices = _make_prices(300)
    labels = volatility_regimes(prices, window=21, n_regimes=3)
    result = conditional_performance(ret, labels, min_obs=100_000)
    assert result == {}


def test_conditional_performance_custom_names():
    ret = _make_returns(500)
    # Simple binary regime: 0 or 1
    labels = pd.Series(
        np.where(ret > 0, 1, 0), index=ret.index
    ).astype(float)
    names = {1.0: "Bull", 0.0: "Bear"}
    result = conditional_performance(ret, labels, regime_names=names)
    assert "Bull" in result or "Bear" in result


# ---------------------------------------------------------------------------
# top_drawdowns
# ---------------------------------------------------------------------------

def test_top_drawdowns_returns_list():
    ret = _make_returns(500)
    dds = top_drawdowns(ret, n=5)
    assert isinstance(dds, list)


def test_top_drawdowns_count():
    ret = _make_returns(500, seed=7)
    dds = top_drawdowns(ret, n=3)
    assert len(dds) <= 3


def test_top_drawdowns_sorted_worst_first():
    ret = _make_returns(500)
    dds = top_drawdowns(ret, n=5)
    depths = [d.depth for d in dds]
    assert depths == sorted(depths)  # most negative first


def test_top_drawdowns_depth_negative():
    ret = _make_returns(500, mean=-0.001)
    dds = top_drawdowns(ret, n=5)
    for d in dds:
        assert d.depth <= 0


def test_top_drawdowns_dates_non_empty():
    ret = _make_returns(500)
    dds = top_drawdowns(ret, n=3)
    for d in dds:
        assert d.peak_date != ""
        assert d.trough_date != ""


def test_top_drawdowns_duration_positive():
    ret = _make_returns(500, mean=-0.001, seed=8)
    dds = top_drawdowns(ret, n=5)
    for d in dds:
        assert d.duration_days >= 0


def test_top_drawdowns_event_dataclass():
    ret = _make_returns(300)
    dds = top_drawdowns(ret, n=2)
    if dds:
        d = dds[0]
        assert isinstance(d, DrawdownEvent)
        assert hasattr(d, "rank")
        assert hasattr(d, "underwater_days")


def test_top_drawdowns_all_negative_returns():
    """Strategy that only loses money → large single drawdown."""
    rets = pd.Series(
        -0.01 * np.ones(100),
        index=pd.bdate_range("2020-01-01", periods=100),
    )
    dds = top_drawdowns(rets, n=5)
    assert len(dds) >= 1
    assert dds[0].depth < -0.5  # at least 50% drawdown


# ---------------------------------------------------------------------------
# format_regime_report
# ---------------------------------------------------------------------------

def test_format_regime_report_markdown():
    ret = _make_returns(500)
    prices = _make_prices(500)
    labels = volatility_regimes(prices, window=21, n_regimes=3)
    vol_stats = conditional_performance(ret, labels)
    dds = top_drawdowns(ret, n=3)
    report = format_regime_report(vol_stats=vol_stats, drawdowns=dds)
    assert "## Regime Analysis" in report
    assert "Volatility Regimes" in report
    assert "Top Drawdowns" in report


def test_format_regime_report_empty():
    report = format_regime_report()
    assert "## Regime Analysis" in report


def test_format_regime_report_hurst_section():
    ret = _make_returns(500)
    prices = _make_prices(500)
    labels = hurst_regime(prices, window=100)
    hurst_stats = conditional_performance(ret, labels)
    report = format_regime_report(hurst_stats=hurst_stats)
    assert "Trend Regimes" in report


# ---------------------------------------------------------------------------
# regime_report_dict
# ---------------------------------------------------------------------------

def test_regime_report_dict_structure():
    ret = _make_returns(500)
    prices = _make_prices(500)
    labels = volatility_regimes(prices, window=21, n_regimes=3)
    vol_stats = conditional_performance(ret, labels)
    dds = top_drawdowns(ret, n=3)
    d = regime_report_dict(vol_stats=vol_stats, drawdowns=dds)
    assert "volatility_regimes" in d
    assert "top_drawdowns" in d
    assert isinstance(d["volatility_regimes"], list)
    assert isinstance(d["top_drawdowns"], list)


def test_regime_report_dict_empty():
    d = regime_report_dict()
    assert d == {}
