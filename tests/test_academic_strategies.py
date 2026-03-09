"""
Tests for academic and econophysics strategies.

Validates signal shape, domain, registry completeness, and metadata.
Note: Hurst/entropy/OU strategies are slow (window fitting) — use short price series.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.registry import StrategyRegistry
from src.strategies.academic import (
    TimeSeriesMomentumStrategy,
    BettingAgainstBetaStrategy,
    LowVolAnomalyStrategy,
    CarryTradeStrategy,
    PostEarningsDriftStrategy,
)
from src.strategies.econophysics import (
    HurstExponentStrategy,
    EntropySignalStrategy,
    OrnsteinUhlenbeckStrategy,
    PowerLawTailStrategy,
)
from src.strategies.econophysics.ornstein_uhlenbeck import _fit_ou_params
from src.strategies.econophysics.power_law_tail import _hill_estimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_prices():
    np.random.seed(7)
    return pd.Series(100 * np.exp(np.cumsum(np.random.randn(600) * 0.01)))


@pytest.fixture
def short_prices():
    """Shorter series for slow econophysics strategies."""
    np.random.seed(7)
    return pd.Series(100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)))


def _assert_signals_valid(signals, prices):
    assert len(signals) == len(prices)
    valid = signals.dropna()
    assert set(valid.unique()).issubset({-1.0, 0.0, 1.0}), (
        f"Unexpected values: {set(valid.unique())}"
    )


# ---------------------------------------------------------------------------
# Academic strategies
# ---------------------------------------------------------------------------

def test_tsmom_signals(random_prices):
    strat = TimeSeriesMomentumStrategy(lookback_long=252, lookback_short=21)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_tsmom_registered():
    StrategyRegistry._load_all()
    assert "time_series_momentum" in StrategyRegistry.names()


def test_tsmom_binary(random_prices):
    """After warmup, TSMOM should be long or short (no flat when formation return != 0)."""
    strat = TimeSeriesMomentumStrategy(lookback_long=100, lookback_short=21)
    signals = strat.generate_signals(random_prices)
    # After warmup, expect mostly non-zero (random walk rarely has exact 0 return)
    assert (signals.iloc[110:] != 0).mean() > 0.9


def test_bab_signals(random_prices):
    strat = BettingAgainstBetaStrategy(beta_lookback=126, signal_lookback=21)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_bab_registered():
    StrategyRegistry._load_all()
    assert "betting_against_beta" in StrategyRegistry.names()


def test_low_vol_signals(random_prices):
    strat = LowVolAnomalyStrategy(vol_lookback=21, rank_lookback=126)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_low_vol_registered():
    StrategyRegistry._load_all()
    assert "low_vol_anomaly" in StrategyRegistry.names()


def test_carry_signals(random_prices):
    strat = CarryTradeStrategy(carry_lookback=126)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_carry_registered():
    StrategyRegistry._load_all()
    assert "carry_trade" in StrategyRegistry.names()


def test_pead_signals(random_prices):
    strat = PostEarningsDriftStrategy(surprise_threshold=0.02, drift_period=10)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_pead_registered():
    StrategyRegistry._load_all()
    assert "post_earnings_drift" in StrategyRegistry.names()


# ---------------------------------------------------------------------------
# Econophysics strategies
# ---------------------------------------------------------------------------

def test_hurst_signals(short_prices):
    strat = HurstExponentStrategy(hurst_window=63, signal_lookback=10)
    signals = strat.generate_signals(short_prices)
    _assert_signals_valid(signals, short_prices)


def test_hurst_registered():
    StrategyRegistry._load_all()
    assert "hurst_exponent" in StrategyRegistry.names()


def test_entropy_signals(short_prices):
    strat = EntropySignalStrategy(entropy_window=30, perm_order=3)
    signals = strat.generate_signals(short_prices)
    _assert_signals_valid(signals, short_prices)


def test_entropy_registered():
    StrategyRegistry._load_all()
    assert "entropy_signal" in StrategyRegistry.names()


def test_ou_signals(short_prices):
    strat = OrnsteinUhlenbeckStrategy(fit_window=100, entry_z=1.5)
    signals = strat.generate_signals(short_prices)
    _assert_signals_valid(signals, short_prices)


def test_ou_registered():
    StrategyRegistry._load_all()
    assert "ornstein_uhlenbeck" in StrategyRegistry.names()


def test_ou_flat_on_random_walk():
    """OU with min_theta > 0 should be mostly flat on a true random walk."""
    np.random.seed(99)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)))
    strat = OrnsteinUhlenbeckStrategy(fit_window=100, min_theta=0.05)
    signals = strat.generate_signals(prices)
    # Random walk should have θ ≈ 0, mostly filtered out
    assert (signals == 0).mean() > 0.5


def test_power_law_signals(short_prices):
    strat = PowerLawTailStrategy(alpha_window=100)
    signals = strat.generate_signals(short_prices)
    _assert_signals_valid(signals, short_prices)


def test_power_law_registered():
    StrategyRegistry._load_all()
    assert "power_law_tail" in StrategyRegistry.names()


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

def test_fit_ou_trending():
    """OU fit on strongly trending series → θ ≈ 0 (no reversion)."""
    np.random.seed(42)
    trend = np.log(np.linspace(100, 200, 200))
    theta, mu, sigma, half_life = _fit_ou_params(trend)
    # Near-unit-root series: b ≈ 1, θ ≈ 0
    assert theta < 0.1, f"Expected low theta for trending series, got {theta:.4f}"


def test_fit_ou_mean_reverting():
    """OU fit on stationary series → positive θ."""
    np.random.seed(42)
    # Simulate OU process: θ=0.1, μ=0, σ=0.01
    x = [0.0]
    for _ in range(299):
        x.append(x[-1] + 0.1 * (0 - x[-1]) + 0.01 * np.random.randn())
    log_prices = np.array(x) + 5  # shift to positive range
    theta, mu, sigma, half_life = _fit_ou_params(log_prices)
    assert theta > 0.0


def test_hill_estimator():
    """Hill estimator should return positive α for reasonable returns."""
    np.random.seed(42)
    returns = np.random.standard_t(df=3, size=500) * 0.01
    alpha = _hill_estimator(returns)
    assert 0.5 <= alpha <= 10.0


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

EXPECTED_ACADEMIC = {
    "time_series_momentum",
    "betting_against_beta",
    "low_vol_anomaly",
    "carry_trade",
    "post_earnings_drift",
}

EXPECTED_ECONOPHYSICS = {
    "hurst_exponent",
    "entropy_signal",
    "ornstein_uhlenbeck",
    "power_law_tail",
}


def test_all_academic_registered():
    StrategyRegistry._load_all()
    registered = set(StrategyRegistry.names())
    missing = EXPECTED_ACADEMIC - registered
    assert not missing, f"Academic strategies not registered: {missing}"


def test_all_econophysics_registered():
    StrategyRegistry._load_all()
    registered = set(StrategyRegistry.names())
    missing = EXPECTED_ECONOPHYSICS - registered
    assert not missing, f"Econophysics strategies not registered: {missing}"


def test_academic_meta_complete():
    StrategyRegistry._load_all()
    for meta in StrategyRegistry.list_by_category("academic"):
        assert meta.source_url, f"Missing source_url for {meta.name}"
        assert meta.hypothesis
        assert meta.expected_result


def test_econophysics_meta_complete():
    StrategyRegistry._load_all()
    for meta in StrategyRegistry.list_by_category("econophysics"):
        assert meta.hypothesis
        assert meta.expected_result


def test_total_strategy_count():
    """Ensure we have 20 total strategies registered."""
    StrategyRegistry._load_all()
    assert len(StrategyRegistry.names()) == 20
