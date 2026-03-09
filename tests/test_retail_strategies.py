"""
Tests for retail / technical analysis strategies.

Validates: signal shape, value domain {-1, 0, 1}, no lookahead (NaN handling),
and that each strategy is registered in StrategyRegistry.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.registry import StrategyRegistry
from src.strategies.retail import (
    BollingerBreakoutStrategy,
    DojiReversalStrategy,
    GapAndGoStrategy,
    GoldenCrossStrategy,
    MACDCrossoverStrategy,
    RSIOverboughtStrategy,
    ThreeBarReversalStrategy,
    VolumeSpikeStrategy,
)


@pytest.fixture
def random_prices():
    np.random.seed(42)
    log_ret = np.random.randn(500) * 0.01
    prices = pd.Series(100 * np.exp(np.cumsum(log_ret)), name="close")
    return prices


@pytest.fixture
def trending_prices():
    """Strongly trending series — useful for testing trend-following strategies."""
    np.random.seed(0)
    trend = np.linspace(100, 200, 500)
    noise = np.random.randn(500) * 0.5
    return pd.Series(trend + noise)


def _assert_signals_valid(signals: pd.Series, prices: pd.Series):
    assert len(signals) == len(prices)
    valid = signals.dropna()
    assert set(valid.unique()).issubset({-1.0, 0.0, 1.0}), (
        f"Unexpected signal values: {set(valid.unique())}"
    )


# ---------------------------------------------------------------------------
# Golden Cross
# ---------------------------------------------------------------------------

def test_golden_cross_signals(random_prices):
    strat = GoldenCrossStrategy(fast=10, slow=30)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_golden_cross_warmup(random_prices):
    strat = GoldenCrossStrategy(fast=10, slow=30)
    signals = strat.generate_signals(random_prices)
    # First slow-1 bars should be NaN or 0 (no enough data for slow MA)
    assert signals.iloc[:29].isna().all() or (signals.iloc[:29] == 0).all()


def test_golden_cross_registered():
    StrategyRegistry._load_all()
    assert "golden_cross" in StrategyRegistry.names()


def test_golden_cross_backtest_returns(random_prices):
    strat = GoldenCrossStrategy(fast=10, slow=30)
    positions, returns = strat.backtest_returns(random_prices)
    assert len(positions) == len(random_prices)
    assert len(returns) == len(random_prices)


# ---------------------------------------------------------------------------
# RSI Overbought
# ---------------------------------------------------------------------------

def test_rsi_signals(random_prices):
    strat = RSIOverboughtStrategy(period=14)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_rsi_registered():
    StrategyRegistry._load_all()
    assert "rsi_overbought" in StrategyRegistry.names()


def test_rsi_extreme_prices():
    """Monotonically increasing prices → RSI stays near 100 → all short signals."""
    prices = pd.Series(np.linspace(100, 200, 200))
    strat = RSIOverboughtStrategy(period=14, overbought=70.0)
    signals = strat.generate_signals(prices)
    # After warmup, should be mostly short (RSI > 70 in a strong uptrend)
    assert (signals.iloc[50:] == -1.0).mean() > 0.5


# ---------------------------------------------------------------------------
# MACD Crossover
# ---------------------------------------------------------------------------

def test_macd_signals(random_prices):
    strat = MACDCrossoverStrategy()
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_macd_registered():
    StrategyRegistry._load_all()
    assert "macd_crossover" in StrategyRegistry.names()


def test_macd_binary(random_prices):
    """MACD is always long or short (never flat except on exact equality)."""
    strat = MACDCrossoverStrategy()
    signals = strat.generate_signals(random_prices)
    # After warmup, MACD line rarely equals signal line exactly — expect <5% flat
    assert (signals.iloc[30:] == 0.0).mean() < 0.05


# ---------------------------------------------------------------------------
# Bollinger Breakout
# ---------------------------------------------------------------------------

def test_bollinger_signals(random_prices):
    strat = BollingerBreakoutStrategy(lookback=20, num_std=2.0)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_bollinger_registered():
    StrategyRegistry._load_all()
    assert "bollinger_breakout" in StrategyRegistry.names()


def test_bollinger_sparsity(random_prices):
    """Most bars should be flat — price rarely outside 2-sigma bands."""
    strat = BollingerBreakoutStrategy(lookback=20, num_std=2.0)
    signals = strat.generate_signals(random_prices)
    assert (signals.iloc[20:] == 0.0).mean() > 0.5


# ---------------------------------------------------------------------------
# Volume Spike
# ---------------------------------------------------------------------------

def test_volume_spike_signals(random_prices):
    strat = VolumeSpikeStrategy(lookback=20, spike_mult=2.0)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_volume_spike_registered():
    StrategyRegistry._load_all()
    assert "volume_spike" in StrategyRegistry.names()


def test_volume_spike_sparsity(random_prices):
    """Spikes are rare — most bars should be flat."""
    strat = VolumeSpikeStrategy(lookback=20, spike_mult=2.0)
    signals = strat.generate_signals(random_prices)
    assert (signals.iloc[20:] == 0.0).mean() > 0.5


# ---------------------------------------------------------------------------
# Doji Reversal
# ---------------------------------------------------------------------------

def test_doji_signals(random_prices):
    strat = DojiReversalStrategy()
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_doji_registered():
    StrategyRegistry._load_all()
    assert "doji_reversal" in StrategyRegistry.names()


# ---------------------------------------------------------------------------
# Three-Bar Reversal
# ---------------------------------------------------------------------------

def test_three_bar_signals(random_prices):
    strat = ThreeBarReversalStrategy(n_bars=3)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_three_bar_registered():
    StrategyRegistry._load_all()
    assert "three_bar_reversal" in StrategyRegistry.names()


def test_three_bar_no_flat_after_run(trending_prices):
    """In a strong uptrend there are many 3-bar runs → signal should fire often."""
    strat = ThreeBarReversalStrategy(n_bars=3)
    signals = strat.generate_signals(trending_prices)
    assert (signals != 0).sum() > 0


# ---------------------------------------------------------------------------
# Gap and Go
# ---------------------------------------------------------------------------

def test_gap_and_go_signals(random_prices):
    strat = GapAndGoStrategy(gap_threshold=0.005)
    signals = strat.generate_signals(random_prices)
    _assert_signals_valid(signals, random_prices)


def test_gap_and_go_registered():
    StrategyRegistry._load_all()
    assert "gap_and_go" in StrategyRegistry.names()


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

EXPECTED_RETAIL = {
    "golden_cross",
    "rsi_overbought",
    "macd_crossover",
    "bollinger_breakout",
    "volume_spike",
    "doji_reversal",
    "three_bar_reversal",
    "gap_and_go",
}


def test_all_retail_registered():
    StrategyRegistry._load_all()
    registered = set(StrategyRegistry.names())
    missing = EXPECTED_RETAIL - registered
    assert not missing, f"Retail strategies not registered: {missing}"


def test_retail_category():
    StrategyRegistry._load_all()
    retail_metas = StrategyRegistry.list_by_category("retail")
    retail_names = {m.name for m in retail_metas}
    assert EXPECTED_RETAIL == retail_names


def test_meta_fields_complete():
    StrategyRegistry._load_all()
    for meta in StrategyRegistry.list_by_category("retail"):
        assert meta.name, f"Missing name for {meta}"
        assert meta.category == "retail"
        assert meta.source, f"Missing source for {meta.name}"
        assert meta.hypothesis, f"Missing hypothesis for {meta.name}"
        assert meta.expected_result, f"Missing expected_result for {meta.name}"
