"""Integration tests: ML strategies plug into the registry and backtester."""

import numpy as np
import pandas as pd

from src.backtest import Backtester
from src.backtest.execution import ExecutionConfig
from src.strategies.registry import StrategyRegistry


def _prices(n=500, seed=5):
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    return pd.Series(px, index=pd.date_range("2016-01-01", periods=n, freq="B"))


def test_ml_strategies_are_registered():
    StrategyRegistry._load_all()
    names = StrategyRegistry.names()
    assert "ml_logistic" in names
    assert "ml_gradient_boost" in names
    assert "ml" in StrategyRegistry.categories()


def test_ml_strategy_produces_valid_signals():
    StrategyRegistry._load_all()
    strat = StrategyRegistry.get("ml_logistic", min_train=150, retrain_every=40)
    sig = strat.generate_signals(_prices())
    assert set(np.unique(sig.values)).issubset({-1.0, 0.0, 1.0})
    assert not sig.isna().any()


def test_ml_strategy_backtests_end_to_end():
    StrategyRegistry._load_all()
    px = _prices()
    strat = StrategyRegistry.get("ml_gradient_boost", min_train=150, retrain_every=60)
    sig = strat.generate_signals(px)
    result = Backtester(annualization_factor=252).run_from_signals(
        px, sig, execution_config=ExecutionConfig(fee_bps=1.0, slippage_bps=2.0)
    )
    # Produces a coherent result object (not asserting profitability on noise).
    assert np.isfinite(result.sharpe_ratio)
    assert result.n_trades >= 0
    assert 0.0 <= result.win_rate <= 1.0


def test_ml_strategy_meta_is_complete():
    StrategyRegistry._load_all()
    for name in ("ml_logistic", "ml_gradient_boost"):
        meta = StrategyRegistry.get(name).meta()
        assert meta.category == "ml"
        assert meta.hypothesis and meta.expected_result
