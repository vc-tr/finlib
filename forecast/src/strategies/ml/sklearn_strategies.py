"""
scikit-learn ML strategies: walk-forward direction classifiers.

Two flavors share one implementation:
  - ``ml_logistic``       : L2 logistic regression (linear baseline)
  - ``ml_gradient_boost`` : shallow gradient-boosted trees (non-linear)

Both predict P(next-bar up) from a causal feature matrix and trade the sign,
retraining monthly on an expanding window.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.ml import (
    SklearnDirectionModel,
    WalkForwardConfig,
    gradient_boost_factory,
    logistic_factory,
    walk_forward_signal,
)
from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


class _SklearnMLStrategy(Strategy):
    """Shared walk-forward ML strategy; subclasses pick the estimator."""

    _factory = staticmethod(logistic_factory)

    def __init__(
        self,
        min_train: int = 252,
        retrain_every: int = 21,
        horizon: int = 1,
        band: float = 0.0,
        seed: int = 0,
    ):
        self.min_train = min_train
        self.retrain_every = retrain_every
        self.horizon = horizon
        self.band = band
        self.seed = seed

    def _config(self) -> WalkForwardConfig:
        return WalkForwardConfig(
            min_train=self.min_train,
            retrain_every=self.retrain_every,
            horizon=self.horizon,
            band=self.band,
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        model = SklearnDirectionModel(self._factory, seed=self.seed)
        return walk_forward_signal(prices, model, self._config())

    def parameter_grid(self) -> Dict[str, List]:
        return {
            "retrain_every": [21, 63],
            "horizon": [1, 5],
            "band": [0.0, 0.02, 0.05],
        }


@StrategyRegistry.register
class MLLogisticStrategy(_SklearnMLStrategy):
    _factory = staticmethod(logistic_factory)

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ml_logistic",
            category="ml",
            source="Logistic regression on causal technical features",
            description="Walk-forward L2 logistic classifier; trades sign of P(next-bar up)",
            hypothesis=(
                "A linear combination of momentum/vol/mean-reversion features "
                "has weak predictive power for next-bar direction"
            ),
            expected_result=(
                "Modest, regime-dependent edge; honest OOS baseline for ML "
                "signals after costs"
            ),
            tags=["ml", "logistic", "walk-forward", "supervised"],
        )


@StrategyRegistry.register
class MLGradientBoostStrategy(_SklearnMLStrategy):
    _factory = staticmethod(gradient_boost_factory)

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ml_gradient_boost",
            category="ml",
            source="Gradient-boosted trees on causal technical features",
            description="Walk-forward gradient-boosting classifier; trades sign of P(next-bar up)",
            hypothesis=(
                "Non-linear interactions among technical features improve "
                "next-bar direction prediction over a linear model"
            ),
            expected_result=(
                "Captures non-linearity but prone to overfitting noise; tends "
                "to need a confidence band to beat costs"
            ),
            tags=["ml", "gradient-boosting", "walk-forward", "supervised"],
        )
