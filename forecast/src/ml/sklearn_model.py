"""
scikit-learn direction models for the walk-forward driver.

A thin wrapper that (a) standardizes features on the *training* rows only,
(b) fits a probabilistic classifier, and (c) returns P(up) for the requested
test rows. Scaler statistics are learned per-refit from past data only, so no
distributional information leaks across the train/test boundary.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def logistic_factory(seed: int = 0) -> LogisticRegression:
    """L2-regularized logistic regression (linear, fast, interpretable)."""
    return LogisticRegression(C=1.0, max_iter=200, random_state=seed)


def gradient_boost_factory(seed: int = 0) -> GradientBoostingClassifier:
    """Shallow gradient-boosted trees for non-linear feature interactions."""
    return GradientBoostingClassifier(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )


class SklearnDirectionModel:
    """Adapts a scikit-learn classifier to the ``DirectionModel`` protocol."""

    def __init__(
        self,
        estimator_factory: Callable[[int], object] = logistic_factory,
        seed: int = 0,
    ):
        self.estimator_factory = estimator_factory
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._model = None
        self._const: float | None = None  # set when training labels are single-class

    def fit(self, features: pd.DataFrame, labels: pd.Series, train_pos: np.ndarray) -> None:
        X = features.to_numpy()[train_pos]
        y = (labels.to_numpy()[train_pos] > 0.5).astype(int)

        self._scaler = StandardScaler().fit(X)
        # Degenerate window (one class only): predict that class as a constant.
        if np.unique(y).size < 2:
            self._const = float(y[0]) if len(y) else 0.5
            self._model = None
            return

        self._const = None
        self._model = self.estimator_factory(self.seed)
        self._model.fit(self._scaler.transform(X), y)

    def predict(self, features: pd.DataFrame, test_pos: np.ndarray) -> np.ndarray:
        if self._const is not None:
            return np.full(len(test_pos), self._const, dtype=float)
        X = self._scaler.transform(features.to_numpy()[test_pos])
        classes = list(self._model.classes_)
        up_col = classes.index(1)
        return self._model.predict_proba(X)[:, up_col]
