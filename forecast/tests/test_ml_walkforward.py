"""
Tests for the walk-forward ML driver.

The headline test is anti-lookahead: changing future prices must leave every
earlier signal bit-for-bit identical. We also verify the pipeline actually
*learns* — both at the model level (linearly separable target) and end-to-end
(a planted momentum series).
"""

import numpy as np
import pandas as pd

from src.ml import SklearnDirectionModel, WalkForwardConfig, walk_forward_signal
from src.ml.sklearn_model import logistic_factory


def _random_walk(n, seed):
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    return pd.Series(px, index=pd.date_range("2015-01-01", periods=n, freq="B"))


def _ar1_momentum(n, rho, sigma, seed):
    """Prices whose returns follow an AR(1) process -> sign is predictable."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = rho * r[t - 1] + eps[t]
    px = 100 * np.exp(np.cumsum(r))
    return pd.Series(px, index=pd.date_range("2014-01-01", periods=n, freq="B"))


def test_signal_values_are_valid():
    px = _random_walk(400, seed=3)
    sig = walk_forward_signal(
        px, SklearnDirectionModel(logistic_factory),
        WalkForwardConfig(min_train=120, retrain_every=30),
    )
    assert sig.index.equals(px.index)
    assert set(np.unique(sig.values)).issubset({-1.0, 0.0, 1.0})
    assert not sig.isna().any()
    # Nothing is traded before the first out-of-sample block.
    assert (sig.iloc[:120] == 0).all()


def test_walkforward_is_anti_lookahead():
    px = _random_walk(420, seed=7)
    k = 360
    px_pert = px.copy()
    px_pert.iloc[k:] = px_pert.iloc[k:] * 1.4  # shock the future

    cfg = WalkForwardConfig(min_train=120, retrain_every=20)
    s0 = walk_forward_signal(px, SklearnDirectionModel(logistic_factory), cfg)
    s1 = walk_forward_signal(px_pert, SklearnDirectionModel(logistic_factory), cfg)

    # Every signal before the perturbation must be unchanged.
    pd.testing.assert_series_equal(s0.iloc[:k], s1.iloc[:k])
    # Sanity: the shock did change later signals (otherwise the test is vacuous).
    assert not s0.iloc[k:].equals(s1.iloc[k:])


def test_walkforward_is_deterministic():
    px = _random_walk(350, seed=11)
    cfg = WalkForwardConfig(min_train=120, retrain_every=30)
    a = walk_forward_signal(px, SklearnDirectionModel(logistic_factory, seed=0), cfg)
    b = walk_forward_signal(px, SklearnDirectionModel(logistic_factory, seed=0), cfg)
    pd.testing.assert_series_equal(a, b)


def test_sklearn_model_learns_separable_target():
    """Model-level sanity: logistic should fit a linearly separable label."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + X["b"] > 0).astype(float))
    model = SklearnDirectionModel(logistic_factory)
    train_pos = np.arange(200)
    test_pos = np.arange(200, 300)
    model.fit(X, y, train_pos)
    proba = model.predict(X, test_pos)
    acc = ((proba > 0.5).astype(float) == y.to_numpy()[test_pos]).mean()
    assert acc > 0.9


def test_walkforward_has_edge_on_momentum_series():
    """End-to-end: directional accuracy beats a coin flip on a momentum series."""
    px = _ar1_momentum(900, rho=0.6, sigma=0.008, seed=42)
    sig = walk_forward_signal(
        px, SklearnDirectionModel(logistic_factory, seed=0),
        WalkForwardConfig(min_train=150, retrain_every=30),
    )
    next_ret = px.pct_change().shift(-1)  # realized return for the position held at t
    mask = (sig != 0) & next_ret.notna()
    assert mask.sum() > 100  # actually traded
    correct = (np.sign(sig[mask]) == np.sign(next_ret[mask])).mean()
    assert correct > 0.52  # genuine, if modest, predictive edge
