"""
Tests for the PyTorch LSTM direction model (skipped if torch is absent).

Mirrors the anti-lookahead and learning guarantees of the sklearn path, plus a
determinism check (same seed -> identical signals) and a sanity check that the
network can fit a separable target.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # skip the whole module if torch is missing

from src.ml import WalkForwardConfig, walk_forward_signal  # noqa: E402
from src.ml.torch_lstm import LSTMDirectionModel  # noqa: E402


def _ar1_momentum(n, rho, sigma, seed):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = rho * r[t - 1] + eps[t]
    px = 100 * np.exp(np.cumsum(r))
    return pd.Series(px, index=pd.date_range("2014-01-01", periods=n, freq="B"))


def _model(seed=0):
    return LSTMDirectionModel(
        seq_len=8, hidden_size=12, epochs=25, seed=seed, device="cpu"
    )


def _cfg():
    return WalkForwardConfig(min_train=140, retrain_every=70)


def test_lstm_produces_valid_signals():
    px = _ar1_momentum(360, rho=0.5, sigma=0.01, seed=1)
    sig = walk_forward_signal(px, _model(), _cfg())
    assert sig.index.equals(px.index)
    assert set(np.unique(sig.values)).issubset({-1.0, 0.0, 1.0})
    assert not sig.isna().any()


def test_lstm_is_deterministic():
    px = _ar1_momentum(360, rho=0.5, sigma=0.01, seed=2)
    a = walk_forward_signal(px, _model(seed=0), _cfg())
    b = walk_forward_signal(px, _model(seed=0), _cfg())
    pd.testing.assert_series_equal(a, b)


def test_lstm_is_anti_lookahead():
    px = _ar1_momentum(380, rho=0.5, sigma=0.01, seed=3)
    k = 320
    px_pert = px.copy()
    px_pert.iloc[k:] = px_pert.iloc[k:] * 1.3

    s0 = walk_forward_signal(px, _model(seed=0), _cfg())
    s1 = walk_forward_signal(px_pert, _model(seed=0), _cfg())
    pd.testing.assert_series_equal(s0.iloc[:k], s1.iloc[:k])


def test_lstm_fits_separable_target():
    """The network can learn a clearly separable label (overfit sanity)."""
    rng = np.random.default_rng(0)
    n = 260
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    # Label depends on a persistent feature so windows carry the signal.
    y = pd.Series((X["a"].rolling(3).mean() > 0).astype(float)).fillna(0.0)
    model = _model(seed=0)
    train_pos = np.arange(40, 200)
    test_pos = np.arange(200, n)
    model.fit(X, y, train_pos)
    proba = model.predict(X, test_pos)
    acc = ((proba > 0.5).astype(float) == y.to_numpy()[test_pos]).mean()
    assert acc > 0.6
