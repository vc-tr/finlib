"""Tests for src/factors/weight_learning.py and combo methods."""

import numpy as np
import pandas as pd
import pytest

from src.factors.weight_learning import (
    learn_weights_ic,
    learn_weights_ridge,
    learn_weights_sharpe,
)
from src.factors.ensemble import combine_factors
from src.backtest.walkforward import generate_folds


def _synthetic_prices(n_dates: int = 100, n_symbols: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    ret = rng.randn(n_dates, n_symbols) * 0.01
    close = 100 * (1 + pd.DataFrame(ret, index=idx, columns=[f"S{i}" for i in range(n_symbols)])).cumprod()
    return close


def test_learn_weights_ic_predictive_factor():
    """One factor predicts returns, others are noise => predictive factor gets higher weight."""
    n_dates, n_symbols = 80, 10
    rng = np.random.RandomState(42)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]

    ret = rng.randn(n_dates, n_symbols) * 0.01
    close = 100 * (1 + pd.DataFrame(ret, index=idx, columns=cols)).cumprod()
    fwd_1 = close.shift(-1) / close - 1

    # Factor A = future return + small noise (predictive)
    factor_a = fwd_1 + rng.randn(n_dates, n_symbols) * 0.001
    # Factor B = pure noise
    factor_b = pd.DataFrame(rng.randn(n_dates, n_symbols), index=idx, columns=cols)

    fac_train = {"A": factor_a, "B": factor_b}
    fwd_train = {1: fwd_1}
    weights = learn_weights_ic(fac_train, fwd_train, horizons=[1])
    assert "A" in weights and "B" in weights
    assert weights["A"] > weights["B"]


def test_learn_weights_ridge():
    """Ridge returns non-negative weights summing to 1."""
    rng = np.random.RandomState(7)
    X = rng.randn(100, 3)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.randn(100) * 0.1
    w = learn_weights_ridge(X, y, l2=1.0)
    assert len(w) == 3
    assert np.all(w >= 0)
    assert np.abs(w.sum() - 1.0) < 1e-6


def test_learn_weights_sharpe():
    """Sharpe returns weights summing to 1."""
    rng = np.random.RandomState(11)
    n = 50
    ret_a = rng.randn(n) * 0.01
    ret_b = rng.randn(n) * 0.02
    weights = learn_weights_sharpe({"A": ret_a, "B": ret_b}, l2=0.5)
    assert "A" in weights and "B" in weights
    assert np.abs(sum(weights.values()) - 1.0) < 1e-6


def test_auto_selects_predictive_factor():
    """Synthetic: one factor predictive, others noise => auto assigns higher weight to predictive."""
    n_dates, n_symbols = 80, 10
    rng = np.random.RandomState(123)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]

    ret = rng.randn(n_dates, n_symbols) * 0.01
    close = 100 * (1 + pd.DataFrame(ret, index=idx, columns=cols)).cumprod()
    fwd_1 = close.shift(-1) / close - 1

    factor_good = fwd_1 + rng.randn(n_dates, n_symbols) * 0.001
    factor_noise = pd.DataFrame(rng.randn(n_dates, n_symbols), index=idx, columns=cols)

    factors = {"good": factor_good, "noise": factor_noise}
    fwd_dict = {1: fwd_1}
    ts = slice(idx[0], idx[59])
    combined, weights, _, meta = combine_factors(
        factors, method="auto", train_slice=ts,
        prices=close, fwd_returns_dict=fwd_dict,
        top_k=2, bottom_k=2, rebalance="M",
    )
    assert meta.get("selected_method") in ["equal", "ic_weighted", "ridge", "sharpe_opt"]
    assert weights["good"] >= weights["noise"] or meta["selected_method"] == "equal"


def test_embargo_shifts_test_start():
    """With embargo_days=2, test_start is 2 days after train_end."""
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=100, freq="B"))
    folds = generate_folds(idx, train_days=20, test_days=10, embargo_days=0)
    folds_emb = generate_folds(idx, train_days=20, test_days=10, embargo_days=2)

    assert len(folds) > 0 and len(folds_emb) > 0
    f0 = folds[0]
    fe = folds_emb[0]
    # train_end same
    assert f0.train_end == fe.train_end
    # test_start with embargo is 2 days after
    dates = idx.sort_values().unique()
    train_end_idx = 19
    test_start_no_emb = dates[train_end_idx + 1]
    test_start_emb = dates[train_end_idx + 1 + 2]
    assert fe.test_start == pd.Timestamp(test_start_emb)
    assert f0.test_start == pd.Timestamp(test_start_no_emb)
