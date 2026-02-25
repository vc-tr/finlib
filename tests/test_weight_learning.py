"""Tests for src/factors/weight_learning.py and combo methods."""

import numpy as np
import pandas as pd
import pytest

from src.factors.weight_learning import (
    apply_shrinkage,
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


def test_apply_shrinkage_moves_toward_equal() -> None:
    """Shrinkage moves weights toward equal: w_final = (1-s)*w + s*w_equal."""
    weights = {"A": 0.7, "B": 0.2, "C": 0.1}
    shrunk = apply_shrinkage(weights, shrinkage=0.5)
    # w_equal = 1/3 each. w_final = 0.5*w + 0.5*(1/3)
    # A: 0.5*0.7 + 0.5/3 = 0.35 + 0.1667 = 0.5167
    # B: 0.5*0.2 + 0.5/3 = 0.1 + 0.1667 = 0.2667
    # C: 0.5*0.1 + 0.5/3 = 0.05 + 0.1667 = 0.2167
    assert abs(sum(shrunk.values()) - 1.0) < 1e-6
    assert shrunk["A"] < weights["A"]
    assert shrunk["C"] > weights["C"]
    # Full shrinkage => equal
    equal = apply_shrinkage(weights, shrinkage=1.0)
    for v in equal.values():
        assert abs(v - 1.0 / 3) < 1e-6


def test_auto_robust_selects_stable_factor() -> None:
    """Factor A predictive in both train and val; B predictive only in early train, not in val.
    auto_robust should prefer A (stable) over B (overfits)."""
    n_dates, n_symbols = 120, 12
    rng = np.random.RandomState(42)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]

    ret = rng.randn(n_dates, n_symbols) * 0.01
    close = 100 * (1 + pd.DataFrame(ret, index=idx, columns=cols)).cumprod()
    fwd_1 = close.shift(-1) / close - 1

    # Factor A: predictive throughout (stable)
    factor_a = fwd_1 + rng.randn(n_dates, n_symbols) * 0.002

    # Factor B: predictive only in first half (early train), noise in second half (val)
    factor_b = pd.DataFrame(rng.randn(n_dates, n_symbols), index=idx, columns=cols)
    # Make B predictive in first 60 dates only
    factor_b.iloc[:60] = fwd_1.iloc[:60].values + rng.randn(60, n_symbols) * 0.002

    factors = {"A": factor_a, "B": factor_b}
    fwd_dict = {1: fwd_1}
    # Train = first 84 dates, val = last 36 (val_split 0.3)
    # train_sub = first 58, val_sub = last 26 of train
    ts = slice(idx[0], idx[83])
    combined, weights, _, meta = combine_factors(
        factors,
        method="auto_robust",
        train_slice=ts,
        prices=close,
        fwd_returns_dict=fwd_dict,
        top_k=2,
        bottom_k=2,
        rebalance="M",
        val_split=0.3,
        shrinkage=0.3,
        auto_metric="val_ic_ir",
    )
    # A should get higher weight than B (A is stable in val, B is not)
    assert meta.get("selected_method") in ["equal", "ic_weighted", "ridge", "sharpe_opt"]
    assert weights["A"] >= weights["B"] or meta["selected_method"] == "equal"
    assert "val_score" in meta
    assert "weights_before_shrink" in meta
    assert "weights_final" in meta


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
