"""Tests for causal ML feature engineering and labels."""

import numpy as np
import pandas as pd

from src.ml.features import FEATURE_NAMES, make_features, make_labels


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    px = 100 * np.exp(np.cumsum(rets))
    return pd.Series(px, index=pd.date_range("2018-01-01", periods=n, freq="B"))


def test_features_shape_and_columns():
    px = _prices()
    feat = make_features(px)
    assert list(feat.columns) == FEATURE_NAMES
    assert feat.index.equals(px.index)
    # After the warm-up window every row is fully populated.
    assert feat.dropna().shape[0] > len(px) - 60


def test_features_are_causal():
    """Perturbing future prices must not change any earlier feature row."""
    px = _prices(n=300, seed=1)
    k = 200
    px_pert = px.copy()
    px_pert.iloc[k:] = px_pert.iloc[k:] * 1.5  # large shock from bar k onward

    f0 = make_features(px)
    f1 = make_features(px_pert)
    # Rows strictly before k depend only on prices < k -> identical (NaN-aware).
    pd.testing.assert_frame_equal(f0.iloc[:k], f1.iloc[:k])
    # Sanity: the perturbation actually changed later rows.
    assert not f0.iloc[k:].equals(f1.iloc[k:])


def test_labels_are_forward_and_have_nan_tail():
    px = pd.Series(
        [100, 101, 100, 103, 102],
        index=pd.date_range("2020-01-01", periods=5, freq="B"),
    )
    lab = make_labels(px, horizon=1)
    # label[t] = 1 if price[t+1] > price[t]
    assert lab.iloc[0] == 1.0   # 101 > 100
    assert lab.iloc[1] == 0.0   # 100 < 101
    assert lab.iloc[2] == 1.0   # 103 > 100
    assert lab.iloc[3] == 0.0   # 102 < 103
    assert np.isnan(lab.iloc[4])  # no future bar


def test_labels_horizon_nan_count():
    px = _prices(n=100)
    assert int(make_labels(px, horizon=5).isna().sum()) == 5


def test_rsi_in_unit_interval():
    feat = make_features(_prices())
    rsi = feat["rsi_14"].dropna()
    assert rsi.between(0.0, 1.0).all()
