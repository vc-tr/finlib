"""
Causal feature engineering for ML signal models.

Every feature at bar ``t`` is a function of prices up to and including ``t``
only. All rolling windows are backward-looking (pandas default), and no
operation uses a negative shift. The single forward-looking quantity in this
module is the *label* (``make_labels``), which is the training target and is
never fed back in as a feature.

The contract is verified by ``tests/test_ml_features.py``: perturbing future
prices must leave every earlier feature row bit-for-bit identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Stable, ordered list of feature columns produced by ``make_features``.
FEATURE_NAMES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "mom_21",
    "vol_10",
    "vol_20",
    "zscore_20",
    "rsi_14",
    "donchian_pos_20",
    "ma_gap_10_50",
]


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI in [0, 1], computed causally."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 1.0 - 1.0 / (1.0 + rs)
    # If there were no losses in the window, RSI is fully overbought (1.0).
    rsi = rsi.where(avg_loss > 0, 1.0)
    # If there were no gains either (flat), neutral 0.5.
    rsi = rsi.where((avg_gain > 0) | (avg_loss > 0), 0.5)
    return rsi


def make_features(prices: pd.Series) -> pd.DataFrame:
    """
    Build the causal feature matrix from a close-price series.

    Args:
        prices: Close prices indexed by a (sorted) DatetimeIndex.

    Returns:
        DataFrame indexed like ``prices`` with columns ``FEATURE_NAMES``.
        Warm-up rows contain NaN; callers drop them (the walk-forward driver
        does this automatically).
    """
    prices = prices.astype(float)
    log_ret = np.log(prices).diff()

    feat = pd.DataFrame(index=prices.index)
    feat["ret_1"] = log_ret
    feat["ret_5"] = log_ret.rolling(5).sum()
    feat["ret_10"] = log_ret.rolling(10).sum()
    # 12-1 style momentum: 21-bar return skipping the most recent bar.
    feat["mom_21"] = log_ret.shift(1).rolling(21).sum()
    feat["vol_10"] = log_ret.rolling(10).std()
    feat["vol_20"] = log_ret.rolling(20).std()

    ma20 = prices.rolling(20).mean()
    sd20 = prices.rolling(20).std()
    feat["zscore_20"] = (prices - ma20) / sd20.replace(0.0, np.nan)

    feat["rsi_14"] = _rsi(prices, 14)

    hi20 = prices.rolling(20).max()
    lo20 = prices.rolling(20).min()
    feat["donchian_pos_20"] = (prices - lo20) / (hi20 - lo20).replace(0.0, np.nan)

    ma10 = prices.rolling(10).mean()
    ma50 = prices.rolling(50).mean()
    feat["ma_gap_10_50"] = (ma10 - ma50) / ma50.replace(0.0, np.nan)

    return feat[FEATURE_NAMES]


def make_labels(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Forward-return direction label: ``1`` if the return over the next
    ``horizon`` bars is positive, else ``0``.

    This is the supervised target and is **forward-looking by design** — the
    last ``horizon`` entries are NaN (unrealized). The walk-forward driver only
    ever trains on rows whose label is already realized strictly before the
    prediction point, so the label never leaks into a prediction.

    Args:
        prices: Close prices.
        horizon: Number of bars ahead the label looks.

    Returns:
        Series of {0.0, 1.0} (NaN where the future return is not yet known).
    """
    fwd_ret = prices.shift(-horizon) / prices - 1.0
    label = (fwd_ret > 0).astype(float)
    label[fwd_ret.isna()] = np.nan
    return label
