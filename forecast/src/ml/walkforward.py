"""
Walk-forward driver that turns any direction model into a causal signal.

This is where the anti-lookahead guarantee lives. Given a price series it:

  1. Builds a causal feature matrix and forward-return labels.
  2. Walks forward in blocks of ``retrain_every`` bars. Before predicting a
     block whose first bar sits at integer position ``p``, it refits the model
     on rows ``i`` with ``i + horizon <= p`` -- i.e. rows whose label is fully
     realized *before* the block starts. This is the embargo that prevents the
     horizon-step label from overlapping the prediction window.
  3. Converts the model's P(up) into a position in {-1, 0, +1}, with an optional
     dead-band around 0.5 that maps low-confidence predictions to flat.

The resulting signal[t] is the desired position for bar t+1; the backtester
applies its own shift(1), exactly like every rule-based strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .features import make_features, make_labels


class DirectionModel(Protocol):
    """Minimal interface a model must satisfy to be used walk-forward.

    Implementations receive the *full* feature frame plus the integer positions
    they are allowed to use, so sequence models can build trailing windows
    without ever seeing a row beyond the one being predicted.
    """

    def fit(self, features: pd.DataFrame, labels: pd.Series, train_pos: np.ndarray) -> None:
        ...

    def predict(self, features: pd.DataFrame, test_pos: np.ndarray) -> np.ndarray:
        """Return P(up) in [0, 1] for each position in ``test_pos``."""
        ...


@dataclass
class WalkForwardConfig:
    """Knobs for the walk-forward predictor."""

    min_train: int = 252          # min realized labels before the first prediction
    retrain_every: int = 21       # refit cadence in bars (≈ one trading month)
    horizon: int = 1              # predict the sign of the next `horizon`-bar return
    band: float = 0.0             # |P(up) - 0.5| below this -> flat (0.0 = always ±1)
    warmup_floor: int = 30        # never fit on fewer than this many rows


def _to_signal(proba_up: np.ndarray, band: float) -> np.ndarray:
    """Map P(up) to {-1, 0, +1} with a symmetric dead-band around 0.5."""
    long = proba_up > 0.5 + band
    short = proba_up < 0.5 - band
    return np.where(long, 1.0, np.where(short, -1.0, 0.0))


def walk_forward_signal(
    prices: pd.Series,
    model: DirectionModel,
    config: WalkForwardConfig | None = None,
    features_fn: Callable[[pd.Series], pd.DataFrame] = make_features,
) -> pd.Series:
    """
    Produce a causal {-1, 0, +1} signal for ``prices`` using ``model``.

    Args:
        prices: Close prices with a sorted DatetimeIndex.
        model: Object implementing the ``DirectionModel`` protocol.
        config: Walk-forward configuration (defaults are sane for daily bars).
        features_fn: Causal feature builder (defaults to ``make_features``).

    Returns:
        Signal Series aligned to ``prices.index``; bars before the first
        out-of-sample prediction (and any low-confidence bars) are 0.0.
    """
    cfg = config or WalkForwardConfig()

    feats_full = features_fn(prices)
    labels_full = make_labels(prices, cfg.horizon)

    # Restrict to rows with a complete feature vector; keep label alignment.
    feats = feats_full.dropna()
    labels = labels_full.reindex(feats.index)
    n = len(feats)

    signal = pd.Series(0.0, index=prices.index)
    if n <= cfg.min_train:
        return signal

    label_arr = labels.to_numpy()
    pos = cfg.min_train
    while pos < n:
        block_end = min(pos + cfg.retrain_every, n)

        # Embargo: only rows whose forward label is realized before this block.
        usable = pos - cfg.horizon
        if usable >= cfg.warmup_floor:
            train_pos = np.arange(usable)
            train_pos = train_pos[~np.isnan(label_arr[train_pos])]
            if len(train_pos) >= cfg.warmup_floor:
                model.fit(feats, labels, train_pos)
                test_pos = np.arange(pos, block_end)
                proba_up = np.asarray(model.predict(feats, test_pos), dtype=float)
                signal.loc[feats.index[test_pos]] = _to_signal(proba_up, cfg.band)

        pos = block_end

    return signal
