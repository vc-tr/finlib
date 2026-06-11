"""
LSTM ML strategy (optional dependency: torch).

Wraps the genuinely-trained ``LSTMDirectionModel`` in the standard ``Strategy``
interface. Importing this module raises ImportError when torch is absent, so the
package ``__init__`` skips registration gracefully on torch-free installs.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.ml.torch_lstm import LSTMDirectionModel, torch_available

if not torch_available():
    raise ImportError("torch is required for the LSTM strategy")

from src.ml import WalkForwardConfig, walk_forward_signal
from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class MLLSTMStrategy(Strategy):
    """Walk-forward LSTM direction classifier."""

    def __init__(
        self,
        min_train: int = 252,
        retrain_every: int = 63,
        horizon: int = 1,
        band: float = 0.0,
        seq_len: int = 10,
        hidden_size: int = 16,
        epochs: int = 60,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.min_train = min_train
        self.retrain_every = retrain_every
        self.horizon = horizon
        self.band = band
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.seed = seed
        self.device = device

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ml_lstm",
            category="ml",
            source="PyTorch LSTM on causal feature sequences",
            description="Walk-forward LSTM classifier (Adam, grad clipping, LR scheduling, early stopping)",
            hypothesis=(
                "Short sequences of technical features carry temporal structure "
                "an LSTM can exploit for next-bar direction"
            ),
            expected_result=(
                "Comparable to GBM after costs on a single daily series; the "
                "rigor is in leak-free walk-forward training, not raw Sharpe"
            ),
            tags=["ml", "lstm", "pytorch", "deep-learning", "walk-forward"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        model = LSTMDirectionModel(
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            epochs=self.epochs,
            seed=self.seed,
            device=self.device,
        )
        cfg = WalkForwardConfig(
            min_train=self.min_train,
            retrain_every=self.retrain_every,
            horizon=self.horizon,
            band=self.band,
        )
        return walk_forward_signal(prices, model, cfg)

    def parameter_grid(self) -> Dict[str, List]:
        return {
            "seq_len": [5, 10, 20],
            "hidden_size": [8, 16, 32],
            "retrain_every": [63, 126],
        }
