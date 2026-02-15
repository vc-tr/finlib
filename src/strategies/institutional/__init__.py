"""Institutional / hedge fund style strategies."""

from .kalman_pairs import KalmanPairsStrategy
from .vwap_reversion import VWAPReversionStrategy
from .atr_breakout import ATRBreakoutStrategy

__all__ = [
    "KalmanPairsStrategy",
    "VWAPReversionStrategy",
    "ATRBreakoutStrategy",
]
