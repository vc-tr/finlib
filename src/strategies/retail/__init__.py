"""
Retail / technical analysis strategies — YouTube/TikTok guru debunking.

Each strategy is decorated with @StrategyRegistry.register.
Importing this package is sufficient to register all retail strategies.
"""

from .golden_cross import GoldenCrossStrategy
from .rsi_overbought import RSIOverboughtStrategy
from .macd_crossover import MACDCrossoverStrategy
from .bollinger_breakout import BollingerBreakoutStrategy
from .volume_spike import VolumeSpikeStrategy
from .doji_reversal import DojiReversalStrategy
from .three_bar_reversal import ThreeBarReversalStrategy
from .gap_and_go import GapAndGoStrategy

__all__ = [
    "GoldenCrossStrategy",
    "RSIOverboughtStrategy",
    "MACDCrossoverStrategy",
    "BollingerBreakoutStrategy",
    "VolumeSpikeStrategy",
    "DojiReversalStrategy",
    "ThreeBarReversalStrategy",
    "GapAndGoStrategy",
]
