"""
Golden Cross / Death Cross — SMA 50/200 crossover.

The most famous retail strategy. Every YouTube trading channel covers it.
Long when SMA(fast) crosses above SMA(slow), short/flat when below.

Expected: Low Sharpe after costs. Extreme lag means entering trends late
and exiting them late. Severely whipsawed in sideways markets.
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class GoldenCrossStrategy(Strategy):
    """SMA fast/slow crossover. Long above, short below."""

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast = fast
        self.slow = slow

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="golden_cross",
            category="retail",
            source="Common retail trading lore",
            description=(
                f"Long when SMA({self.fast}) > SMA({self.slow}), "
                f"short when SMA({self.fast}) < SMA({self.slow})"
            ),
            hypothesis=(
                "Moving average crossover captures medium-term trend continuation. "
                "The 50/200 crossover ('golden cross' = bull, 'death cross' = bear) "
                "is widely cited as a reliable trend filter."
            ),
            expected_result=(
                "Fails OOS: extreme lag means buying tops and selling bottoms. "
                "Whipsawed heavily in choppy markets. Marginal even before costs."
            ),
            source_url=None,
            tags=["moving-average", "trend", "retail", "debunk", "crossover"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        fast_ma = prices.rolling(self.fast).mean()
        slow_ma = prices.rolling(self.slow).mean()
        signals = pd.Series(0.0, index=prices.index)
        signals[fast_ma > slow_ma] = 1.0
        signals[fast_ma < slow_ma] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "fast": [10, 20, 50],
            "slow": [50, 100, 200],
        }
