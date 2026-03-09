"""
Doji Reversal — fade the trend after an indecision candle.

A doji is a candlestick where open ≈ close, indicating market indecision.
Retail lore: a doji after a strong trend signals exhaustion and reversal.

Without OHLCV data, we approximate a doji as a day with very small
close-to-close return following a strong directional move.

Expected: No statistical edge. The bar pattern is an arbitrary visual
classification. Any apparent pattern is noise in the distribution of returns.
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class DojiReversalStrategy(Strategy):
    """
    Fade the trend after a doji-like indecision day.

    Doji proxy: |pct_change| < doji_threshold.
    Trend: prior day |pct_change| > trend_threshold.
    Signal: fade the prior trend direction on the day after the doji.
    """

    def __init__(
        self,
        doji_threshold: float = 0.001,
        trend_threshold: float = 0.005,
    ):
        """
        Args:
            doji_threshold: Max absolute return to classify as doji-like (0.001 = 0.1%)
            trend_threshold: Min absolute return on prior day to require trending context
        """
        self.doji_threshold = doji_threshold
        self.trend_threshold = trend_threshold

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="doji_reversal",
            category="retail",
            source="Japanese candlestick analysis — Nison (1991), Steve Nison",
            description=(
                "Fade trend after doji-like indecision candle "
                f"(|ret| < {self.doji_threshold:.1%} following |ret| > {self.trend_threshold:.1%})"
            ),
            hypothesis=(
                "A doji after a directional move signals buyer/seller equilibrium, "
                "indicating trend exhaustion and likely reversal."
            ),
            expected_result=(
                "No edge. Close-price doji approximation is noisy. "
                "Even true doji patterns have no predictive power OOS — "
                "p-values do not survive multiple testing correction."
            ),
            source_url=None,
            tags=["candlestick", "reversal", "pattern", "retail", "debunk"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ret = prices.pct_change()
        abs_ret = ret.abs()

        is_doji = abs_ret < self.doji_threshold
        is_trending = abs_ret.shift(1) > self.trend_threshold
        prior_direction = ret.shift(1).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

        signals = pd.Series(0.0, index=prices.index)
        reversal_condition = is_doji & is_trending
        signals[reversal_condition] = -prior_direction[reversal_condition]
        return signals

    def parameter_grid(self):
        return {
            "doji_threshold": [0.0005, 0.001, 0.002],
            "trend_threshold": [0.003, 0.005, 0.01],
        }
