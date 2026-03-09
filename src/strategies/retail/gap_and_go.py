"""
Gap and Go — trade in the direction of an overnight price gap.

Retail day trading strategy: when a stock gaps up/down at the open relative
to the prior close, trade in the direction of the gap expecting continuation.

Without intraday data, we approximate an "open gap" as a large close-to-close
move (the gap component dominates large overnight moves). A more precise
implementation would use open vs. prior close.

Expected: Inconsistent. True gaps from news/earnings mean-revert intraday
as the information is absorbed. Close-to-close approximation is noisy.
No robust OOS edge, especially after costs.
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class GapAndGoStrategy(Strategy):
    """
    Trade in the direction of large close-to-close moves (gap proxy).

    Note: A production version should use open prices to measure the true gap.
    This implementation uses overnight close-to-close return as a proxy.
    """

    def __init__(self, gap_threshold: float = 0.01, lookback: int = 20):
        """
        Args:
            gap_threshold: Minimum absolute return to classify as a gap (0.01 = 1%)
            lookback: Rolling window for volatility-adjusted gap detection
        """
        self.gap_threshold = gap_threshold
        self.lookback = lookback

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="gap_and_go",
            category="retail",
            source="Retail day trading — YouTube: Warrior Trading, Ross Cameron style",
            description=(
                f"Long on gap-up > {self.gap_threshold:.0%}, "
                f"short on gap-down > {self.gap_threshold:.0%} (close-to-close proxy)"
            ),
            hypothesis=(
                "Gaps driven by news/catalysts continue in the gap direction "
                "as price discovery occurs throughout the trading session."
            ),
            expected_result=(
                "Inconsistent OOS results. True gaps often fill intraday "
                "(mean reversion). Close-to-close proxy cannot distinguish "
                "true gaps from gradual moves. High variance, low Sharpe."
            ),
            source_url=None,
            tags=["gap", "momentum", "retail", "debunk", "day-trading"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ret = prices.pct_change()
        rolling_vol = ret.rolling(self.lookback).std()

        # Volatility-adjusted gap detection: gap if |ret| > threshold AND > 1σ
        abs_ret = ret.abs()
        is_gap = (abs_ret > self.gap_threshold) & (abs_ret > rolling_vol)

        signals = pd.Series(0.0, index=prices.index)
        signals[is_gap & (ret > 0)] = 1.0
        signals[is_gap & (ret < 0)] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "gap_threshold": [0.005, 0.01, 0.02],
            "lookback": [10, 20, 40],
        }
