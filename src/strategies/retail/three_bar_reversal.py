"""
Three-Bar Reversal — fade three consecutive bars in the same direction.

Retail pattern trading premise: three consecutive up (or down) closes signal
exhaustion. Fade on the fourth bar expecting mean reversion.

Expected: Random noise. Runs of 3+ consecutive same-direction bars are common
in trending markets. Fading them there is catastrophic. No statistical edge OOS.
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class ThreeBarReversalStrategy(Strategy):
    """Fade N consecutive same-direction closes."""

    def __init__(self, n_bars: int = 3):
        """
        Args:
            n_bars: Number of consecutive same-direction bars to trigger signal
        """
        self.n_bars = n_bars

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="three_bar_reversal",
            category="retail",
            source="Retail pattern trading — common in price action courses",
            description=(
                f"Fade after {self.n_bars} consecutive same-direction closes"
            ),
            hypothesis=(
                f"{self.n_bars} consecutive bars in one direction exhausts the move. "
                "Mean reversion expected on the following bar."
            ),
            expected_result=(
                "No persistent edge. In trending markets (which dominate equities), "
                "three-bar runs are frequently followed by continuation. "
                "Positive serial correlation in returns makes fading costly."
            ),
            source_url=None,
            tags=["pattern", "reversal", "mean-reversion", "retail", "debunk"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ret = prices.pct_change()
        up = (ret > 0).astype(float)
        down = (ret < 0).astype(float)

        # Rolling sum: n_bars consecutive up/down
        run_up = up.rolling(self.n_bars).sum() == self.n_bars
        run_down = down.rolling(self.n_bars).sum() == self.n_bars

        signals = pd.Series(0.0, index=prices.index)
        signals[run_up] = -1.0    # fade after n consecutive up bars
        signals[run_down] = 1.0   # fade after n consecutive down bars
        return signals

    def parameter_grid(self):
        return {
            "n_bars": [2, 3, 4, 5],
        }
