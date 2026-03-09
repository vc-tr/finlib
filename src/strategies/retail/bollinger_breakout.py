"""
Bollinger Band Breakout — momentum on band breach.

Long when price breaks above the upper Bollinger Band (2 std above 20-day SMA).
Short when price breaks below the lower band.

The breakout (momentum) version — as opposed to the mean-reversion version
where you fade at the bands. Both fail for different reasons.

Expected: False breakouts dominate. Most band breaches revert immediately.
Buying on volatility expansion means entering after the move is already over.
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class BollingerBreakoutStrategy(Strategy):
    """Trade in the direction of Bollinger Band breakouts."""

    def __init__(self, lookback: int = 20, num_std: float = 2.0):
        self.lookback = lookback
        self.num_std = num_std

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="bollinger_breakout",
            category="retail",
            source="Bollinger (1983) — Bollinger on Bollinger Bands",
            description=(
                f"Long on upper band breach, short on lower band breach "
                f"(SMA-{self.lookback} ± {self.num_std}σ)"
            ),
            hypothesis=(
                "Price breaking outside the bands signals unusual momentum that "
                "continues in the breakout direction. Bands adapt to volatility."
            ),
            expected_result=(
                "Dominated by false breakouts. Most breaches revert within 1-3 bars. "
                "High turnover + costs = negative net return. "
                "The mean-reversion version slightly better, both bad OOS."
            ),
            source_url=None,
            tags=["bollinger", "breakout", "volatility", "retail", "debunk"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        mid = prices.rolling(self.lookback).mean()
        std = prices.rolling(self.lookback).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        signals = pd.Series(0.0, index=prices.index)
        signals[prices > upper] = 1.0
        signals[prices < lower] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "lookback": [10, 20, 50],
            "num_std": [1.5, 2.0, 2.5],
        }
