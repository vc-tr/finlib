"""
Mean reversion trading strategy.

Uses z-score of price vs rolling mean to generate signals.
When price deviates significantly from mean, expect reversion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class MeanReversionStrategy(Strategy):
    """
    Z-score based mean reversion strategy.

    Signal: (price - rolling_mean) / rolling_std
    Long when z < -entry_z, short when z > entry_z.
    Exit when z reverts toward zero past exit_z.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        """
        Args:
            lookback: Rolling window for mean/std
            entry_z: Z-score threshold to enter (abs value)
            exit_z: Z-score threshold to exit (reversion to mean)
        """
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="mean_reversion",
            category="stats",
            source="Classic statistical mean reversion",
            description="Trade z-score of price vs rolling mean; enter on extremes, exit on reversion",
            hypothesis="Short-term price deviations from rolling mean revert",
            expected_result="Works in ranging markets; fails in trending regimes; sensitive to lookback",
            tags=["mean-reversion", "z-score", "statistical"],
        )

    def compute_zscore(self, series: pd.Series) -> pd.Series:
        """Compute rolling z-score."""
        mean = series.rolling(self.lookback).mean()
        std = series.rolling(self.lookback).std()
        return (series - mean) / std.replace(0, np.nan)

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate trading signals: 1 = long, -1 = short, 0 = flat.
        Signal at close t = position held during bar t+1.
        """
        z = self.compute_zscore(prices)
        signals = pd.Series(0.0, index=prices.index)

        position = 0
        for i in range(self.lookback, len(prices)):
            if np.isnan(z.iloc[i]):
                continue
            if position == 0:
                if z.iloc[i] < -self.entry_z:
                    position = 1
                elif z.iloc[i] > self.entry_z:
                    position = -1
            elif position == 1:
                if z.iloc[i] >= -self.exit_z:
                    position = 0
            elif position == -1:
                if z.iloc[i] <= self.exit_z:
                    position = 0
            signals.iloc[i] = position

        return signals

    def backtest_returns(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Returns (signals, strategy_returns)."""
        signals = self.generate_signals(prices)
        returns = prices.pct_change()
        strategy_returns = signals.shift(1) * returns
        return signals, strategy_returns

    def parameter_grid(self) -> Dict[str, List]:
        return {
            "lookback": [10, 20, 50],
            "entry_z": [1.5, 2.0, 2.5],
            "exit_z": [0.0, 0.5, 1.0],
        }
