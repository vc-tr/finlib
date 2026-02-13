"""
Mean reversion trading strategy.

Uses z-score of price vs rolling mean to generate signals.
When price deviates significantly from mean, expect reversion.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class MeanReversionStrategy:
    """
    Z-score based mean reversion strategy.
    
    Signal: (price - rolling_mean) / rolling_std
    Long when z < -threshold, short when z > threshold.
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
    
    def compute_zscore(self, series: pd.Series) -> pd.Series:
        """Compute rolling z-score."""
        mean = series.rolling(self.lookback).mean()
        std = series.rolling(self.lookback).std()
        return (series - mean) / std.replace(0, np.nan)
    
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate trading signals: 1 = long, -1 = short, 0 = flat.
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
    
    def backtest_returns(
        self, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute strategy returns.
        
        Returns:
            (signals, strategy_returns)
        """
        signals = self.generate_signals(prices)
        returns = prices.pct_change()
        strategy_returns = signals.shift(1) * returns  # shift to avoid lookahead
        return signals, strategy_returns
