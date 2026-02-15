"""
Volume + price momentum as a proxy for "crowd sentiment".

When you don't have explicit sentiment data, use volume spikes + price direction
as a proxy for retail/influencer-driven moves (e.g. WSB-style pumps).
"""

import numpy as np
import pandas as pd
from typing import Tuple


class VolumeSentimentStrategy:
    """
    Proxy for crowd sentiment: volume spike + price momentum.
    
    Signal: when volume > N * rolling_avg(volume) and price is rising/falling,
    trade in that direction. Mimics retail-driven momentum.
    """

    def __init__(
        self,
        volume_multiplier: float = 2.0,
        lookback: int = 20,
        momentum_period: int = 5,
    ):
        self.volume_multiplier = volume_multiplier
        self.lookback = lookback
        self.momentum_period = momentum_period

    def generate_signals(
        self,
        prices: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Generate signals from volume spike + momentum."""
        vol_ma = volume.rolling(self.lookback).mean()
        vol_spike = volume > (self.volume_multiplier * vol_ma)
        momentum = prices.pct_change(self.momentum_period)
        signals = pd.Series(0.0, index=prices.index)
        signals[vol_spike & (momentum > 0)] = 1
        signals[vol_spike & (momentum < 0)] = -1
        return signals.shift(1).fillna(0)

    def backtest_returns(
        self,
        prices: pd.Series,
        volume: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Returns (signals, strategy_returns)."""
        signals = self.generate_signals(prices, volume)
        returns = prices.pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
