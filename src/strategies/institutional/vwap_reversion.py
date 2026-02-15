"""
VWAP reversion strategy.

Institutional benchmark: trade mean reversion to VWAP.
Price above VWAP + stretched = short (expect reversion down).
Price below VWAP + stretched = long.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class VWAPReversionStrategy:
    """
    Mean reversion to VWAP.
    
    Z-score of (price - VWAP) / std(price - VWAP).
    Long when z < -threshold, short when z > threshold.
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 20,
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback

    def _vwap(self, df: pd.DataFrame) -> pd.Series:
        """Volume-weighted average price."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
        return vwap.ffill()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """df: OHLCV."""
        close = df["close"]
        vwap = self._vwap(df)
        deviation = close - vwap
        mean = deviation.rolling(self.lookback).mean()
        std = deviation.rolling(self.lookback).std()
        z = (deviation - mean) / std.replace(0, np.nan)
        signals = pd.Series(0.0, index=df.index)
        position = 0
        for i in range(self.lookback, len(df)):
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
        return signals.shift(1).fillna(0)

    def backtest_returns(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        signals = self.generate_signals(df)
        returns = df["close"].pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
