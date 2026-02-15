"""
ATR-based volatility breakout strategy.

Uses Average True Range for dynamic support/resistance.
Breakout when price moves beyond ATR multiple from recent high/low.
Common in institutional volatility targeting.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class ATRBreakoutStrategy:
    """
    Volatility breakout: trade when price breaks ATR-based bands.
    
    Upper band = recent high + k * ATR
    Lower band = recent low - k * ATR
    Long on break above upper, short on break below lower.
    """

    def __init__(
        self,
        atr_period: int = 14,
        lookback: int = 20,
        atr_multiplier: float = 2.0,
    ):
        self.atr_period = atr_period
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        """Average True Range."""
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """df: OHLCV."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        atr = self._atr(df)
        recent_high = high.rolling(self.lookback).max().shift(1)
        recent_low = low.rolling(self.lookback).min().shift(1)
        upper = recent_high + self.atr_multiplier * atr
        lower = recent_low - self.atr_multiplier * atr
        long_break = (close > upper) & upper.notna()
        short_break = (close < lower) & lower.notna()
        signals = pd.Series(0.0, index=df.index)
        position = 0
        for i in range(len(df)):
            if long_break.iloc[i]:
                position = 1
            elif short_break.iloc[i]:
                position = -1
            signals.iloc[i] = position
        return signals.shift(1).fillna(0)

    def backtest_returns(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        signals = self.generate_signals(df)
        returns = df["close"].pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
