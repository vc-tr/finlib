"""
Opening Range Breakout (ORB) strategy.

Records high/low of first N bars (e.g. first 15 min of session), then trades breakouts.
Popular with day traders. Works on intraday or daily data.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class OpeningRangeBreakoutStrategy:
    """
    ORB: trade breakouts of the opening range.
    
    Long when close > opening_range_high, short when close < opening_range_low.
    For intraday: first N bars = opening range.
    For daily: first N days = "range" (rolling window).
    """

    def __init__(
        self,
        orb_bars: int = 15,
        hold_until_end: bool = True,
    ):
        """
        Args:
            orb_bars: Number of bars for opening range
            hold_until_end: If True, hold position until end of data/session
        """
        self.orb_bars = orb_bars
        self.hold_until_end = hold_until_end

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        df: DataFrame with 'high', 'low', 'close' (and optionally 'volume').
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        n = self.orb_bars
        signals = pd.Series(0.0, index=df.index)
        # Rolling opening range: for each bar i, range = bars [i-n:i]
        orb_high = high.rolling(n).max().shift(1)
        orb_low = low.rolling(n).min().shift(1)
        long_break = (close > orb_high) & (orb_high.notna())
        short_break = (close < orb_low) & (orb_low.notna())
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
