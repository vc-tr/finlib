"""
EMA + Stochastic Oscillator strategy.

Classic day trading combo: EMA for trend, Stochastic for overbought/oversold.
Long when: price > EMA and Stochastic crosses up from oversold.
Short when: price < EMA and Stochastic crosses down from overbought.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class EmaStochasticStrategy:
    """
    EMA trend filter + Stochastic for entries.
    
    - EMA(20): trend direction
    - Stochastic(14,3): overbought (>80), oversold (<20)
    - Long: price > EMA, Stoch crosses above 20
    - Short: price < EMA, Stoch crosses below 80
    """

    def __init__(
        self,
        ema_period: int = 20,
        stoch_k: int = 14,
        stoch_d: int = 3,
        overbought: float = 80,
        oversold: float = 20,
    ):
        self.ema_period = ema_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.overbought = overbought
        self.oversold = oversold

    def _stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        lowest = low.rolling(self.stoch_k).min()
        highest = high.rolling(self.stoch_k).max()
        k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
        return k.rolling(self.stoch_d).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """df: OHLCV with high, low, close."""
        close = df["close"]
        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        stoch = self._stochastic(df["high"], df["low"], close)
        above_ema = close > ema
        below_ema = close < ema
        stoch_cross_up = (stoch > self.oversold) & (stoch.shift(1) <= self.oversold)
        stoch_cross_down = (stoch < self.overbought) & (stoch.shift(1) >= self.overbought)
        long_signal = above_ema & stoch_cross_up
        short_signal = below_ema & stoch_cross_down
        signals = pd.Series(0.0, index=df.index)
        position = 0
        for i in range(len(df)):
            if long_signal.iloc[i]:
                position = 1
            elif short_signal.iloc[i]:
                position = -1
            signals.iloc[i] = position
        return signals.shift(1).fillna(0)

    def backtest_returns(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        signals = self.generate_signals(df)
        returns = df["close"].pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
