"""
Scalping strategy: capture small price moves.

Uses fractional price movement - buy near support, sell near resistance.
Signal-based: EMA crossover + short-term momentum.
Best on intraday (1m, 5m) data.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class ScalpingStrategy:
    """
    Scalping: quick in-and-out trades on small moves.
    
    Entry: Fast EMA crosses above slow EMA (long) or below (short).
    Exit: Opposite crossover or fixed period.
    """

    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        min_move_pct: float = 0.001,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.min_move_pct = min_move_pct

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """EMA crossover signals."""
        fast = prices.ewm(span=self.fast_ema, adjust=False).mean()
        slow = prices.ewm(span=self.slow_ema, adjust=False).mean()
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        signals = pd.Series(0.0, index=prices.index)
        signals[cross_up] = 1
        signals[cross_down] = -1
        # Hold until opposite signal
        pos = 0
        out = pd.Series(0.0, index=prices.index)
        for i in range(len(prices)):
            if cross_up.iloc[i]:
                pos = 1
            elif cross_down.iloc[i]:
                pos = -1
            out.iloc[i] = pos
        return out.shift(1).fillna(0)

    def backtest_returns(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        signals = self.generate_signals(prices)
        returns = prices.pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
