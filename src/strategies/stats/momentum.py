"""
Momentum (trend-following) strategy.

Buys winners, sells losers. Uses past returns as signal.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class MomentumStrategy:
    """
    Momentum strategy based on lookback returns.
    
    Signal = sign(return over lookback period)
    Long when past return > 0, short when < 0.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 0.0,
    ):
        """
        Args:
            lookback: Period for momentum calculation
            threshold: Minimum return to trigger (0 = any positive/negative)
        """
        self.lookback = lookback
        self.threshold = threshold
    
    def compute_momentum(self, prices: pd.Series) -> pd.Series:
        """Momentum = (price / price_n_lookback_ago) - 1"""
        return prices.pct_change(self.lookback)
    
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate signals: 1 = long, -1 = short, 0 = flat.
        """
        mom = self.compute_momentum(prices)
        signals = pd.Series(0.0, index=prices.index)
        signals[mom > self.threshold] = 1
        signals[mom < -self.threshold] = -1
        return signals.shift(1)  # avoid lookahead
    
    def backtest_returns(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Returns (signals, strategy_returns)."""
        signals = self.generate_signals(prices)
        returns = prices.pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
