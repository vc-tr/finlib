"""
Long-Term Reversal (De Bondt & Thaler 1985, 1987).

Paper: "Does the Stock Market Overreact?", Journal of Finance, 40(3), 793-805.
       "Further Evidence on Investor Overreaction and Stock Market Seasonality", JF 1987.

Strategy: Contrarian - buy past LOSERS (3-5 year), sell past WINNERS.
          Formation: rank by 3-5 year past return.
          Losers outperform winners over next 3-5 years (reversal).

For single-asset: proxy as reversal of long-term momentum.
Signal = -sign(formation-period return)  (opposite of momentum).
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Paper: 3-5 year formation and holding
# For daily: 3yr ≈ 756, 5yr ≈ 1260
PAPER_FORMATION_DAYS = 756  # ~3 years
PAPER_HOLDING_DAYS = 756


class DeBondtThalerReversal:
    """
    Long-term reversal (contrarian) per De Bondt & Thaler.
    
    Signal = -sign(formation-period return)
    Buy past losers, sell past winners. Hold for holding_period.
    """

    def __init__(
        self,
        formation_period: int = PAPER_FORMATION_DAYS,
        holding_period: int = PAPER_HOLDING_DAYS,
    ):
        """
        Args:
            formation_period: Lookback for ranking (paper: 3-5 years)
            holding_period: Hold period (paper: 3-5 years)
        """
        self.formation_period = formation_period
        self.holding_period = holding_period

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """Contrarian: signal = -sign(past return)."""
        ret = prices.pct_change(self.formation_period)
        raw_signal = -np.sign(ret)
        raw_signal = raw_signal.replace(0, np.nan).ffill().fillna(0)
        signals = pd.Series(0.0, index=prices.index)
        pos = 0
        rebalance_bar = 0
        for i in range(self.formation_period, len(prices)):
            if i >= rebalance_bar:
                pos = raw_signal.iloc[i]
                rebalance_bar = i + self.holding_period
            signals.iloc[i] = pos
        return signals.shift(1).fillna(0)

    def backtest_returns(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Returns (signals, strategy_returns)."""
        signals = self.generate_signals(prices)
        returns = prices.pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
