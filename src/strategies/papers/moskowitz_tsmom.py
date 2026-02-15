"""
Time Series Momentum (Moskowitz, Ooi, Pedersen 2012).

Paper: "Time Series Momentum", Journal of Financial Economics, 104(2), 228-250.
DOI: 10.1016/j.jfineco.2011.11.003

Strategy: Signal_t = sign(r_{t-12,t-1})  (12-month past return)
          Position held for 1 month (or until next rebalance).
          Works on single asset; paper tested 58 futures/forwards.

Key finding: Past 12-month return positively predicts next month's return
across equity indices, currencies, commodities, bonds.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# Paper parameters (monthly): J=12 formation, K=1 holding
# For daily: ~252 formation, ~21 holding
PAPER_FORMATION_DAYS = 252
PAPER_HOLDING_DAYS = 21


class MoskowitzTimeSeriesMomentum:
    """
    Time series momentum per Moskowitz, Ooi, Pedersen (2012).
    
    Signal = sign(cumulative return over formation period)
    Rebalance every holding_period bars.
    """

    def __init__(
        self,
        formation_period: int = PAPER_FORMATION_DAYS,
        holding_period: int = PAPER_HOLDING_DAYS,
    ):
        """
        Args:
            formation_period: Lookback for momentum (paper: 12 months ≈ 252 days)
            holding_period: Bars between rebalances (paper: 1 month ≈ 21 days)
        """
        self.formation_period = formation_period
        self.holding_period = holding_period

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """Signal = sign(formation-period return)."""
        ret = prices.pct_change(self.formation_period)
        raw_signal = np.sign(ret)
        raw_signal = raw_signal.replace(0, np.nan).ffill().fillna(0)
        # Paper: hold for K periods. Implement as holding until next rebalance.
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
