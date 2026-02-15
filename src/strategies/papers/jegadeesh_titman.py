"""
Cross-Sectional Momentum (Jegadeesh & Titman 1993).

Paper: "Returns to Buying Winners and Selling Losers: Implications for Stock
       Market Efficiency", Journal of Finance, 48(1), 65-91.

Strategy: Rank stocks by past J-month return. Long top decile (winners),
          short bottom decile (losers). Hold for K months. Rebalance monthly.

Requires: DataFrame of asset returns (columns = assets).
For single-asset: use MoskowitzTimeSeriesMomentum instead.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Paper: J=3,6,9,12 formation; K=3,6,9,12 holding
PAPER_FORMATION_MONTHS = 6
PAPER_HOLDING_MONTHS = 6


class JegadeeshTitmanMomentum:
    """
    Cross-sectional momentum per Jegadeesh & Titman (1993).
    
    Works on DataFrame of returns (columns = assets, index = dates).
    Long top decile, short bottom decile by formation-period return.
    """

    def __init__(
        self,
        formation_period: int = 126,  # ~6 months daily
        holding_period: int = 126,
        n_long: int = 1,
        n_short: int = 1,
    ):
        """
        Args:
            formation_period: Bars for momentum ranking
            holding_period: Bars to hold
            n_long: Number of top assets to long (1 = top decile if 10 assets)
            n_short: Number of bottom assets to short
        """
        self.formation_period = formation_period
        self.holding_period = holding_period
        self.n_long = n_long
        self.n_short = n_short

    def generate_signals(self, returns: pd.DataFrame) -> pd.Series:
        """
        returns: DataFrame (dates x assets). Each column = asset return series.
        Returns: Series of portfolio returns (dollar-neutral: long - short).
        """
        cum_ret = (1 + returns).rolling(self.formation_period).apply(
            lambda x: np.prod(1 + x) - 1,
            raw=True,
        )
        signals = pd.Series(0.0, index=returns.index)
        for i in range(self.formation_period, len(returns), self.holding_period):
            row = cum_ret.iloc[i]
            valid = row.dropna()
            if len(valid) < self.n_long + self.n_short:
                continue
            ranked = valid.rank(ascending=False)
            winners = ranked[ranked <= self.n_long].index.tolist()
            losers = ranked[ranked > len(valid) - self.n_short].index.tolist()
            for j in range(min(self.holding_period, len(returns) - i)):
                idx = i + j
                if idx >= len(returns):
                    break
                port_ret = 0.0
                for w in winners:
                    if w in returns.columns:
                        port_ret += returns.loc[returns.index[idx], w] / self.n_long
                for l in losers:
                    if l in returns.columns:
                        port_ret -= returns.loc[returns.index[idx], l] / self.n_short
                signals.iloc[idx] = port_ret
        return signals

    def backtest_returns(
        self, returns: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Returns (portfolio_returns_series, same as strategy returns).
        The signal here IS the portfolio return (long-short).
        """
        strat_returns = self.generate_signals(returns)
        return strat_returns, strat_returns
