"""
Pairs Trading (Gatev, Goetzmann, Rouwenhorst 2006).

Paper: "Pairs Trading: Performance of a Relative-Value Arbitrage Rule",
       Review of Financial Studies, 19(3), 797-827.

Strategy: Match pairs by minimum sum of squared deviations of normalized prices.
          Trade when spread exceeds 2 std; close when it reverts to 0.
          Uses normalized price = price / first_price (not cointegration).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from itertools import combinations


class GatevGoetzmannRouwenhorstPairs:
    """
    Pairs trading per Gatev, Goetzmann, Rouwenhorst (2006).
    
    Formation: Normalize prices to start at 1. Find pairs with minimum SSD.
    Trading: When spread (A_norm - B_norm) exceeds 2σ, open; close when crosses 0.
    """

    def __init__(
        self,
        formation_period: int = 252,
        entry_std: float = 2.0,
        exit_at_zero: bool = True,
    ):
        """
        Args:
            formation_period: Bars for pair formation and SSD calculation
            entry_std: Enter when |spread| > entry_std * rolling_std
            exit_at_zero: Close when spread crosses 0 (paper default)
        """
        self.formation_period = formation_period
        self.entry_std = entry_std
        self.exit_at_zero = exit_at_zero

    def _normalize(self, series: pd.Series) -> pd.Series:
        """Normalize to start at 1."""
        return series / series.iloc[0]

    def _ssd(self, a: np.ndarray, b: np.ndarray) -> float:
        """Sum of squared deviations between normalized series."""
        a_norm = a / a[0]
        b_norm = b / b[0]
        return np.sum((a_norm - b_norm) ** 2)

    def find_best_pair(
        self, prices: pd.DataFrame
    ) -> Optional[Tuple[str, str, float]]:
        """
        Find pair with minimum SSD over formation period.
        prices: DataFrame with columns = assets.
        Returns (asset_a, asset_b, ssd) or None.
        """
        cols = prices.columns.tolist()
        if len(cols) < 2:
            return None
        best = None
        best_ssd = np.inf
        for i, j in combinations(cols, 2):
            a = prices[i].iloc[: self.formation_period].values
            b = prices[j].iloc[: self.formation_period].values
            if np.any(np.isnan(a)) or np.any(np.isnan(b)):
                continue
            ssd = self._ssd(a, b)
            if ssd < best_ssd:
                best_ssd = ssd
                best = (i, j, ssd)
        return best

    def generate_signals(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        GGR-style: spread = normalized A - normalized B.
        Enter when |spread| > 2*std, exit when spread crosses 0.
        """
        a_norm = self._normalize(price_a)
        b_norm = self._normalize(price_b)
        a_norm, b_norm = a_norm.align(b_norm, join="inner")
        spread = a_norm - b_norm
        mean = spread.rolling(self.formation_period).mean()
        std = spread.rolling(self.formation_period).std()
        z = (spread - mean) / std.replace(0, np.nan)
        signals = pd.Series(0.0, index=spread.index)
        position = 0
        for i in range(self.formation_period, len(spread)):
            if np.isnan(z.iloc[i]):
                continue
            if position == 0:
                if z.iloc[i] > self.entry_std:
                    position = -1
                elif z.iloc[i] < -self.entry_std:
                    position = 1
            elif position == 1:
                if self.exit_at_zero and spread.iloc[i] >= 0:
                    position = 0
                elif not self.exit_at_zero and z.iloc[i] >= -0.5:
                    position = 0
            elif position == -1:
                if self.exit_at_zero and spread.iloc[i] <= 0:
                    position = 0
                elif not self.exit_at_zero and z.iloc[i] <= 0.5:
                    position = 0
            signals.iloc[i] = position
        return signals, spread

    def backtest_returns(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Returns (signals, strategy_returns). Strategy return from spread."""
        signals, spread = self.generate_signals(price_a, price_b)
        scale = price_a.rolling(20).mean().abs().replace(0, np.nan).ffill().bfill()
        spread_ret = (signals.shift(1) * spread.diff()) / scale
        spread_ret = spread_ret.fillna(0).clip(-0.1, 0.1)
        return signals, spread_ret
