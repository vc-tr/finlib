"""
Pairs trading (statistical arbitrage) strategy.

Identifies cointegrated pairs and trades the spread when it deviates.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.stattools import coint


class PairsTradingStrategy:
    """
    Pairs trading using cointegration and z-score on the spread.
    
    Spread = price_A - hedge_ratio * price_B
    Trade when spread deviates from mean (z-score).
    """
    
    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 60,
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
    
    def find_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        """OLS hedge ratio: y = alpha + beta*x => beta is hedge ratio."""
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
        return float(model.coef_[0])
    
    def test_cointegration(
        self, series_a: pd.Series, series_b: pd.Series
    ) -> Tuple[bool, float]:
        """
        Engle-Granger cointegration test.
        
        Returns:
            (is_cointegrated, p_value)
        """
        series_a = series_a.dropna()
        series_b = series_b.dropna()
        common_idx = series_a.index.intersection(series_b.index)
        a = series_a.loc[common_idx].values
        b = series_b.loc[common_idx].values
        score, pvalue, _ = coint(a, b)
        return pvalue < 0.05, pvalue
    
    def compute_spread(
        self, price_a: pd.Series, price_b: pd.Series, hedge_ratio: float
    ) -> pd.Series:
        """Spread = A - hedge_ratio * B"""
        a, b = price_a.align(price_b, join="inner")
        return a - hedge_ratio * b
    
    def generate_signals(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, float]:
        """
        Generate pairs trading signals.
        
        Returns:
            (signals, spread, hedge_ratio)
            signals: 1 = long A/short B, -1 = short A/long B, 0 = flat
        """
        if hedge_ratio is None:
            hedge_ratio = self.find_hedge_ratio(price_a, price_b)
        
        spread = self.compute_spread(price_a, price_b, hedge_ratio)
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        z = (spread - mean) / std.replace(0, np.nan)
        
        signals = pd.Series(0.0, index=spread.index)
        position = 0
        for i in range(self.lookback, len(spread)):
            if np.isnan(z.iloc[i]):
                continue
            if position == 0:
                if z.iloc[i] < -self.entry_z:
                    position = 1  # long spread = long A, short B
                elif z.iloc[i] > self.entry_z:
                    position = -1
            elif position == 1:
                if z.iloc[i] >= -self.exit_z:
                    position = 0
            elif position == -1:
                if z.iloc[i] <= self.exit_z:
                    position = 0
            signals.iloc[i] = position
        
        return signals, spread, hedge_ratio
