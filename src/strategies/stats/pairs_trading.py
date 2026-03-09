"""
Pairs trading (statistical arbitrage) strategy.

Identifies cointegrated pairs and trades the spread when it deviates.
Note: This is a multi-asset strategy; generate_signals() takes two price series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from statsmodels.tsa.stattools import coint

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class PairsTradingStrategy(Strategy):
    """
    Pairs trading using cointegration and z-score on the spread.

    Spread = price_A - hedge_ratio * price_B
    Trade when spread deviates from mean (z-score).

    Note: This is a multi-asset strategy. Call generate_signals(price_a, price_b)
    rather than the single-price interface used by single-asset strategies.
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

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="pairs_trading",
            category="stats",
            source="Statistical arbitrage — cointegration-based pairs trading",
            description="Trade the spread between two cointegrated assets when it deviates from mean",
            hypothesis="Cointegrated pair spread is stationary; deviations are temporary",
            expected_result="Works when cointegration is stable; breaks down in regime changes",
            tags=["pairs", "stat-arb", "cointegration", "multi-asset"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Single-asset interface stub. Use generate_pair_signals() for actual trading.
        Returns zeros — this strategy requires two price series.
        """
        return pd.Series(0.0, index=prices.index)

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

    def generate_pair_signals(
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
                    position = 1
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

    def parameter_grid(self) -> Dict[str, List]:
        return {
            "entry_z": [1.5, 2.0, 2.5],
            "exit_z": [0.0, 0.5, 1.0],
            "lookback": [30, 60, 120],
        }
