"""
Portfolio allocation across multiple strategies or assets.

Supports: equal weight, risk parity, inverse volatility, custom weights.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional, Dict, Union


class AllocationMethod(str, Enum):
    """Allocation methods for multi-strategy/asset portfolios."""
    EQUAL = "equal"
    RISK_PARITY = "risk_parity"
    INVERSE_VOLATILITY = "inverse_volatility"
    CUSTOM = "custom"


class PortfolioAllocator:
    """
    Allocate weights across strategies (or assets) using various methods.
    
    Input: DataFrame of returns (columns = strategies, index = dates)
    Output: weights per strategy, used to compute portfolio returns.
    """

    def __init__(
        self,
        method: AllocationMethod = AllocationMethod.EQUAL,
        custom_weights: Optional[Dict[str, float]] = None,
        lookback: int = 63,
        rebalance_freq: int = 21,
    ):
        """
        Args:
            method: Allocation method
            custom_weights: For CUSTOM method, dict of strategy_name -> weight
            lookback: Bars for volatility/covariance estimation
            rebalance_freq: Rebalance every N bars (0 = no rebalance, use initial)
        """
        self.method = method
        self.custom_weights = custom_weights or {}
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq

    def _equal_weights(self, n: int) -> np.ndarray:
        """Equal weight: 1/n for each."""
        return np.ones(n) / n

    def _inverse_volatility_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Weight inversely proportional to volatility."""
        vol = returns.tail(self.lookback).std()
        vol = vol.replace(0, np.nan)
        inv_vol = 1 / vol
        w = inv_vol / inv_vol.sum()
        return w.fillna(0).values

    def _risk_parity_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Risk parity: each asset contributes equally to portfolio risk.
        Simplified: iterative inverse-vol with correlation adjustment.
        """
        cov = returns.tail(self.lookback).cov()
        vol = np.sqrt(np.diag(cov))
        vol = np.where(vol < 1e-10, 1e-10, vol)
        # Start with inverse vol, then iterate toward equal risk contribution
        w = 1 / vol
        w = w / w.sum()
        for _ in range(20):
            marginal_contrib = cov.values @ w
            risk_contrib = w * marginal_contrib
            total_risk = risk_contrib.sum()
            if total_risk < 1e-10:
                break
            target = total_risk / len(w)
            ratio = np.clip(target / (risk_contrib + 1e-10), 0.1, 10)
            w = w * np.sqrt(ratio)
            w = np.clip(w, 0, 1)
            w = w / w.sum()
        return w

    def compute_weights(
        self,
        returns: pd.DataFrame,
        t: int,
    ) -> np.ndarray:
        """
        Compute weights at time t based on method and lookback.
        
        returns: DataFrame (dates x strategies)
        t: current index (0-based)
        """
        cols = returns.columns.tolist()
        n = len(cols)
        if n == 0:
            return np.array([])
        start = max(0, t - self.lookback)
        window = returns.iloc[start:t + 1]
        if self.method == AllocationMethod.EQUAL:
            return self._equal_weights(n)
        if self.method == AllocationMethod.CUSTOM:
            w = np.array([self.custom_weights.get(c, 0) for c in cols])
            if w.sum() == 0:
                return self._equal_weights(n)
            return w / w.sum()
        if self.method == AllocationMethod.INVERSE_VOLATILITY:
            if len(window) < 5:
                return self._equal_weights(n)
            return self._inverse_volatility_weights(window)
        if self.method == AllocationMethod.RISK_PARITY:
            if len(window) < 10:
                return self._equal_weights(n)
            return self._risk_parity_weights(window)
        return self._equal_weights(n)

    def allocate(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce weight series for each strategy over time.
        
        Returns: DataFrame (dates x strategies) of weights.
        """
        weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        for i in range(len(returns)):
            if self.rebalance_freq == 0 and i > 0:
                weights.iloc[i] = weights.iloc[0]
            elif self.rebalance_freq > 0 and i % self.rebalance_freq != 0 and i > 0:
                weights.iloc[i] = weights.iloc[i - 1]
            else:
                w = self.compute_weights(returns, i)
                weights.iloc[i] = w
        return weights.ffill().fillna(0)

    def portfolio_returns(
        self,
        returns: pd.DataFrame,
        weights: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute portfolio returns: sum over strategies of (weight * strategy_return).
        """
        if weights is None:
            weights = self.allocate(returns)
        aligned = returns.reindex(weights.index).fillna(0)
        port_ret = (weights * aligned).sum(axis=1)
        return port_ret
