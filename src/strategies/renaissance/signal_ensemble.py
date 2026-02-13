"""
Renaissance-style signal ensemble strategy.

Inspired by Renaissance Technologies / Medallion Fund approach:
- Combine multiple uncorrelated signals
- Use statistical pattern recognition
- Systematic, data-driven (no discretionary overlay)
- Focus on short-term, high-frequency patterns

This is a simplified, transparent implementation of the general philosophy.
The actual Medallion strategies are proprietary.
"""

import numpy as np
import pandas as pd
from typing import List, Callable, Optional


class RenaissanceSignalEnsemble:
    """
    Combines multiple signals with optional weighting.
    
    Philosophy: Identify patterns that repeat; combine signals for robustness.
    Uses: momentum, mean reversion, volatility regime, cross-sectional rank.
    """
    
    def __init__(
        self,
        signal_weights: Optional[dict] = None,
        min_signal_agreement: float = 0.5,
    ):
        """
        Args:
            signal_weights: Dict mapping signal name -> weight (default equal)
            min_signal_agreement: Min fraction of signals agreeing to trade
        """
        self.signal_weights = signal_weights or {}
        self.min_signal_agreement = min_signal_agreement
    
    def _momentum_signal(self, prices: pd.Series, lookback: int = 10) -> pd.Series:
        """Short-term momentum: sign of return."""
        ret = prices.pct_change(lookback)
        return np.sign(ret)
    
    def _mean_reversion_signal(
        self, prices: pd.Series, lookback: int = 20
    ) -> pd.Series:
        """Z-score: negative z -> long, positive z -> short."""
        mean = prices.rolling(lookback).mean()
        std = prices.rolling(lookback).std()
        z = (prices - mean) / std.replace(0, np.nan)
        return -np.sign(z)  # mean revert: buy low (neg z), sell high (pos z)
    
    def _volatility_regime_signal(
        self, prices: pd.Series, lookback: int = 20
    ) -> pd.Series:
        """Low vol -> trend follow; high vol -> reduce position."""
        ret = prices.pct_change()
        vol = ret.rolling(lookback).std()
        vol_rank = vol.rolling(lookback * 2).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
        )
        # In high vol, dampen; use inverse of vol rank as confidence
        return 1 - vol_rank  # higher when vol is low
    
    def _cross_sectional_rank_signal(
        self, prices: pd.DataFrame, lookback: int = 5
    ) -> pd.Series:
        """
        Rank by recent return (if multiple assets).
        For single asset, use percentile of rolling return.
        """
        if isinstance(prices, pd.Series):
            ret = prices.pct_change(lookback)
            rank = ret.rolling(lookback * 10).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10)
            )
            return 2 * rank - 1  # map to [-1, 1]
        # Multi-asset: rank each row
        ret = prices.pct_change(lookback)
        rank = ret.rank(axis=1, pct=True)
        return (2 * rank - 1).mean(axis=1)
    
    def get_default_signals(self, prices: pd.Series) -> dict:
        """Generate default signal set for single asset."""
        return {
            "momentum_10": self._momentum_signal(prices, 10),
            "momentum_20": self._momentum_signal(prices, 20),
            "mean_reversion": self._mean_reversion_signal(prices, 20),
            "vol_regime": self._volatility_regime_signal(prices, 20),
        }
    
    def combine_signals(
        self,
        signals: dict,
        weights: Optional[dict] = None,
    ) -> pd.Series:
        """
        Combine signals into single score in [-1, 1].
        """
        w = weights or self.signal_weights
        if not w:
            w = {k: 1.0 / len(signals) for k in signals}
        
        combined = pd.Series(0.0, index=list(signals.values())[0].index)
        total_w = 0
        for name, s in signals.items():
            weight = w.get(name, 1.0 / len(signals))
            combined = combined.add(s.fillna(0) * weight, fill_value=0)
            total_w += weight
        if total_w > 0:
            combined = combined / total_w
        return combined.clip(-1, 1)
    
    def generate_signals(
        self,
        prices: pd.Series,
        custom_signals: Optional[dict] = None,
    ) -> pd.Series:
        """
        Generate final trading signals: 1, -1, or 0.
        
        Uses min_signal_agreement: only trade when |combined| > threshold.
        """
        signals = custom_signals or self.get_default_signals(prices)
        combined = self.combine_signals(signals, self.signal_weights)
        
        out = pd.Series(0.0, index=prices.index)
        out[combined > self.min_signal_agreement] = 1
        out[combined < -self.min_signal_agreement] = -1
        return out.shift(1)  # avoid lookahead
    
    def backtest_returns(self, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Returns (signals, strategy_returns)."""
        signals = self.generate_signals(prices)
        returns = prices.pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
