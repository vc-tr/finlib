"""
Low-Volatility Anomaly — Ang, Hodrick, Xing & Zhang (2006).

Low-volatility / low-idiosyncratic-volatility stocks earn higher returns
than high-volatility stocks, violating the standard risk-return tradeoff.
A single-asset implementation: go long in low-vol regimes, reduce in high-vol.

Paper: "The Cross-Section of Volatility and Expected Returns"
Ang, Hodrick, Xing & Zhang (2006)
Journal of Finance, 61(1), 259-299.

Finding: Stocks in the highest idiosyncratic volatility quintile earn
0.31% less per month than stocks in the lowest quintile (1963-2000, CRSP).
The anomaly persists internationally and survives FF3 adjustment.

Mechanism: Lottery preferences, investor overconfidence in high-vol stocks,
benchmark constraints that prevent shorting volatile names.

Expected: Positive alpha in cross-sectional tests. Single-asset implementation
shows vol-regime timing — hold during low-vol periods, reduce/short during
high-vol spikes (left-tail risk).
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class LowVolAnomalyStrategy(Strategy):
    """
    Low-volatility regime timing: long in calm markets, reduce in turbulence.

    Approximates the cross-sectional low-vol anomaly in a single-asset context
    by comparing current realized volatility to its historical distribution.
    """

    def __init__(
        self,
        vol_lookback: int = 21,
        rank_lookback: int = 252,
        low_vol_quantile: float = 0.33,
        high_vol_quantile: float = 0.67,
    ):
        """
        Args:
            vol_lookback: Window to estimate realized vol (days)
            rank_lookback: Window for vol percentile rank (days)
            low_vol_quantile: Below this rank → long (low-vol regime)
            high_vol_quantile: Above this rank → short (high-vol regime)
        """
        self.vol_lookback = vol_lookback
        self.rank_lookback = rank_lookback
        self.low_vol_quantile = low_vol_quantile
        self.high_vol_quantile = high_vol_quantile

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="low_vol_anomaly",
            category="academic",
            source="Ang, Hodrick, Xing & Zhang (2006)",
            description=(
                f"Long when {self.vol_lookback}d vol is in bottom "
                f"{self.low_vol_quantile:.0%} of {self.rank_lookback}d history; "
                f"short in top {1-self.high_vol_quantile:.0%}"
            ),
            hypothesis=(
                "Low-volatility assets earn higher risk-adjusted returns due to "
                "lottery preferences and benchmark constraints on institutional investors. "
                "Low vol regime ≈ favorable risk/reward environment."
            ),
            expected_result=(
                "Moderate Sharpe on timing the vol regime. "
                "Better as cross-sectional stock selection (needs universe). "
                "Single-asset version captures vol clustering but not the anomaly directly."
            ),
            source_url="https://doi.org/10.1111/j.1540-6261.2006.00836.x",
            tags=["volatility", "anomaly", "factor", "academic", "low-vol"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change()
        realized_vol = returns.rolling(self.vol_lookback).std()
        vol_rank = realized_vol.rolling(self.rank_lookback).rank(pct=True)

        signals = pd.Series(0.0, index=prices.index)
        signals[vol_rank <= self.low_vol_quantile] = 1.0
        signals[vol_rank >= self.high_vol_quantile] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "vol_lookback": [10, 21, 42],
            "rank_lookback": [126, 252],
            "low_vol_quantile": [0.25, 0.33],
            "high_vol_quantile": [0.67, 0.75],
        }
