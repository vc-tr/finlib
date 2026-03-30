"""
Betting Against Beta (BAB) — Frazzini & Pedersen (2014).

Long low-beta assets (leveraged), short high-beta assets (de-leveraged).
Exploits the empirical finding that the Security Market Line (SML) is flatter
than CAPM predicts — low-beta assets earn higher risk-adjusted returns.

Paper: "Betting Against Beta" — Frazzini & Pedersen (2014)
Journal of Financial Economics, 111(1), 1-23.

Mechanism: Leverage-constrained investors (pension funds, mutual funds)
overpay for high-beta assets, depressing their risk-adjusted returns.
Unconstrained investors can earn the alpha by shorting high-beta.

Implementation note: True BAB requires a universe of securities to estimate
beta against the market. This single-asset implementation uses rolling beta
estimated from returns vs a hypothetical market proxy (the asset's own
smoothed trend) as a demonstration of the concept. Cross-sectional BAB
is implemented in src/factors/ for multi-asset contexts.

Expected: Positive risk-adjusted alpha in cross-sectional studies.
Single-asset implementation is approximate — use factor-based BAB for
rigorous testing.
"""

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class BettingAgainstBetaStrategy(Strategy):
    """
    Single-asset BAB proxy: fade high-momentum (high-beta-like) periods,
    load up during low-volatility (low-beta-like) regimes.

    Uses rolling volatility as a beta proxy: low vol ≈ low beta → long,
    high vol ≈ high beta → reduce/short position.

    For true cross-sectional BAB across a stock universe, see:
    src/factors/factors.py — factor_beta_rank.
    """

    def __init__(
        self,
        beta_lookback: int = 252,
        signal_lookback: int = 63,
        beta_quantile: float = 0.33,
    ):
        """
        Args:
            beta_lookback: Window to estimate rolling beta (days)
            signal_lookback: Window for current beta percentile rank
            beta_quantile: Threshold: low-beta if rank < quantile, high if > 1-quantile
        """
        self.beta_lookback = beta_lookback
        self.signal_lookback = signal_lookback
        self.beta_quantile = beta_quantile

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="betting_against_beta",
            category="academic",
            source="Frazzini & Pedersen (2014)",
            description=(
                "Long in low-volatility (low-beta proxy) regimes, "
                "short in high-volatility (high-beta proxy) regimes"
            ),
            hypothesis=(
                "Leverage-constrained investors bid up high-beta assets, "
                "creating a flat SML. Low-beta assets earn positive alpha "
                "on a risk-adjusted basis (CAPM alpha positive for low beta)."
            ),
            expected_result=(
                "Modest positive risk-adjusted return. "
                "Single-asset implementation is an approximation — "
                "beta is inherently cross-sectional. True BAB requires universe."
            ),
            source_url="https://doi.org/10.1016/j.jfineco.2013.10.005",
            tags=["beta", "factor", "low-volatility", "academic", "leverage"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change()
        # Rolling volatility as beta proxy
        rolling_vol = returns.rolling(self.beta_lookback).std()
        # Percentile rank of current vol over signal lookback
        vol_rank = rolling_vol.rolling(self.signal_lookback).rank(pct=True)

        signals = pd.Series(0.0, index=prices.index)
        signals[vol_rank < self.beta_quantile] = 1.0       # low beta → long
        signals[vol_rank > (1 - self.beta_quantile)] = -1.0  # high beta → short
        return signals

    def parameter_grid(self):
        return {
            "beta_lookback": [126, 252],
            "signal_lookback": [21, 63],
            "beta_quantile": [0.25, 0.33, 0.40],
        }


def rolling_beta(asset_returns: pd.Series, market_returns: pd.Series,
                 window: int = 252) -> pd.Series:
    """
    Utility: compute rolling beta of asset vs market.

    Args:
        asset_returns: Asset return series
        market_returns: Market (benchmark) return series
        window: Rolling window in days

    Returns:
        pd.Series of rolling beta estimates
    """
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var.replace(0, np.nan)
