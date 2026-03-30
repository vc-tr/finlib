"""
Post-Earnings Announcement Drift (PEAD) — Ball & Brown (1968).

Prices drift in the direction of earnings surprises for weeks to months
after announcement. One of the oldest and most robust anomalies in finance.

Paper: "An Empirical Evaluation of Accounting Income Numbers"
Ball & Brown (1968), Journal of Accounting Research, 6(2), 159-178.

Mechanism: Initial underreaction to earnings news — investors fail to
fully incorporate the earnings surprise immediately. Gradual information
diffusion (analysts revisions, media coverage) drives drift.

Implementation note: True PEAD requires earnings announcement dates and
analyst consensus estimates (SUE = Standardized Unexpected Earnings).
This implementation uses a simplified proxy: large one-day return as a
"surprise" event, then rides the subsequent drift.

Expected: Robust but shrinking alpha. Strongest in small-caps where
information diffuses slowly. Near-zero OOS after 2010 for large-caps
(faster information processing, lower transaction costs).
"""

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class PostEarningsDriftStrategy(Strategy):
    """
    PEAD proxy: ride the momentum after a large single-day return event.

    A large positive return on day 0 proxies for a positive earnings surprise.
    Strategy goes long for drift_period days, then exits.

    True implementation requires Compustat earnings data + IBES consensus.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.03,
        drift_period: int = 20,
        vol_lookback: int = 63,
    ):
        """
        Args:
            surprise_threshold: Min absolute return to classify as "surprise" (0.03 = 3%)
            drift_period: Days to hold position after surprise
            vol_lookback: Window for volatility-adjusted threshold
        """
        self.surprise_threshold = surprise_threshold
        self.drift_period = drift_period
        self.vol_lookback = vol_lookback

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="post_earnings_drift",
            category="academic",
            source="Ball & Brown (1968)",
            description=(
                f"Long {self.drift_period}d after large positive return "
                f"(> {self.surprise_threshold:.0%}), "
                f"short after large negative return (PEAD proxy)"
            ),
            hypothesis=(
                "Market underreacts to earnings surprises. Prices drift in the "
                "direction of the surprise for weeks after the announcement as "
                "information gradually diffuses through the market."
            ),
            expected_result=(
                "Historically robust in academic studies. Severely degraded in "
                "large-caps post-2010 due to faster information processing. "
                "Proxy implementation misses true earnings events — noisy signal."
            ),
            source_url="https://doi.org/10.2307/2490232",
            tags=["earnings", "drift", "event-driven", "academic", "pead"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ret = prices.pct_change()
        # Volatility-adjusted surprise detection
        rolling_vol = ret.rolling(self.vol_lookback).std()
        threshold = self.surprise_threshold + rolling_vol  # adaptive threshold

        # Detect surprise events
        is_pos_surprise = ret > threshold
        is_neg_surprise = ret < -threshold

        # Hold signal for drift_period after event
        # Use rolling max to propagate signal forward
        pos_signal = is_pos_surprise.astype(float)
        neg_signal = is_neg_surprise.astype(float)

        # Forward-fill for drift_period bars after event
        pos_drift = pos_signal.rolling(self.drift_period, min_periods=1).max()
        neg_drift = neg_signal.rolling(self.drift_period, min_periods=1).max()

        signals = pd.Series(0.0, index=prices.index)
        signals[pos_drift > 0] = 1.0
        signals[neg_drift > 0] = -1.0
        # If both triggered (conflicting), flat
        signals[pos_drift * neg_drift > 0] = 0.0
        return signals

    def parameter_grid(self):
        return {
            "surprise_threshold": [0.02, 0.03, 0.05],
            "drift_period": [10, 20, 40],
        }


# Placeholder for true SUE-based PEAD
def compute_sue(earnings: pd.Series, forecast: pd.Series,
                price: pd.Series) -> pd.Series:
    """
    Standardized Unexpected Earnings (SUE).

    SUE = (Actual EPS - Forecast EPS) / std(prior forecast errors)

    Requires Compustat + IBES data. Not available in this price-only implementation.
    Shown for documentation purposes.

    Args:
        earnings: Actual reported EPS
        forecast: Analyst consensus EPS forecast
        price: Close price (for scaling)

    Returns:
        SUE series
    """
    surprise = earnings - forecast
    std_error = (earnings - forecast).rolling(8).std()  # 8 quarters
    return surprise / std_error.replace(0, np.nan)
