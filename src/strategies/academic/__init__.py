"""
Academic strategies — research paper replications with proper citations.

Each strategy is decorated with @StrategyRegistry.register.
Importing this package is sufficient to register all academic strategies.
"""

from .time_series_momentum import TimeSeriesMomentumStrategy
from .betting_against_beta import BettingAgainstBetaStrategy
from .low_vol_anomaly import LowVolAnomalyStrategy
from .carry_trade import CarryTradeStrategy
from .post_earnings_drift import PostEarningsDriftStrategy

__all__ = [
    "TimeSeriesMomentumStrategy",
    "BettingAgainstBetaStrategy",
    "LowVolAnomalyStrategy",
    "CarryTradeStrategy",
    "PostEarningsDriftStrategy",
]
