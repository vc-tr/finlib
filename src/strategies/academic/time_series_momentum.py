"""
Time-Series Momentum (TSMOM) — Moskowitz, Ooi & Pedersen (2012).

Long if the asset's 12-month return is positive, short if negative.
Uses 12-1 formation (excludes most recent month to avoid short-term reversal).

Paper: "Time Series Momentum" — Moskowitz, Ooi, Pedersen (2012)
Journal of Financial Economics, 104(2), 228-250.

Key finding: Positive autocorrelation in monthly returns over 1-12 month horizons
across 58 liquid instruments (futures). Evidence of trend-following premium.

Expected: Positive OOS Sharpe on liquid instruments with long histories.
Works better on futures than equities. Degrades post-2012 as strategy became
crowded. Beta to trend-following CTAs is high.
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class TimeSeriesMomentumStrategy(Strategy):
    """
    TSMOM: Long if 12-month return > 0, short if < 0.

    Formation: lookback_long - lookback_short months (excludes recent reversal).
    Default: 12-1 = 11 months of signal, skip most recent month.
    """

    def __init__(self, lookback_long: int = 252, lookback_short: int = 21):
        """
        Args:
            lookback_long: Long lookback in trading days (default 252 = ~12 months)
            lookback_short: Short lookback to exclude in days (default 21 = ~1 month)
        """
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="time_series_momentum",
            category="academic",
            source="Moskowitz, Ooi & Pedersen (2012)",
            description=(
                f"Long if {self.lookback_long}d return > 0, "
                f"short if < 0 (excludes most recent {self.lookback_short}d)"
            ),
            hypothesis=(
                "Positive autocorrelation in returns at 1-12 month horizons. "
                "Initial underreaction to information is followed by gradual "
                "price discovery, creating predictable trend continuation."
            ),
            expected_result=(
                "Positive Sharpe on liquid futures/large-cap equities. "
                "Degrades after ~2012 due to strategy crowding. "
                "Significant drawdowns in trend reversals (2009, 2020)."
            ),
            source_url="https://doi.org/10.1016/j.jfineco.2011.11.003",
            tags=["momentum", "trend", "academic", "time-series", "futures"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # 12-month return excluding most recent 1 month
        long_ret = prices.pct_change(self.lookback_long)
        short_ret = prices.pct_change(self.lookback_short)
        formation_ret = (1 + long_ret) / (1 + short_ret) - 1

        signals = pd.Series(0.0, index=prices.index)
        signals[formation_ret > 0] = 1.0
        signals[formation_ret < 0] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "lookback_long": [126, 252, 504],    # 6m, 12m, 24m
            "lookback_short": [0, 21, 42],        # no exclusion, 1m, 2m
        }
