"""
Volume Spike Momentum — buy large moves as proxies for volume surges.

True implementation requires OHLCV data (volume). Since this strategy module
operates on close-price series, we use large absolute returns as a proxy for
volume spikes — days with anomalously large price moves often coincide with
high volume (earnings, news, institutional activity).

Long when proxy-spike + positive return; short when proxy-spike + negative return.

Expected: Chases momentum after the fact. By the time a "volume spike" is
confirmed at close, the move is over. High turnover, costs dominate.
"""

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class VolumeSpikeStrategy(Strategy):
    """
    Trade in direction of anomalously large daily moves (volume spike proxy).

    A true volume spike strategy needs intraday OHLCV data. This implementation
    uses |pct_change| > spike_mult × rolling median |pct_change| as a proxy.
    """

    def __init__(self, lookback: int = 20, spike_mult: float = 2.0):
        """
        Args:
            lookback: Rolling window for median absolute return calculation
            spike_mult: Threshold multiplier — spike if |ret| > spike_mult × median(|ret|)
        """
        self.lookback = lookback
        self.spike_mult = spike_mult

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="volume_spike",
            category="retail",
            source="Retail day trading lore — common YouTube/TikTok strategy",
            description=(
                f"Long on large positive move ({self.spike_mult}x median), "
                f"short on large negative move; uses return magnitude as volume proxy"
            ),
            hypothesis=(
                "High volume + directional move signals institutional conviction. "
                "Retail traders 'buy the action' expecting continuation."
            ),
            expected_result=(
                "Negative expected value after costs. Spike days have high mean-reversion "
                "(news absorbed at open, fades intraday). Close-to-close chasing "
                "systematically buys the top and sells the bottom of spike days."
            ),
            source_url=None,
            tags=["volume", "momentum", "retail", "debunk", "spike"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        ret = prices.pct_change()
        abs_ret = ret.abs()
        median_abs = abs_ret.rolling(self.lookback).median()
        spike = abs_ret > self.spike_mult * median_abs

        direction = np.sign(ret)
        signals = pd.Series(0.0, index=prices.index)
        signals[spike & (direction > 0)] = 1.0
        signals[spike & (direction < 0)] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "lookback": [10, 20, 40],
            "spike_mult": [1.5, 2.0, 3.0],
        }
