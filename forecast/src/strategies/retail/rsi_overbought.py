"""
RSI Overbought/Oversold — contrarian mean-reversion.

Buy when RSI(14) < 30 (oversold), sell/short when RSI(14) > 70 (overbought).
Taught in every retail trading course as a reliable reversal signal.

Expected: Marginal edge at best — often fades momentum in trending markets,
leading to catching falling knives. Completely arbitrary threshold levels.
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI via exponential smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # When avg_loss = 0: all gains, no losses → RSI = 100
    rsi = avg_gain.copy()
    mask = avg_loss != 0
    rsi[mask] = 100 - (100 / (1 + avg_gain[mask] / avg_loss[mask]))
    rsi[~mask] = 100.0
    return rsi


@StrategyRegistry.register
class RSIOverboughtStrategy(Strategy):
    """Contrarian RSI: long oversold, short overbought."""

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="rsi_overbought",
            category="retail",
            source="Wilder (1978) — New Concepts in Technical Trading Systems",
            description=(
                f"Long when RSI({self.period}) < {self.oversold}, "
                f"short when RSI({self.period}) > {self.overbought}, flat otherwise"
            ),
            hypothesis=(
                "Assets oscillate between overbought and oversold states. "
                "Extreme RSI readings indicate exhausted moves likely to reverse."
            ),
            expected_result=(
                "Near-zero OOS Sharpe. Misses sustained trends by fading momentum. "
                "Catches 'falling knives' in bear markets. Threshold levels are arbitrary."
            ),
            source_url=None,
            tags=["rsi", "mean-reversion", "oscillator", "retail", "debunk"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        rsi = _rsi(prices, self.period)
        signals = pd.Series(0.0, index=prices.index)
        signals[rsi < self.oversold] = 1.0
        signals[rsi > self.overbought] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "period": [7, 14, 21],
            "oversold": [20.0, 30.0],
            "overbought": [70.0, 80.0],
        }
