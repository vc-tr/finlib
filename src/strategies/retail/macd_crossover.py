"""
MACD Crossover — Moving Average Convergence Divergence.

Long when MACD line crosses above signal line.
Short when MACD line crosses below signal line.

One of the most widely taught indicators in retail trading.

Expected: Noise trading. MACD is a lagged derivative of price — crosses
fire frequently and mostly on noise. Costs destroy any apparent edge.
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class MACDCrossoverStrategy(Strategy):
    """MACD line vs signal line crossover."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="macd_crossover",
            category="retail",
            source="Appel (1979) — Technical Analysis — Power Tools for Active Investors",
            description=(
                f"Long when MACD({self.fast},{self.slow}) crosses above "
                f"signal({self.signal}), short when below"
            ),
            hypothesis=(
                "MACD crossovers identify shifts in momentum before they are "
                "visible in price alone. Crossing above signal = bullish, below = bearish."
            ),
            expected_result=(
                "Poor OOS Sharpe. Extremely high turnover at default parameters. "
                "MACD is a derivative of EMAs, adding noise rather than information. "
                "Near-random crossover frequency on daily data."
            ),
            source_url=None,
            tags=["macd", "momentum", "crossover", "retail", "debunk"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        fast_ema = prices.ewm(span=self.fast, adjust=False).mean()
        slow_ema = prices.ewm(span=self.slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        signals = pd.Series(0.0, index=prices.index)
        signals[macd_line > signal_line] = 1.0
        signals[macd_line < signal_line] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "fast": [8, 12, 16],
            "slow": [20, 26, 34],
            "signal": [7, 9, 12],
        }
