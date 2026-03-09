"""
Momentum (trend-following) strategy.

Buys winners, sells losers. Uses past returns as signal.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


def _signals_to_position_with_hold(signals: pd.Series, min_hold_bars: int) -> pd.Series:
    """
    Convert raw signals to position with minimum hold.
    Position changes only when: (a) flat->directional, or (b) held min_hold_bars and signal differs.
    Reduces trade spam from bar-to-bar signal flips.
    """
    if min_hold_bars <= 1:
        return signals
    out = pd.Series(0.0, index=signals.index)
    pos = 0.0
    bars_held = 0
    for i in range(len(signals)):
        sig = signals.iloc[i]
        if np.isnan(sig):
            sig = 0.0
        if pos == 0:
            pos = sig
            bars_held = 1 if sig != 0 else 0
        elif sig != pos:
            if bars_held >= min_hold_bars or sig == 0:
                pos = sig
                bars_held = 1 if sig != 0 else 0
            else:
                bars_held += 1
        else:
            bars_held += 1
        out.iloc[i] = pos
    return out


@StrategyRegistry.register
class MomentumStrategy(Strategy):
    """
    Momentum strategy based on lookback returns.

    Signal = sign(return over lookback period)
    Long when past return > threshold, short when < -threshold.
    Uses min_hold_bars to reduce churn (default 1 for daily, 5 for minute).
    """

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 0.0,
        min_hold_bars: int = 1,
    ):
        """
        Args:
            lookback: Period for momentum calculation
            threshold: Minimum return to trigger (0 = any positive/negative)
            min_hold_bars: Min bars to hold position before changing (1 for 1d, 5 for 1m)
        """
        self.lookback = lookback
        self.threshold = threshold
        self.min_hold_bars = min_hold_bars

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="momentum",
            category="stats",
            source="Classic time-series momentum",
            description="Long recent winners, short recent losers based on lookback returns",
            hypothesis="Trend continuation — assets with positive recent returns keep outperforming",
            expected_result="Positive OOS Sharpe on daily data; sensitive to lookback and costs",
            tags=["trend", "momentum", "time-series"],
        )

    def compute_momentum(self, prices: pd.Series) -> pd.Series:
        """Momentum = (price / price_n_lookback_ago) - 1"""
        return prices.pct_change(self.lookback)

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate raw signals: 1 = long, -1 = short, 0 = flat.
        Signal at close t = position held during bar t+1 (backtester applies shift).
        """
        mom = self.compute_momentum(prices)
        signals = pd.Series(0.0, index=prices.index)
        signals[mom > self.threshold] = 1
        signals[mom < -self.threshold] = -1
        return signals

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        """Generate position series with min_hold applied (use for backtest)."""
        raw = self.generate_signals(prices)
        return _signals_to_position_with_hold(raw, self.min_hold_bars)

    def backtest_returns(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Returns (positions, strategy_returns). Positions have min_hold applied."""
        positions = self.generate_positions(prices)
        returns = prices.pct_change()
        strategy_returns = positions.shift(1).fillna(0) * returns
        return positions, strategy_returns

    def parameter_grid(self) -> Dict[str, List]:
        return {
            "lookback": [5, 10, 20, 50],
            "threshold": [0.0, 0.005, 0.01],
            "min_hold_bars": [1, 5, 10],
        }
