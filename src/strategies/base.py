"""
Base class and metadata for all Quant Lab strategies.

Every strategy must inherit from Strategy and implement:
  - meta() -> StrategyMeta
  - generate_signals(prices) -> pd.Series of {-1, 0, 1}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class StrategyMeta:
    """Metadata for strategy cataloging, reporting, and attribution."""
    name: str
    category: str          # "stats" | "retail" | "academic" | "econophysics"
    source: str            # e.g. "Jegadeesh & Titman (1993)" or "YouTube: Delta Trading"
    description: str = ""
    hypothesis: str = ""   # What market inefficiency does this exploit?
    expected_result: str = ""  # What do we expect from rigorous OOS testing?
    source_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class Strategy(ABC):
    """Base class for all strategies in the Quant Lab."""

    @abstractmethod
    def meta(self) -> StrategyMeta:
        """Return strategy metadata for catalog and reporting."""
        ...

    @abstractmethod
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate trading signals from a price series.

        Args:
            prices: Close prices with DatetimeIndex.

        Returns:
            pd.Series of float in {-1.0, 0.0, 1.0}.
            Signal at index t represents the desired position for bar t+1.
            The backtester applies shift(1) internally to prevent lookahead.
            NaN values are treated as 0 (flat).
        """
        ...

    def generate_positions(self, prices: pd.Series) -> pd.Series:
        """
        Generate positions, optionally with hold logic applied.
        Default: same as generate_signals(). Override to add min-hold smoothing.
        """
        return self.generate_signals(prices)

    def backtest_returns(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Convenience method: return (positions, strategy_returns).
        Applies shift(1) to prevent lookahead.
        """
        positions = self.generate_positions(prices)
        ret = prices.pct_change()
        strategy_returns = positions.shift(1).fillna(0) * ret
        return positions, strategy_returns

    def parameter_grid(self) -> Dict[str, list]:
        """
        Return parameter grid for sensitivity analysis.
        Override in strategies to enable --sweep mode.

        Example:
            return {"lookback": [10, 20, 50], "threshold": [0.0, 0.01]}
        """
        return {}
