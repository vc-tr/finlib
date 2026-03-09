"""
Quant Lab strategy library.

Subpackages:
- stats:        Core statistical strategies (momentum, mean reversion, pairs trading)
- retail:       Retail/technical strategies — YouTube/TikTok guru debunking
- academic:     Research paper replications with proper citations
- econophysics: Physics-inspired quantitative approaches

All strategies inherit from Strategy (base.py) and self-register via
@StrategyRegistry.register on import.
"""

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry

__all__ = [
    "Strategy",
    "StrategyMeta",
    "StrategyRegistry",
    # Legacy names — lazy-loaded for backward compatibility
    "MeanReversionStrategy",
    "PairsTradingStrategy",
    "MomentumStrategy",
]


def __getattr__(name: str):
    """Lazy imports for backward compatibility."""
    if name == "MomentumStrategy":
        from src.strategies.stats.momentum import MomentumStrategy
        return MomentumStrategy
    if name == "MeanReversionStrategy":
        from src.strategies.stats.mean_reversion import MeanReversionStrategy
        return MeanReversionStrategy
    if name == "PairsTradingStrategy":
        from src.strategies.stats.pairs_trading import PairsTradingStrategy
        return PairsTradingStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
