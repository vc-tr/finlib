"""
Quantitative finance strategies.

Submodules:
- stats: Mean reversion, pairs trading, momentum
"""

__all__ = [
    "MeanReversionStrategy",
    "PairsTradingStrategy",
    "MomentumStrategy",
]


def __getattr__(name: str):
    """Lazy imports."""
    if name == "MeanReversionStrategy":
        from .stats.mean_reversion import MeanReversionStrategy
        return MeanReversionStrategy
    if name == "PairsTradingStrategy":
        from .stats.pairs_trading import PairsTradingStrategy
        return PairsTradingStrategy
    if name == "MomentumStrategy":
        from .stats.momentum import MomentumStrategy
        return MomentumStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
