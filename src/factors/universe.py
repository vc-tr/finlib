"""
Universe definitions for cross-sectional factor backtests.

Hardcoded liquid ETFs to avoid survivorship bias in backtests.
"""

from typing import List

# Liquid ETFs (high AUM, tight spreads) - snapshot as of common listing
LIQUID_ETFS: List[str] = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
    "GLD", "SLV", "USO", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "BND",
    "AGG", "TIP", "VNQ", "VYM", "VUG", "IVV", "IJH", "IJR", "MDY", "RSP",
    "SCHA", "SCHB", "SCHX", "SCHF", "SCHE", "SCHZ", "SCHO", "SCHP", "SCHD",
]


def get_universe(name: str = "liquid_etfs", n: int = 50) -> List[str]:
    """
    Return symbol list for the given universe.

    Args:
        name: Universe name (only "liquid_etfs" supported)
        n: Max symbols to return (default 50)

    Returns:
        List of ticker symbols
    """
    if name == "liquid_etfs":
        return LIQUID_ETFS[:n]
    raise ValueError(f"Unknown universe: {name}")
