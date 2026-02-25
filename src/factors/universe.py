"""
Universe definitions for cross-sectional factor backtests.

UniverseRegistry provides symbols + metadata for multiple universes.
Hardcoded ETFs to avoid survivorship bias in backtests.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class UniverseMeta:
    """Metadata for a universe."""
    name: str
    description: str
    category: str  # e.g. "equity", "bond", "commodity", "sector", "international"


# Liquid ETFs (high AUM, tight spreads) - broad mix
LIQUID_ETFS: List[str] = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
    "GLD", "SLV", "USO", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "BND",
    "AGG", "TIP", "VNQ", "VYM", "VUG", "IVV", "IJH", "IJR", "MDY", "RSP",
    "SCHA", "SCHB", "SCHX", "SCHF", "SCHE", "SCHZ", "SCHO", "SCHP", "SCHD",
]

# S&P sector SPDRs
SECTOR_ETFS: List[str] = [
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
]

# Bond ETFs (Treasury, corporate, aggregate)
BOND_ETFS: List[str] = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "BND", "AGG", "TIP", "BNDX",
    "SCHO", "SCHP", "SCHZ", "VCIT", "VCLT", "VGSH", "VGIT", "VGLT",
]

# Commodity ETFs
COMMODITY_ETFS: List[str] = [
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "IAU", "SIVR", "PALL", "PPLT",
]

# International equity ETFs
INTERNATIONAL_ETFS: List[str] = [
    "EFA", "EEM", "VEA", "VWO", "IEFA", "IEMG", "SCHF", "SCHE", "VXUS",
    "IXUS", "VEU", "VSS", "VGK", "VPL", "EWJ", "EWZ", "EWG", "EWU",
]


class UniverseRegistry:
    """Registry of universes with symbols and metadata."""

    _universes: Dict[str, tuple[List[str], UniverseMeta]] = {}

    @classmethod
    def _init(cls) -> None:
        if cls._universes:
            return
        cls._universes = {
            "liquid_etfs": (LIQUID_ETFS, UniverseMeta(
                name="liquid_etfs",
                description="Broad mix of liquid US equity, bond, commodity ETFs",
                category="mixed",
            )),
            "sector_etfs": (SECTOR_ETFS, UniverseMeta(
                name="sector_etfs",
                description="S&P 500 sector SPDRs (XLK, XLF, XLV, etc.)",
                category="sector",
            )),
            "bond_etfs": (BOND_ETFS, UniverseMeta(
                name="bond_etfs",
                description="Treasury, corporate, and aggregate bond ETFs",
                category="bond",
            )),
            "commodity_etfs": (COMMODITY_ETFS, UniverseMeta(
                name="commodity_etfs",
                description="Precious metals, energy, agriculture commodity ETFs",
                category="commodity",
            )),
            "international_etfs": (INTERNATIONAL_ETFS, UniverseMeta(
                name="international_etfs",
                description="Developed and emerging market equity ETFs",
                category="international",
            )),
        }

    @classmethod
    def list_names(cls) -> List[str]:
        """Return sorted list of universe names."""
        cls._init()
        return sorted(cls._universes.keys())

    @classmethod
    def get(cls, name: str, n: Optional[int] = None) -> Tuple[List[str], UniverseMeta]:
        """
        Return (symbols, metadata) for the given universe.

        Args:
            name: Universe name
            n: Max symbols to return (None = all)

        Returns:
            (symbols, UniverseMeta)

        Raises:
            ValueError: if universe unknown
        """
        cls._init()
        if name not in cls._universes:
            raise ValueError(
                f"Unknown universe: {name}. Available: {', '.join(cls.list_names())}"
            )
        symbols, meta = cls._universes[name]
        if n is not None:
            symbols = symbols[:n]
        return symbols, meta

    @classmethod
    def get_symbols(cls, name: str, n: Optional[int] = None) -> List[str]:
        """Convenience: return symbols only."""
        symbols, _ = cls.get(name, n)
        return symbols

    @classmethod
    def get_meta(cls, name: str) -> UniverseMeta:
        """Return metadata only."""
        cls._init()
        if name not in cls._universes:
            raise ValueError(
                f"Unknown universe: {name}. Available: {', '.join(cls.list_names())}"
            )
        _, meta = cls._universes[name]
        return meta


def get_universe(name: str = "liquid_etfs", n: int = 50) -> List[str]:
    """
    Return symbol list for the given universe (backward compatible).

    Args:
        name: Universe name
        n: Max symbols to return (default 50)

    Returns:
        List of ticker symbols
    """
    return UniverseRegistry.get_symbols(name, n)
