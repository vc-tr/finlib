"""
Strategies from peer-reviewed academic papers.

Each strategy includes paper citation and methodology notes.
"""

from .moskowitz_tsmom import MoskowitzTimeSeriesMomentum
from .jegadeesh_titman import JegadeeshTitmanMomentum
from .gatev_pairs import GatevGoetzmannRouwenhorstPairs
from .de_bondt_thaler import DeBondtThalerReversal

__all__ = [
    "MoskowitzTimeSeriesMomentum",
    "JegadeeshTitmanMomentum",
    "GatevGoetzmannRouwenhorstPairs",
    "DeBondtThalerReversal",
]
