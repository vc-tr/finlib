"""Cross-sectional factor engine."""

from .universe import get_universe
from .factors import compute_factor, compute_factors
from .ranking import cross_sectional_rank
from .portfolio import build_portfolio
from .ensemble import combine_factors
from .research import forward_returns, information_coefficient, summarize_ic
from .risk import estimate_beta

__all__ = [
    "get_universe", "compute_factor", "compute_factors", "cross_sectional_rank",
    "build_portfolio", "combine_factors",
    "forward_returns", "information_coefficient", "summarize_ic",
    "estimate_beta",
]
