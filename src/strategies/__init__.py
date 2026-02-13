"""
Quantitative finance strategies and models.

Submodules:
- options: Black-Scholes, Monte Carlo option pricing
- stochastic: Brownian motion, GBM, Ornstein-Uhlenbeck
- stats: Mean reversion, pairs trading, momentum
- renaissance: Renaissance-style pattern/signal strategies
"""

__all__ = [
    "BlackScholes",
    "MonteCarloPricer",
    "GeometricBrownianMotion",
    "BrownianMotion",
    "OrnsteinUhlenbeck",
    "MeanReversionStrategy",
    "PairsTradingStrategy",
    "MomentumStrategy",
    "RenaissanceSignalEnsemble",
]


def __getattr__(name: str):
    """Lazy imports for optional dependencies."""
    if name == "BlackScholes":
        from .options.black_scholes import BlackScholes
        return BlackScholes
    if name == "MonteCarloPricer":
        from .options.monte_carlo import MonteCarloPricer
        return MonteCarloPricer
    if name == "GeometricBrownianMotion":
        from .stochastic.brownian import GeometricBrownianMotion
        return GeometricBrownianMotion
    if name == "BrownianMotion":
        from .stochastic.brownian import BrownianMotion
        return BrownianMotion
    if name == "OrnsteinUhlenbeck":
        from .stochastic.ornstein_uhlenbeck import OrnsteinUhlenbeck
        return OrnsteinUhlenbeck
    if name == "MeanReversionStrategy":
        from .stats.mean_reversion import MeanReversionStrategy
        return MeanReversionStrategy
    if name == "PairsTradingStrategy":
        from .stats.pairs_trading import PairsTradingStrategy
        return PairsTradingStrategy
    if name == "MomentumStrategy":
        from .stats.momentum import MomentumStrategy
        return MomentumStrategy
    if name == "RenaissanceSignalEnsemble":
        from .renaissance.signal_ensemble import RenaissanceSignalEnsemble
        return RenaissanceSignalEnsemble
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
