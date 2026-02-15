"""
Quantitative finance strategies and models.

Submodules:
- options: Black-Scholes, Monte Carlo option pricing
- stochastic: Brownian motion, GBM, Ornstein-Uhlenbeck
- stats: Mean reversion, pairs trading, momentum
- renaissance: Renaissance-style pattern/signal strategies
- influencer: Sentiment, volume-sentiment (Reddit/Twitter proxy)
- daytrader: Scalping, ORB, EMA+Stochastic
- institutional: Kalman pairs (DE Shaw), VWAP reversion, ATR breakout
- papers: Academic paper strategies (Moskowitz, JT, GGR, De Bondt-Thaler)
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
    "SentimentStrategy",
    "VolumeSentimentStrategy",
    "ScalpingStrategy",
    "OpeningRangeBreakoutStrategy",
    "EmaStochasticStrategy",
    "KalmanPairsStrategy",
    "VWAPReversionStrategy",
    "ATRBreakoutStrategy",
    "MoskowitzTimeSeriesMomentum",
    "JegadeeshTitmanMomentum",
    "GatevGoetzmannRouwenhorstPairs",
    "DeBondtThalerReversal",
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
    if name == "SentimentStrategy":
        from .influencer.sentiment import SentimentStrategy
        return SentimentStrategy
    if name == "VolumeSentimentStrategy":
        from .influencer.volume_sentiment import VolumeSentimentStrategy
        return VolumeSentimentStrategy
    if name == "ScalpingStrategy":
        from .daytrader.scalping import ScalpingStrategy
        return ScalpingStrategy
    if name == "OpeningRangeBreakoutStrategy":
        from .daytrader.opening_range_breakout import OpeningRangeBreakoutStrategy
        return OpeningRangeBreakoutStrategy
    if name == "EmaStochasticStrategy":
        from .daytrader.ema_stochastic import EmaStochasticStrategy
        return EmaStochasticStrategy
    if name == "KalmanPairsStrategy":
        from .institutional.kalman_pairs import KalmanPairsStrategy
        return KalmanPairsStrategy
    if name == "VWAPReversionStrategy":
        from .institutional.vwap_reversion import VWAPReversionStrategy
        return VWAPReversionStrategy
    if name == "ATRBreakoutStrategy":
        from .institutional.atr_breakout import ATRBreakoutStrategy
        return ATRBreakoutStrategy
    if name == "MoskowitzTimeSeriesMomentum":
        from .papers.moskowitz_tsmom import MoskowitzTimeSeriesMomentum
        return MoskowitzTimeSeriesMomentum
    if name == "JegadeeshTitmanMomentum":
        from .papers.jegadeesh_titman import JegadeeshTitmanMomentum
        return JegadeeshTitmanMomentum
    if name == "GatevGoetzmannRouwenhorstPairs":
        from .papers.gatev_pairs import GatevGoetzmannRouwenhorstPairs
        return GatevGoetzmannRouwenhorstPairs
    if name == "DeBondtThalerReversal":
        from .papers.de_bondt_thaler import DeBondtThalerReversal
        return DeBondtThalerReversal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
