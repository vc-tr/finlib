"""
Econophysics strategies — physics-inspired quantitative approaches.

Each strategy is decorated with @StrategyRegistry.register.
Importing this package is sufficient to register all econophysics strategies.
"""

from .hurst_exponent import HurstExponentStrategy
from .entropy_signal import EntropySignalStrategy
from .ornstein_uhlenbeck import OrnsteinUhlenbeckStrategy
from .power_law_tail import PowerLawTailStrategy

__all__ = [
    "HurstExponentStrategy",
    "EntropySignalStrategy",
    "OrnsteinUhlenbeckStrategy",
    "PowerLawTailStrategy",
]
