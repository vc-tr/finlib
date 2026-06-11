"""
Machine-learning strategies.

These wrap the walk-forward ML driver (``src.ml``) in the standard ``Strategy``
interface so they share the platform's backtester, walk-forward harness,
execution model, and tearsheets. Every signal is produced causally — models are
retrained on past data only — so the platform's anti-lookahead guarantees hold
for learned signals exactly as they do for rule-based ones.

The LSTM strategy registers only when PyTorch is installed (optional dependency).
"""

from . import sklearn_strategies  # noqa: F401  (triggers @register)

try:  # LSTM is optional — only register it if torch is available.
    from . import lstm_strategy  # noqa: F401
except ImportError:
    pass
