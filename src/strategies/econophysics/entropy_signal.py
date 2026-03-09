"""
Entropy Signal — Shannon entropy of return distribution.

Shannon entropy measures the disorder/uncertainty in a distribution.
Low entropy (ordered market) → trend-following conditions.
High entropy (disordered market) → mean-reversion or flat.

Intuition from statistical physics: markets have phases between
ordered (low entropy, trending) and disordered (high entropy, random).
Entropy is a natural measure of market "complexity."

Reference:
  - Risso (2008) — "The informational efficiency and the financial crashes"
  - Zunino et al. (2009) — "Forbidden patterns, permutation entropy, and stock market inefficiency"
  - Mantegna & Stanley (1999) — "Introduction to Econophysics"

Implementation: Rolling permutation entropy on discretized return series.
Low permutation entropy → fewer unique ordinal patterns → trending.
High permutation entropy → many patterns → random/mean-reverting.

Expected: Regime detection works in theory but discretization and
estimation noise limit OOS edge. Interesting as a research tool.
"""

import math
import numpy as np
import pandas as pd
from itertools import permutations

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


def _permutation_entropy(x: np.ndarray, order: int = 3, normalize: bool = True) -> float:
    """
    Compute permutation entropy of order m.

    Maps each consecutive m-tuple to its ordinal pattern,
    counts pattern frequencies, computes Shannon entropy.

    Args:
        x: 1D array
        order: Embedding dimension (3-6 typical)
        normalize: If True, divide by log(m!) to get H ∈ [0, 1]

    Returns:
        Permutation entropy H
    """
    n = len(x)
    if n < order:
        return np.nan

    # Extract ordinal patterns
    patterns = []
    for i in range(n - order + 1):
        window = x[i: i + order]
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)

    # Count pattern frequencies
    all_patterns = list(permutations(range(order)))
    counts = {p: 0 for p in all_patterns}
    for p in patterns:
        counts[p] = counts.get(p, 0) + 1

    total = sum(counts.values())
    probs = [c / total for c in counts.values() if c > 0]

    # Shannon entropy
    H = -sum(p * np.log2(p) for p in probs)
    if normalize:
        H = H / np.log2(math.factorial(order))
    return H


@StrategyRegistry.register
class EntropySignalStrategy(Strategy):
    """
    Trend-follow in low-entropy regimes; mean-revert in high-entropy regimes.

    Rolling permutation entropy classifies the current market state:
    - Low H (ordered): momentum signal
    - High H (disordered): contrarian signal
    - Mid H: flat
    """

    def __init__(
        self,
        entropy_window: int = 50,
        perm_order: int = 3,
        low_entropy_quantile: float = 0.33,
        high_entropy_quantile: float = 0.67,
        signal_lookback: int = 10,
    ):
        """
        Args:
            entropy_window: Rolling window for entropy computation (days)
            perm_order: Embedding dimension for permutation entropy (3-5)
            low_entropy_quantile: Below this rolling rank → trending regime
            high_entropy_quantile: Above this rolling rank → random regime
            signal_lookback: Return lookback for direction (days)
        """
        self.entropy_window = entropy_window
        self.perm_order = perm_order
        self.low_entropy_quantile = low_entropy_quantile
        self.high_entropy_quantile = high_entropy_quantile
        self.signal_lookback = signal_lookback

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="entropy_signal",
            category="econophysics",
            source="Risso (2008) / Zunino et al. (2009) — Permutation entropy",
            description=(
                f"Trend-follow in low permutation entropy regimes "
                f"(H rank < {self.low_entropy_quantile:.0%}), "
                f"mean-revert in high entropy (H rank > {self.high_entropy_quantile:.0%})"
            ),
            hypothesis=(
                "Markets alternate between ordered (low entropy, trending) and "
                "disordered (high entropy, near-random) phases. "
                "Permutation entropy detects these regime transitions from price alone."
            ),
            expected_result=(
                "Mixed results OOS. Entropy estimation requires sufficient data. "
                "Regime transitions are often too slow to provide timely signals. "
                "Most interesting as a market characterization tool."
            ),
            source_url=None,
            tags=["entropy", "information-theory", "regime", "econophysics", "complexity"],
        )

    def _compute_rolling_entropy(self, returns: pd.Series) -> pd.Series:
        """Compute rolling permutation entropy over entropy_window."""
        entropy_vals = pd.Series(np.nan, index=returns.index)
        arr = returns.values
        for i in range(self.entropy_window, len(arr) + 1):
            window = arr[i - self.entropy_window: i]
            entropy_vals.iloc[i - 1] = _permutation_entropy(window, self.perm_order)
        return entropy_vals

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change().fillna(0)
        entropy = self._compute_rolling_entropy(returns)

        # Percentile rank of current entropy
        rank_window = min(252, len(entropy) // 2)
        entropy_rank = entropy.rolling(rank_window).rank(pct=True)

        # Direction from recent return
        direction = returns.rolling(self.signal_lookback).sum().apply(
            lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        )

        signals = pd.Series(0.0, index=prices.index)
        is_low_entropy = entropy_rank < self.low_entropy_quantile
        is_high_entropy = entropy_rank > self.high_entropy_quantile

        signals[is_low_entropy] = direction[is_low_entropy]  # trend-follow
        signals[is_high_entropy] = -direction[is_high_entropy]  # mean-revert
        return signals

    def parameter_grid(self):
        return {
            "entropy_window": [30, 50, 100],
            "perm_order": [3, 4],
            "low_entropy_quantile": [0.25, 0.33],
        }
