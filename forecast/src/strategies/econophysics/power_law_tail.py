"""
Power Law Tail Strategy — fat tail detection for tail risk/opportunity.

Financial returns are NOT normally distributed — they follow power law tails.
The tail exponent α characterizes the severity of extreme events:

  P(|X| > x) ~ x^(-α)

  α > 3: Finite variance (but still fat-tailed vs Normal)
  2 < α < 3: Infinite variance, finite mean (Cauchy-like)
  α ≈ 3: "Cubic law" — empirical finding for equity returns (Gopikrishnan 1999)

Strategy: When the estimated tail exponent drops (fatter tails = higher crash risk),
reduce exposure. When tails are thin (normal conditions), maintain full exposure.

Reference:
  - Gopikrishnan et al. (1999) — "Scaling of the distribution of fluctuations
    of financial market indices" — Physical Review E, 60(5)
  - Mantegna & Stanley (1999) — "Introduction to Econophysics"
  - Gabaix et al. (2003) — "A theory of power-law distributions in financial
    market fluctuations" — Nature, 423

Expected: Useful as a tail-risk filter rather than standalone strategy.
Fat tail detection is noisy — requires long return histories for reliable α.
Works better as a position-sizing overlay than a binary signal.
"""

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


def _hill_estimator(returns: np.ndarray, k: int = None) -> float:
    """
    Hill estimator for tail exponent α from return magnitudes.

    Uses the k largest absolute returns. Standard estimator for
    Pareto tail index: α_hat = k / Σ(log|X|_(i) - log|X|_(k+1))

    Args:
        returns: Return array
        k: Number of tail observations (default: sqrt(n))

    Returns:
        Estimated tail exponent α (higher = thinner tails)
    """
    n = len(returns)
    if n < 20:
        return 3.0  # default to empirical equity α

    abs_ret = np.abs(returns)
    abs_ret = abs_ret[abs_ret > 0]
    if len(abs_ret) < 10:
        return 3.0

    abs_ret_sorted = np.sort(abs_ret)[::-1]  # descending

    if k is None:
        k = max(5, int(np.sqrt(len(abs_ret_sorted))))
    k = min(k, len(abs_ret_sorted) - 1)

    threshold = abs_ret_sorted[k]
    tail_obs = abs_ret_sorted[:k]

    if threshold <= 0:
        return 3.0

    alpha = k / np.sum(np.log(tail_obs / threshold))
    return float(np.clip(alpha, 0.5, 10.0))


@StrategyRegistry.register
class PowerLawTailStrategy(Strategy):
    """
    Tail-risk adjusted position sizing: reduce when fat tails detected.

    When rolling tail exponent α drops below normal_alpha_threshold,
    conditions indicate elevated tail risk. Strategy goes flat/short
    to reduce crash exposure.

    When α is high (thin tails, normal conditions), full long position.
    """

    def __init__(
        self,
        alpha_window: int = 252,
        normal_alpha: float = 3.5,
        danger_alpha: float = 2.5,
        hill_k: int = None,
    ):
        """
        Args:
            alpha_window: Rolling window for tail estimation (days)
            normal_alpha: Above this α → normal tails → long
            danger_alpha: Below this α → fat tails → reduce/short
            hill_k: Hill estimator k (None = sqrt(n))
        """
        self.alpha_window = alpha_window
        self.normal_alpha = normal_alpha
        self.danger_alpha = danger_alpha
        self.hill_k = hill_k

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="power_law_tail",
            category="econophysics",
            source="Gopikrishnan et al. (1999) / Mantegna & Stanley (1999)",
            description=(
                f"Long when rolling tail exponent α > {self.normal_alpha} "
                f"(thin tails); flat/short when α < {self.danger_alpha} "
                f"(fat tails, crash risk)"
            ),
            hypothesis=(
                "Financial returns follow power laws with α ≈ 3. "
                "Periods with lower α (fatter tails) have elevated crash risk. "
                "Detecting tail regime shifts allows proactive risk reduction."
            ),
            expected_result=(
                "Works better as risk filter than alpha source. "
                "Hill estimator needs long windows — signals are slow. "
                "Most valuable during regime shifts (2008, 2020). "
                "Modest Sharpe improvement when combined with momentum."
            ),
            source_url="https://doi.org/10.1103/PhysRevE.60.5305",
            tags=["power-law", "fat-tail", "risk", "econophysics", "extreme-values"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change().values
        signals = np.zeros(len(prices))

        for i in range(self.alpha_window, len(returns)):
            window = returns[i - self.alpha_window: i]
            alpha = _hill_estimator(window, self.hill_k)

            if alpha >= self.normal_alpha:
                signals[i] = 1.0   # thin tails → long
            elif alpha <= self.danger_alpha:
                signals[i] = -1.0  # fat tails → defensive short
            # else: in between → flat

        return pd.Series(signals, index=prices.index)

    def parameter_grid(self):
        return {
            "alpha_window": [126, 252],
            "normal_alpha": [3.0, 3.5, 4.0],
            "danger_alpha": [2.0, 2.5, 3.0],
        }
