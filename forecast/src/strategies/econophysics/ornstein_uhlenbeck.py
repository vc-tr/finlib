"""
Ornstein-Uhlenbeck Mean Reversion — Fitting a stochastic process.

The Ornstein-Uhlenbeck (OU) process is the canonical model for mean-reverting
stochastic processes:

  dX_t = θ(μ - X_t)dt + σ dW_t

where:
  θ = mean-reversion speed (half-life = ln(2)/θ)
  μ = long-run mean
  σ = volatility
  W_t = standard Brownian motion

Strategy: Fit OU parameters to rolling log-prices, then trade proportionally
to the z-score of the current price relative to the estimated OU mean.

Reference:
  - Ornstein & Uhlenbeck (1930)
  - Avellaneda & Lee (2010) — "Statistical Arbitrage in the US Equities Market"
  - Elliott, van der Hoek & Malcolm (2005) — "Pairs Trading"

Key insight: The OU process provides a principled statistical test for
mean reversion — the estimated θ tells us HOW FAST the process reverts.
A high θ (fast reversion) → aggressive trading; low θ → patient positions.

Expected: Works in genuine mean-reverting assets (spread series in pairs
trading, interest rates). Applied to equity prices (near-random walks),
the OU fit will show θ ≈ 0 — no reversion. The estimation itself is
diagnostically valuable.
"""

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


def _fit_ou_params(log_prices: np.ndarray) -> tuple:
    """
    Fit OU parameters via MLE/regression on discretized process.

    For discrete observations:  X[t] = a + b * X[t-1] + ε
    where:
      b = exp(-θ * dt)  →  θ = -log(b) / dt
      μ = a / (1 - b)
      σ² = var(ε) / (1 - b²) * 2θ

    Args:
        log_prices: Log price array

    Returns:
        (theta, mu, sigma, half_life_days) — all in daily units
    """
    x = log_prices[:-1]
    y = log_prices[1:]

    # OLS: y = a + b * x
    cov = np.cov(x, y)
    b = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 1.0
    b = np.clip(b, 0.0, 0.9999)  # ensure mean reversion (b < 1)
    a = np.mean(y) - b * np.mean(x)

    theta = -np.log(b)  # reversion speed (daily)
    mu = a / (1 - b) if abs(1 - b) > 1e-10 else np.mean(log_prices)

    residuals = y - (a + b * x)
    sigma = np.std(residuals) / np.sqrt(1 - b**2 + 1e-10)

    half_life = np.log(2) / theta if theta > 0 else np.inf

    return float(theta), float(mu), float(sigma), float(half_life)


@StrategyRegistry.register
class OrnsteinUhlenbeckStrategy(Strategy):
    """
    OU-based mean reversion: trade z-score of price vs OU equilibrium.

    Fits OU process to rolling log-prices, computes z-score:
      z = (log(price) - μ_OU) / σ_OU

    Long when z < -entry_z (below equilibrium),
    Short when z > +entry_z (above equilibrium).
    Exit when |z| < exit_z.
    """

    def __init__(
        self,
        fit_window: int = 252,
        entry_z: float = 1.5,
        exit_z: float = 0.5,
        min_theta: float = 0.01,
    ):
        """
        Args:
            fit_window: Rolling window to fit OU parameters (days)
            entry_z: Z-score threshold to enter position
            exit_z: Z-score threshold to exit position
            min_theta: Minimum θ to trade — filters out near-random-walk assets
        """
        self.fit_window = fit_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.min_theta = min_theta

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ornstein_uhlenbeck",
            category="econophysics",
            source="Ornstein & Uhlenbeck (1930) / Avellaneda & Lee (2010)",
            description=(
                f"Mean reversion via OU z-score: "
                f"long below -{self.entry_z}σ, short above +{self.entry_z}σ, "
                f"exit at ±{self.exit_z}σ. Min θ filter: {self.min_theta}"
            ),
            hypothesis=(
                "Prices in mean-reverting series follow an OU process. "
                "Trading the z-score of the current price vs the OU equilibrium "
                "captures the statistically-grounded expected reversion."
            ),
            expected_result=(
                "Works well on pairs/spread series (where mean reversion holds). "
                "Applied to raw equity prices: OU fit gives θ ≈ 0 (near-RW), "
                "θ filter correctly keeps strategy flat. "
                "Diagnostically valuable even when not trading."
            ),
            source_url=None,
            tags=["ou", "mean-reversion", "stochastic", "econophysics", "pairs"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        log_p = np.log(prices.values)
        signals = np.zeros(len(prices))
        pos = 0.0

        for i in range(self.fit_window, len(log_p)):
            window = log_p[i - self.fit_window: i]
            theta, mu, sigma, _ = _fit_ou_params(window)

            if theta < self.min_theta or sigma < 1e-10:
                signals[i] = 0.0
                pos = 0.0
                continue

            z = (log_p[i] - mu) / sigma

            if pos == 0.0:
                if z < -self.entry_z:
                    pos = 1.0   # long (below equilibrium)
                elif z > self.entry_z:
                    pos = -1.0  # short (above equilibrium)
            elif pos == 1.0 and z >= -self.exit_z:
                pos = 0.0      # exit long
            elif pos == -1.0 and z <= self.exit_z:
                pos = 0.0      # exit short

            signals[i] = pos

        return pd.Series(signals, index=prices.index)

    def parameter_grid(self):
        return {
            "fit_window": [126, 252],
            "entry_z": [1.0, 1.5, 2.0],
            "exit_z": [0.0, 0.5],
        }
