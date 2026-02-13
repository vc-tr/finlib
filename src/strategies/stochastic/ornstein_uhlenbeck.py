"""
Ornstein-Uhlenbeck process for mean-reverting dynamics.

dX_t = θ(μ - X_t)dt + σdW_t

Used for: interest rates (Vasicek), pairs trading spreads, volatility.
"""

import numpy as np
from typing import Optional


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process: mean-reverting stochastic process.
    
    Parameters:
        theta: Speed of mean reversion (higher = faster reversion)
        mu: Long-term mean
        sigma: Volatility
        x0: Initial value
    """
    
    def __init__(
        self,
        theta: float,
        mu: float,
        sigma: float,
        x0: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = mu if x0 is None else x0
        self.rng = np.random.default_rng(seed)
    
    def simulate(
        self, T: float, n_steps: int = 252, n_paths: int = 1
    ) -> np.ndarray:
        """
        Simulate OU paths using Euler-Maruyama.
        
        X_{t+dt} = X_t + θ(μ - X_t)dt + σ√dt * Z
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.x0
        
        for i in range(n_steps):
            dW = self.rng.standard_normal(n_paths) * np.sqrt(dt)
            paths[:, i + 1] = paths[:, i] + self.theta * (
                self.mu - paths[:, i]
            ) * dt + self.sigma * dW
        
        return paths
    
    def expected_value(self, t: float, x0: Optional[float] = None) -> float:
        """E[X_t] = μ + (x0 - μ)exp(-θt)"""
        x = self.x0 if x0 is None else x0
        return self.mu + (x - self.mu) * np.exp(-self.theta * t)
    
    def variance(self, t: float) -> float:
        """Var[X_t] = σ²/(2θ) * (1 - exp(-2θt))"""
        if self.theta <= 0:
            return np.inf
        return (self.sigma**2 / (2 * self.theta)) * (
            1 - np.exp(-2 * self.theta * t)
        )
    
    @staticmethod
    def calibrate_from_series(prices: np.ndarray, dt: float = 1.0 / 252) -> dict:
        """
        Calibrate OU parameters from price series using OLS.
        
        Discrete form: X_{t+1} - X_t = α + β*X_t + ε
        Where: α = θμΔt, β = -θΔt  =>  θ = -β/Δt, μ = -α/β
        """
        X = prices[:-1]
        Y = np.diff(prices)
        # Y = α + β*X => [1, X] @ [α, β]' = Y
        A = np.column_stack([np.ones_like(X), X])
        coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        alpha, beta = coeffs
        theta = -beta / dt
        mu = -alpha / beta if abs(beta) > 1e-10 else np.mean(prices)
        residuals = Y - (alpha + beta * X)
        sigma = np.std(residuals) / np.sqrt(dt)
        return {"theta": theta, "mu": mu, "sigma": sigma}
