"""
Brownian motion and Geometric Brownian Motion (GBM).

Standard Brownian: dX = dW (Wiener process)
GBM: dS = μS dt + σS dW  =>  S_t = S_0 exp((μ - σ²/2)t + σW_t)
"""

import numpy as np
from typing import Optional


class BrownianMotion:
    """
    Standard Brownian motion (Wiener process).
    
    X_t = X_0 + W_t, where W_t ~ N(0, t)
    """
    
    def __init__(self, x0: float = 0.0, seed: Optional[int] = None):
        self.x0 = x0
        self.rng = np.random.default_rng(seed)
    
    def simulate(
        self, T: float, n_steps: int = 252, n_paths: int = 1
    ) -> np.ndarray:
        """
        Simulate Brownian motion paths.
        
        Args:
            T: Total time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            
        Returns:
            Array of shape (n_paths, n_steps+1)
        """
        dt = T / n_steps
        dW = self.rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
        W = np.cumsum(dW, axis=1)
        W = np.concatenate([np.zeros((n_paths, 1)), W], axis=1)
        return self.x0 + W


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion for asset prices.
    
    dS = μS dt + σS dW
    S_t = S_0 * exp((μ - σ²/2)t + σW_t)
    
    Used in Black-Scholes and many option pricing models.
    """
    
    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        seed: Optional[int] = None,
    ):
        """
        Args:
            S0: Initial price
            mu: Drift (expected return)
            sigma: Volatility
            seed: Random seed
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
    
    def simulate(
        self, T: float, n_steps: int = 252, n_paths: int = 1
    ) -> np.ndarray:
        """
        Simulate GBM paths.
        
        Returns:
            Array of shape (n_paths, n_steps+1)
        """
        dt = T / n_steps
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)
        
        Z = self.rng.standard_normal((n_paths, n_steps))
        log_returns = drift + vol * Z
        log_paths = np.concatenate(
            [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
        )
        return self.S0 * np.exp(log_paths)
    
    def expected_value(self, t: float) -> float:
        """E[S_t] = S_0 * exp(μt)"""
        return self.S0 * np.exp(self.mu * t)
    
    def variance(self, t: float) -> float:
        """Var[S_t] = S_0² exp(2μt)(exp(σ²t) - 1)"""
        return (
            self.S0**2
            * np.exp(2 * self.mu * t)
            * (np.exp(self.sigma**2 * t) - 1)
        )
