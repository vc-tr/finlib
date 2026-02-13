"""
Monte Carlo simulation for option pricing.

Supports European, Asian, and path-dependent options.
Uses Geometric Brownian Motion for underlying price paths.
"""

import numpy as np
from typing import Literal, Callable, Optional


class MonteCarloPricer:
    """
    Monte Carlo pricer for options using risk-neutral valuation.
    
    Simulates asset paths under GBM: S_T = S_0 * exp((r - 0.5*σ²)T + σ√T Z)
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_paths: int = 100_000,
        n_steps: int = 252,
        seed: Optional[int] = None,
    ):
        """
        Args:
            S: Initial asset price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path (for path-dependent options)
            seed: Random seed for reproducibility
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)
    
    def _simulate_paths(self) -> np.ndarray:
        """Simulate price paths under GBM. Returns (n_paths, n_steps+1)."""
        dt = self.T / self.n_steps
        drift = (self.r - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)
        
        # Log returns
        Z = self.rng.standard_normal((self.n_paths, self.n_steps))
        log_returns = drift + vol * Z
        
        # Price paths: S_t = S_0 * exp(sum of log returns)
        log_paths = np.concatenate(
            [np.zeros((self.n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
        )
        paths = self.S * np.exp(log_paths)
        return paths
    
    def european_payoff(
        self, paths: np.ndarray, option_type: Literal["call", "put"] = "call"
    ) -> np.ndarray:
        """Payoff at expiry for European option (uses final price)."""
        S_T = paths[:, -1]
        if option_type == "call":
            return np.maximum(S_T - self.K, 0)
        else:
            return np.maximum(self.K - S_T, 0)
    
    def asian_payoff(
        self, paths: np.ndarray, option_type: Literal["call", "put"] = "call"
    ) -> np.ndarray:
        """Payoff for Asian option (arithmetic average of path)."""
        S_avg = paths[:, 1:].mean(axis=1)  # exclude S_0
        if option_type == "call":
            return np.maximum(S_avg - self.K, 0)
        else:
            return np.maximum(self.K - S_avg, 0)
    
    def barrier_payoff(
        self,
        paths: np.ndarray,
        barrier: float,
        option_type: Literal["call", "put"] = "call",
        barrier_type: Literal["up_and_out", "down_and_out", "up_and_in", "down_and_in"] = "up_and_out",
    ) -> np.ndarray:
        """Payoff for barrier option."""
        S_T = paths[:, -1]
        touched = np.any(paths >= barrier, axis=1) if "up" in barrier_type else np.any(
            paths <= barrier, axis=1
        )
        
        if option_type == "call":
            vanilla = np.maximum(S_T - self.K, 0)
        else:
            vanilla = np.maximum(self.K - S_T, 0)
        
        if "out" in barrier_type:
            return np.where(touched, 0, vanilla)
        else:
            return np.where(touched, vanilla, 0)
    
    def price(
        self,
        option_style: Literal["european", "asian"] = "european",
        option_type: Literal["call", "put"] = "call",
        custom_payoff: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> float:
        """
        Price option via Monte Carlo.
        
        Args:
            option_style: 'european' or 'asian'
            option_type: 'call' or 'put'
            custom_payoff: Optional function paths -> payoffs
            
        Returns:
            Discounted expected payoff (option price)
        """
        paths = self._simulate_paths()
        
        if custom_payoff is not None:
            payoffs = custom_payoff(paths)
        elif option_style == "european":
            payoffs = self.european_payoff(paths, option_type)
        elif option_style == "asian":
            payoffs = self.asian_payoff(paths, option_type)
        else:
            raise ValueError(f"Unknown option_style: {option_style}")
        
        return np.exp(-self.r * self.T) * np.mean(payoffs)
    
    def price_with_std(
        self,
        option_style: Literal["european", "asian"] = "european",
        option_type: Literal["call", "put"] = "call",
    ) -> tuple[float, float]:
        """Return (price, standard_error)."""
        paths = self._simulate_paths()
        if option_style == "european":
            payoffs = self.european_payoff(paths, option_type)
        else:
            payoffs = self.asian_payoff(paths, option_type)
        
        discounted = np.exp(-self.r * self.T) * payoffs
        return float(np.mean(discounted)), float(np.std(discounted) / np.sqrt(self.n_paths))
