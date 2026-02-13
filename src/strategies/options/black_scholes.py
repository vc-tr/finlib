"""
Black-Scholes-Merton option pricing model.

Prices European call and put options using the closed-form solution.
Also computes Greeks: delta, gamma, theta, vega, rho.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


@dataclass
class BlackScholesResult:
    """Result of Black-Scholes pricing."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    d1: float
    d2: float


class BlackScholes:
    """
    Black-Scholes-Merton model for European option pricing.
    
    Assumes underlying follows Geometric Brownian Motion:
    dS = μS dt + σS dW
    
    Under risk-neutral measure: μ = r (risk-free rate).
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ):
        """
        Args:
            S: Current stock/asset price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate (annualized)
            sigma: Volatility (annualized)
            q: Dividend yield (annualized, default 0)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def _d1_d2(self) -> tuple[float, float]:
        """Compute d1 and d2 for Black-Scholes formula."""
        if self.T <= 0:
            raise ValueError("Time to expiration must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        
        d1 = (
            np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2
    
    def call_price(self) -> float:
        """Price of European call option."""
        d1, d2 = self._d1_d2()
        return self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(
            -self.r * self.T
        ) * norm.cdf(d2)
    
    def put_price(self) -> float:
        """Price of European put option (via put-call parity)."""
        return self.call_price() - self.S * np.exp(-self.q * self.T) + self.K * np.exp(
            -self.r * self.T
        )
    
    def price(self, option_type: Literal["call", "put"] = "call") -> float:
        """Price option by type."""
        if option_type == "call":
            return self.call_price()
        elif option_type == "put":
            return self.put_price()
        raise ValueError("option_type must be 'call' or 'put'")
    
    def delta(self, option_type: Literal["call", "put"] = "call") -> float:
        """Delta: ∂V/∂S."""
        d1, _ = self._d1_d2()
        if option_type == "call":
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        else:  # put
            return np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
    
    def gamma(self) -> float:
        """Gamma: ∂²V/∂S² (same for call and put)."""
        d1, _ = self._d1_d2()
        if self.S <= 0 or self.sigma <= 0 or self.T <= 0:
            return 0.0
        return (
            np.exp(-self.q * self.T)
            * norm.pdf(d1)
            / (self.S * self.sigma * np.sqrt(self.T))
        )
    
    def theta(self, option_type: Literal["call", "put"] = "call") -> float:
        """Theta: ∂V/∂t (per year, negative = time decay)."""
        d1, d2 = self._d1_d2()
        term1 = -self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma / (
            2 * np.sqrt(self.T)
        )
        if option_type == "call":
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            term3 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            term3 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)
        return (term1 + term2 + term3) / 365  # per day
    
    def vega(self) -> float:
        """Vega: ∂V/∂σ (per 1% change in vol)."""
        d1, _ = self._d1_d2()
        return self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T) / 100
    
    def rho(self, option_type: Literal["call", "put"] = "call") -> float:
        """Rho: ∂V/∂r (per 1% change in rate)."""
        _, d2 = self._d1_d2()
        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
    
    def full_result(
        self, option_type: Literal["call", "put"] = "call"
    ) -> BlackScholesResult:
        """Return price and all Greeks."""
        d1, d2 = self._d1_d2()
        return BlackScholesResult(
            price=self.price(option_type),
            delta=self.delta(option_type),
            gamma=self.gamma(),
            theta=self.theta(option_type),
            vega=self.vega(),
            rho=self.rho(option_type),
            d1=d1,
            d2=d2,
        )
    
    def implied_volatility(
        self, market_price: float, option_type: Literal["call", "put"] = "call"
    ) -> float:
        """
        Find implied volatility via Newton-Raphson.
        
        Args:
            market_price: Observed option price
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility (annualized)
        """
        vol = 0.3  # initial guess
        for _ in range(100):
            self.sigma = vol
            price = self.price(option_type)
            vega_val = self.vega() * 100  # vega is per 1%, so scale
            if abs(vega_val) < 1e-10:
                break
            vol = vol - (price - market_price) / vega_val
            if vol <= 0:
                vol = 0.01
            if abs(price - market_price) < 1e-6:
                break
        return vol
