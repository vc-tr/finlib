#!/usr/bin/env python3
"""
Demo: Black-Scholes and Monte Carlo option pricing.

Usage:
    python scripts/run_options_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.options import BlackScholes, MonteCarloPricer


def main():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    print("\n=== Black-Scholes Option Pricing ===\n")
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    call = bs.call_price()
    put = bs.put_price()
    print(f"Call price: {call:.4f}")
    print(f"Put price:  {put:.4f}")
    result = bs.full_result("call")
    print(f"Delta: {result.delta:.4f}, Gamma: {result.gamma:.6f}, Vega: {result.vega:.4f}")

    print("\n=== Monte Carlo (100k paths) ===\n")
    mc = MonteCarloPricer(S=S, K=K, T=T, r=r, sigma=sigma, n_paths=100_000, seed=42)
    euro_call, se = mc.price_with_std("european", "call")
    print(f"European call: {euro_call:.4f} ± {se:.4f}")
    asian_call = mc.price("asian", "call")
    print(f"Asian call:    {asian_call:.4f}")


if __name__ == "__main__":
    main()
