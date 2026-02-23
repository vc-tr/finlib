#!/usr/bin/env python3
"""
Backtest multi-strategy portfolio with configurable allocation.

Usage:
    python scripts/backtest_portfolio.py --symbol SPY --period 2y
    python scripts/backtest_portfolio.py --symbol SPY --period 2y --alloc risk_parity
    python scripts/backtest_portfolio.py --symbol SPY --period 2y --alloc equal --rebalance 21
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.backtest import Backtester
from src.portfolio import MultiStrategyPortfolio, PortfolioAllocator, AllocationMethod
from src.portfolio.manager import build_default_portfolio


def main():
    parser = argparse.ArgumentParser(description="Backtest multi-strategy portfolio")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--alloc", default="equal", choices=["equal", "risk_parity", "inverse_vol"])
    parser.add_argument("--rebalance", type=int, default=21, help="Rebalance every N days (0=no rebal)")
    parser.add_argument("--lookback", type=int, default=63, help="Lookback for vol estimation")
    args = parser.parse_args()

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, "1d", period=args.period)
    df = df.dropna()
    prices = df["close"]

    method = {
        "equal": AllocationMethod.EQUAL,
        "risk_parity": AllocationMethod.RISK_PARITY,
        "inverse_vol": AllocationMethod.INVERSE_VOLATILITY,
    }[args.alloc]

    allocator = PortfolioAllocator(
        method=method,
        lookback=args.lookback,
        rebalance_freq=args.rebalance if args.rebalance > 0 else 0,
    )

    portfolio = build_default_portfolio(prices, df)
    returns_df, weights_df, port_returns = portfolio.run(allocator=allocator)

    backtester = Backtester(annualization_factor=252)
    result = backtester.run(port_returns)

    print(f"\n=== Multi-Strategy Portfolio: {args.symbol} ({args.period}) ===\n")
    print(f"Allocation: {args.alloc}  |  Rebalance: every {args.rebalance}d  |  Lookback: {args.lookback}d\n")
    print("Individual strategies:")
    for col in returns_df.columns:
        r = backtester.run(returns_df[col])
        print(f"  {col:<20} Sharpe: {r.sharpe_ratio:>6.2f}  Return: {r.total_return:>8.2%}  MaxDD: {r.max_drawdown:>6.2%}")
    print("\n" + "-" * 55)
    print(f"  {'PORTFOLIO':<20} Sharpe: {result.sharpe_ratio:>6.2f}  Return: {result.total_return:>8.2%}  MaxDD: {result.max_drawdown:>6.2%}")
    print(f"\nAvg weight per strategy (last period):")
    last_w = weights_df.iloc[-1]
    for k, v in last_w.items():
        print(f"  {k}: {v:.2%}")


if __name__ == "__main__":
    main()
