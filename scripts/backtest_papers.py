#!/usr/bin/env python3
"""
Backtest strategies from academic papers.

Usage:
    python scripts/backtest_papers.py --symbol SPY --period 5y
    python scripts/backtest_papers.py --symbol1 SPY --symbol2 IVV --period 2y  # GGR pairs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.backtest import Backtester
from src.strategies import (
    MoskowitzTimeSeriesMomentum,
    DeBondtThalerReversal,
    GatevGoetzmannRouwenhorstPairs,
)


def main():
    parser = argparse.ArgumentParser(description="Backtest paper strategies")
    parser.add_argument("--symbol", default="SPY", help="Single-asset symbol")
    parser.add_argument("--symbol1", help="First symbol for pairs")
    parser.add_argument("--symbol2", help="Second symbol for pairs")
    parser.add_argument("--period", default="5y", help="Data period (need 3-5y for De Bondt-Thaler)")
    args = parser.parse_args()

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    backtester = Backtester(annualization_factor=252)

    print("\n=== Academic Paper Strategies Backtest ===\n")

    # Single-asset: Moskowitz TSMOM, De Bondt-Thaler
    if args.symbol:
        df = fetcher.fetch_ohlcv(args.symbol, "1d", period=args.period)
        prices = df["close"].dropna()
        if len(prices) < 300:
            print(f"Need at least 300 days; got {len(prices)}. Use --period 2y or more.")
        else:
            # Moskowitz (12m formation, 1m hold)
            tsmom = MoskowitzTimeSeriesMomentum(formation_period=252, holding_period=21)
            sig, ret = tsmom.backtest_returns(prices)
            res = backtester.run_from_signals(prices, sig)
            print(f"Moskowitz TSMOM (2012)   Sharpe: {res.sharpe_ratio:.2f}  Return: {res.total_return:.2%}  MaxDD: {res.max_drawdown:.2%}")

            # De Bondt-Thaler (3yr formation - needs long history)
            if len(prices) >= 800:
                dbt = DeBondtThalerReversal(formation_period=756, holding_period=756)
                sig, ret = dbt.backtest_returns(prices)
                res = backtester.run_from_signals(prices, sig)
                print(f"De Bondt-Thaler (1985)   Sharpe: {res.sharpe_ratio:.2f}  Return: {res.total_return:.2%}  MaxDD: {res.max_drawdown:.2%}")
            else:
                print(f"De Bondt-Thaler: need 800+ days (use --period 5y)")

    # Pairs: GGR
    if args.symbol1 and args.symbol2:
        df1 = fetcher.fetch_ohlcv(args.symbol1, "1d", period=args.period)
        df2 = fetcher.fetch_ohlcv(args.symbol2, "1d", period=args.period)
        a, b = df1["close"].align(df2["close"], join="inner")
        a, b = a.dropna(), b.dropna()
        if len(a) < 300:
            print(f"GGR Pairs: need 300+ days; got {len(a)}")
        else:
            ggr = GatevGoetzmannRouwenhorstPairs(formation_period=252, entry_std=2.0)
            sig, strat_ret = ggr.backtest_returns(a, b)
            res = backtester.run(strat_ret)
            print(f"GGR Pairs (2006)          Sharpe: {res.sharpe_ratio:.2f}  Return: {res.total_return:.2%}  MaxDD: {res.max_drawdown:.2%}")


if __name__ == "__main__":
    main()
