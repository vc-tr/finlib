#!/usr/bin/env python3
"""
Demo runner for the walk-forward ML strategies.

Trains a direction model walk-forward (retrained on past data only) and reports
out-of-sample performance against buy-and-hold. Works on real data (Yahoo) or a
deterministic synthetic series for an offline smoke test.

Usage:
    # scikit-learn logistic model on real SPY data
    python scripts/run_ml_demo.py --strategy ml_logistic --symbol SPY --period 8y

    # gradient-boosted trees
    python scripts/run_ml_demo.py --strategy ml_gradient_boost --symbol QQQ --period 8y

    # PyTorch LSTM (requires: pip install -r requirements-ml.txt)
    python scripts/run_ml_demo.py --strategy ml_lstm --symbol SPY --period 10y

    # offline, deterministic (used as a CI smoke test)
    python scripts/run_ml_demo.py --strategy ml_logistic --synthetic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest import Backtester
from src.backtest.execution import ExecutionConfig
from src.strategies.registry import StrategyRegistry


def _synthetic_prices(n: int = 1500, seed: int = 0) -> pd.Series:
    """Deterministic AR(1)-momentum series (no network needed)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.15 * r[t - 1] + eps[t]  # mild, realistic autocorrelation
    px = 100 * np.exp(np.cumsum(r))
    return pd.Series(px, index=pd.date_range("2014-01-01", periods=n, freq="B"))


def _fetch_prices(symbol: str, period: str) -> pd.Series:
    from src.pipeline.data_fetcher_yahoo import YahooDataFetcher

    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(symbol, interval="1d", period=period).dropna()
    if df.empty:
        print(f"[ERROR] No data for {symbol} {period}")
        sys.exit(1)
    return df["close"]


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward ML strategy demo")
    p.add_argument("--strategy", default="ml_logistic",
                   choices=["ml_logistic", "ml_gradient_boost", "ml_lstm"])
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--period", default="8y")
    p.add_argument("--synthetic", action="store_true",
                   help="Use a deterministic offline series instead of Yahoo data")
    p.add_argument("--min-train", type=int, default=252)
    p.add_argument("--retrain-every", type=int, default=21)
    p.add_argument("--band", type=float, default=0.0,
                   help="Flat dead-band around P(up)=0.5")
    args = p.parse_args()

    StrategyRegistry._load_all()
    if args.strategy not in StrategyRegistry.names():
        print(f"[ERROR] '{args.strategy}' is not registered "
              f"(LSTM needs: pip install -r requirements-ml.txt)")
        sys.exit(1)

    if args.synthetic:
        prices = _synthetic_prices()
        label = "synthetic AR(1) series"
    else:
        prices = _fetch_prices(args.symbol, args.period)
        label = f"{args.symbol} ({args.period})"

    print(f"Data: {label} — {len(prices)} bars "
          f"[{prices.index[0].date()} → {prices.index[-1].date()}]")

    strat = StrategyRegistry.get(
        args.strategy,
        min_train=args.min_train,
        retrain_every=args.retrain_every,
        band=args.band,
    )
    meta = strat.meta()
    print(f"Strategy: {args.strategy} ({meta.category}) — {meta.description}")
    print("Training walk-forward (past-only, leak-free)...")

    signals = strat.generate_signals(prices)

    bt = Backtester(annualization_factor=252)
    exec_cfg = ExecutionConfig(fee_bps=1.0, slippage_bps=2.0, spread_bps=1.0)
    res = bt.run_from_signals(prices, signals, execution_config=exec_cfg)

    bh = bt.run(prices.pct_change().fillna(0.0))  # buy-and-hold benchmark
    n_oos = int((signals != 0).sum())

    print("\n" + "=" * 56)
    print(f"{'Metric':<22}{'ML strategy':>16}{'Buy & hold':>18}")
    print("-" * 56)
    print(f"{'Sharpe':<22}{res.sharpe_ratio:>16.2f}{bh.sharpe_ratio:>18.2f}")
    print(f"{'Total return':<22}{res.total_return:>15.1%}{bh.total_return:>17.1%}")
    print(f"{'Max drawdown':<22}{res.max_drawdown:>15.1%}{bh.max_drawdown:>17.1%}")
    print(f"{'Win rate':<22}{res.win_rate:>15.1%}{'—':>18}")
    print(f"{'Trades':<22}{res.n_trades:>16}{'—':>18}")
    print(f"{'OOS bars traded':<22}{n_oos:>16}{'—':>18}")
    print("=" * 56)
    print(f"Expected (honest): {meta.expected_result}")


if __name__ == "__main__":
    main()
