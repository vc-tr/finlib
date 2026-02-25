#!/usr/bin/env python3
"""
One-command demo: data download → strategy → backtest → tear-sheet.

Usage:
    python scripts/run_demo.py --symbol SPY --period 2y
    python scripts/run_demo.py --symbol SPY --period 30d  # smoke test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.backtest import Backtester
from src.backtest.execution import ExecutionConfig
from src.strategies import MomentumStrategy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Forecast demo: data → strategy → backtest"
    )
    parser.add_argument("--config", help="Path to JSON config (overrides CLI)")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--period", default="2y", help="Data period (e.g. 30d, 2y)")
    parser.add_argument("--interval", default="1d", help="Data interval (1d)")
    parser.add_argument("--lookback", type=int, default=20, help="Momentum lookback")
    parser.add_argument("--fee-bps", type=float, default=5.0, help="Fee in bps per trade")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage in bps")
    parser.add_argument("--no-execution", action="store_true", help="Disable fees/slippage")
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        import json
        with open(args.config) as f:
            cfg = json.load(f)
        args.symbol = cfg.get("symbol", args.symbol)
        args.period = cfg.get("period", args.period)
        args.interval = cfg.get("interval", args.interval)
        sp = cfg.get("strategy_params", {})
        args.lookback = sp.get("lookback", args.lookback)
        ex = cfg.get("execution", {})
        if ex and not args.no_execution:
            args.fee_bps = ex.get("fee_bps", args.fee_bps)
            args.slippage_bps = ex.get("slippage_bps", args.slippage_bps)

    # 1) Data download
    print(f"[1/4] Fetching {args.symbol} ({args.period}, {args.interval})...")
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(args.symbol, args.interval, period=args.period)
    df = df.dropna()
    close = df["close"]
    print(f"      Loaded {len(close)} bars from {close.index[0].date()} to {close.index[-1].date()}")

    # 2) Strategy + execution (flagship: Momentum)
    print(f"[2/4] Running Momentum strategy (lookback={args.lookback})...")
    strategy = MomentumStrategy(lookback=args.lookback)
    signals, _ = strategy.backtest_returns(close)

    # 3) Backtest (with execution realism)
    print("[3/4] Backtesting...")
    annualization = 252 if args.interval == "1d" else 252 * 6.5 * 60
    backtester = Backtester(annualization_factor=annualization)
    exec_config = None if args.no_execution else ExecutionConfig(
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps
    )
    result = backtester.run_from_signals(close, signals, execution_config=exec_config)

    # 4) Summary
    print("[4/4] Results:")
    print("-" * 50)
    print(f"  Sharpe (ann.):  {result.sharpe_ratio:.2f}")
    print(f"  Total Return:   {result.total_return:.2%}")
    print(f"  Max Drawdown:   {result.max_drawdown:.2%}")
    print(f"  Trades:         {result.n_trades}")
    print(f"  Win Rate:       {result.win_rate:.1%}")
    print("-" * 50)

    # 5) Tear-sheet
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    from src.reporting.tearsheet import generate_tearsheet
    generate_tearsheet(result, close, signals, out_dir, annualization=annualization)
    print(f"\n  Tear-sheet saved to {out_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
