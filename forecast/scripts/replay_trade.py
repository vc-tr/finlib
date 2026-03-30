#!/usr/bin/env python3
"""
Event-driven paper trading replay engine.

Replays historical bars chronologically, generates target weights on rebalance dates,
creates orders, simulates fills with costs, updates portfolio state.

Usage:
    python scripts/replay_trade.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M --start 2024-01-01 --end 2025-12-31
    python scripts/replay_trade.py --strategy factors --factor combo --combo "momentum_12_1,reversal_5d,lowvol_20d" --combo-method auto_robust --rebalance M
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.factors import get_universe
from src.paper.runner import run_replay
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.utils.cli import build_replay_parser
from src.utils.io import fetch_universe_ohlcv, make_output_dir, timestamp_for_run
from src.utils.runlock import RunLock


def main() -> None:
    parser = build_replay_parser()
    args = parser.parse_args()

    if args.factor == "combo" and not args.combo:
        print("[ERROR] --factor combo requires --combo")
        sys.exit(1)
    combo_list = [f.strip() for f in args.combo.split(",")] if args.combo else None

    start_ts = pd.Timestamp(args.start) if args.start else None
    end_ts = pd.Timestamp(args.end) if args.end else None

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = make_output_dir("output/runs", f"{timestamp_for_run()}_replay_{args.factor}_{args.rebalance}")

    symbols = get_universe(args.universe, n=30)
    print(f"[1/3] Fetching {len(symbols)} symbols ({args.period})...")
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df_by_symbol = fetch_universe_ohlcv(symbols, "1d", args.period, fetcher, min_bars=10, warn_fn=print)
    if len(df_by_symbol) < args.top_k + args.bottom_k:
        print(f"[ERROR] Need at least {args.top_k + args.bottom_k} symbols")
        sys.exit(1)
    print(f"      Loaded {len(df_by_symbol)} symbols")

    def _run() -> None:
        result = run_replay(
            df_by_symbol,
            strategy=args.strategy,
            factor=args.factor,
            combo_list=combo_list,
            combo_method=args.combo_method,
            start=start_ts,
            end=end_ts,
            rebalance=args.rebalance,
            initial_cash=args.initial_cash,
            fill_mode=args.fill_mode,
            cost_model=args.cost_model,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            spread_bps=args.spread_bps,
            top_k=args.top_k,
            bottom_k=args.bottom_k,
            output_dir=output_dir,
        )
        if "error" in result:
            print(f"[ERROR] {result['error']}")
            sys.exit(1)
        print("[3/3] Results:")
        print("-" * 40)
        print(f"  Total Return:  {result['total_return']:.2%}")
        print(f"  Sharpe:        {result['sharpe']:.2f}")
        print(f"  Max Drawdown:  {result['max_drawdown']:.2%}")
        print(f"  Orders:        {result['n_orders']}")
        print("-" * 40)
        print(f"Output: {output_dir}")

    if args.no_lock:
        _run()
    else:
        with RunLock(lock_path=str(Path(__file__).resolve().parent.parent / ".runlock"), timeout_s=args.lock_timeout):
            _run()


if __name__ == "__main__":
    main()
