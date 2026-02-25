#!/usr/bin/env python3
"""
Production daily pipeline: generate tomorrow's target portfolio + staged orders.

Dry-run by default; use --apply to update data/state/current_portfolio.json.

Usage:
    python scripts/daily_run.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M
    python scripts/daily_run.py --factor combo --combo "momentum_12_1,reversal_5d" --combo-method auto_robust --asof 2024-06-30
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.factors import get_universe
from src.factors.factors import get_prices_wide
from src.ops import build_run_meta, run_daily, write_run_meta
from src.ops.daily import DEFAULT_STATE_PATH, MARKET_SYMBOL
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.utils.cli import build_daily_parser
from src.utils.io import fetch_universe_ohlcv, make_output_dir, timestamp_for_run
from src.utils.runlock import RunLock


def main() -> None:
    parser = build_daily_parser()
    args = parser.parse_args()

    if args.factor == "combo" and not args.combo:
        print("[ERROR] --factor combo requires --combo")
        sys.exit(1)
    combo_list = [f.strip() for f in args.combo.split(",")] if args.combo else None

    root = Path(__file__).resolve().parent.parent
    state_path = Path(args.state_path) if args.state_path else root / DEFAULT_STATE_PATH
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = make_output_dir("output/runs", f"{timestamp_for_run()}_daily_{args.factor}_{args.rebalance}")

    symbols = get_universe(args.universe, n=30)
    if MARKET_SYMBOL not in symbols:
        symbols = [MARKET_SYMBOL] + [s for s in symbols if s != MARKET_SYMBOL]
    print(f"[1/3] Fetching {len(symbols)} symbols ({args.period})...")
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df_by_symbol = fetch_universe_ohlcv(symbols, "1d", args.period, fetcher, warn_fn=print)
    if len(df_by_symbol) < args.top_k + args.bottom_k:
        print(f"[ERROR] Need at least {args.top_k + args.bottom_k} symbols")
        sys.exit(1)
    print(f"      Loaded {len(df_by_symbol)} symbols")

    prices = get_prices_wide(df_by_symbol)
    asof = pd.Timestamp(args.asof) if args.asof else prices.index.max()
    if asof not in prices.index:
        asof = prices.index[prices.index <= asof].max() if len(prices.index[prices.index <= asof]) > 0 else prices.index.max()

    state_path_resolved = state_path.resolve()
    state_loaded = state_path_resolved.exists()

    def _write_meta_and_exit(result_or_error: dict) -> None:
        if "error" in result_or_error:
            meta = build_run_meta(
                output_dir=output_dir,
                state_path=state_path_resolved,
                state_loaded=state_loaded,
                asof_requested=args.asof,
                asof_trading=str(asof.date()),
                rebalance=args.rebalance,
                universe=args.universe,
                factor=args.factor,
                combo=args.combo,
                combo_method=args.combo_method,
                cost_model=args.cost_model,
                apply=args.apply,
                status="error",
                error=result_or_error["error"],
            )
        else:
            meta = build_run_meta(
                output_dir=output_dir,
                state_path=state_path_resolved,
                state_loaded=state_loaded,
                asof_requested=args.asof,
                asof_trading=str(asof.date()),
                rebalance=args.rebalance,
                universe=args.universe,
                factor=args.factor,
                combo=args.combo,
                combo_method=args.combo_method,
                cost_model=args.cost_model,
                apply=args.apply,
                result=result_or_error,
            )
        write_run_meta(output_dir, meta)
        print(f"Run meta: {output_dir / 'run_meta.json'}")

    def _run() -> None:
        print(f"[2/3] Running daily pipeline (asof={asof.date()})...")
        try:
            result = run_daily(
                df_by_symbol,
                strategy=args.strategy,
                factor=args.factor,
                combo_list=combo_list,
                combo_method=args.combo_method,
                asof=asof,
                rebalance=args.rebalance,
                initial_cash=args.initial_cash,
                cost_model=args.cost_model,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                spread_bps=args.spread_bps,
                top_k=args.top_k,
                bottom_k=args.bottom_k,
                max_gross=args.max_gross,
                max_net=args.max_net,
                max_position_weight=args.max_position_weight,
                beta_threshold=args.beta_threshold,
                state_path=state_path,
                output_dir=output_dir,
                apply=args.apply,
                force_rebalance=args.force_rebalance,
            )
            if "error" in result:
                (output_dir / "ERROR.txt").write_text(result["error"], encoding="utf-8")
                _write_meta_and_exit(result)
                print(f"[ERROR] {result['error']}")
                sys.exit(1)
            _write_meta_and_exit(result)
            print("[3/3] Done:")
            print("-" * 40)
            print(f"  Orders:     {result['n_orders']}")
            print(f"  Turnover:   {result['turnover']:.2%}")
            print(f"  Beta:       {result['risk_checks']['portfolio_beta']:.2f}")
            print(f"  Applied:    {result['applied']}")
            print("-" * 40)
            print(f"Output: {output_dir}")
        except Exception as e:
            meta = build_run_meta(
                output_dir=output_dir,
                state_path=state_path_resolved,
                state_loaded=state_loaded,
                asof_requested=args.asof,
                asof_trading=str(asof.date()),
                rebalance=args.rebalance,
                universe=args.universe,
                factor=args.factor,
                combo=args.combo,
                combo_method=args.combo_method,
                cost_model=args.cost_model,
                apply=args.apply,
                status="error",
                error=str(e),
            )
            write_run_meta(output_dir, meta)
            (output_dir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise

    if args.no_lock:
        _run()
    else:
        with RunLock(lock_path=str(root / ".runlock"), timeout_s=args.lock_timeout):
            _run()


if __name__ == "__main__":
    main()
