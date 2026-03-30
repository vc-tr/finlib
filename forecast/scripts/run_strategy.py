#!/usr/bin/env python3
"""
Unified Quant Lab strategy runner.

Run any registered strategy (or all in a category) by name.
Fetches data, runs backtest, generates tearsheet.

Usage:
    # Single strategy
    python scripts/run_strategy.py --strategy golden_cross --symbol SPY --period 5y

    # All retail strategies
    python scripts/run_strategy.py --category retail --symbol SPY --period 5y

    # All 20 strategies, comparison table
    python scripts/run_strategy.py --all --symbol SPY --period 5y --compare

    # List registered strategies
    python scripts/run_strategy.py --list

    # List strategies in a category
    python scripts/run_strategy.py --list --category academic
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.strategies.registry import StrategyRegistry


def _load_all_strategies() -> None:
    """Ensure all strategy modules are imported (triggers @register decorators)."""
    StrategyRegistry._load_all()


def _print_catalog(category: str = None) -> None:
    """Print the strategy catalog table."""
    _load_all_strategies()
    if category:
        metas = StrategyRegistry.list_by_category(category)
        print(f"\nRegistered strategies — category: {category}")
    else:
        metas = StrategyRegistry.list_all()
        print(f"\nRegistered strategies ({len(metas)} total):")
    print("-" * 80)
    for m in metas:
        url = f" [{m.source_url}]" if m.source_url else ""
        tags = f"  [{', '.join(m.tags[:3])}]" if m.tags else ""
        print(f"  [{m.category:12s}] {m.name:30s} — {m.source[:40]}{url}")
        if m.expected_result:
            print(f"    {m.expected_result[:78]}")
        if tags:
            print(f"    tags: {tags.strip()}")
    print("-" * 80)
    print(f"  Categories: {', '.join(StrategyRegistry.categories())}")
    print(f"  Total strategies: {len(metas)}")


def _fetch_prices(symbol: str, period: str) -> pd.Series:
    """Download close prices for the symbol."""
    from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df = fetcher.fetch_ohlcv(symbol, interval="1d", period=period)
    df = df.dropna()
    if df.empty:
        print(f"[ERROR] No data for {symbol} {period}")
        sys.exit(1)
    return df["close"]


def _run_one(strategy_name: str, prices: pd.Series, symbol: str, period: str,
             output_dir: Path, fee_bps: float, slippage_bps: float,
             verbose: bool = True) -> dict:
    """
    Run a single strategy and return result dict.

    Returns:
        dict with keys: name, sharpe, total_return, max_drawdown, n_trades,
                        win_rate, category, source
    """
    from src.backtest import Backtester
    from src.backtest.execution import ExecutionConfig

    _load_all_strategies()
    strategy = StrategyRegistry.get(strategy_name)
    meta = strategy.meta()

    if verbose:
        print(f"\n[{meta.category}] {strategy_name}")
        print(f"  Source: {meta.source}")
        print(f"  Hypothesis: {meta.hypothesis[:80]}")

    try:
        positions = strategy.generate_positions(prices)
    except Exception as e:
        print(f"  [WARN] generate_positions failed: {e}. Using generate_signals.")
        positions = strategy.generate_signals(prices)

    exec_config = ExecutionConfig(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=1.0,
        execution_delay_bars=1,
    )
    backtester = Backtester(annualization_factor=252)
    result = backtester.run_from_signals(prices, positions, execution_config=exec_config)

    if verbose:
        print(f"  Sharpe: {result.sharpe_ratio:.2f}  "
              f"Return: {result.total_return:.1%}  "
              f"MaxDD: {result.max_drawdown:.1%}  "
              f"Trades: {result.n_trades}")
        print(f"  Expected: {meta.expected_result[:80]}")

    # Generate tearsheet
    if output_dir is not None:
        strat_dir = output_dir / strategy_name
        strat_dir.mkdir(parents=True, exist_ok=True)
        try:
            from src.reporting.tearsheet import generate_tearsheet
            generate_tearsheet(
                result, prices, positions, strat_dir,
                annualization=252,
                config={
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "period": period,
                    "fee_bps": fee_bps,
                    "slippage_bps": slippage_bps,
                    "category": meta.category,
                    "source": meta.source,
                },
            )
        except Exception as e:
            print(f"  [WARN] Tearsheet generation failed: {e}")

    return {
        "name": strategy_name,
        "category": meta.category,
        "source": meta.source[:40],
        "sharpe": result.sharpe_ratio,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "n_trades": result.n_trades,
        "win_rate": result.win_rate,
        "expected": meta.expected_result[:60],
    }


def _print_comparison_table(results: list) -> None:
    """Print a formatted comparison table of all strategy results."""
    if not results:
        return

    # Sort by Sharpe descending
    results = sorted(results, key=lambda r: r["sharpe"], reverse=True)

    print("\n" + "=" * 95)
    print(f"{'Strategy':<30} {'Cat':>10} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} "
          f"{'Trades':>7} {'WinRate':>8}")
    print("-" * 95)
    for r in results:
        sharpe_str = f"{r['sharpe']:+.2f}"
        print(
            f"{r['name']:<30} {r['category']:>10} {sharpe_str:>7} "
            f"{r['total_return']:>7.1%} {r['max_drawdown']:>7.1%} "
            f"{r['n_trades']:>7} {r['win_rate']:>7.1%}"
        )
    print("=" * 95)
    positive = sum(1 for r in results if r["sharpe"] > 0)
    print(f"\nPositive Sharpe: {positive}/{len(results)} strategies ({positive/len(results):.0%})")
    best = results[0]
    worst = results[-1]
    print(f"Best:  {best['name']} (Sharpe={best['sharpe']:.2f})")
    print(f"Worst: {worst['name']} (Sharpe={worst['sharpe']:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Lab — run any registered strategy by name or category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--strategy", "-s", help="Strategy name (e.g. golden_cross)")
    parser.add_argument("--category", "-c",
                        help="Run all strategies in category (stats|retail|academic|econophysics)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run all 20 registered strategies")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List registered strategies and exit")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--period", default="5y", help="Data period (e.g. 2y, 5y, 10y)")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table after running multiple strategies")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Fee in bps (default: 1.0)")
    parser.add_argument("--slippage-bps", type=float, default=2.0,
                        help="Slippage in bps (default: 2.0)")
    parser.add_argument("--output-dir",
                        help="Output directory (default: output/runs/<timestamp>_<strategy>)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-bar strategy output")
    args = parser.parse_args()

    _load_all_strategies()

    # --list: show catalog and exit
    if args.list:
        _print_catalog(args.category)
        return

    # Determine which strategies to run
    if args.all:
        strategy_names = StrategyRegistry.names()
    elif args.category:
        metas = StrategyRegistry.list_by_category(args.category)
        strategy_names = [m.name for m in metas]
        if not strategy_names:
            print(f"[ERROR] No strategies in category '{args.category}'")
            print(f"  Available categories: {StrategyRegistry.categories()}")
            sys.exit(1)
    elif args.strategy:
        if args.strategy not in StrategyRegistry.names():
            print(f"[ERROR] Unknown strategy '{args.strategy}'")
            print(f"  Available: {StrategyRegistry.names()}")
            sys.exit(1)
        strategy_names = [args.strategy]
    else:
        print("[ERROR] Specify --strategy, --category, --all, or --list")
        parser.print_help()
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = args.strategy or args.category or "all"
        output_dir = Path("output") / "runs" / f"{ts}_{tag}_{args.symbol}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data once (shared across all strategies)
    print(f"Fetching {args.symbol} ({args.period})...")
    prices = _fetch_prices(args.symbol, args.period)
    print(f"  {len(prices)} bars from {prices.index[0].date()} to {prices.index[-1].date()}")

    # Run strategies
    results = []
    for name in strategy_names:
        try:
            r = _run_one(
                name, prices, args.symbol, args.period,
                output_dir=output_dir,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                verbose=not args.quiet,
            )
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    # Comparison table
    if (args.all or args.category or len(strategy_names) > 1) or args.compare:
        _print_comparison_table(results)

    print(f"\nArtifacts saved to: {output_dir}/")


if __name__ == "__main__":
    main()
