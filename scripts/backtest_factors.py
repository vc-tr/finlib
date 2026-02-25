#!/usr/bin/env python3
"""
Cross-sectional factor backtest.

Usage:
    python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1 --rebalance M --top-k 10 --bottom-k 10
    python scripts/backtest_factors.py --factor lowvol_20d --period 5y --walkforward
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import Backtester
from src.factors import compute_factor, cross_sectional_rank, build_portfolio, get_universe
from src.factors.factors import get_prices_wide
from src.factors.portfolio import apply_rebalance_costs, _resample_weights_to_rebalance
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.reporting.tearsheet import generate_tearsheet


def _parse_period_days(period: str) -> int:
    p = period.lower().strip()
    if p.endswith("d"):
        return int(p[:-1])
    if p.endswith("y"):
        return int(p[:-1]) * 252
    if p.endswith("mo"):
        return int(p[:-2]) * 21
    return 252


def _fetch_universe(
    symbols: list[str],
    interval: str,
    period: str,
    fetcher: YahooDataFetcher,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for each symbol, return df_by_symbol."""
    df_by_symbol = {}
    for sym in symbols:
        try:
            df = fetcher.fetch_ohlcv(sym, interval, period=period)
            df = df.dropna()
            if len(df) >= 30:
                df_by_symbol[sym] = df
        except Exception as e:
            print(f"  [WARN] Skip {sym}: {e}")
    return df_by_symbol


def _run_factor_backtest(
    df_by_symbol: dict[str, pd.DataFrame],
    factor: str,
    top_k: int,
    bottom_k: int,
    rebalance: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    annualization: float,
) -> tuple:
    """Run factor backtest, return (result, positions, prices)."""
    factor_df = compute_factor(df_by_symbol, factor)
    weights = cross_sectional_rank(
        factor_df, top_k=top_k, bottom_k=bottom_k,
        method="zscore", long_short=True, gross_leverage=1.0, max_weight=0.1,
    )
    prices = get_prices_wide(df_by_symbol)
    port_ret = build_portfolio(weights, prices, rebalance=rebalance, execution_delay=1)
    w_held = _resample_weights_to_rebalance(weights, rebalance).shift(1).fillna(0)
    port_ret = apply_rebalance_costs(
        port_ret, w_held,
        cost_bps=fee_bps + slippage_bps + spread_bps,
    )
    bt = Backtester(annualization_factor=annualization)
    result = bt.run(port_ret)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0)
    positions = w_held.sum(axis=1)
    return result, positions, prices, turnover


def _run_walkforward(
    df_by_symbol: dict[str, pd.DataFrame],
    factor: str,
    top_k: int,
    bottom_k: int,
    rebalance: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    annualization: float,
    folds: int,
    train_days: int,
    test_days: int,
) -> dict:
    """Run walk-forward: use history up to test_end, evaluate on test window only."""
    from src.backtest.walkforward import generate_folds

    prices_wide = get_prices_wide(df_by_symbol)
    index = prices_wide.index
    fold_list = generate_folds(index, train_days, test_days, test_days, max_folds=folds)

    per_fold = []
    all_returns = []
    for fold in fold_list:
        test_start, test_end = fold.test_start, fold.test_end
        df_by_hist = {
            sym: df.loc[:test_end]
            for sym, df in df_by_symbol.items()
            if len(df.loc[:test_end]) >= 30
        }
        if len(df_by_hist) < top_k + bottom_k:
            continue
        result, _, _, _ = _run_factor_backtest(
            df_by_hist, factor, top_k, bottom_k, rebalance,
            fee_bps, slippage_bps, spread_bps, annualization,
        )
        test_ret = result.returns.loc[test_start:test_end].dropna()
        if len(test_ret) < 5:
            continue
        fold_result = Backtester(annualization_factor=annualization).run(test_ret)
        per_fold.append({
            "fold_idx": fold.fold_idx,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "sharpe": fold_result.sharpe_ratio,
            "total_return": fold_result.total_return,
            "max_drawdown": fold_result.max_drawdown,
        })
        all_returns.append(test_ret)

    if not all_returns:
        agg = {"mean_sharpe": 0, "median_sharpe": 0, "agg_sharpe": 0, "n_folds": 0}
    else:
        agg_ret = pd.concat(all_returns).sort_index()
        agg_ret = agg_ret[~agg_ret.index.duplicated(keep="first")]
        bt = Backtester(annualization_factor=annualization)
        agg_result = bt.run(agg_ret)
        sharpes = [r["sharpe"] for r in per_fold]
        agg = {
            "mean_sharpe": float(sum(sharpes) / len(sharpes)),
            "median_sharpe": float(sorted(sharpes)[len(sharpes) // 2]),
            "agg_sharpe": agg_result.sharpe_ratio,
            "agg_total_return": agg_result.total_return,
            "n_folds": len(per_fold),
        }
    return {"per_fold": per_fold, "aggregated": agg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional factor backtest")
    parser.add_argument("--universe", default="liquid_etfs", help="Universe name")
    parser.add_argument("--factor", default="momentum_12_1",
                        choices=["momentum_12_1", "reversal_5d", "lowvol_20d"])
    parser.add_argument("--rebalance", default="M", choices=["D", "W", "M"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d", choices=["1d", "1h"])
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--lock-timeout", type=float, default=0)
    args = parser.parse_args()

    symbols = get_universe(args.universe, n=50)
    annualization = 252 if args.interval == "1d" else 252 * 6.5

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / "runs" / f"{ts}_factors_{args.factor}_{args.rebalance}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Fetching {len(symbols)} symbols ({args.period}, {args.interval})...")
    fetcher = YahooDataFetcher(max_retries=2, retry_delay=1)
    df_by_symbol = _fetch_universe(symbols, args.interval, args.period, fetcher)
    if len(df_by_symbol) < args.top_k + args.bottom_k:
        print(f"[ERROR] Need at least {args.top_k + args.bottom_k} symbols, got {len(df_by_symbol)}")
        sys.exit(1)
    print(f"      Loaded {len(df_by_symbol)} symbols")

    if args.walkforward:
        print(f"[2/4] Walk-forward ({args.folds} folds)...")
        wf_result = _run_walkforward(
            df_by_symbol, args.factor, args.top_k, args.bottom_k, args.rebalance,
            args.fee_bps, args.slippage_bps, args.spread_bps, annualization,
            args.folds, args.train_days, args.test_days,
        )
        agg = wf_result["aggregated"]
        print(f"  Mean Sharpe: {agg['mean_sharpe']:.2f}")
        print(f"  Agg Sharpe:  {agg['agg_sharpe']:.2f}")
        summary = {"walkforward": True, "aggregated": agg, "per_fold": wf_result["per_fold"]}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Output: {output_dir}")
        return

    print(f"[2/4] Computing factor {args.factor}...")
    result, positions, prices, turnover = _run_factor_backtest(
        df_by_symbol, args.factor, args.top_k, args.bottom_k, args.rebalance,
        args.fee_bps, args.slippage_bps, args.spread_bps, annualization,
    )

    print("[3/4] Results:")
    print("-" * 50)
    print(f"  Sharpe:      {result.sharpe_ratio:.2f}")
    print(f"  Total Ret:   {result.total_return:.2%}")
    print(f"  Max DD:      {result.max_drawdown:.2%}")
    print(f"  Trades:      {result.n_trades}")
    print(f"  Turnover:    {turnover.mean() * annualization:.2f} (ann.)")
    print("-" * 50)

    print("[4/4] Writing tear-sheet...")
    prices_1d = prices.mean(axis=1)
    generate_tearsheet(
        result, prices_1d, positions,
        output_dir, annualization=annualization,
        config={
            "factor": args.factor,
            "rebalance": args.rebalance,
            "top_k": args.top_k,
            "bottom_k": args.bottom_k,
        },
    )
    summary = {
        "factor": args.factor,
        "rebalance": args.rebalance,
        "top_k": args.top_k,
        "bottom_k": args.bottom_k,
        "sharpe": result.sharpe_ratio,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "n_trades": result.n_trades,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--no-lock", action="store_true")
    _p.add_argument("--lock-timeout", type=float, default=0)
    _pre, _ = _p.parse_known_args()
    if _pre.no_lock:
        main()
    else:
        from src.utils.runlock import RunLock
        with RunLock(lock_path=str(Path(__file__).resolve().parent.parent / ".runlock"), timeout_s=_pre.lock_timeout):
            main()
