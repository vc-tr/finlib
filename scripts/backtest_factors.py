#!/usr/bin/env python3
"""
Cross-sectional factor backtest.

Usage:
    python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1 --rebalance M --top-k 10 --bottom-k 10
    python scripts/backtest_factors.py --factor lowvol_20d --period 5y --walkforward
    python scripts/backtest_factors.py --factor combo --combo "momentum_12_1,reversal_5d,lowvol_20d" --combo-method equal --walkforward
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.backtest import Backtester
from src.factors import compute_factor, compute_factors, cross_sectional_rank, build_portfolio, get_universe, estimate_beta
from src.factors.factors import get_prices_wide
from src.factors.ensemble import combine_factors
from src.factors.portfolio import apply_rebalance_costs, _resample_weights_to_rebalance, apply_beta_neutral, apply_constraints
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
    factor_df: pd.DataFrame,
    top_k: int,
    bottom_k: int,
    rebalance: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    annualization: float,
    beta_neutral: bool = False,
    market_symbol: str = "SPY",
    beta_window: int = 252,
    max_gross: float | None = None,
    max_net: float | None = None,
) -> tuple:
    """
    Run factor backtest.

    Returns (order matters for robust unpacking):
        0: result (Backtester result)
        1: positions (Series)
        2: prices (DataFrame)
        3: turnover (Series)
        4: w_held (DataFrame)
        5: beta_before (Series or None)
        6: beta_after (Series or None)
        7: hedge_weight (Series or None)
    """
    weights = cross_sectional_rank(
        factor_df,
        top_k=top_k,
        bottom_k=bottom_k,
        method="zscore",
        long_short=True,
        gross_leverage=1.0,
        max_weight=0.1,
    )
    prices = get_prices_wide(df_by_symbol)
    beta_before = None
    beta_after = None
    hedge_weight = None

    if beta_neutral:
        if market_symbol not in df_by_symbol:
            raise ValueError(f"Beta-neutral requires {market_symbol} in universe; add it to --universe or fetch it.")
        returns = prices.pct_change()
        mr = returns[market_symbol] if market_symbol in returns.columns else prices[market_symbol].pct_change()
        betas = estimate_beta(returns, mr, window=beta_window)
        out = build_portfolio(
            weights, prices,
            rebalance=rebalance, execution_delay=1,
            max_gross=max_gross, max_net=max_net,
            beta_neutral=True, betas=betas, market_symbol=market_symbol,
        )
        port_ret, beta_before, beta_after, hedge_weight = out
        w_raw = weights.copy()
        if max_gross is not None or max_net is not None:
            w_raw = apply_constraints(w_raw, max_gross=max_gross, max_net=max_net, gross_leverage=1.0)
        w_raw, _, _ = apply_beta_neutral(w_raw, betas, market_symbol=market_symbol)
        w_held = _resample_weights_to_rebalance(w_raw, rebalance).shift(1).fillna(0)
    else:
        port_ret = build_portfolio(
            weights, prices,
            rebalance=rebalance, execution_delay=1,
            max_gross=max_gross, max_net=max_net,
        )
        w_held = _resample_weights_to_rebalance(weights, rebalance).shift(1).fillna(0)
    port_ret = apply_rebalance_costs(
        port_ret, w_held,
        cost_bps=fee_bps + slippage_bps + spread_bps,
    )
    bt = Backtester(annualization_factor=annualization)
    result = bt.run(port_ret)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0)
    positions = w_held.sum(axis=1)
    return result, positions, prices, turnover, w_held, beta_before, beta_after, hedge_weight


def _get_factor_df(
    df_by_symbol: dict[str, pd.DataFrame],
    factor: str,
    combo_list: list[str] | None,
    combo_method: str,
    train_slice: slice | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Get factor DataFrame for backtest. For single factor or combo.
    Returns (factor_df, combo_weights or None).
    """
    if combo_list is None:
        factor_df = compute_factor(df_by_symbol, factor)
        return factor_df, None

    factors_dict = compute_factors(df_by_symbol, combo_list)
    prices = get_prices_wide(df_by_symbol)
    fwd_returns = prices.pct_change().shift(-1)

    if combo_method == "equal":
        combined, weights = combine_factors(factors_dict, method="equal")
    else:
        if train_slice is None:
            idx = prices.index
            n = int(len(idx) * 0.7)
            if n < 30:
                combined, weights = combine_factors(factors_dict, method="equal")
            else:
                ts = slice(idx[0], idx[n - 1])
                combined, weights = combine_factors(
                    factors_dict,
                    method=combo_method,
                    train_slice=ts,
                    fwd_returns=fwd_returns,
                )
        else:
            combined, weights = combine_factors(
                factors_dict,
                method=combo_method,
                train_slice=train_slice,
                fwd_returns=fwd_returns,
            )
    return combined, weights


def _run_walkforward(
    df_by_symbol: dict[str, pd.DataFrame],
    factor: str,
    combo_list: list[str] | None,
    combo_method: str,
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
    combo_weights_per_fold = []

    for fold in fold_list:
        test_start, test_end = fold.test_start, fold.test_end
        df_by_hist = {
            sym: df.loc[:test_end]
            for sym, df in df_by_symbol.items()
            if len(df.loc[:test_end]) >= 30
        }
        if len(df_by_hist) < top_k + bottom_k:
            continue

        if factor == "combo" and combo_list:
            train_slice = slice(fold.train_start, fold.train_end)
            factor_df, combo_weights = _get_factor_df(
                df_by_hist, factor, combo_list, combo_method, train_slice
            )
            if combo_weights is not None:
                combo_weights_per_fold.append({
                    "fold_idx": fold.fold_idx,
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "weights": combo_weights,
                })
        else:
            factor_df, _ = _get_factor_df(df_by_hist, factor, None, "equal", None)

        out = _run_factor_backtest(
            df_by_hist, factor_df, top_k, bottom_k, rebalance,
            fee_bps, slippage_bps, spread_bps, annualization,
        )
        result = out[0]
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
    out = {"per_fold": per_fold, "aggregated": agg}
    if combo_weights_per_fold:
        out["combo_weights_per_fold"] = combo_weights_per_fold
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional factor backtest")
    parser.add_argument("--universe", default="liquid_etfs", help="Universe name")
    parser.add_argument("--factor", default="momentum_12_1",
                        choices=["momentum_12_1", "reversal_5d", "lowvol_20d", "combo"])
    parser.add_argument("--combo", default=None,
                        help='Comma-separated factors for combo (e.g. "momentum_12_1,reversal_5d,lowvol_20d")')
    parser.add_argument("--combo-method", default="equal",
                        choices=["equal", "ic_weighted", "ridge"],
                        help="Combo weighting: equal, ic_weighted, ridge")
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
    parser.add_argument("--beta-neutral", action="store_true", help="Hedge portfolio beta with market (SPY)")
    parser.add_argument("--market-symbol", default="SPY", help="Market symbol for beta hedge (default SPY)")
    parser.add_argument("--beta-window", type=int, default=252, help="Rolling window for beta estimation")
    parser.add_argument("--max-gross", type=float, default=None, help="Cap gross exposure per rebalance")
    parser.add_argument("--max-net", type=float, default=None, help="Cap net exposure per rebalance")
    args = parser.parse_args()

    symbols = get_universe(args.universe, n=50)
    if args.beta_neutral and args.market_symbol not in symbols:
        symbols = [args.market_symbol] + [s for s in symbols if s != args.market_symbol]
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

    combo_list = None
    if args.factor == "combo":
        if not args.combo:
            print("[ERROR] --factor combo requires --combo (e.g. --combo momentum_12_1,reversal_5d,lowvol_20d)")
            sys.exit(1)
        combo_list = [f.strip() for f in args.combo.split(",") if f.strip()]

    if args.walkforward:
        print(f"[2/4] Walk-forward ({args.folds} folds)...")
        wf_result = _run_walkforward(
            df_by_symbol, args.factor, combo_list, args.combo_method,
            args.top_k, args.bottom_k, args.rebalance,
            args.fee_bps, args.slippage_bps, args.spread_bps, annualization,
            args.folds, args.train_days, args.test_days,
        )
        agg = wf_result["aggregated"]
        print(f"  Mean Sharpe: {agg['mean_sharpe']:.2f}")
        print(f"  Agg Sharpe:  {agg['agg_sharpe']:.2f}")
        summary = {"walkforward": True, "aggregated": agg, "per_fold": wf_result["per_fold"]}
        if "combo_weights_per_fold" in wf_result:
            summary["combo_weights_per_fold"] = wf_result["combo_weights_per_fold"]
            (output_dir / "combo_weights.json").write_text(
                json.dumps(wf_result["combo_weights_per_fold"], indent=2), encoding="utf-8"
            )
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Output: {output_dir}")
        return

    print(f"[2/4] Computing factor {args.factor}...")
    factor_df, combo_weights = _get_factor_df(
        df_by_symbol, args.factor, combo_list, args.combo_method, None
    )
    out = _run_factor_backtest(
        df_by_symbol, factor_df, args.top_k, args.bottom_k, args.rebalance,
        args.fee_bps, args.slippage_bps, args.spread_bps, annualization,
        beta_neutral=args.beta_neutral,
        market_symbol=args.market_symbol,
        beta_window=args.beta_window,
        max_gross=args.max_gross,
        max_net=args.max_net,
    )
    result = out[0]
    positions = out[1]
    prices = out[2]
    turnover = out[3]
    w_held = out[4]
    beta_before = out[5] if len(out) > 5 else None
    beta_after = out[6] if len(out) > 6 else None
    hedge_weight = out[7] if len(out) > 7 else None

    print("[3/4] Results:")
    print("-" * 50)
    print(f"  Sharpe:      {result.sharpe_ratio:.2f}")
    print(f"  Total Ret:   {result.total_return:.2%}")
    print(f"  Max DD:      {result.max_drawdown:.2%}")
    print(f"  Trades:      {result.n_trades}")
    print(f"  Turnover:    {turnover.mean() * annualization:.2f} (ann.)")
    print("-" * 50)

    if combo_weights is not None:
        (output_dir / "combo_weights.json").write_text(
            json.dumps(combo_weights, indent=2), encoding="utf-8"
        )

    print("[4/4] Writing tear-sheet...")
    prices_1d = prices.mean(axis=1)
    config = {
        "factor": args.factor,
        "rebalance": args.rebalance,
        "top_k": args.top_k,
        "bottom_k": args.bottom_k,
    }
    if combo_weights is not None:
        config["combo_weights"] = combo_weights
        config["combo"] = args.combo
        config["combo_method"] = args.combo_method
    generate_tearsheet(
        result, prices_1d, positions,
        output_dir, annualization=annualization,
        config=config,
        weights=w_held,
        turnover_series=turnover,
        prices_wide=prices,
        portfolio_beta_before=beta_before,
        portfolio_beta_after=beta_after,
        hedge_weight=hedge_weight,
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
    if combo_weights is not None:
        summary["combo_weights"] = combo_weights
        summary["combo"] = args.combo
        summary["combo_method"] = args.combo_method
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
