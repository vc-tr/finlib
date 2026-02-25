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
from src.factors import (
    compute_factor,
    compute_factors,
    build_portfolio,
    get_universe,
    UniverseRegistry,
    estimate_beta,
    rolling_portfolio_beta,
    forward_returns,
    cross_sectional_ic,
    summarize_ic,
)
from src.factors.factors import get_prices_wide
from src.factors.ensemble import combine_factors
from src.factors.portfolio import (
    apply_rebalance_costs,
    _resample_weights_to_rebalance,
    apply_beta_neutral,
    apply_constraints,
    rebalance_dates,
    weights_at_rebalance,
)
from src.backtest.cost_models import (
    FixedBpsCostModel,
    LiquidityAwareCostModel,
    build_trades_from_weights,
    apply_costs_from_trades,
    compute_capacity_report,
)
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
    cost_model: str = "fixed",
    impact_k: float = 10.0,
    impact_alpha: float = 0.5,
    max_impact_bps: float = 50.0,
    adv_window: int = 20,
    portfolio_value: float = 1e6,
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
        8: n_rebalance_events (int)
    """
    weights = weights_at_rebalance(
        factor_df,
        rebalance=rebalance,
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

    cost_config = {
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "spread_bps": spread_bps,
        "impact_k": impact_k,
        "impact_alpha": impact_alpha,
        "max_impact_bps": max_impact_bps,
        "adv_window": adv_window,
    }
    trades_df = build_trades_from_weights(w_held, prices, portfolio_value=portfolio_value)
    if cost_model == "liquidity":
        model = LiquidityAwareCostModel()
    else:
        model = FixedBpsCostModel()
    trades_df = model.estimate_costs(trades_df, df_by_symbol, cost_config)
    port_ret = apply_costs_from_trades(port_ret, trades_df)

    bt = Backtester(annualization_factor=annualization)
    result = bt.run(port_ret)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0)
    positions = w_held.sum(axis=1)
    rb_dates = rebalance_dates(factor_df.index, rebalance)
    n_rebalance = len(rb_dates)
    capacity_report = compute_capacity_report(
        trades_df, df_by_symbol, cost_config,
        adv_window=adv_window, target_impact_bps=10.0,
    )
    return result, positions, prices, turnover, w_held, beta_before, beta_after, hedge_weight, n_rebalance, capacity_report


def _get_factor_df(
    df_by_symbol: dict[str, pd.DataFrame],
    factor: str,
    combo_list: list[str] | None,
    combo_method: str,
    train_slice: slice | None = None,
    top_k: int = 10,
    bottom_k: int = 10,
    rebalance: str = "M",
    cost_bps: float = 4.0,
    auto_metric: str = "val_ic_ir",
    val_split: float = 0.3,
    shrinkage: float = 0.5,
) -> tuple[pd.DataFrame, dict | None, dict[str, pd.DataFrame] | None, dict]:
    """
    Get factor DataFrame for backtest. For single factor or combo.
    Returns (factor_df, combo_weights or None, zscored_dict or None, meta).
    """
    if combo_list is None:
        factor_df = compute_factor(df_by_symbol, factor)
        return factor_df, None, None, {}

    factors_dict = compute_factors(df_by_symbol, combo_list)
    prices = get_prices_wide(df_by_symbol)
    fwd_returns = prices.pct_change().shift(-1)
    fwd_returns_dict = forward_returns(prices, horizons=[1, 5, 21])

    combo_kw = dict(
        fwd_returns=fwd_returns,
        fwd_returns_dict=fwd_returns_dict,
        prices=prices,
        top_k=top_k,
        bottom_k=bottom_k,
        rebalance=rebalance,
        cost_bps=cost_bps,
    )
    if combo_method == "auto_robust":
        combo_kw["auto_metric"] = auto_metric
        combo_kw["val_split"] = val_split
        combo_kw["shrinkage"] = shrinkage

    if combo_method == "equal":
        combined, weights, zscored, meta = combine_factors(factors_dict, method="equal")
    else:
        if train_slice is None:
            idx = prices.index
            n = int(len(idx) * 0.7)
            if n < 30:
                combined, weights, zscored, meta = combine_factors(factors_dict, method="equal")
            else:
                ts = slice(idx[0], idx[n - 1])
                combined, weights, zscored, meta = combine_factors(
                    factors_dict,
                    method=combo_method,
                    train_slice=ts,
                    **combo_kw,
                )
        else:
            combined, weights, zscored, meta = combine_factors(
                factors_dict,
                method=combo_method,
                train_slice=train_slice,
                **combo_kw,
            )
    return combined, weights, zscored, meta


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
    embargo_days: int = 1,
    auto_metric: str = "val_ic_ir",
    val_split: float = 0.3,
    shrinkage: float = 0.5,
    beta_neutral: bool = False,
    market_symbol: str = "SPY",
    beta_window: int = 252,
    max_gross: float | None = None,
    max_net: float | None = None,
    cost_model: str = "fixed",
    impact_k: float = 10.0,
    impact_alpha: float = 0.5,
    max_impact_bps: float = 50.0,
    adv_window: int = 20,
    portfolio_value: float = 1e6,
) -> dict:
    """Run walk-forward: use history up to test_end, evaluate on test window only."""
    from src.backtest.walkforward import generate_folds

    prices_wide = get_prices_wide(df_by_symbol)
    index = prices_wide.index
    fold_list = generate_folds(
        index, train_days, test_days, test_days,
        max_folds=folds, embargo_days=embargo_days,
    )

    per_fold = []
    all_returns = []
    combo_weights_per_fold = []
    factor_df_per_fold = []

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
            cost_bps = fee_bps + slippage_bps + spread_bps
            factor_df, combo_weights, _, meta = _get_factor_df(
                df_by_hist, factor, combo_list, combo_method, train_slice,
                top_k=top_k, bottom_k=bottom_k, rebalance=rebalance,
                cost_bps=cost_bps,
                auto_metric=auto_metric, val_split=val_split, shrinkage=shrinkage,
            )
            if combo_weights is not None:
                entry = {
                    "fold_idx": fold.fold_idx,
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "weights": combo_weights,
                }
                if meta.get("selected_method"):
                    entry["selected_method"] = meta["selected_method"]
                if meta.get("val_score") is not None:
                    entry["val_score"] = meta["val_score"]
                if meta.get("weights_before_shrink"):
                    entry["weights_before_shrink"] = meta["weights_before_shrink"]
                if meta.get("weights_final"):
                    entry["weights_final"] = meta["weights_final"]
                combo_weights_per_fold.append(entry)
        else:
            factor_df, _, _, _ = _get_factor_df(df_by_hist, factor, None, "equal", None)
        out = _run_factor_backtest(
            df_by_hist, factor_df, top_k, bottom_k, rebalance,
            fee_bps, slippage_bps, spread_bps, annualization,
            beta_neutral=beta_neutral,
            market_symbol=market_symbol,
            beta_window=beta_window,
            max_gross=max_gross,
            max_net=max_net,
            cost_model=cost_model,
            impact_k=impact_k,
            impact_alpha=impact_alpha,
            max_impact_bps=max_impact_bps,
            adv_window=adv_window,
            portfolio_value=portfolio_value,
        )
        result = out[0]
        test_ret = result.returns.loc[test_start:test_end].dropna()
        if len(test_ret) < 5:
            continue
        factor_df_per_fold.append((fold.fold_idx, factor_df))
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
        agg_ret = None
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
    if agg_ret is not None:
        out["all_returns"] = agg_ret
    if factor_df_per_fold:
        out["factor_df_per_fold"] = factor_df_per_fold
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional factor backtest")
    parser.add_argument(
        "--universe",
        default="liquid_etfs",
        help=f"Universe name (choices: {', '.join(UniverseRegistry.list_names())})",
    )
    parser.add_argument(
        "--list-universes",
        action="store_true",
        help="List available universes and exit",
    )
    parser.add_argument("--factor", default="momentum_12_1",
                        choices=["momentum_12_1", "reversal_5d", "lowvol_20d", "combo"])
    parser.add_argument("--combo", default=None,
                        help='Comma-separated factors for combo (e.g. "momentum_12_1,reversal_5d,lowvol_20d")')
    parser.add_argument("--combo-method", default="equal",
                        choices=["equal", "ic_weighted", "ridge", "sharpe_opt", "auto", "auto_robust"],
                        help="Combo weighting: equal, ic_weighted, ridge, sharpe_opt, auto, auto_robust")
    parser.add_argument("--auto-metric", default="val_ic_ir", choices=["val_sharpe", "val_ic_ir"],
                        help="For auto_robust: selection metric (default val_ic_ir)")
    parser.add_argument("--val-split", type=float, default=0.3,
                        help="For auto_robust: validation fraction of train window (default 0.3)")
    parser.add_argument("--shrinkage", type=float, default=0.5,
                        help="For auto_robust: weight shrinkage toward equal (default 0.5)")
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
    parser.add_argument("--embargo-days", type=int, default=1,
                        help="Days between train end and test start (for IC/fwd returns)")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--lock-timeout", type=float, default=0)
    parser.add_argument("--beta-neutral", action="store_true", help="Hedge portfolio beta with market (SPY)")
    parser.add_argument("--market-symbol", default="SPY", help="Market symbol for beta hedge (default SPY)")
    parser.add_argument("--beta-window", type=int, default=252, help="Rolling window for beta estimation")
    parser.add_argument("--max-gross", type=float, default=None, help="Cap gross exposure per rebalance")
    parser.add_argument("--max-net", type=float, default=None, help="Cap net exposure per rebalance")
    parser.add_argument("--cost-model", default="fixed", choices=["fixed", "liquidity"],
                        help="Cost model: fixed (bps) or liquidity-aware")
    parser.add_argument("--impact-k", type=float, default=10.0, help="Impact model coefficient")
    parser.add_argument("--impact-alpha", type=float, default=0.5, help="Impact model exponent")
    parser.add_argument("--max-impact-bps", type=float, default=50.0, help="Cap impact cost in bps")
    parser.add_argument("--adv-window", type=int, default=20, help="Rolling window for ADV")
    parser.add_argument("--portfolio-value", type=float, default=1e6, help="AUM for capacity/impact")
    parser.add_argument("--report-ic", action="store_true", help="Compute and report IC/IR metrics")
    parser.add_argument("--ic-horizons", default="1,5,21", help="Comma-separated IC horizons (default 1,5,21)")
    parser.add_argument("--ic-method", default="spearman", choices=["spearman", "pearson"], help="IC correlation method")
    args = parser.parse_args()

    if args.list_universes:
        print("Available universes:")
        for name in UniverseRegistry.list_names():
            meta = UniverseRegistry.get_meta(name)
            print(f"  {name}: {meta.description} ({meta.category})")
        return

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
            embargo_days=args.embargo_days,
            auto_metric=args.auto_metric,
            val_split=args.val_split,
            shrinkage=args.shrinkage,
            beta_neutral=args.beta_neutral,
            market_symbol=args.market_symbol,
            beta_window=args.beta_window,
            max_gross=args.max_gross,
            max_net=args.max_net,
            cost_model=args.cost_model,
            impact_k=args.impact_k,
            impact_alpha=args.impact_alpha,
            max_impact_bps=args.max_impact_bps,
            adv_window=args.adv_window,
            portfolio_value=args.portfolio_value,
        )
        agg = wf_result["aggregated"]
        print(f"  Mean Sharpe: {agg['mean_sharpe']:.2f}")
        print(f"  Agg Sharpe:  {agg['agg_sharpe']:.2f}")

        # Beta series for walkforward: market returns vs aggregated OOS returns
        prices_wide = get_prices_wide(df_by_symbol)
        market_sym = args.market_symbol
        if market_sym in prices_wide.columns:
            market_ret = prices_wide[market_sym].pct_change()
            if "all_returns" in wf_result:
                agg_ret = wf_result["all_returns"]
                beta_p = rolling_portfolio_beta(agg_ret, market_ret, window=args.beta_window)
                beta_p.to_csv(output_dir / "beta_series.csv", header=["beta_p"])
                valid = beta_p.dropna()
                agg["beta_mean"] = float(valid.mean()) if len(valid) > 0 else 0.0
                agg["beta_std"] = float(valid.std()) if len(valid) > 1 else 0.0
                agg["beta_max_abs"] = float(beta_p.abs().max()) if len(beta_p) > 0 else 0.0

        # IC reporting for walkforward: compute on concatenated test windows only
        if args.report_ic and "all_returns" in wf_result and "factor_df_per_fold" in wf_result:
            ic_horizons = [int(x.strip()) for x in args.ic_horizons.split(",") if x.strip()] or [1, 5, 21]
            fwd = forward_returns(prices_wide, horizons=ic_horizons)
            factor_by_fold = {fidx: fdf for fidx, fdf in wf_result["factor_df_per_fold"]}
            ic_summary_wf = {}
            for h in ic_horizons:
                ic_list = []
                for fold in wf_result.get("per_fold", []):
                    fidx = fold["fold_idx"]
                    if fidx not in factor_by_fold:
                        continue
                    test_start = pd.Timestamp(fold["test_start"])
                    test_end = pd.Timestamp(fold["test_end"])
                    factor_slice = factor_by_fold[fidx].loc[test_start:test_end]
                    fwd_slice = fwd[h].loc[test_start:test_end]
                    ic_s = cross_sectional_ic(factor_slice, fwd_slice, method=args.ic_method)
                    ic_list.append(ic_s)
                ic_concat = pd.concat(ic_list).sort_index()
                ic_concat = ic_concat[~ic_concat.index.duplicated(keep="first")]
                ic_summary_wf[str(h)] = summarize_ic(ic_concat)
                ic_concat.to_csv(output_dir / f"ic_h{h}.csv", header=["ic"])
            ic_config = {
                "horizons": ic_horizons,
                "method": args.ic_method,
                "scope": "walkforward_test_windows_only",
                "note": "IC computed on concatenated OOS test windows; no train data used.",
            }
            (output_dir / "ic_summary.json").write_text(
                json.dumps({"config": ic_config, "summary": ic_summary_wf}, indent=2), encoding="utf-8"
            )
        elif args.report_ic and "all_returns" in wf_result:
            # Simpler: compute IC on full factor_df aligned with test dates
            ic_horizons = [int(x.strip()) for x in args.ic_horizons.split(",") if x.strip()] or [1, 5, 21]
            fwd = forward_returns(prices_wide, horizons=ic_horizons)
            agg_ret = wf_result["all_returns"]
            test_dates = agg_ret.index
            factor_full, _, _, _ = _get_factor_df(df_by_symbol, args.factor, combo_list, args.combo_method, None)
            factor_test = factor_full.reindex(test_dates).dropna(how="all")
            ic_summary_wf = {}
            for h in ic_horizons:
                fwd_h = fwd[h].reindex(factor_test.index)
                ic_s = cross_sectional_ic(factor_test, fwd_h, method=args.ic_method)
                ic_s.to_csv(output_dir / f"ic_h{h}.csv", header=["ic"])
                ic_summary_wf[str(h)] = summarize_ic(ic_s)
            ic_config = {
                "horizons": ic_horizons,
                "method": args.ic_method,
                "scope": "walkforward_test_windows_only",
                "note": "IC computed on dates in concatenated OOS test windows.",
            }
            (output_dir / "ic_summary.json").write_text(
                json.dumps({"config": ic_config, "summary": ic_summary_wf}, indent=2), encoding="utf-8"
            )

        summary = {"walkforward": True, "aggregated": agg, "per_fold": wf_result["per_fold"]}
        if "combo_weights_per_fold" in wf_result:
            summary["combo_weights_per_fold"] = wf_result["combo_weights_per_fold"]
            (output_dir / "combo_weights.json").write_text(
                json.dumps(wf_result["combo_weights_per_fold"], indent=2), encoding="utf-8"
            )
            # REPORT.md for combo walkforward
            cw = wf_result["combo_weights_per_fold"]
            report_lines = [
                "# Walk-Forward Combo Report",
                "",
                "## Summary",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Mean Sharpe | {agg['mean_sharpe']:.2f} |",
                f"| Agg Sharpe | {agg['agg_sharpe']:.2f} |",
                f"| Agg Total Return | {agg['agg_total_return']:.2%} |",
                f"| Folds | {agg['n_folds']} |",
                "",
            ]
            if args.combo_method == "auto_robust":
                report_lines.extend([
                    "## Auto-Robust Config",
                    "",
                    f"| Setting | Value |",
                    f"|---------|-------|",
                    f"| Selection metric | {args.auto_metric} |",
                    f"| Val split | {args.val_split} |",
                    f"| Shrinkage | {args.shrinkage} |",
                    "",
                ])
            report_lines.extend([
                "## Combo Weights by Fold",
                "",
                "| Fold | Test Start | Test End | Selected Method | Val Score |",
                "|------|------------|----------|-----------------|-----------|",
            ])
            for e in cw:
                method = e.get("selected_method", args.combo_method)
                val_score = e.get("val_score", "")
                val_str = f"{val_score:.4f}" if isinstance(val_score, (int, float)) else str(val_score)
                report_lines.append(f"| {e['fold_idx']} | {e['test_start']} | {e['test_end']} | {method} | {val_str} |")
            report_lines.append("")
            # Average and std of weights
            factor_names = list(cw[0]["weights"].keys()) if cw else []
            if factor_names:
                w_df = pd.DataFrame([e["weights"] for e in cw]).fillna(0)
                avg_w = w_df.mean()
                std_w = w_df.std()
                report_lines.extend([
                    "## Average Weights Across Folds",
                    "",
                    "| Factor | Mean | Std |",
                    "|--------|------|-----|",
                ])
                for f in factor_names:
                    report_lines.append(f"| {f} | {avg_w.get(f, 0):.4f} | {std_w.get(f, 0):.4f} |")
                report_lines.append("")
            report_lines.extend([
                "- [combo_weights.json](combo_weights.json) — full per-fold weights",
                "",
            ])
            (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Output: {output_dir}")
        return

    print(f"[2/4] Computing factor {args.factor}...")
    factor_df, combo_weights, zscored, _ = _get_factor_df(
        df_by_symbol, args.factor, combo_list, args.combo_method, None,
        top_k=args.top_k, bottom_k=args.bottom_k, rebalance=args.rebalance,
        cost_bps=args.fee_bps + args.slippage_bps + args.spread_bps,
    )
    out = _run_factor_backtest(
        df_by_symbol, factor_df, args.top_k, args.bottom_k, args.rebalance,
        args.fee_bps, args.slippage_bps, args.spread_bps, annualization,
        beta_neutral=args.beta_neutral,
        market_symbol=args.market_symbol,
        beta_window=args.beta_window,
        max_gross=args.max_gross,
        max_net=args.max_net,
        cost_model=args.cost_model,
        impact_k=args.impact_k,
        impact_alpha=args.impact_alpha,
        max_impact_bps=args.max_impact_bps,
        adv_window=args.adv_window,
        portfolio_value=args.portfolio_value,
    )
    result = out[0]
    positions = out[1]
    prices = out[2]
    turnover = out[3]
    w_held = out[4]
    beta_before = out[5] if len(out) > 5 else None
    beta_after = out[6] if len(out) > 6 else None
    hedge_weight = out[7] if len(out) > 7 else None
    n_rebalance = out[8] if len(out) > 8 else 0
    capacity_report = out[9] if len(out) > 9 else None

    if capacity_report is not None:
        (output_dir / "capacity_report.json").write_text(
            json.dumps(capacity_report, indent=2), encoding="utf-8"
        )

    # Turnover at rebalance/execution timestamps only
    turnover_at_rb = turnover[turnover > 1e-10]
    avg_turnover_per_rebalance = turnover_at_rb.mean() if len(turnover_at_rb) > 0 else 0.0
    n_years = max(1e-6, len(result.returns) / annualization)
    rebalances_per_year = n_rebalance / n_years
    annual_turnover = avg_turnover_per_rebalance * rebalances_per_year

    print("[3/4] Results:")
    print("-" * 50)
    print(f"  Sharpe:      {result.sharpe_ratio:.2f}")
    print(f"  Total Ret:   {result.total_return:.2%}")
    print(f"  Max DD:      {result.max_drawdown:.2%}")
    print(f"  Trades:      {result.n_trades}")
    print(f"  Rebalances:  {n_rebalance}")
    print(f"  Avg turnover per rebalance: {avg_turnover_per_rebalance:.2f}")
    print(f"  Annual turnover: {annual_turnover:.2f}")
    print("-" * 50)

    if combo_weights is not None:
        (output_dir / "combo_weights.json").write_text(
            json.dumps(combo_weights, indent=2), encoding="utf-8"
        )

    # Factor attribution (combo only): exposures + corr with returns
    factor_attribution = None
    if combo_weights is not None and zscored is not None:
        common_idx = w_held.index.intersection(result.returns.index)
        common_cols = w_held.columns
        w_aligned = w_held.reindex(index=common_idx, columns=common_cols).fillna(0)
        port_ret = result.returns.reindex(common_idx).fillna(0)
        exposures = {}
        for fname, zdf in zscored.items():
            cols_f = common_cols.intersection(zdf.columns)
            idx_f = common_idx.intersection(zdf.index)
            w_f = w_aligned.reindex(index=idx_f, columns=cols_f).fillna(0)
            z_f = zdf.reindex(index=idx_f, columns=cols_f).fillna(0)
            exp = (w_f * z_f).sum(axis=1)
            exposures[fname] = exp
        exp_df = pd.DataFrame(exposures)
        exp_df.to_csv(output_dir / "factor_exposures.csv")
        att = {}
        for fname in exp_df.columns:
            ex = exp_df[fname].dropna()
            ex = ex.reindex(port_ret.index).dropna()
            ret_aligned = port_ret.reindex(ex.index).fillna(0)
            if len(ex) >= 5 and ex.std() > 1e-12:
                corr = ex.corr(ret_aligned)
            else:
                corr = float("nan")
            att[fname] = {
                "mean_exposure": float(ex.mean()),
                "std_exposure": float(ex.std()) if len(ex) > 1 else 0.0,
                "corr_with_returns": float(corr) if pd.notna(corr) else None,
            }
        factor_attribution = att
        (output_dir / "factor_attribution.json").write_text(
            json.dumps(factor_attribution, indent=2), encoding="utf-8"
        )

    # IC research metrics (when --report-ic)
    ic_summary = None
    ic_preview = None
    if args.report_ic:
        ic_horizons = [int(x.strip()) for x in args.ic_horizons.split(",") if x.strip()]
        if not ic_horizons:
            ic_horizons = [1, 5, 21]
        fwd = forward_returns(prices, horizons=ic_horizons)
        ic_summary = {}
        ic_preview = {}
        for h in ic_horizons:
            ic_series = cross_sectional_ic(factor_df, fwd[h], method=args.ic_method)
            ic_series.to_csv(output_dir / f"ic_h{h}.csv", header=["ic"])
            ic_summary[str(h)] = summarize_ic(ic_series)
            ic_preview[str(h)] = ic_series.dropna().tail(10).tolist()
        ic_config = {
            "horizons": ic_horizons,
            "method": args.ic_method,
            "scope": "full_backtest_window",
        }
        (output_dir / "ic_summary.json").write_text(
            json.dumps({"config": ic_config, "summary": ic_summary}, indent=2), encoding="utf-8"
        )

    # Market beta exposure: rolling beta of portfolio vs market
    beta_series = None
    market_sym = args.market_symbol
    if market_sym in prices.columns:
        market_ret = prices[market_sym].pct_change()
        beta_series = rolling_portfolio_beta(
            result.returns, market_ret, window=args.beta_window
        )
        beta_series.to_csv(output_dir / "beta_series.csv", header=["beta_p"])

    print("[4/4] Writing tear-sheet...")
    prices_1d = prices.mean(axis=1)
    cmd = "python scripts/backtest_factors.py " + " ".join(sys.argv[1:])
    config = {
        "factor": args.factor,
        "universe": args.universe,
        "period": args.period,
        "interval": args.interval,
        "rebalance": args.rebalance,
        "top_k": args.top_k,
        "bottom_k": args.bottom_k,
        "cmd": cmd,
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
        ic_summary=ic_summary,
        ic_preview=ic_preview,
        combo_weights=combo_weights,
        factor_attribution=factor_attribution,
        portfolio_beta_before=beta_before,
        portfolio_beta_after=beta_after,
        hedge_weight=hedge_weight,
        beta_series=beta_series,
        beta_neutral=args.beta_neutral,
        capacity_report=capacity_report,
    )
    summary = {
        "factor": args.factor,
        "universe": args.universe,
        "period": args.period,
        "interval": args.interval,
        "rebalance": args.rebalance,
        "top_k": args.top_k,
        "bottom_k": args.bottom_k,
        "sharpe": result.sharpe_ratio,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "n_trades": result.n_trades,
        "n_rebalance": n_rebalance,
        "avg_turnover_per_rebalance": avg_turnover_per_rebalance,
        "annual_turnover": annual_turnover,
        "cmd": cmd,
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
