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
from src.backtest.factor_backtest import (
    get_factor_df,
    run_factor_backtest,
    run_factor_walkforward,
)
from src.factors import (
    get_universe,
    UniverseRegistry,
    rolling_portfolio_beta,
    forward_returns,
    cross_sectional_ic,
    summarize_ic,
)
from src.factors.factors import get_prices_wide
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.reporting.tearsheet import generate_tearsheet
from src.utils.io import fetch_universe_ohlcv


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
    df_by_symbol = fetch_universe_ohlcv(symbols, args.interval, args.period, fetcher, warn_fn=print)
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
        wf_result = run_factor_walkforward(
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
            factor_full, _, _, _ = get_factor_df(df_by_symbol, args.factor, combo_list, args.combo_method, None)
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
    factor_df, combo_weights, zscored, _ = get_factor_df(
        df_by_symbol, args.factor, combo_list, args.combo_method, None,
        top_k=args.top_k, bottom_k=args.bottom_k, rebalance=args.rebalance,
        cost_bps=args.fee_bps + args.slippage_bps + args.spread_bps,
    )
    out = run_factor_backtest(
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
