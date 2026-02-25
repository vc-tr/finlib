"""
Factor backtest runner: CLI-friendly main(args).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.factor_backtest import get_factor_df, run_factor_backtest, run_factor_walkforward
from src.factors import (
    UniverseRegistry,
    cross_sectional_ic,
    forward_returns,
    get_universe,
    rolling_portfolio_beta,
    summarize_ic,
)
from src.factors.factors import get_prices_wide
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.reporting.tearsheet import generate_tearsheet
from src.utils.io import fetch_universe_ohlcv


def main(args: Any, cmd: str | None = None) -> Path:
    """
    Run factor backtest (or walkforward). Returns output_dir.
    Exits with sys.exit(1) on error.
    """
    if args.list_universes:
        print("Available universes:")
        for name in UniverseRegistry.list_names():
            meta = UniverseRegistry.get_meta(name)
            print(f"  {name}: {meta.description} ({meta.category})")
        sys.exit(0)

    symbols = get_universe(args.universe, n=50)
    if getattr(args, "beta_neutral", False) and getattr(args, "market_symbol", "SPY") not in symbols:
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

    if getattr(args, "walkforward", False):
        return _run_walkforward(args, df_by_symbol, combo_list, annualization, output_dir)

    return _run_single(args, df_by_symbol, combo_list, annualization, output_dir, cmd)


def _run_walkforward(
    args: Any,
    df_by_symbol: dict[str, pd.DataFrame],
    combo_list: list[str] | None,
    annualization: float,
    output_dir: Path,
) -> Path:
    """Walk-forward path."""
    wf_result = run_factor_walkforward(
        df_by_symbol,
        args.factor,
        combo_list,
        args.combo_method,
        args.top_k,
        args.bottom_k,
        args.rebalance,
        args.fee_bps,
        args.slippage_bps,
        args.spread_bps,
        annualization,
        args.folds,
        args.train_days,
        args.test_days,
        embargo_days=getattr(args, "embargo_days", 1),
        auto_metric=getattr(args, "auto_metric", "val_ic_ir"),
        val_split=getattr(args, "val_split", 0.3),
        shrinkage=getattr(args, "shrinkage", 0.5),
        beta_neutral=getattr(args, "beta_neutral", False),
        market_symbol=getattr(args, "market_symbol", "SPY"),
        beta_window=getattr(args, "beta_window", 252),
        max_gross=getattr(args, "max_gross", None),
        max_net=getattr(args, "max_net", None),
        cost_model=getattr(args, "cost_model", "fixed"),
        impact_k=getattr(args, "impact_k", 10.0),
        impact_alpha=getattr(args, "impact_alpha", 0.5),
        max_impact_bps=getattr(args, "max_impact_bps", 50.0),
        adv_window=getattr(args, "adv_window", 20),
        portfolio_value=getattr(args, "portfolio_value", 1e6),
    )
    agg = wf_result["aggregated"]
    print(f"  Mean Sharpe: {agg['mean_sharpe']:.2f}")
    print(f"  Agg Sharpe:  {agg['agg_sharpe']:.2f}")

    prices_wide = get_prices_wide(df_by_symbol)
    market_sym = getattr(args, "market_symbol", "SPY")
    if market_sym in prices_wide.columns and "all_returns" in wf_result:
        market_ret = prices_wide[market_sym].pct_change()
        agg_ret = wf_result["all_returns"]
        beta_p = rolling_portfolio_beta(agg_ret, market_ret, window=getattr(args, "beta_window", 252))
        beta_p.to_csv(output_dir / "beta_series.csv", header=["beta_p"])
        valid = beta_p.dropna()
        agg["beta_mean"] = float(valid.mean()) if len(valid) > 0 else 0.0
        agg["beta_std"] = float(valid.std()) if len(valid) > 1 else 0.0
        agg["beta_max_abs"] = float(beta_p.abs().max()) if len(beta_p) > 0 else 0.0

    ic_horizons = [int(x.strip()) for x in getattr(args, "ic_horizons", "1,5,21").split(",") if x.strip()] or [1, 5, 21]
    ic_method = getattr(args, "ic_method", "spearman")

    if getattr(args, "report_ic", False) and "all_returns" in wf_result and "factor_df_per_fold" in wf_result:
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
                ic_s = cross_sectional_ic(factor_slice, fwd_slice, method=ic_method)
                ic_list.append(ic_s)
            ic_concat = pd.concat(ic_list).sort_index()
            ic_concat = ic_concat[~ic_concat.index.duplicated(keep="first")]
            ic_summary_wf[str(h)] = summarize_ic(ic_concat)
            ic_concat.to_csv(output_dir / f"ic_h{h}.csv", header=["ic"])
        ic_config = {"horizons": ic_horizons, "method": ic_method, "scope": "walkforward_test_windows_only", "note": "IC computed on concatenated OOS test windows; no train data used."}
        (output_dir / "ic_summary.json").write_text(json.dumps({"config": ic_config, "summary": ic_summary_wf}, indent=2), encoding="utf-8")
    elif getattr(args, "report_ic", False) and "all_returns" in wf_result:
        fwd = forward_returns(prices_wide, horizons=ic_horizons)
        agg_ret = wf_result["all_returns"]
        factor_full, _, _, _ = get_factor_df(df_by_symbol, args.factor, combo_list, args.combo_method, None)
        factor_test = factor_full.reindex(agg_ret.index).dropna(how="all")
        ic_summary_wf = {}
        for h in ic_horizons:
            fwd_h = fwd[h].reindex(factor_test.index)
            ic_s = cross_sectional_ic(factor_test, fwd_h, method=ic_method)
            ic_s.to_csv(output_dir / f"ic_h{h}.csv", header=["ic"])
            ic_summary_wf[str(h)] = summarize_ic(ic_s)
        ic_config = {"horizons": ic_horizons, "method": ic_method, "scope": "walkforward_test_windows_only", "note": "IC computed on dates in concatenated OOS test windows."}
        (output_dir / "ic_summary.json").write_text(json.dumps({"config": ic_config, "summary": ic_summary_wf}, indent=2), encoding="utf-8")

    summary = {"walkforward": True, "aggregated": agg, "per_fold": wf_result["per_fold"]}
    if "combo_weights_per_fold" in wf_result:
        summary["combo_weights_per_fold"] = wf_result["combo_weights_per_fold"]
        (output_dir / "combo_weights.json").write_text(json.dumps(wf_result["combo_weights_per_fold"], indent=2), encoding="utf-8")
        cw = wf_result["combo_weights_per_fold"]
        report_lines = [
            "# Walk-Forward Combo Report", "",
            "## Summary", "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean Sharpe | {agg['mean_sharpe']:.2f} |",
            f"| Agg Sharpe | {agg['agg_sharpe']:.2f} |",
            f"| Agg Total Return | {agg['agg_total_return']:.2%} |",
            f"| Folds | {agg['n_folds']} |", "",
        ]
        if args.combo_method == "auto_robust":
            report_lines.extend([
                "## Auto-Robust Config", "",
                f"| Setting | Value |",
                f"|---------|-------|",
                f"| Selection metric | {getattr(args, 'auto_metric', 'val_ic_ir')} |",
                f"| Val split | {getattr(args, 'val_split', 0.3)} |",
                f"| Shrinkage | {getattr(args, 'shrinkage', 0.5)} |", "",
            ])
        report_lines.extend([
            "## Combo Weights by Fold", "",
            "| Fold | Test Start | Test End | Selected Method | Val Score |",
            "|------|------------|----------|-----------------|-----------|",
        ])
        for e in cw:
            method = e.get("selected_method", args.combo_method)
            val_score = e.get("val_score", "")
            val_str = f"{val_score:.4f}" if isinstance(val_score, (int, float)) else str(val_score)
            report_lines.append(f"| {e['fold_idx']} | {e['test_start']} | {e['test_end']} | {method} | {val_str} |")
        report_lines.append("")
        factor_names = list(cw[0]["weights"].keys()) if cw else []
        if factor_names:
            w_df = pd.DataFrame([e["weights"] for e in cw]).fillna(0)
            avg_w, std_w = w_df.mean(), w_df.std()
            report_lines.extend(["## Average Weights Across Folds", "", "| Factor | Mean | Std |", "|--------|------|-----|"])
            for f in factor_names:
                report_lines.append(f"| {f} | {avg_w.get(f, 0):.4f} | {std_w.get(f, 0):.4f} |")
            report_lines.append("")
        report_lines.extend(["- [combo_weights.json](combo_weights.json) — full per-fold weights", ""])
        (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir


def _run_single(
    args: Any,
    df_by_symbol: dict[str, pd.DataFrame],
    combo_list: list[str] | None,
    annualization: float,
    output_dir: Path,
    cmd: str | None,
) -> Path:
    """Single backtest path."""
    print(f"[2/4] Computing factor {args.factor}...")
    factor_df, combo_weights, zscored, _ = get_factor_df(
        df_by_symbol,
        args.factor,
        combo_list,
        args.combo_method,
        None,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        rebalance=args.rebalance,
        cost_bps=args.fee_bps + args.slippage_bps + args.spread_bps,
    )
    out = run_factor_backtest(
        df_by_symbol,
        factor_df,
        args.top_k,
        args.bottom_k,
        args.rebalance,
        args.fee_bps,
        args.slippage_bps,
        args.spread_bps,
        annualization,
        beta_neutral=getattr(args, "beta_neutral", False),
        market_symbol=getattr(args, "market_symbol", "SPY"),
        beta_window=getattr(args, "beta_window", 252),
        max_gross=getattr(args, "max_gross", None),
        max_net=getattr(args, "max_net", None),
        cost_model=getattr(args, "cost_model", "fixed"),
        impact_k=getattr(args, "impact_k", 10.0),
        impact_alpha=getattr(args, "impact_alpha", 0.5),
        max_impact_bps=getattr(args, "max_impact_bps", 50.0),
        adv_window=getattr(args, "adv_window", 20),
        portfolio_value=getattr(args, "portfolio_value", 1e6),
    )
    result, positions, prices, turnover, w_held = out[0], out[1], out[2], out[3], out[4]
    beta_before = out[5] if len(out) > 5 else None
    beta_after = out[6] if len(out) > 6 else None
    hedge_weight = out[7] if len(out) > 7 else None
    n_rebalance = out[8] if len(out) > 8 else 0
    capacity_report = out[9] if len(out) > 9 else None

    if capacity_report:
        (output_dir / "capacity_report.json").write_text(json.dumps(capacity_report, indent=2), encoding="utf-8")

    turnover_at_rb = turnover[turnover > 1e-10]
    avg_turnover_per_rebalance = turnover_at_rb.mean() if len(turnover_at_rb) > 0 else 0.0
    n_years = max(1e-6, len(result.returns) / annualization)
    annual_turnover = avg_turnover_per_rebalance * (n_rebalance / n_years)

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

    if combo_weights:
        (output_dir / "combo_weights.json").write_text(json.dumps(combo_weights, indent=2), encoding="utf-8")

    factor_attribution = None
    if combo_weights and zscored:
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
            exposures[fname] = (w_f * z_f).sum(axis=1)
        exp_df = pd.DataFrame(exposures)
        exp_df.to_csv(output_dir / "factor_exposures.csv")
        att = {}
        for fname in exp_df.columns:
            ex = exp_df[fname].dropna().reindex(port_ret.index).dropna()
            ret_aligned = port_ret.reindex(ex.index).fillna(0)
            corr = ex.corr(ret_aligned) if len(ex) >= 5 and ex.std() > 1e-12 else float("nan")
            att[fname] = {"mean_exposure": float(ex.mean()), "std_exposure": float(ex.std()) if len(ex) > 1 else 0.0, "corr_with_returns": float(corr) if pd.notna(corr) else None}
        factor_attribution = att
        (output_dir / "factor_attribution.json").write_text(json.dumps(factor_attribution, indent=2), encoding="utf-8")

    ic_summary = None
    ic_preview = None
    if getattr(args, "report_ic", False):
        ic_horizons = [int(x.strip()) for x in getattr(args, "ic_horizons", "1,5,21").split(",") if x.strip()] or [1, 5, 21]
        ic_method = getattr(args, "ic_method", "spearman")
        fwd = forward_returns(prices, horizons=ic_horizons)
        ic_summary = {}
        ic_preview = {}
        for h in ic_horizons:
            ic_series = cross_sectional_ic(factor_df, fwd[h], method=ic_method)
            ic_series.to_csv(output_dir / f"ic_h{h}.csv", header=["ic"])
            ic_summary[str(h)] = summarize_ic(ic_series)
            ic_preview[str(h)] = ic_series.dropna().tail(10).tolist()
        (output_dir / "ic_summary.json").write_text(
            json.dumps({"config": {"horizons": ic_horizons, "method": ic_method, "scope": "full_backtest_window"}, "summary": ic_summary}, indent=2
        ), encoding="utf-8")

    beta_series = None
    market_sym = getattr(args, "market_symbol", "SPY")
    if market_sym in prices.columns:
        market_ret = prices[market_sym].pct_change()
        beta_series = rolling_portfolio_beta(result.returns, market_ret, window=getattr(args, "beta_window", 252))
        beta_series.to_csv(output_dir / "beta_series.csv", header=["beta_p"])

    print("[4/4] Writing tear-sheet...")
    prices_1d = prices.mean(axis=1)
    cmd_str = cmd or "python scripts/backtest_factors.py"
    config = {"factor": args.factor, "universe": args.universe, "period": args.period, "interval": args.interval, "rebalance": args.rebalance, "top_k": args.top_k, "bottom_k": args.bottom_k, "cmd": cmd_str}
    if combo_weights:
        config["combo_weights"] = combo_weights
        config["combo"] = args.combo
        config["combo_method"] = args.combo_method
    generate_tearsheet(
        result, prices_1d, positions, output_dir, annualization=annualization,
        config=config, weights=w_held, turnover_series=turnover, prices_wide=prices,
        ic_summary=ic_summary, ic_preview=ic_preview, combo_weights=combo_weights,
        factor_attribution=factor_attribution, portfolio_beta_before=beta_before,
        portfolio_beta_after=beta_after, hedge_weight=hedge_weight, beta_series=beta_series,
        beta_neutral=getattr(args, "beta_neutral", False), capacity_report=capacity_report,
    )
    summary = {
        "factor": args.factor, "universe": args.universe, "period": args.period, "interval": args.interval,
        "rebalance": args.rebalance, "top_k": args.top_k, "bottom_k": args.bottom_k,
        "sharpe": result.sharpe_ratio, "total_return": result.total_return, "max_drawdown": result.max_drawdown,
        "n_trades": result.n_trades, "n_rebalance": n_rebalance,
        "avg_turnover_per_rebalance": avg_turnover_per_rebalance, "annual_turnover": annual_turnover,
        "cmd": cmd_str,
    }
    if combo_weights:
        summary["combo_weights"] = combo_weights
        summary["combo"] = args.combo
        summary["combo_method"] = args.combo_method
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir
