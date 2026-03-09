"""
Tear-sheet report generation for backtest results.

Produces: summary.json, REPORT.md, equity_curve.png, drawdown.png, rolling_sharpe.png,
returns_hist.png, turnover.png, positions.png, tearsheet.html
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestResult
from src.backtest.execution import compute_turnover


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _median_adv(capacity_report: Dict[str, Any]) -> float:
    """Median of per-symbol avg ADV from capacity report."""
    per_sym = capacity_report.get("per_symbol_adv", {})
    if not per_sym:
        return 0.0
    advs = [v.get("avg_adv", 0) for v in per_sym.values() if v.get("avg_adv", 0) > 0]
    return float(np.median(advs)) if advs else 0.0


def _generate_index_md(out: Path, config: Dict[str, Any]) -> None:
    """Generate INDEX.md landing page with quick links and reproduce command."""
    parts = []
    factor = config.get("factor") or config.get("symbol") or "backtest"
    universe = config.get("universe", "")
    period = config.get("period", "")
    interval = config.get("interval", "")
    title_parts = [str(factor)]
    if universe:
        title_parts.append(universe)
    if period:
        title_parts.append(period)
    if interval:
        title_parts.append(interval)
    title = " / ".join(title_parts)

    lines = [
        f"# {title}",
        "",
        "## Quick links",
        "",
        "- [REPORT.md](REPORT.md)",
        "- [tearsheet.html](tearsheet.html)",
        "",
        "## Research",
        "",
    ]
    # IC research
    if (out / "ic_summary.json").exists():
        lines.append("- [ic_summary.json](ic_summary.json)")
        for h in [1, 5, 21]:
            if (out / f"ic_h{h}.csv").exists():
                lines.append(f"- [ic_h{h}.csv](ic_h{h}.csv)")
        lines.append("")
    # Beta
    if (out / "beta_series.csv").exists():
        lines.append("- [beta_series.csv](beta_series.csv)")
    if (out / "rolling_beta.png").exists():
        lines.append("- [rolling_beta.png](rolling_beta.png)")
    if (out / "beta_series.csv").exists() or (out / "rolling_beta.png").exists():
        lines.append("")
    # Holdings
    if (out / "holdings_by_date.csv").exists():
        lines.append("- [holdings_by_date.csv](holdings_by_date.csv)")
        lines.append("")
    # Combo weights
    if (out / "combo_weights.json").exists():
        lines.append("- [combo_weights.json](combo_weights.json)")
        lines.append("")
    # Capacity
    if (out / "capacity_report.json").exists():
        lines.append("- [capacity_report.json](capacity_report.json)")
        lines.append("")

    cmd = config.get("cmd")
    if cmd:
        lines.extend([
            "## How to reproduce",
            "",
            "```bash",
            cmd,
            "```",
            "",
        ])

    (out / "INDEX.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _summary_metrics(result: BacktestResult, annualization: float = 252) -> Dict[str, Any]:
    """Compute summary metrics (numeric + formatted strings)."""
    ret = result.returns
    if len(ret) == 0:
        return {}

    vol = ret.std() * np.sqrt(annualization) if ret.std() > 1e-10 else 0.0
    downside = ret[ret < 0].std() * np.sqrt(annualization) if (ret < 0).any() else 0.0
    sortino = (
        (ret.mean() * annualization) / downside
        if downside > 1e-10
        else 0.0
    )
    wins = (ret > 0).sum()
    losses = (ret < 0).sum()
    avg_win = ret[ret > 0].mean() if wins > 0 else 0.0
    avg_loss = ret[ret < 0].mean() if losses > 0 else 0.0

    n_years = len(ret) / annualization
    cagr = (1 + result.total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    return {
        "CAGR": f"{cagr:.2%}",
        "Sharpe": f"{result.sharpe_ratio:.2f}",
        "Sortino": f"{sortino:.2f}",
        "MaxDD": f"{result.max_drawdown:.2%}",
        "Vol": f"{vol:.2%}",
        "Win Rate": f"{result.win_rate:.1%}",
        "Avg Win": f"{avg_win:.2%}",
        "Avg Loss": f"{avg_loss:.2%}",
        "Trades": str(result.n_trades),
        "_cagr": cagr,
        "_sharpe": result.sharpe_ratio,
        "_sortino": sortino,
        "_max_dd": result.max_drawdown,
        "_vol": vol,
        "_trades": result.n_trades,
    }


def generate_tearsheet(
    result: BacktestResult,
    prices: pd.Series,
    signals: pd.Series,
    output_dir: Union[str, Path],
    rolling_window: Optional[int] = None,
    annualization: float = 252,
    config: Optional[Dict[str, Any]] = None,
    weights: Optional[pd.DataFrame] = None,
    turnover_series: Optional[pd.Series] = None,
    prices_wide: Optional[pd.DataFrame] = None,
    ic_summary: Optional[Dict[str, Dict[str, float]]] = None,
    ic_preview: Optional[Dict[str, list]] = None,
    combo_weights: Optional[Dict[str, float]] = None,
    factor_attribution: Optional[Dict[str, Dict[str, float]]] = None,
    portfolio_beta_before: Optional[pd.Series] = None,
    portfolio_beta_after: Optional[pd.Series] = None,
    hedge_weight: Optional[pd.Series] = None,
    beta_series: Optional[pd.Series] = None,
    beta_neutral: bool = False,
    capacity_report: Optional[Dict[str, Any]] = None,
    n_trials: int = 1,
) -> None:
    """
    Generate tear-sheet: summary.json, REPORT.md, PNG charts, tearsheet.html.

    For portfolio/universe backtests, pass weights (date x symbol), turnover_series,
    and prices_wide to add exposures, turnover plot, per-symbol contribution, holdings CSV.

    Args:
        result: BacktestResult from Backtester.run or run_from_signals
        prices: Price series (for alignment)
        signals: Position/signal series
        output_dir: Directory for output files
        rolling_window: Window for rolling Sharpe (default: ~63 bars or 20% of data)
        annualization: Annualization factor (252 for daily)
        config: Optional run config for summary.json
        weights: Optional weights DataFrame (date x symbol) for portfolio reporting
        turnover_series: Optional turnover series (use when weights provided)
        prices_wide: Optional prices DataFrame (date x symbol) for contribution
        ic_summary: Optional dict horizon -> {mean_ic, std_ic, ir, t_stat, n} for REPORT
        ic_preview: Optional dict horizon -> last 10 IC values for REPORT
        portfolio_beta_before: Portfolio beta before hedge (beta-neutral)
        portfolio_beta_after: Portfolio beta after hedge (beta-neutral)
        hedge_weight: Hedge weight (e.g. SPY) over time (beta-neutral)
        beta_series: Rolling portfolio beta vs market (beta_p(t))
        beta_neutral: Whether beta-neutral mode was enabled
    """
    out = Path(output_dir)
    _ensure_dir(out)

    cum = result.cumulative_returns.reindex(prices.index).ffill().bfill().fillna(1.0)
    returns = result.returns.reindex(prices.index).fillna(0)
    if turnover_series is not None:
        turnover = turnover_series.reindex(prices.index).fillna(0)
    else:
        turnover = compute_turnover(signals).reindex(prices.index).fillna(0)

    rw = rolling_window or min(63, max(20, len(returns) // 5))

    # 1) equity_curve.png
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum.index, cum.values, color="steelblue", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative Return")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=100)
    plt.close()

    # 2) drawdown.png
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="coral", alpha=0.7)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "drawdown.png", dpi=100)
    plt.close()

    # 3) rolling_sharpe.png
    if len(returns) >= rw:
        rolling_sharpe = (
            returns.rolling(rw).mean()
            / returns.rolling(rw).std().replace(0, np.nan)
            * np.sqrt(annualization)
        )
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="green", alpha=0.8)
        ax.set_title(f"Rolling Sharpe ({rw} bars)")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rolling_sharpe.png", dpi=100)
        plt.close()

    # 4) returns_hist.png
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(returns.dropna(), bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("Returns Distribution")
    ax.set_xlabel("Return")
    fig.tight_layout()
    fig.savefig(out / "returns_hist.png", dpi=100)
    plt.close()

    # 5) positions.png
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(signals.index, signals.values, 0, color="steelblue", alpha=0.5)
    ax.set_title("Position (Exposure)")
    ax.set_ylabel("Signal")
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "positions.png", dpi=100)
    plt.close()

    # 6) turnover.png
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(turnover.index, turnover.values, width=1, color="gray", alpha=0.6)
    ax.set_title("Turnover")
    ax.set_ylabel("|ΔPosition|")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "turnover.png", dpi=100)
    plt.close()

    # 6b) portfolio_beta.png (when beta-neutral)
    if portfolio_beta_before is not None and portfolio_beta_after is not None:
        common = prices.index.intersection(portfolio_beta_before.index).intersection(portfolio_beta_after.index)
        b_bef = portfolio_beta_before.reindex(common).ffill().fillna(0)
        b_aft = portfolio_beta_after.reindex(common).ffill().fillna(0)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(b_bef.index, b_bef.values, label="Beta (before hedge)", color="coral", alpha=0.8)
        ax.plot(b_aft.index, b_aft.values, label="Beta (after hedge)", color="steelblue", alpha=0.8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Portfolio Beta")
        ax.set_ylabel("Beta")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "portfolio_beta.png", dpi=100)
        plt.close()
    if hedge_weight is not None:
        hw = hedge_weight.reindex(prices.index).ffill().fillna(0)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(hw.index, hw.values, color="green", alpha=0.8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Hedge Weight (Market)")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "hedge_weight.png", dpi=100)
        plt.close()

    # Rolling portfolio beta vs market
    if beta_series is not None and len(beta_series.dropna()) > 0:
        bs = beta_series.reindex(prices.index).ffill().fillna(0)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(bs.index, bs.values, color="steelblue", alpha=0.8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Rolling Portfolio Beta vs Market")
        ax.set_ylabel("Beta")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rolling_beta.png", dpi=100)
        plt.close()

    # Portfolio/universe extras when weights provided
    exposures_stats: Optional[Dict[str, Any]] = None
    contribution_top: Optional[pd.Series] = None
    contribution_bottom: Optional[pd.Series] = None
    if weights is not None and not weights.empty:
        # Exposures: long count, short count, gross, net
        long_count = (weights > 0).sum(axis=1)
        short_count = (weights < 0).sum(axis=1)
        gross = weights.abs().sum(axis=1)
        net = weights.sum(axis=1)
        exposures_stats = {
            "long_count_mean": float(long_count.mean()),
            "short_count_mean": float(short_count.mean()),
            "gross_mean": f"{float(gross.mean()):.2%}",
            "net_mean": f"{float(net.mean()):.2%}",
        }
        # holdings_by_date.csv
        weights.reindex(prices.index).ffill().fillna(0).to_csv(out / "holdings_by_date.csv")
        # Per-symbol contribution (when prices_wide provided)
        if prices_wide is not None and not prices_wide.empty:
            ret_wide = prices_wide.pct_change().reindex(weights.index).fillna(0)
            common = weights.columns.intersection(ret_wide.columns)
            w = weights.reindex(columns=common).fillna(0)
            r = ret_wide.reindex(columns=common).fillna(0)
            contrib = (w * r).sum(axis=0)
            contrib = contrib.sort_values(ascending=False)
            contribution_top = contrib.head(10)
            contribution_bottom = contrib.tail(10)

    metrics = _summary_metrics(result, annualization)
    summary = {k: v for k, v in metrics.items() if not k.startswith("_")}
    if config:
        summary["config"] = config

    # Significance testing (Lo 2002 SE + bootstrap CI + DSR)
    sig_report: Optional[Dict[str, Any]] = None
    try:
        from src.research.significance import significance_report as _sig_report
        sig_report = _sig_report(
            returns,
            strategy_name=config.get("strategy", "strategy") if config else "strategy",
            n_trials=n_trials,
            annualization=int(annualization),
            n_bootstrap=500,
        )
        (out / "significance.json").write_text(
            json.dumps(sig_report, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        pass  # significance module optional — don't break tearsheet

    # 7) summary.json
    summary_json = {
        "sharpe": result.sharpe_ratio,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "n_trades": result.n_trades,
        "win_rate": result.win_rate,
        "cagr": metrics.get("_cagr"),
        "vol": metrics.get("_vol"),
        "config": config or {},
    }
    (out / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    # 8) REPORT.md
    pngs = ["equity_curve", "drawdown", "rolling_sharpe", "returns_hist", "positions", "turnover"]
    if portfolio_beta_before is not None:
        pngs.extend(["portfolio_beta", "hedge_weight"])
    if beta_series is not None and len(beta_series.dropna()) > 0:
        pngs.append("rolling_beta")
    report_lines = [
        "# Backtest Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in summary.items():
        if k != "config" and isinstance(v, (str, int, float)):
            report_lines.append(f"| {k} | {v} |")

    cfg = config or {}
    weights_for_attribution = combo_weights or cfg.get("combo_weights")
    if weights_for_attribution or factor_attribution:
        report_lines.extend(["", "## Combo Attribution", ""])
        if weights_for_attribution:
            report_lines.extend([
                "### Factor Weights",
                "",
                "| Factor | Weight |",
                "|--------|--------|",
            ])
            for name, w in weights_for_attribution.items():
                report_lines.append(f"| {name} | {w:.4f} |")
            report_lines.append("")
        if factor_attribution:
            report_lines.extend([
                "### Exposures & Attribution",
                "",
                "| Factor | mean_exposure | std_exposure | corr_with_returns |",
                "|--------|---------------|--------------|-------------------|",
            ])
            for name, att in factor_attribution.items():
                me = att.get("mean_exposure", 0)
                se = att.get("std_exposure", 0)
                corr = att.get("corr_with_returns")
                corr_str = f"{corr:.4f}" if isinstance(corr, (int, float)) and corr == corr else "—"
                report_lines.append(f"| {name} | {me:.4f} | {se:.4f} | {corr_str} |")
            report_lines.extend([
                "",
                "- [factor_exposures.csv](factor_exposures.csv) — portfolio exposure to each factor by date",
                "",
            ])

    if portfolio_beta_before is not None and portfolio_beta_after is not None:
        b_bef = portfolio_beta_before.dropna()
        b_aft = portfolio_beta_after.dropna()
        beta_lines = [
            "",
            "## Beta-Neutral",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| mean(|beta|) before hedge | {float(b_bef.abs().mean()):.2f} |",
            f"| mean(|beta|) after hedge | {float(b_aft.abs().mean()):.2f} |",
            f"| Beta (before hedge, mean) | {float(b_bef.mean()):.2f} |",
            f"| Beta (after hedge, mean) | {float(b_aft.mean()):.2f} |",
            f"| |Beta| reduction | {float(b_bef.abs().mean() - b_aft.abs().mean()):.2f} |",
        ]
        if hedge_weight is not None:
            hw = hedge_weight.dropna()
            beta_lines.append(f"| Hedge weight (mean) | {float(hw.mean()):.2%} |")
        beta_lines.append("")
        report_lines.extend(beta_lines)

    if beta_series is not None and len(beta_series.dropna()) > 0:
        bs = beta_series.dropna()
        bs_std = float(bs.std()) if len(bs) > 1 else 0.0
        beta_exp_lines = [
            "",
            "## Market Beta Exposure",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| mean beta | {float(bs.mean()):.2f} |",
            f"| std beta | {bs_std:.2f} |",
            f"| max |beta| | {float(bs.abs().max()):.2f} |",
            "",
            "- [beta_series.csv](beta_series.csv) — rolling beta by date",
            "",
        ]
        report_lines.extend(beta_exp_lines)

    if capacity_report is not None:
        ib = capacity_report.get("impact_bps", {})
        cap_lines = [
            "",
            "## Liquidity & Capacity",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| avg ADV (median symbol) | ${_median_adv(capacity_report):,.0f} |",
            f"| median impact bps | {ib.get('median', 0):.1f} |",
            f"| estimated capacity (10 bps) | ${capacity_report.get('capacity_notional_at_target_bps', 0):,.0f} |",
            "",
            "- [capacity_report.json](capacity_report.json) — per-symbol ADV, impact distribution",
            "",
        ]
        report_lines.extend(cap_lines)

    if exposures_stats:
        report_lines.extend([
            "",
            "## Exposures (portfolio/universe)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Long count (mean) | {exposures_stats['long_count_mean']:.1f} |",
            f"| Short count (mean) | {exposures_stats['short_count_mean']:.1f} |",
            f"| Gross (mean) | {exposures_stats['gross_mean']} |",
            f"| Net (mean) | {exposures_stats['net_mean']} |",
            "",
        ])

    if contribution_top is not None and contribution_bottom is not None:
        report_lines.extend([
            "## Per-Symbol Contribution (top / bottom)",
            "",
            "| Symbol | Contribution |",
            "|--------|--------------|",
        ])
        for sym, val in contribution_top.items():
            report_lines.append(f"| {sym} | {val:.4%} |")
        report_lines.append("| ... | ... |")
        for sym, val in contribution_bottom.items():
            report_lines.append(f"| {sym} | {val:.4%} |")
        report_lines.append("")

    if weights is not None:
        report_lines.extend([
            "## Holdings",
            "",
            "- [holdings_by_date.csv](holdings_by_date.csv) — weights by date (audit)",
            "",
        ])

    if ic_summary and ic_preview:
        report_lines.extend([
            "## Factor Research (IC/IR)",
            "",
            "| Horizon | mean_ic | std_ic | IR | t_stat | n |",
            "|---------|---------|--------|-----|--------|---|",
        ])
        for h, s in sorted(ic_summary.items(), key=lambda x: int(x[0])):
            report_lines.append(
                f"| {h} | {s.get('mean_ic', 0):.4f} | {s.get('std_ic', 0):.4f} | "
                f"{s.get('ir', 0):.2f} | {s.get('t_stat', 0):.2f} | {s.get('n', 0)} |"
            )
        report_lines.append("")
        for h, vals in sorted(ic_preview.items(), key=lambda x: int(x[0])):
            report_lines.append(f"**IC(h={h}) last 10:** " + ", ".join(f"{v:.4f}" for v in vals))
            report_lines.append("")

    # Significance section
    if sig_report is not None:
        sr = sig_report
        sig_lines = [
            "",
            "## Statistical Significance",
            "",
            "> Lo (2002) Sharpe SE · Block Bootstrap 95% CI"
            " · Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Observations | {sr.get('n_obs', '—')} |",
            f"| Sharpe (ann.) | {sr.get('sharpe', float('nan')):.4f} |",
            f"| Sharpe SE (Lo 2002) | {sr.get('sharpe_se', float('nan')):.4f} |",
            f"| t-statistic | {sr.get('t_stat', float('nan')):.3f} |",
            f"| p-value (SR=0) | {sr.get('p_value', float('nan')):.4f} |",
            f"| Significant (5%) | {'Yes ✓' if sr.get('significant_5pct') else 'No ✗'} |",
            f"| 95% CI (bootstrap) | [{sr.get('ci_95_lower', float('nan')):.2f},"
            f" {sr.get('ci_95_upper', float('nan')):.2f}] |",
            f"| Skewness | {sr.get('skewness', float('nan')):.3f} |",
            f"| Excess Kurtosis | {sr.get('excess_kurtosis', float('nan')):.3f} |",
        ]
        if "dsr" in sr:
            sig_lines += [
                f"| Deflated Sharpe Ratio | {sr.get('dsr', float('nan')):.4f} |",
                "| DSR significant (95%) | "
                f"{'Yes ✓' if sr.get('dsr_significant') else 'No ✗'} |",
                f"| Min required SR* | {sr.get('sr_star', float('nan')):.4f} |",
                f"| E[max SR | {sr.get('n_trials', 1)} trials]"
                f" | {sr.get('expected_max_sharpe', float('nan')):.4f} |",
            ]
        sig_lines += [
            "",
            "- [significance.json](significance.json) — full significance report",
            "",
        ]
        report_lines.extend(sig_lines)

    report_lines.extend(["", "## Charts", ""])
    for p in pngs:
        if (out / f"{p}.png").exists():
            report_lines.append(f"![{p}]({p}.png)")
            report_lines.append("")

    # Notes section
    notes = [
        "- **delay_bars** prevents lookahead: signals execute at bar t + delay.",
        "- **Costs** (fee/slip/spread) are applied per trade.",
        "- **decision_interval_bars** reduces churn by throttling position changes.",
    ]
    if cfg.get("interval") == "1m":
        notes.append(
            "- **Warning**: Naive intraday momentum at 1m is typically cost-sensitive; tune via `sweep_momentum`."
        )
    report_lines.extend(["", "## Notes", ""] + notes + [""])
    (out / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    # 8b) INDEX.md (run directory landing page)
    _generate_index_md(out, config or {})

    # 9) tearsheet.html (legacy)
    rows = "\n".join(
        f"    <tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in summary.items()
        if k != "config" and isinstance(v, (str, int, float))
    )
    img_refs = "\n".join(
        f'  <p><img src="{f.name}" alt="{f.stem}" width="800"/></p>'
        for f in sorted(out.glob("*.png"))
    )
    html = f"""<!DOCTYPE html>
<html>
<head><title>Backtest Tear-Sheet</title></head>
<body>
  <h1>Backtest Tear-Sheet</h1>
  <h2>Summary</h2>
  <table border="1" cellpadding="8">
{rows}
  </table>
  <h2>Charts</h2>
{img_refs}
</body>
</html>
"""
    (out / "tearsheet.html").write_text(html, encoding="utf-8")
