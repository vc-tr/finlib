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
) -> None:
    """
    Generate tear-sheet: summary.json, REPORT.md, PNG charts, tearsheet.html.

    Args:
        result: BacktestResult from Backtester.run or run_from_signals
        prices: Price series (for alignment)
        signals: Position/signal series
        output_dir: Directory for output files
        rolling_window: Window for rolling Sharpe (default: ~63 bars or 20% of data)
        annualization: Annualization factor (252 for daily)
        config: Optional run config for summary.json
    """
    out = Path(output_dir)
    _ensure_dir(out)

    cum = result.cumulative_returns.reindex(prices.index).ffill().bfill().fillna(1.0)
    returns = result.returns.reindex(prices.index).fillna(0)
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

    metrics = _summary_metrics(result, annualization)
    summary = {k: v for k, v in metrics.items() if not k.startswith("_")}
    if config:
        summary["config"] = config

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
    report_lines.extend(["", "## Charts", ""])
    for p in pngs:
        if (out / f"{p}.png").exists():
            report_lines.append(f"![{p}]({p}.png)")
            report_lines.append("")
    (out / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

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
