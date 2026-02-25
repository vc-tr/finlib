#!/usr/bin/env python3
"""
Monitor last N daily runs: summary + alerts.

Scans output/runs for *_daily_* directories and produces:
- output/monitor/summary.csv
- output/monitor/monitor_report.md

Alerts: missing files, turnover > threshold, beta abs > threshold, cost spike.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _scan_daily_runs(runs_dir: Path, n: int) -> list[Path]:
    """Return last N daily run directories, newest first."""
    if not runs_dir.exists():
        return []
    dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "_daily_" in d.name]
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dirs[:n]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_metrics(run_dir: Path) -> dict:
    """Extract metrics from a daily run directory."""
    summary = _load_json(run_dir / "summary.json")
    risk = _load_json(run_dir / "risk_checks.json")
    orders_path = run_dir / "orders_to_place.csv"
    has_orders = orders_path.exists() and orders_path.stat().st_size > 50

    asof = run_dir.name.split("_daily_")[0][:8] if "_daily_" in run_dir.name else ""
    turnover = 0.0
    n_orders = 0
    expected_costs = 0.0
    beta = 0.0

    if summary:
        asof = summary.get("asof", asof)
        turnover = summary.get("turnover", 0.0)
        n_orders = summary.get("n_orders", 0)
        expected_costs = summary.get("expected_costs", 0.0)
        beta = summary.get("portfolio_beta", 0.0)
    elif risk:
        beta = risk.get("portfolio_beta", 0.0)

    if orders_path.exists() and n_orders == 0:
        try:
            import pandas as pd
            df = pd.read_csv(orders_path)
            n_orders = len(df) if not df.empty and len(df.columns) > 1 else 0
        except Exception:
            pass

    return {
        "run_dir": str(run_dir.name),
        "path": str(run_dir),
        "asof": asof,
        "n_orders": n_orders,
        "turnover": turnover,
        "expected_costs": expected_costs,
        "beta": beta,
        "has_risk_checks": (run_dir / "risk_checks.json").exists(),
        "has_orders": has_orders,
        "has_report": (run_dir / "daily_report.md").exists(),
        "has_summary": (run_dir / "summary.json").exists(),
    }


def _run_monitor(
    runs_dir: Path,
    output_dir: Path,
    n: int,
    turnover_threshold: float,
    beta_threshold: float,
    cost_spike_factor: float,
) -> dict:
    """Scan runs, produce summary and report."""
    run_dirs = _scan_daily_runs(runs_dir, n)
    rows = []
    for d in run_dirs:
        m = _run_metrics(d)
        rows.append(m)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        summary_df = __import__("pandas").DataFrame(columns=["run_dir", "asof", "n_orders", "turnover", "expected_costs", "beta"])
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "monitor_report.md").write_text("# Monitor Report\n\nNo daily runs found.\n", encoding="utf-8")
        return {"n_runs": 0, "alerts": []}

    import pandas as pd
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    alerts = []
    for r in rows:
        run_name = r["run_dir"]
        if not r["has_risk_checks"]:
            alerts.append({"run": run_name, "type": "missing_file", "detail": "risk_checks.json"})
        if not r["has_orders"] and r["n_orders"] == 0:
            pass  # No orders is ok
        if not r["has_report"]:
            alerts.append({"run": run_name, "type": "missing_file", "detail": "daily_report.md"})
        if r["turnover"] > turnover_threshold:
            alerts.append({"run": run_name, "type": "turnover", "detail": f"turnover {r['turnover']:.2%} > {turnover_threshold:.2%}"})
        if abs(r["beta"]) > beta_threshold:
            alerts.append({"run": run_name, "type": "beta", "detail": f"|beta| {abs(r['beta']):.2f} > {beta_threshold}"})

    # Cost spike: compare to median of previous runs
    costs = [r["expected_costs"] for r in rows if r["expected_costs"] > 0]
    if len(costs) >= 2 and cost_spike_factor > 0:
        import numpy as np
        median_cost = float(np.median(costs[1:]))  # Exclude latest
        if median_cost > 0 and costs[0] > median_cost * cost_spike_factor:
            alerts.append({"run": rows[0]["run_dir"], "type": "cost_spike", "detail": f"costs ${costs[0]:,.0f} vs median ${median_cost:,.0f}"})

    report_lines = [
        "# Monitor Report",
        "",
        f"**Runs scanned:** {len(rows)}",
        "",
        "## Summary",
        "",
        "| Run | As-of | Orders | Turnover | Costs | Beta |",
        "|-----|-------|--------|----------|-------|------|",
    ]
    for r in rows:
        report_lines.append(f"| {r['run_dir']} | {r['asof']} | {r['n_orders']} | {r['turnover']:.2%} | ${r['expected_costs']:,.0f} | {r['beta']:.2f} |")
    report_lines.extend(["", "## Alerts", ""])
    if alerts:
        for a in alerts:
            report_lines.append(f"- **{a['type']}** ({a['run']}): {a['detail']}")
    else:
        report_lines.append("No alerts.")
    report_lines.append("")

    (output_dir / "monitor_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {"n_runs": len(rows), "alerts": alerts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor daily runs")
    parser.add_argument("--runs-dir", default=None, help="Default: output/runs")
    parser.add_argument("--output-dir", default=None, help="Default: output/monitor")
    parser.add_argument("-n", type=int, default=10, help="Last N runs to scan")
    parser.add_argument("--turnover-threshold", type=float, default=0.5, help="Alert if turnover > this")
    parser.add_argument("--beta-threshold", type=float, default=0.5, help="Alert if |beta| > this")
    parser.add_argument("--cost-spike-factor", type=float, default=2.0, help="Alert if costs > median * this")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    runs_dir = Path(args.runs_dir) if args.runs_dir else root / "output" / "runs"
    output_dir = Path(args.output_dir) if args.output_dir else root / "output" / "monitor"

    result = _run_monitor(
        runs_dir,
        output_dir,
        n=args.n,
        turnover_threshold=args.turnover_threshold,
        beta_threshold=args.beta_threshold,
        cost_spike_factor=args.cost_spike_factor,
    )

    print(f"Scanned {result['n_runs']} runs. Alerts: {len(result['alerts'])}")
    for a in result["alerts"]:
        print(f"  - {a['type']}: {a['detail']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
