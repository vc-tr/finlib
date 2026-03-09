"""
Monitor last N runs: summary + alerts.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_TYPE_PATTERNS = ("_daily_", "_replay_", "_factors_")
REQUIRED_FILES = {
    "daily": ["orders_to_place.csv", "daily_report.md", "risk_checks.json"],
    "replay": ["orders.csv", "blotter.csv", "equity_curve.csv", "replay_report.md"],
    "factors": ["REPORT.md", "summary.json"],
}


def _run_type(folder_name: str) -> str:
    """Detect run type from folder name."""
    for pattern, rtype in [
        ("_daily_", "daily"),
        ("_replay_", "replay"),
        ("_factors_", "factors"),
    ]:
        if pattern in folder_name:
            return rtype
    return "other"


def _parse_timestamp_prefix(folder_name: str) -> float | None:
    """Parse YYYYMMDD_HHMMSS from folder name prefix. Returns Unix timestamp or None."""
    m = re.match(r"^(\d{8})_(\d{6})_", folder_name)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        return dt.timestamp()
    except ValueError:
        return None


def _scan_runs(
    runs_dir: Path,
    n: int,
    only_type: str = "all",
    since_hours: float | None = None,
    since_days: float | None = None,
) -> list[Path]:
    """Return last N run directories matching filters, newest first."""
    if not runs_dir.exists():
        return []
    now = time.time()
    if since_hours is not None:
        cutoff = now - since_hours * 3600
    elif since_days is not None:
        cutoff = now - since_days * 86400
    else:
        cutoff = None
    dirs = []
    for d in runs_dir.iterdir():
        if not d.is_dir() or not any(p in d.name for p in RUN_TYPE_PATTERNS):
            continue
        if only_type != "all" and _run_type(d.name) != only_type:
            continue
        ts = _parse_timestamp_prefix(d.name)
        if ts is None:
            continue
        if cutoff is not None and ts < cutoff:
            continue
        dirs.append(d)
    dirs.sort(key=lambda x: _parse_timestamp_prefix(x.name) or 0, reverse=True)
    return dirs[:n]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_metrics(run_dir: Path) -> dict[str, Any]:
    """Extract metrics from a run directory."""
    import pandas as pd

    summary = _load_json(run_dir / "summary.json")
    risk = _load_json(run_dir / "risk_checks.json")
    orders_path = run_dir / "orders_to_place.csv"
    has_orders = orders_path.exists() and orders_path.stat().st_size > 50

    run_meta = _load_json(run_dir / "run_meta.json")
    rtype = run_meta.get("run_type") if run_meta else None
    if not rtype:
        rtype = _run_type(run_dir.name)
    asof = ""
    for p in RUN_TYPE_PATTERNS:
        if p in run_dir.name:
            asof = run_dir.name.split(p)[0][:8]
            break
    turnover = 0.0
    n_orders = 0
    expected_costs = 0.0
    beta = 0.0
    state_bootstrap = False

    if summary:
        asof = summary.get("asof", asof)
        turnover = summary.get("turnover", 0.0)
        n_orders = summary.get("n_orders", 0)
        expected_costs = summary.get("expected_costs", 0.0)
        beta = summary.get("portfolio_beta", 0.0)
    elif risk:
        beta = risk.get("portfolio_beta", 0.0)

    if risk:
        state_bootstrap = bool(risk.get("state_bootstrap", False))

    if orders_path.exists() and n_orders == 0:
        try:
            df = pd.read_csv(orders_path)
            n_orders = len(df) if not df.empty and len(df.columns) > 1 else 0
        except Exception:
            pass

    required = REQUIRED_FILES.get(rtype, [])
    has_required = all((run_dir / f).exists() for f in required)

    return {
        "run_dir": str(run_dir.name),
        "path": str(run_dir),
        "run_type": rtype,
        "asof": asof,
        "n_orders": n_orders,
        "turnover": turnover,
        "expected_costs": expected_costs,
        "beta": beta,
        "state_bootstrap": state_bootstrap,
        "has_risk_checks": (run_dir / "risk_checks.json").exists(),
        "has_orders": has_orders,
        "has_report": (run_dir / "daily_report.md").exists(),
        "has_summary": (run_dir / "summary.json").exists(),
        "has_required_files": has_required,
        "has_error_txt": (run_dir / "ERROR.txt").exists(),
        "error_summary": (run_dir / "ERROR.txt").read_text(encoding="utf-8").strip()[:200]
        if (run_dir / "ERROR.txt").exists()
        else None,
    }


def run_monitor(
    runs_dir: Path,
    output_dir: Path,
    n: int = 10,
    turnover_threshold: float = 0.5,
    beta_threshold: float = 0.5,
    cost_spike_factor: float = 2.0,
    only_type: str = "all",
    since_hours: float | None = None,
    since_days: float | None = None,
    turnover_applies_to: str = "daily",
    ignore_initial_deploy: bool = True,
) -> dict[str, Any]:
    """Scan runs, produce summary and report. Returns {n_runs, alerts}."""
    import numpy as np
    import pandas as pd

    run_dirs = _scan_runs(
        runs_dir, n, only_type=only_type, since_hours=since_hours, since_days=since_days
    )
    rows = [_run_metrics(d) for d in run_dirs]

    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        summary_df = pd.DataFrame(columns=["run_dir", "asof", "n_orders", "turnover", "expected_costs", "beta"])
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "monitor_report.md").write_text("# Monitor Report\n\nNo runs found.\n", encoding="utf-8")
        return {"n_runs": 0, "alerts": []}

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    alerts = []
    for r in rows:
        run_name = r["run_dir"]
        run_path = Path(r["path"])

        if r["has_error_txt"]:
            alerts.append({"run": run_name, "type": "failed_run", "detail": r["error_summary"] or "ERROR.txt present"})
            continue

        if not r["has_required_files"]:
            required = REQUIRED_FILES.get(r["run_type"], [])
            missing = [f for f in required if not (run_path / f).exists()]
            for f in missing:
                alerts.append({"run": run_name, "type": "missing_file", "detail": f})

        if turnover_applies_to == "all" or r["run_type"] in turnover_applies_to:
            if not (ignore_initial_deploy and r.get("state_bootstrap")) and r["turnover"] > turnover_threshold:
                alerts.append({"run": run_name, "type": "turnover", "detail": f"turnover {r['turnover']:.2%} > {turnover_threshold:.2%}"})  # noqa: E501

        if abs(r["beta"]) > beta_threshold:
            alerts.append({"run": run_name, "type": "beta", "detail": f"|beta| {abs(r['beta']):.2f} > {beta_threshold}"})  # noqa: E501

    costs = [r["expected_costs"] for r in rows if r["expected_costs"] > 0]
    if len(costs) >= 2 and cost_spike_factor > 0:
        median_cost = float(np.median(costs[1:]))
        if median_cost > 0 and costs[0] > median_cost * cost_spike_factor:
            alerts.append({"run": rows[0]["run_dir"], "type": "cost_spike", "detail": f"costs ${costs[0]:,.0f} vs median ${median_cost:,.0f}"})  # noqa: E501

    report_lines = [
        "# Monitor Report",
        "",
        f"**Runs scanned:** {len(rows)}",
        "",
        "## Summary",
        "",
        "| Run | Type | As-of | Orders | Turnover | Costs | Beta |",
        "|-----|------|-------|--------|----------|-------|------|",
    ]
    for r in rows:
        report_lines.append(f"| {r['run_dir']} | {r['run_type']} | {r['asof']} | {r['n_orders']} | {r['turnover']:.2%} | ${r['expected_costs']:,.0f} | {r['beta']:.2f} |")  # noqa: E501
    report_lines.extend(["", "## Alerts", ""])
    if alerts:
        for a in alerts:
            report_lines.append(f"- **{a['type']}** ({a['run']}): {a['detail']}")
    else:
        report_lines.append("No alerts.")
    report_lines.append("")

    (output_dir / "monitor_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {"n_runs": len(rows), "alerts": alerts}
