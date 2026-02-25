"""Tests for monitor_runs (no network)."""

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.monitor_runs import _run_monitor, _scan_daily_runs, _run_metrics


def test_monitor_empty_runs_dir(tmp_path: Path) -> None:
    """Monitor with no runs produces empty summary."""
    out = tmp_path / "monitor"
    result = _run_monitor(tmp_path, out, n=5, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0)
    assert result["n_runs"] == 0
    assert (out / "summary.csv").exists()
    assert (out / "monitor_report.md").exists()


def test_monitor_with_mock_runs(tmp_path: Path) -> None:
    """Monitor scans daily run dirs and produces summary."""
    run1 = tmp_path / "20240601_daily_momentum_M"
    run1.mkdir()
    (run1 / "risk_checks.json").write_text(json.dumps({"portfolio_beta": 0.2}), encoding="utf-8")
    (run1 / "summary.json").write_text(
        json.dumps({"asof": "2024-06-01", "n_orders": 5, "turnover": 0.3, "expected_costs": 100.0, "portfolio_beta": 0.2}),
        encoding="utf-8",
    )
    (run1 / "orders_to_place.csv").write_text("symbol,side,quantity\nSPY,buy,10\n", encoding="utf-8")
    (run1 / "daily_report.md").write_text("# Report", encoding="utf-8")

    out = tmp_path / "monitor"
    result = _run_monitor(tmp_path, out, n=5, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0)
    assert result["n_runs"] == 1
    assert (out / "summary.csv").exists()
    content = (out / "summary.csv").read_text()
    assert "20240601_daily_momentum_M" in content


def test_monitor_alerts_on_turnover(tmp_path: Path) -> None:
    """Monitor alerts when turnover exceeds threshold."""
    run1 = tmp_path / "20240601_daily_momentum_M"
    run1.mkdir()
    (run1 / "summary.json").write_text(
        json.dumps({"asof": "2024-06-01", "n_orders": 20, "turnover": 0.8, "expected_costs": 200.0, "portfolio_beta": 0.1}),
        encoding="utf-8",
    )
    (run1 / "risk_checks.json").write_text("{}", encoding="utf-8")

    out = tmp_path / "monitor"
    result = _run_monitor(tmp_path, out, n=5, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0)
    turnover_alerts = [a for a in result["alerts"] if a["type"] == "turnover"]
    assert len(turnover_alerts) >= 1
