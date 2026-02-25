"""Tests for monitor_runs (no network)."""

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.monitor_runs import _run_monitor, _run_metrics, _run_type, _scan_runs


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


def test_run_type_detection() -> None:
    """Run type is detected from folder name."""
    assert _run_type("20240601_daily_momentum_M") == "daily"
    assert _run_type("20240601_replay_SPY_1m") == "replay"
    assert _run_type("20240601_factors_combo_M") == "factors"
    assert _run_type("20240601_sweep_SPY") == "other"


def test_missing_file_alerts_only_for_daily_runs(tmp_path: Path) -> None:
    """missing_file alerts only apply to daily runs, not replay or factors."""
    # Daily run with missing artifacts -> should get missing_file alerts
    daily_run = tmp_path / "20240601_daily_momentum_M"
    daily_run.mkdir()
    (daily_run / "summary.json").write_text(json.dumps({"asof": "2024-06-01", "turnover": 0.1, "portfolio_beta": 0.0}), encoding="utf-8")
    # No risk_checks.json, daily_report.md, orders_to_place.csv

    # Replay run with missing artifacts -> should NOT get missing_file alerts
    replay_run = tmp_path / "20240601_replay_SPY_1m"
    replay_run.mkdir()
    (replay_run / "summary.json").write_text(json.dumps({"asof": "2024-06-01"}), encoding="utf-8")
    # No risk_checks, daily_report, orders_to_place

    # Factors run with missing artifacts -> should NOT get missing_file alerts
    factors_run = tmp_path / "20240601_factors_combo_M"
    factors_run.mkdir()
    (factors_run / "summary.json").write_text(json.dumps({"asof": "2024-06-01"}), encoding="utf-8")
    # No risk_checks, daily_report, orders_to_place

    out = tmp_path / "monitor"
    result = _run_monitor(tmp_path, out, n=10, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0)

    missing_alerts = [a for a in result["alerts"] if a["type"] == "missing_file"]
    missing_runs = {a["run"] for a in missing_alerts}

    assert "20240601_daily_momentum_M" in missing_runs
    assert "20240601_replay_SPY_1m" not in missing_runs
    assert "20240601_factors_combo_M" not in missing_runs
    assert len(missing_alerts) >= 3  # daily run missing risk_checks, daily_report, orders_to_place
