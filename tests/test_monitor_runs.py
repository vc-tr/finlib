"""Tests for monitor_runs (no network)."""

import json
from datetime import datetime
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ops import (
    REQUIRED_FILES,
    _parse_timestamp_prefix,
    _run_metrics,
    _run_type,
    _scan_runs,
    run_monitor as _run_monitor,
)


def _recent_ts_prefix() -> str:
    """Timestamp prefix for runs within last 6 hours."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def test_parse_timestamp_prefix() -> None:
    """Parse YYYYMMDD_HHMMSS from folder name."""
    ts = _parse_timestamp_prefix("20260225_165510_daily_momentum_M")
    assert ts is not None
    assert datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S") == "20260225_165510"

    assert _parse_timestamp_prefix("20240601_daily_M") is None  # No HHMMSS
    assert _parse_timestamp_prefix("bad_folder") is None
    assert _parse_timestamp_prefix("20240231_120000_daily") is None  # Invalid date (Feb 31)


def test_since_hours_filters_correctly(tmp_path: Path) -> None:
    """Only include run dirs whose timestamp prefix is within since window."""
    from datetime import timedelta

    now = datetime.now()
    recent = (now - timedelta(hours=2)).strftime("%Y%m%d_%H%M%S")
    old = (now - timedelta(hours=10)).strftime("%Y%m%d_%H%M%S")

    r1 = tmp_path / f"{recent}_daily_momentum_M"
    r2 = tmp_path / f"{old}_daily_momentum_M"
    r1.mkdir()
    r2.mkdir()
    for r in [r1, r2]:
        (r / "summary.json").write_text("{}", encoding="utf-8")
        (r / "orders_to_place.csv").write_text("x", encoding="utf-8")

    # With since_hours=6, only recent should be included
    dirs = _scan_runs(tmp_path, n=10, since_hours=6)
    assert len(dirs) == 1
    assert recent in dirs[0].name

    # With since_hours=12, both included
    dirs = _scan_runs(tmp_path, n=10, since_hours=12)
    assert len(dirs) == 2

    # With --no-since (since_hours=None), both included
    dirs = _scan_runs(tmp_path, n=10, since_hours=None, since_days=None)
    assert len(dirs) == 2


def test_skip_dirs_with_unparseable_timestamp(tmp_path: Path) -> None:
    """Dirs without valid YYYYMMDD_HHMMSS prefix are skipped."""
    ts = _recent_ts_prefix()
    (tmp_path / f"{ts}_daily_momentum_M").mkdir()
    (tmp_path / "20240601_daily_M").mkdir()  # No HHMMSS - unparseable
    (tmp_path / "nobadge_replay_x").mkdir()  # No timestamp

    dirs = _scan_runs(tmp_path, n=10, since_hours=None, since_days=None)
    assert len(dirs) == 1
    assert ts in dirs[0].name


def test_required_files_are_type_specific(tmp_path: Path) -> None:
    """Required file checks are type-specific."""
    ts = _recent_ts_prefix()

    # Daily: needs orders_to_place.csv, daily_report.md, risk_checks.json
    daily = tmp_path / f"{ts}_daily_momentum_M"
    daily.mkdir()
    (daily / "orders_to_place.csv").write_text("x", encoding="utf-8")
    (daily / "daily_report.md").write_text("x", encoding="utf-8")
    (daily / "risk_checks.json").write_text("{}", encoding="utf-8")
    assert all((daily / f).exists() for f in REQUIRED_FILES["daily"])

    # Replay: needs orders.csv, blotter.csv, equity_curve.csv, replay_report.md
    replay = tmp_path / f"{ts}_replay_SPY_1m"
    replay.mkdir()
    for f in REQUIRED_FILES["replay"]:
        (replay / f).write_text("x", encoding="utf-8")

    # Factors: needs REPORT.md, summary.json
    factors = tmp_path / f"{ts}_factors_combo_M"
    factors.mkdir()
    (factors / "REPORT.md").write_text("x", encoding="utf-8")
    (factors / "summary.json").write_text("{}", encoding="utf-8")

    out = tmp_path / "monitor"
    result = _run_monitor(
        tmp_path, out, n=10, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0,
        since_hours=None, since_days=None,
    )
    missing = [a for a in result["alerts"] if a["type"] == "missing_file"]
    assert len(missing) == 0


def test_monitor_empty_runs_dir(tmp_path: Path) -> None:
    """Monitor with no runs produces empty summary."""
    out = tmp_path / "monitor"
    result = _run_monitor(tmp_path, out, n=5, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0)
    assert result["n_runs"] == 0
    assert (out / "summary.csv").exists()
    assert (out / "monitor_report.md").exists()


def test_monitor_with_mock_runs(tmp_path: Path) -> None:
    """Monitor scans daily run dirs and produces summary."""
    ts = _recent_ts_prefix()
    run1 = tmp_path / f"{ts}_daily_momentum_M"
    run1.mkdir()
    (run1 / "risk_checks.json").write_text(json.dumps({"portfolio_beta": 0.2}), encoding="utf-8")
    (run1 / "summary.json").write_text(
        json.dumps({"asof": "2024-06-01", "n_orders": 5, "turnover": 0.3, "expected_costs": 100.0, "portfolio_beta": 0.2}),
        encoding="utf-8",
    )
    (run1 / "orders_to_place.csv").write_text("symbol,side,quantity\nSPY,buy,10\n", encoding="utf-8")
    (run1 / "daily_report.md").write_text("# Report", encoding="utf-8")

    out = tmp_path / "monitor"
    result = _run_monitor(
        tmp_path, out, n=5, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0,
        since_hours=None, since_days=None,
    )
    assert result["n_runs"] == 1
    assert (out / "summary.csv").exists()
    content = (out / "summary.csv").read_text()
    assert f"{ts}_daily_momentum_M" in content


def test_monitor_alerts_on_turnover(tmp_path: Path) -> None:
    """Monitor alerts when turnover exceeds threshold (and not state_bootstrap)."""
    ts = _recent_ts_prefix()
    run1 = tmp_path / f"{ts}_daily_momentum_M"
    run1.mkdir()
    (run1 / "summary.json").write_text(
        json.dumps({"asof": "2024-06-01", "n_orders": 20, "turnover": 0.8, "expected_costs": 200.0, "portfolio_beta": 0.1}),
        encoding="utf-8",
    )
    (run1 / "risk_checks.json").write_text(json.dumps({"portfolio_beta": 0.1, "state_bootstrap": False}), encoding="utf-8")
    (run1 / "orders_to_place.csv").write_text("x", encoding="utf-8")
    (run1 / "daily_report.md").write_text("x", encoding="utf-8")

    out = tmp_path / "monitor"
    result = _run_monitor(
        tmp_path, out, n=5, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0,
        since_hours=None, since_days=None, ignore_initial_deploy=True,
    )
    turnover_alerts = [a for a in result["alerts"] if a["type"] == "turnover"]
    assert len(turnover_alerts) >= 1


def test_ignore_initial_deploy_skips_turnover_alert(tmp_path: Path) -> None:
    """When state_bootstrap is True, turnover alert is suppressed."""
    ts = _recent_ts_prefix()
    run1 = tmp_path / f"{ts}_daily_momentum_M"
    run1.mkdir()
    (run1 / "summary.json").write_text(
        json.dumps({"asof": "2024-06-01", "n_orders": 20, "turnover": 0.8, "expected_costs": 200.0, "portfolio_beta": 0.1}),
        encoding="utf-8",
    )
    (run1 / "risk_checks.json").write_text(json.dumps({"portfolio_beta": 0.1, "state_bootstrap": True}), encoding="utf-8")
    (run1 / "orders_to_place.csv").write_text("x", encoding="utf-8")
    (run1 / "daily_report.md").write_text("x", encoding="utf-8")

    out = tmp_path / "monitor"
    result = _run_monitor(
        tmp_path, out, n=5, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0,
        since_hours=None, since_days=None, ignore_initial_deploy=True,
    )
    turnover_alerts = [a for a in result["alerts"] if a["type"] == "turnover"]
    assert len(turnover_alerts) == 0


def test_error_txt_suppresses_missing_file_alerts(tmp_path: Path) -> None:
    """When ERROR.txt exists, report failed_run instead of missing_file."""
    ts = _recent_ts_prefix()
    run1 = tmp_path / f"{ts}_daily_momentum_M"
    run1.mkdir()
    (run1 / "ERROR.txt").write_text("No price data", encoding="utf-8")
    # No orders_to_place, daily_report, risk_checks

    out = tmp_path / "monitor"
    result = _run_monitor(
        tmp_path, out, n=5, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0,
        since_hours=None, since_days=None,
    )
    missing = [a for a in result["alerts"] if a["type"] == "missing_file"]
    failed = [a for a in result["alerts"] if a["type"] == "failed_run"]
    assert len(missing) == 0
    assert len(failed) == 1
    assert "No price data" in failed[0]["detail"]


def test_run_type_detection() -> None:
    """Run type is detected from folder name."""
    assert _run_type("20240601_120000_daily_momentum_M") == "daily"
    assert _run_type("20240601_120000_replay_SPY_1m") == "replay"
    assert _run_type("20240601_120000_factors_combo_M") == "factors"
    assert _run_type("20240601_120000_sweep_SPY") == "other"


def test_missing_file_alerts_type_specific(tmp_path: Path) -> None:
    """missing_file alerts are type-specific: daily gets daily required, replay gets replay required."""
    ts = _recent_ts_prefix()

    # Daily run with missing artifacts -> should get missing_file alerts
    daily_run = tmp_path / f"{ts}_daily_momentum_M"
    daily_run.mkdir()
    (daily_run / "summary.json").write_text(json.dumps({"asof": "2024-06-01", "turnover": 0.1, "portfolio_beta": 0.0}), encoding="utf-8")
    # Missing: risk_checks.json, daily_report.md, orders_to_place.csv

    # Replay run with all required files -> no missing_file
    replay_run = tmp_path / f"{ts}_replay_SPY_1m"
    replay_run.mkdir()
    for f in REQUIRED_FILES["replay"]:
        (replay_run / f).write_text("x", encoding="utf-8")

    # Factors run with all required files -> no missing_file
    factors_run = tmp_path / f"{ts}_factors_combo_M"
    factors_run.mkdir()
    (factors_run / "REPORT.md").write_text("x", encoding="utf-8")
    (factors_run / "summary.json").write_text(json.dumps({"asof": "2024-06-01"}), encoding="utf-8")

    out = tmp_path / "monitor"
    result = _run_monitor(
        tmp_path, out, n=10, turnover_threshold=0.5, beta_threshold=0.5, cost_spike_factor=2.0,
        since_hours=None, since_days=None,
    )

    missing_alerts = [a for a in result["alerts"] if a["type"] == "missing_file"]
    missing_runs = {a["run"] for a in missing_alerts}

    assert daily_run.name in missing_runs
    assert replay_run.name not in missing_runs
    assert factors_run.name not in missing_runs
    assert len(missing_alerts) >= 3  # daily run missing risk_checks, daily_report, orders_to_place
