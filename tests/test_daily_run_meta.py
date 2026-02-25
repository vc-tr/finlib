"""Tests for daily run metadata (run_meta.json)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.daily_run import _build_run_meta, _write_run_meta


REQUIRED_KEYS = [
    "run_type",
    "asof_requested",
    "asof_trading",
    "state_path",
    "state_loaded",
    "rebalance_day",
    "rebalance",
    "interval",
    "universe",
    "factor",
    "combo",
    "combo_method",
    "cost_model",
    "fill_mode",
    "orders_count",
    "turnover",
    "beta",
    "applied",
    "output_dir",
    "timestamp_utc",
]


def test_write_run_meta_creates_file(tmp_path: Path) -> None:
    """run_meta.json is written and contains required keys."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    meta = _build_run_meta(
        output_dir=output_dir,
        state_path=tmp_path / "state.json",
        state_loaded=True,
        asof_requested="2024-07-02",
        asof_trading="2024-06-28",
        rebalance="M",
        universe="liquid_etfs",
        factor="momentum_12_1",
        combo=None,
        combo_method=None,
        cost_model="fixed",
        apply=False,
        result={
            "n_orders": 5,
            "turnover": 0.15,
            "risk_checks": {"portfolio_beta": 0.2},
            "rebalance_day": True,
        },
    )

    path = _write_run_meta(output_dir, meta)
    assert path.exists()
    assert path.name == "run_meta.json"

    loaded = json.loads(path.read_text())
    for key in REQUIRED_KEYS:
        assert key in loaded, f"Missing key: {key}"


def test_run_meta_serializes_numpy_pandas(tmp_path: Path) -> None:
    """run_meta with numpy.bool_ and pandas.Timestamp serializes correctly."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    meta = _build_run_meta(
        output_dir=output_dir,
        state_path=tmp_path / "state.json",
        state_loaded=np.bool_(True),
        asof_requested=None,
        asof_trading="2024-06-28",
        rebalance="M",
        universe="liquid_etfs",
        factor="combo",
        combo="m1,r1",
        combo_method="equal",
        cost_model="liquidity",
        apply=np.bool_(False),
        result={
            "n_orders": 0,
            "turnover": np.float64(0.0),
            "risk_checks": {"portfolio_beta": np.float64(0.1)},
            "rebalance_day": np.bool_(False),
        },
    )
    # Ensure we have numpy/pandas types that need conversion
    meta["state_loaded"] = np.bool_(True)
    meta["rebalance_day"] = np.bool_(False)
    meta["beta"] = np.float64(0.1)

    path = _write_run_meta(output_dir, meta)
    loaded = json.loads(path.read_text())

    assert loaded["state_loaded"] is True
    assert loaded["rebalance_day"] is False
    assert loaded["beta"] == 0.1
    assert "run_type" in loaded
    assert loaded["run_type"] == "daily"


def test_run_meta_error_status(tmp_path: Path) -> None:
    """run_meta with status=error includes status and error fields."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    meta = _build_run_meta(
        output_dir=output_dir,
        state_path=tmp_path / "state.json",
        state_loaded=False,
        asof_requested="2024-07-02",
        asof_trading="2024-06-28",
        rebalance="W",
        universe="liquid_etfs",
        factor="momentum_12_1",
        combo=None,
        combo_method=None,
        cost_model="fixed",
        apply=False,
        status="error",
        error="No price data",
    )

    path = _write_run_meta(output_dir, meta)
    loaded = json.loads(path.read_text())

    assert loaded["status"] == "error"
    assert loaded["error"] == "No price data"
    assert loaded["rebalance_day"] is False
    assert loaded["orders_count"] == 0
    assert loaded["beta"] is None
