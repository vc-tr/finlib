"""Tests for research bundle: INDEX generation and smoke run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_bundle_index_generation(tmp_path: Path) -> None:
    """INDEX.md generation with mocked summary dicts produces correct structure."""
    import importlib.util
    import sys

    proj_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "make_research_bundle", proj_root / "scripts" / "make_research_bundle.py"
    )
    bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundle)

    run_dir = tmp_path / "bundle"
    run_dir.mkdir()
    (run_dir / "daily_demo").mkdir()
    (run_dir / "walkforward").mkdir()
    (run_dir / "intraday_demo").mkdir()
    (run_dir / "sweep").mkdir()

    configs = {
        "daily_demo": {"symbol": "SPY", "period": "2y", "interval": "1d"},
        "walkforward": {"folds": 4, "train_days": 90, "test_days": 30, "interval": "1d"},
        "intraday_demo": {"symbol": "SPY", "period": "7d", "interval": "1m"},
    }
    summaries = {
        "daily_demo": {"sharpe": 1.5, "total_return": 0.12},
        "walkforward": {"aggregated": {"mean_sharpe": 0.8, "agg_total_return": 0.05, "n_folds": 4}},
        "intraday_demo": {"sharpe": -0.5, "total_return": -0.02},
    }

    bundle._write_index(run_dir, configs, summaries)

    index_path = run_dir / "INDEX.md"
    assert index_path.exists()
    content = index_path.read_text()
    assert "Research Bundle" in content
    assert "daily_demo" in content
    assert "walkforward" in content
    assert "intraday_demo" in content
    assert "Sharpe=1.5" in content or "1.5" in content
    assert "daily_demo/REPORT.md" in content
    assert "walkforward/WALKFORWARD_REPORT.md" in content


def test_bundle_smoke_synthetic(tmp_path: Path, monkeypatch) -> None:
    """Bundle runs to completion with mocked fetcher (synthetic data, no network)."""
    import importlib.util
    import sys

    # Synthetic OHLCV
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    prices = pd.Series(100 + np.arange(300) * 0.1, index=idx)
    df = pd.DataFrame({
        "open": prices - 0.5,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": 1000000,
    })

    def fake_fetch(*args, **kwargs):
        return df.copy()

    monkeypatch.setattr(
        "src.pipeline.data_fetcher_yahoo.YahooDataFetcher.fetch_ohlcv",
        fake_fetch,
    )
    monkeypatch.setattr(
        "src.pipeline.pipeline.reindex_and_backfill",
        lambda df_in, **kw: df_in,
    )

    run_dir = tmp_path / "bundle_out"
    sys.argv = [
        "make_research_bundle",
        "--symbol", "SPY",
        "--output-dir", str(run_dir),
        "--no-lock",
        "--quick",
    ]

    proj_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "make_research_bundle", proj_root / "scripts" / "make_research_bundle.py"
    )
    bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundle)
    bundle.main()

    assert (run_dir / "INDEX.md").exists()
    assert (run_dir / "daily_demo" / "summary.json").exists()
    assert (run_dir / "walkforward" / "walkforward_summary.json").exists()
    assert (run_dir / "intraday_demo" / "summary.json").exists()
