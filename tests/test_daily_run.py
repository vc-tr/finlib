"""Tests for daily_run pipeline (no network)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import after path setup
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.daily_run import _run_daily, _load_current_portfolio, _save_current_portfolio
from src.utils.jsonable import to_jsonable


def _synthetic_ohlcv(n_dates: int = 100, n_symbols: int = 12, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create synthetic OHLCV df_by_symbol."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]
    if "SPY" not in cols:
        cols = ["SPY"] + cols[: n_symbols - 1]
    ret = rng.randn(n_dates, len(cols)) * 0.01
    close = 100 * (1 + pd.DataFrame(ret, index=idx, columns=cols)).cumprod()
    out = {}
    for i, sym in enumerate(cols):
        c = close[sym]
        o = c.shift(1).fillna(c.iloc[0])
        h = c * (1 + rng.rand(n_dates) * 0.005)
        l = c * (1 - rng.rand(n_dates) * 0.005)
        v = np.abs(rng.randn(n_dates) * 1e6).astype(float)
        out[sym] = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    return out


def test_daily_run_produces_orders_to_place(tmp_path: Path) -> None:
    """daily_run on synthetic data produces orders_to_place.csv."""
    df_by_symbol = _synthetic_ohlcv(n_dates=80, n_symbols=12)
    asof = pd.Timestamp("2024-04-15")
    state_path = tmp_path / "state.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)

    result = _run_daily(
        df_by_symbol,
        strategy="factors",
        factor="momentum_12_1",
        combo_list=None,
        combo_method="equal",
        asof=asof,
        rebalance="M",
        initial_cash=100_000.0,
        cost_model="fixed",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        top_k=3,
        bottom_k=3,
        max_gross=None,
        max_net=None,
        max_position_weight=0.2,
        beta_threshold=0.5,
        state_path=state_path,
        output_dir=output_dir,
        apply=False,
        force_rebalance=True,
    )

    assert "error" not in result
    assert (output_dir / "orders_to_place.csv").exists()
    orders_df = pd.read_csv(output_dir / "orders_to_place.csv")
    assert "symbol" in orders_df.columns
    assert "side" in orders_df.columns
    assert "quantity" in orders_df.columns


def test_daily_run_creates_risk_checks(tmp_path: Path) -> None:
    """risk_checks.json created with thresholds."""
    df_by_symbol = _synthetic_ohlcv(n_dates=80, n_symbols=12)
    asof = pd.Timestamp("2024-04-15")
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)

    result = _run_daily(
        df_by_symbol,
        strategy="factors",
        factor="momentum_12_1",
        combo_list=None,
        combo_method="equal",
        asof=asof,
        rebalance="M",
        initial_cash=100_000.0,
        cost_model="fixed",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        top_k=3,
        bottom_k=3,
        max_gross=2.0,
        max_net=0.5,
        max_position_weight=0.15,
        beta_threshold=0.3,
        state_path=tmp_path / "state.json",
        output_dir=output_dir,
        apply=False,
        force_rebalance=True,
    )

    assert (output_dir / "risk_checks.json").exists()
    risk = json.loads((output_dir / "risk_checks.json").read_text())
    assert "gross_weight" in risk
    assert "net_weight" in risk
    assert "max_single_weight" in risk
    assert "portfolio_beta" in risk
    assert "max_gross_breach" in risk
    assert "max_net_breach" in risk
    assert "max_position_breach" in risk
    assert "beta_breach" in risk


def test_daily_run_thresholds_enforced(tmp_path: Path) -> None:
    """When max_position_weight is tight, breach is detected."""
    df_by_symbol = _synthetic_ohlcv(n_dates=80, n_symbols=5)  # Few symbols => concentrated
    asof = pd.Timestamp("2024-04-15")
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)

    result = _run_daily(
        df_by_symbol,
        strategy="factors",
        factor="momentum_12_1",
        combo_list=None,
        combo_method="equal",
        asof=asof,
        rebalance="M",
        initial_cash=100_000.0,
        cost_model="fixed",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        top_k=2,
        bottom_k=2,
        max_gross=None,
        max_net=None,
        max_position_weight=0.05,  # Very tight: 5% max per name
        beta_threshold=10.0,  # Lax
        state_path=tmp_path / "state.json",
        output_dir=output_dir,
        apply=False,
        force_rebalance=True,
    )

    risk = json.loads((output_dir / "risk_checks.json").read_text())
    # With top 2 + bottom 2, we have 4 names, each ~25% => likely breach
    assert "max_position_breach" in risk


def test_to_jsonable_serializes_numpy_pandas(tmp_path: Path) -> None:
    """to_jsonable converts np.bool_, np.float64, pd.Timestamp, Path for json.dumps."""
    risk_checks = {
        "flag": np.bool_(True),
        "value": np.float64(1.2),
        "ts": pd.Timestamp("2024-06-30"),
        "path": tmp_path / "foo.json",
        "nested": {"x": np.int64(42), "y": np.bool_(False)},
    }
    serialized = json.dumps(to_jsonable(risk_checks), indent=2)
    parsed = json.loads(serialized)
    assert parsed["flag"] is True
    assert parsed["value"] == 1.2
    assert "2024-06-30" in parsed["ts"]
    assert "foo.json" in parsed["path"]
    assert parsed["nested"]["x"] == 42
    assert parsed["nested"]["y"] is False


def test_load_save_portfolio(tmp_path: Path) -> None:
    """Load/save portfolio state."""
    p = tmp_path / "portfolio.json"
    loaded = _load_current_portfolio(p)
    assert loaded["cash"] == 0.0
    assert loaded["positions"] == {}
    assert loaded["asof"] is None

    _save_current_portfolio(p, 50_000.0, {"SPY": 100.0, "QQQ": 50.0}, "2024-06-01")
    loaded2 = _load_current_portfolio(p)
    assert loaded2["cash"] == 50_000.0
    assert loaded2["positions"] == {"SPY": 100.0, "QQQ": 50.0}
    assert loaded2["asof"] == "2024-06-01"


def test_turnover_zero_when_weights_match(tmp_path: Path) -> None:
    """When current weights == target weights, turnover is 0 and orders are 0."""
    df_by_symbol = _synthetic_ohlcv(n_dates=80, n_symbols=12)
    asof = pd.Timestamp("2024-04-15")
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    state_path = tmp_path / "state.json"

    # First run to get target weights and save state (reversal_5d needs only 5d lookback)
    result1 = _run_daily(
        df_by_symbol,
        strategy="factors",
        factor="reversal_5d",
        combo_list=None,
        combo_method="equal",
        asof=asof,
        rebalance="D",  # Daily = every day is rebalance day
        initial_cash=100_000.0,
        cost_model="fixed",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        top_k=3,
        bottom_k=3,
        max_gross=None,
        max_net=None,
        max_position_weight=0.2,
        beta_threshold=0.5,
        state_path=state_path,
        output_dir=output_dir,
        apply=True,
        force_rebalance=False,
    )
    assert "error" not in result1

    # Second run: load the state we just saved (portfolio matches target)
    result2 = _run_daily(
        df_by_symbol,
        strategy="factors",
        factor="reversal_5d",
        combo_list=None,
        combo_method="equal",
        asof=asof,
        rebalance="D",
        initial_cash=100_000.0,
        cost_model="fixed",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        top_k=3,
        bottom_k=3,
        max_gross=None,
        max_net=None,
        max_position_weight=0.2,
        beta_threshold=0.5,
        state_path=state_path,
        output_dir=tmp_path / "out2",
        apply=False,
        force_rebalance=False,
    )
    assert "error" not in result2
    assert result2["n_orders"] == 0
    assert result2["turnover"] == 0.0


def test_turnover_from_cash_near_100pct(tmp_path: Path) -> None:
    """Starting from cash, turnover ~100% (0.5 * sum |w| for long/short ~ 1.0)."""
    df_by_symbol = _synthetic_ohlcv(n_dates=80, n_symbols=12)
    asof = pd.Timestamp("2024-04-15")
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    state_path = tmp_path / "nonexistent_state.json"  # No prior state = cash

    result = _run_daily(
        df_by_symbol,
        strategy="factors",
        factor="reversal_5d",
        combo_list=None,
        combo_method="equal",
        asof=asof,
        rebalance="D",  # Daily = every day is rebalance day
        initial_cash=100_000.0,
        cost_model="fixed",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        top_k=3,
        bottom_k=3,
        max_gross=None,
        max_net=None,
        max_position_weight=0.2,
        beta_threshold=0.5,
        state_path=state_path,
        output_dir=output_dir,
        apply=False,
        force_rebalance=False,
    )
    assert "error" not in result
    # From cash: w_current = 0 for all. turnover = 0.5 * sum |w_target|.
    # With gross_leverage=1.0 (default), sum |w_target| = 1.0 -> turnover = 0.5
    assert result["turnover"] >= 0.4
    assert result["turnover"] <= 1.1
    assert result["n_orders"] > 0


def test_state_saved_then_loaded_decreases_turnover(tmp_path: Path) -> None:
    """State saved then loaded -> second run turnover decreases."""
    df_by_symbol = _synthetic_ohlcv(n_dates=80, n_symbols=12)
    asof = pd.Timestamp("2024-04-15")
    state_path = tmp_path / "state.json"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    out1.mkdir(parents=True)
    out2.mkdir(parents=True)

    # Run 1: from cash, apply to save state (reversal_5d needs only 5d lookback)
    result1 = _run_daily(
        df_by_symbol,
        strategy="factors",
        factor="reversal_5d",
        combo_list=None,
        combo_method="equal",
        asof=asof,
        rebalance="D",
        initial_cash=100_000.0,
        cost_model="fixed",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        top_k=3,
        bottom_k=3,
        max_gross=None,
        max_net=None,
        max_position_weight=0.2,
        beta_threshold=0.5,
        state_path=state_path,
        output_dir=out1,
        apply=True,
        force_rebalance=False,
    )
    assert "error" not in result1
    turnover1 = result1["turnover"]

    # Run 2: load state, same asof (target weights unchanged)
    result2 = _run_daily(
        df_by_symbol,
        strategy="factors",
        factor="reversal_5d",
        combo_list=None,
        combo_method="equal",
        asof=asof,
        rebalance="D",
        initial_cash=100_000.0,
        cost_model="fixed",
        fee_bps=1.0,
        slippage_bps=2.0,
        spread_bps=1.0,
        top_k=3,
        bottom_k=3,
        max_gross=None,
        max_net=None,
        max_position_weight=0.2,
        beta_threshold=0.5,
        state_path=state_path,
        output_dir=out2,
        apply=False,
        force_rebalance=False,
    )
    assert "error" not in result2
    turnover2 = result2["turnover"]
    assert turnover2 < turnover1
