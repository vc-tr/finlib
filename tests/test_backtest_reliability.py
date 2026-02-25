"""
Reliability tests for backtest engine: no lookahead, costs, trade counting, tearsheet.

Uses deterministic synthetic OHLCV data (no network). Tests run fast (<2s).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.backtest import Backtester
from src.backtest.execution import ExecutionConfig, throttle_positions
from src.reporting.tearsheet import generate_tearsheet


def _rising_prices(n: int = 20, start: float = 100.0) -> pd.Series:
    """Deterministic rising prices: 100, 101, 102, ..."""
    return pd.Series(start + np.arange(n), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _oscillating_prices(n: int = 20, base: float = 100.0) -> pd.Series:
    """Deterministic oscillating: 100, 101, 100, 101, ..."""
    return pd.Series(
        base + np.array([1, -1] * (n // 2) + ([1] if n % 2 else []))[:n],
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _constant_prices(n: int = 20, value: float = 100.0) -> pd.Series:
    """Constant prices."""
    return pd.Series(
        np.full(n, value),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_execution_delay_no_lookahead() -> None:
    """If signal becomes 1 at bar 5, first fill must occur at bar 6."""
    prices = _rising_prices(15)
    signals = pd.Series(0, index=prices.index)
    signals.iloc[5:] = 1  # signal becomes 1 at bar 5 (index 5)

    bt = Backtester()
    result = bt.run_from_signals(prices, signals)

    # pos = signals.shift(1): bar 0..4 pos=0, bar 5 pos=0, bar 6.. pos=1
    # First bar with position=1 is bar 6. Strategy return at bar 6 = pos[6]*returns[6]
    # returns[6] = (106-105)/105 != 0, so result.returns.iloc[6] != 0
    returns = result.returns.reindex(prices.index).fillna(0)
    assert returns.iloc[5] == 0, "Bar 5 must have 0 return (no fill yet)"
    assert returns.iloc[6] != 0, "Bar 6 must have nonzero return (first fill)"


def test_costs_reduce_returns() -> None:
    """Same strategy: total return with costs=0 must exceed total return with costs>0."""
    prices = _oscillating_prices(50)  # oscillating triggers more trades
    signals = pd.Series(0, index=prices.index)
    signals.iloc[10:25] = 1
    signals.iloc[25:40] = -1

    bt = Backtester()
    r_zero = bt.run_from_signals(prices, signals, execution_config=None)
    r_costs = bt.run_from_signals(
        prices,
        signals,
        execution_config=ExecutionConfig(fee_bps=50, slippage_bps=50),
    )
    assert r_costs.total_return < r_zero.total_return, "Costs must reduce returns"


def test_trade_count_on_position_change_only() -> None:
    """If signal stays 1 for N bars, trades==1 (enter) not N."""
    prices = _rising_prices(30)
    signals = pd.Series(0, index=prices.index)
    signals.iloc[5:25] = 1  # long from bar 5 to 24 (20 bars), then exit at 25

    bt = Backtester()
    result = bt.run_from_signals(prices, signals)

    # Position changes: 0->1 at bar 5 (executed bar 6), 1->0 at bar 25 (executed bar 26)
    # n_trades = 2 (one enter, one exit)
    assert result.n_trades == 2, "Should count 2 trades (enter + exit), not 20 bars"


def test_backtester_accepts_dataframe() -> None:
    """Backtester.run_from_signals accepts DataFrame with 'close' column."""
    prices = _rising_prices(20)
    df = pd.DataFrame({"open": prices - 0.5, "high": prices + 1, "low": prices - 1, "close": prices})
    signals = pd.Series(0, index=df.index)
    signals.iloc[5:15] = 1  # enter at 6, exit at 16 -> 2 trades

    bt = Backtester()
    result = bt.run_from_signals(df, signals)
    assert result.total_return != 0
    assert result.n_trades == 2


def test_decision_interval_reduces_trades() -> None:
    """decision_interval_bars throttles position changes; fewer trades vs baseline."""
    prices = _oscillating_prices(60)  # oscillating triggers more flips
    # Oscillating signal: +1, -1, +1, -1, ... (flips every bar)
    signals = pd.Series(0, index=prices.index)
    for i in range(len(signals)):
        signals.iloc[i] = 1 if i % 2 == 0 else -1

    bt = Backtester()
    r_baseline = bt.run_from_signals(prices, signals)

    throttled = throttle_positions(signals, decision_interval_bars=5)
    r_throttled = bt.run_from_signals(prices, throttled)

    assert r_throttled.n_trades < r_baseline.n_trades, (
        "decision_interval should reduce trades"
    )


def test_cost_sensitivity_worse_or_equal_with_costs() -> None:
    """With costs, total return must be <= zero-cost return."""
    prices = _oscillating_prices(50)
    signals = pd.Series(0, index=prices.index)
    signals.iloc[10:25] = 1
    signals.iloc[25:40] = -1

    bt = Backtester()
    r_zero = bt.run_from_signals(prices, signals, execution_config=None)
    r_costs = bt.run_from_signals(
        prices,
        signals,
        execution_config=ExecutionConfig(fee_bps=10, slippage_bps=10, spread_bps=5),
    )
    assert r_costs.total_return <= r_zero.total_return, (
        "Costs must produce worse-or-equal return"
    )


def test_tearsheet_outputs_files(tmp_path: Path) -> None:
    """Run reporting function and assert expected files exist."""
    prices = _rising_prices(100)
    signals = pd.Series(0, index=prices.index)
    signals.iloc[20:80] = 1

    bt = Backtester()
    result = bt.run_from_signals(prices, signals)

    generate_tearsheet(result, prices, signals, tmp_path)

    assert (tmp_path / "tearsheet.html").exists()
    assert (tmp_path / "equity_curve.png").exists()
    assert (tmp_path / "drawdown.png").exists()
    assert (tmp_path / "returns_hist.png").exists()
    assert (tmp_path / "positions.png").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "turnover.png").exists()


def test_tearsheet_respects_temp_output_dir(tmp_path: Path) -> None:
    """Tear-sheet generation writes all artifacts only to the specified output directory."""
    prices = _rising_prices(50)
    signals = pd.Series(0, index=prices.index)
    signals.iloc[10:40] = 1

    bt = Backtester()
    result = bt.run_from_signals(prices, signals)

    out_subdir = tmp_path / "custom_output"
    out_subdir.mkdir()
    generate_tearsheet(result, prices, signals, out_subdir)

    # All artifacts must be in the specified dir, not in tmp_path root
    expected = [
        "tearsheet.html",
        "equity_curve.png",
        "drawdown.png",
        "returns_hist.png",
        "positions.png",
        "summary.json",
        "REPORT.md",
        "turnover.png",
    ]
    for name in expected:
        assert (out_subdir / name).exists(), f"Expected {name} in {out_subdir}"
    assert not any((tmp_path / f).exists() for f in expected), (
        "Artifacts must not leak to parent of output_dir"
    )


def test_run_demo_writes_to_output_dir_without_touching_root(tmp_path: Path, monkeypatch) -> None:
    """run_demo with --output-dir writes all artifacts to that dir, not output/ root."""
    import importlib.util
    import sys

    # Fake OHLCV data
    prices = _rising_prices(50)
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

    out_dir = tmp_path / "demo_out"
    out_dir.mkdir()
    sys.argv = [
        "run_demo",
        "--symbol", "SPY",
        "--period", "5d",
        "--interval", "1d",
        "--output-dir", str(out_dir),
        "--no-lock",
        "--no-cost-sensitivity",
    ]

    proj_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_demo", proj_root / "scripts" / "run_demo.py"
    )
    run_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_demo)
    run_demo.main()

    expected = ["tearsheet.html", "equity_curve.png", "drawdown.png", "summary.json", "REPORT.md"]
    for name in expected:
        assert (out_dir / name).exists(), f"Expected {name} in {out_dir}"
    # All artifacts must be in the specified output_dir, not leaked elsewhere
    assert not any((tmp_path / name).exists() for name in expected if tmp_path != out_dir), (
        "Artifacts must not leak to tmp_path root when output_dir is a subdir"
    )
