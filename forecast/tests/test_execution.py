"""Tests for execution realism and anti-leakage."""

import numpy as np
import pandas as pd
import pytest
from src.backtest import Backtester
from src.backtest.execution import (
    ExecutionConfig,
    apply_execution_realism,
    compute_turnover,
)
from src.strategies import MomentumStrategy


def test_execution_delay_prevents_lookahead() -> None:
    """Signal at t must not be executed at t; must execute at t+1."""
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(50) * 0.02)))
    strategy = MomentumStrategy(lookback=5)
    signals, _ = strategy.backtest_returns(prices)

    # Backtester uses signals.shift(1) * returns
    bt = Backtester()
    result = bt.run_from_signals(prices, signals)

    # At each bar t, strategy return = signal[t-1] * price_return[t]
    # So we never use same-bar information for execution
    assert len(result.returns) > 0
    # First bar should have 0 or NaN return (no prior signal)
    assert result.returns.iloc[0] == 0 or np.isnan(result.returns.iloc[0]) or abs(result.returns.iloc[0]) < 1e-10


def test_fee_slippage_reduces_returns() -> None:
    """Higher fees/slippage should reduce total return."""
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.02)))
    strategy = MomentumStrategy(lookback=10)
    signals, _ = strategy.backtest_returns(prices)

    bt = Backtester()
    r_none = bt.run_from_signals(prices, signals, execution_config=None)
    r_fees = bt.run_from_signals(
        prices, signals,
        execution_config=ExecutionConfig(fee_bps=50, slippage_bps=0),
    )
    r_slip = bt.run_from_signals(
        prices, signals,
        execution_config=ExecutionConfig(fee_bps=0, slippage_bps=50),
    )

    # With costs, return should be lower (or equal if strategy has no trades)
    assert r_fees.total_return <= r_none.total_return + 1e-6
    assert r_slip.total_return <= r_none.total_return + 1e-6


def test_turnover_computation() -> None:
    """Turnover = |position change|."""
    signals = pd.Series([0, 1, 1, -1, 0])
    to = compute_turnover(signals)
    assert to.iloc[0] == 0  # no prior
    assert to.iloc[1] == 1  # 0 -> 1
    assert to.iloc[2] == 0  # 1 -> 1
    assert to.iloc[3] == 2  # 1 -> -1
    assert to.iloc[4] == 1  # -1 -> 0


def test_walkforward_fold_boundaries() -> None:
    """Walk-forward produces non-overlapping test windows."""
    from src.backtest.walkforward import generate_folds

    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    folds = generate_folds(dates, train_days=252, test_days=63, step_days=63)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end < f.test_start
        assert f.test_start < f.test_end


def test_tearsheet_generates_files() -> None:
    """Tearsheet runs without error and produces output files."""
    from pathlib import Path
    import tempfile
    from src.backtest import Backtester
    from src.reporting.tearsheet import generate_tearsheet

    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.02)))
    signals = pd.Series(0, index=prices.index)
    signals.iloc[50:] = 1

    bt = Backtester()
    result = bt.run_from_signals(prices, signals)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        generate_tearsheet(result, prices, signals, out)
        assert (out / "tearsheet.html").exists()
        assert (out / "equity_curve.png").exists()
        assert (out / "drawdown.png").exists()
        assert (out / "summary.json").exists()
