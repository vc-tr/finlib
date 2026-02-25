"""Tests for scripts/backtest_factors.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Load backtest_factors module
_script_path = Path(__file__).resolve().parent.parent / "scripts" / "backtest_factors.py"
_spec = importlib.util.spec_from_file_location("backtest_factors", _script_path)
_backtest_mod = importlib.util.module_from_spec(_spec)
sys.modules["backtest_factors"] = _backtest_mod
_spec.loader.exec_module(_backtest_mod)


def _synthetic_df_by_symbol(n_dates: int = 400, n_symbols: int = 15, seed: int = 42) -> dict:
    """Create synthetic df_by_symbol for testing."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    df_by = {}
    for i in range(n_symbols):
        sym = f"S{i}"
        ret = rng.randn(n_dates) * 0.01
        close = 100 * (1 + pd.Series(ret, index=idx)).cumprod()
        df_by[sym] = pd.DataFrame({"close": close})
    return df_by


def test_run_walkforward_robust_to_many_return_values():
    """_run_walkforward works when _run_factor_backtest returns >5 values (mock)."""
    import unittest.mock as mock

    _run_walkforward = _backtest_mod._run_walkforward
    _run_factor_backtest = _backtest_mod._run_factor_backtest

    df_by = _synthetic_df_by_symbol(300, 12)
    idx = df_by["S0"].index
    # Mock _run_factor_backtest to return 10 values (more than 5)
    fake_result = type("R", (), {})()
    fake_result.returns = pd.Series(0.001, index=idx)

    with mock.patch.object(_backtest_mod, "_run_factor_backtest") as m:
        m.return_value = (
            fake_result,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "extra1",
            "extra2",
        )
        out = _run_walkforward(
            df_by,
            factor="momentum_12_1",
            combo_list=None,
            combo_method="equal",
            top_k=2,
            bottom_k=2,
            rebalance="M",
            fee_bps=1.0,
            slippage_bps=2.0,
            spread_bps=1.0,
            annualization=252,
            folds=3,
            train_days=100,
            test_days=30,
        )
    assert "per_fold" in out
    assert "aggregated" in out
    assert m.called
