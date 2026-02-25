"""Tests for cross-sectional factor engine. Uses synthetic df_by_symbol."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import Backtester
from src.factors import compute_factor, cross_sectional_rank, build_portfolio, get_universe
from src.factors.factors import get_prices_wide
from src.factors.portfolio import apply_rebalance_costs, _resample_weights_to_rebalance


def _synthetic_df_by_symbol(n_dates: int = 300, n_symbols: int = 20, seed: int = 42) -> dict:
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


def test_get_universe() -> None:
    """Universe returns list of symbols."""
    syms = get_universe("liquid_etfs", n=10)
    assert len(syms) == 10
    assert "SPY" in syms
    assert "QQQ" in syms


def test_compute_factor_momentum() -> None:
    """Momentum factor has correct shape."""
    df_by = _synthetic_df_by_symbol(300, 15)
    factor_df = compute_factor(df_by, "momentum_12_1")
    assert factor_df.shape[0] <= 300
    assert factor_df.shape[1] == 15
    assert factor_df.notna().any().any()


def test_compute_factor_reversal() -> None:
    """Reversal factor has correct shape."""
    df_by = _synthetic_df_by_symbol(100, 10)
    factor_df = compute_factor(df_by, "reversal_5d")
    assert factor_df.shape[1] == 10


def test_compute_factor_lowvol() -> None:
    """Low-vol factor has correct shape."""
    df_by = _synthetic_df_by_symbol(100, 10)
    factor_df = compute_factor(df_by, "lowvol_20d")
    assert factor_df.shape[1] == 10


def test_cross_sectional_rank() -> None:
    """Ranking produces weights with correct structure."""
    df_by = _synthetic_df_by_symbol(100, 15)
    factor_df = compute_factor(df_by, "reversal_5d")
    weights = cross_sectional_rank(factor_df, top_k=3, bottom_k=3)
    assert weights.shape == factor_df.shape
    long_sum = (weights > 0).sum(axis=1)
    short_sum = (weights < 0).sum(axis=1)
    assert (long_sum <= 3).all() or (long_sum == 0).all()
    assert (short_sum <= 3).all() or (short_sum == 0).all()


def test_build_portfolio() -> None:
    """Portfolio returns have correct index."""
    df_by = _synthetic_df_by_symbol(100, 10)
    factor_df = compute_factor(df_by, "reversal_5d")
    weights = cross_sectional_rank(factor_df, top_k=2, bottom_k=2)
    prices = get_prices_wide(df_by)
    port_ret = build_portfolio(weights, prices, rebalance="M")
    assert len(port_ret) > 0
    assert port_ret.index.equals(prices.index)


def test_apply_rebalance_costs() -> None:
    """Costs reduce returns when turnover > 0."""
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    port_ret = pd.Series(0.001, index=idx)
    w = np.zeros((50, 3))
    w[:25, 0] = 0.5
    w[25:, 1] = 0.5
    weights = pd.DataFrame(w, index=idx, columns=["A", "B", "C"])
    after = apply_rebalance_costs(port_ret, weights, cost_bps=10)
    assert after.iloc[25] < port_ret.iloc[25]


def test_portfolio_tearsheet_exposures_and_holdings(tmp_path: Path) -> None:
    """Portfolio tearsheet with weights produces exposures, holdings CSV, contribution table."""
    from src.reporting.tearsheet import generate_tearsheet

    df_by = _synthetic_df_by_symbol(100, 10)
    factor_df = compute_factor(df_by, "reversal_5d")
    weights = cross_sectional_rank(factor_df, top_k=2, bottom_k=2)
    prices = get_prices_wide(df_by)
    w_held = _resample_weights_to_rebalance(weights, "M").shift(1).fillna(0)
    port_ret = build_portfolio(weights, prices, rebalance="M")
    port_ret = apply_rebalance_costs(port_ret, w_held, cost_bps=4)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0)
    positions = w_held.sum(axis=1)

    bt = Backtester(annualization_factor=252)
    result = bt.run(port_ret)
    prices_1d = prices.mean(axis=1)

    generate_tearsheet(
        result, prices_1d, positions,
        tmp_path, annualization=252,
        config={"factor": "reversal_5d", "top_k": 2, "bottom_k": 2},
        weights=w_held,
        turnover_series=turnover,
        prices_wide=prices,
    )

    assert (tmp_path / "holdings_by_date.csv").exists()
    report = (tmp_path / "REPORT.md").read_text()
    assert "Exposures" in report
    assert "Long count" in report
    assert "holdings_by_date.csv" in report
    assert "Per-Symbol Contribution" in report
