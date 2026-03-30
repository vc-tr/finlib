"""Tests for cross-sectional factor engine. Uses synthetic df_by_symbol."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import Backtester
from src.factors import (
    compute_factor,
    compute_factors,
    cross_sectional_rank,
    build_portfolio,
    get_universe,
    UniverseRegistry,
    forward_returns,
    cross_sectional_ic,
    summarize_ic,
)
from src.factors.factors import get_prices_wide
from src.factors.portfolio import (
    apply_rebalance_costs,
    _resample_weights_to_rebalance,
    apply_beta_neutral,
    rebalance_dates,
    weights_at_rebalance,
)
from src.factors.risk import estimate_beta, rolling_portfolio_beta


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


def test_universe_registry() -> None:
    """UniverseRegistry lists and returns all universes."""
    names = UniverseRegistry.list_names()
    assert "liquid_etfs" in names
    assert "sector_etfs" in names
    assert "bond_etfs" in names
    assert "commodity_etfs" in names
    assert "international_etfs" in names

    symbols, meta = UniverseRegistry.get("sector_etfs", n=5)
    assert len(symbols) == 5
    assert meta.category == "sector"
    assert "XLK" in symbols or "XLF" in symbols

    syms = UniverseRegistry.get_symbols("bond_etfs", n=3)
    assert len(syms) == 3
    bond_all = UniverseRegistry.get_symbols("bond_etfs")
    assert all(s in bond_all for s in syms)

    with pytest.raises(ValueError, match="Unknown universe"):
        UniverseRegistry.get("nonexistent")


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


def test_compute_factors_multiple() -> None:
    """compute_factors returns dict of factor DataFrames."""
    df_by = _synthetic_df_by_symbol(100, 10)
    factors = compute_factors(df_by, ["momentum_12_1", "reversal_5d", "lowvol_20d"])
    assert len(factors) == 3
    assert "momentum_12_1" in factors
    assert "reversal_5d" in factors
    assert "lowvol_20d" in factors
    for f in factors.values():
        assert f.shape[1] == 10


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


def test_rolling_portfolio_beta_portfolio_equals_market_times_two() -> None:
    """Portfolio = market * 2 => rolling beta ~ 2."""
    rng = np.random.RandomState(42)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    market_ret = pd.Series(rng.randn(n) * 0.01, index=idx)
    port_ret = market_ret * 2  # exact 2x exposure
    beta_p = rolling_portfolio_beta(port_ret, market_ret, window=252)
    # After warmup, beta should be ~2
    beta_tail = beta_p.dropna().iloc[-50:]
    assert 1.8 < beta_tail.mean() < 2.2, f"Expected beta ~2, got {beta_tail.mean()}"


def test_estimate_beta_known_relation() -> None:
    """Beta estimator recovers known synthetic relation: asset = 1.5 * market + noise."""
    rng = np.random.RandomState(42)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    mr = pd.Series(rng.randn(n) * 0.01, index=idx)
    ar = pd.DataFrame({
        "A": mr * 1.5 + rng.randn(n) * 0.002,
        "B": mr * 0.5 + rng.randn(n) * 0.002,
    }, index=idx)
    betas = estimate_beta(ar, mr, window=100)
    assert betas.shape == (n, 2)
    beta_a = betas["A"].dropna().iloc[-1]
    beta_b = betas["B"].dropna().iloc[-1]
    assert 1.2 < beta_a < 1.8
    assert 0.3 < beta_b < 0.7


def test_beta_neutral_reduces_absolute_beta() -> None:
    """Beta-neutralization reduces |portfolio beta|."""
    rng = np.random.RandomState(42)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B", "SPY"])
    weights.loc[:, "A"] = 0.5
    weights.loc[:, "B"] = -0.5
    betas = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    betas["A"] = 1.2
    betas["B"] = 0.8
    w_adj, beta_before, beta_after = apply_beta_neutral(weights, betas, market_symbol="SPY")
    assert (beta_after.abs() < beta_before.abs() + 0.01).all()


def test_rebalance_dates_monthly() -> None:
    """Monthly rebalance produces dates only at month boundaries."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rb = rebalance_dates(idx, "M")
    assert len(rb) > 0
    for d in rb:
        assert d == idx[idx <= d][-1]
    months = rb.to_series().dt.to_period("M").unique()
    assert len(months) == len(rb)


def test_weights_at_rebalance_changes_only_at_month_boundaries() -> None:
    """Monthly rebalance: weight changes only at month boundaries."""
    df_by = _synthetic_df_by_symbol(100, 10, seed=42)
    factor_df = compute_factor(df_by, "reversal_5d")
    weights = weights_at_rebalance(factor_df, "M", top_k=2, bottom_k=2)
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    rb = rebalance_dates(weights.index, "M")
    non_rb = weights.index.difference(rb)
    if len(non_rb) > 0:
        turnover_non_rb = turnover.reindex(non_rb).dropna()
        assert (turnover_non_rb < 1e-10).all(), "Turnover should be ~0 on non-rebalance days"


def test_turnover_zero_on_non_rebalance_days() -> None:
    """Turnover is zero (within tolerance) on non-execution days."""
    df_by = _synthetic_df_by_symbol(80, 8, seed=123)
    factor_df = compute_factor(df_by, "momentum_12_1")
    weights = weights_at_rebalance(factor_df, "M", top_k=2, bottom_k=2)
    w_held = _resample_weights_to_rebalance(weights, "M").shift(1).fillna(0)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0)
    idx = turnover.index
    rb = rebalance_dates(idx, "M")
    exec_dates = set()
    for d in rb:
        later = idx[idx > d]
        if len(later) > 0:
            exec_dates.add(later[0])
    non_exec = idx.difference(exec_dates)
    non_exec = non_exec[1:]
    if len(non_exec) > 0:
        to_non = turnover.reindex(non_exec).dropna()
        assert (to_non < 1e-8).all(), f"Turnover on non-exec days: {to_non[to_non >= 1e-8]}"


def test_ic_positive_when_factor_predicts_returns() -> None:
    """Synthetic dataset where factor perfectly predicts returns => positive mean IC."""
    rng = np.random.RandomState(42)
    n_dates, n_symbols = 200, 30
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    signal = rng.randn(n_dates, n_symbols) * 0.5
    fwd_ret = signal + rng.randn(n_dates, n_symbols) * 0.2
    factor_df = pd.DataFrame(signal, index=idx, columns=[f"S{i}" for i in range(n_symbols)])
    fwd_ret_df = pd.DataFrame(fwd_ret, index=idx, columns=factor_df.columns)
    ic_series = cross_sectional_ic(factor_df, fwd_ret_df, method="spearman")
    mean_ic = ic_series.dropna().mean()
    assert mean_ic > 0.5, f"Expected mean IC > 0.5 when factor predicts returns, got {mean_ic}"


def test_ic_constant_factor_no_crash() -> None:
    """Constant factor => IC all NaN but code does not crash; summary handles n=0 gracefully."""
    rng = np.random.RandomState(43)
    n_dates, n_symbols = 100, 15
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    factor_df = pd.DataFrame(
        np.ones((n_dates, n_symbols)),
        index=idx,
        columns=[f"S{i}" for i in range(n_symbols)],
    )
    fwd_ret_df = pd.DataFrame(
        rng.randn(n_dates, n_symbols) * 0.01,
        index=idx,
        columns=factor_df.columns,
    )
    ic_series = cross_sectional_ic(factor_df, fwd_ret_df, method="spearman")
    assert ic_series.isna().all(), "Constant factor should have all NaN IC"
    out = summarize_ic(ic_series)
    assert out["n"] == 0
    assert np.isnan(out["mean_ic"])
    assert np.isnan(out["t_stat"])
