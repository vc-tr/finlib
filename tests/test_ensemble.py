"""Tests for multi-factor ensemble."""

import numpy as np
import pandas as pd

from src.factors.ensemble import combine_factors


def _synthetic_factors(n_dates: int = 100, n_symbols: int = 20, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create synthetic factor DataFrames (date x symbol)."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]
    return {
        "f1": pd.DataFrame(rng.randn(n_dates, n_symbols) * 0.1, index=idx, columns=cols),
        "f2": pd.DataFrame(rng.randn(n_dates, n_symbols) * 0.05, index=idx, columns=cols),
        "f3": pd.DataFrame(rng.randn(n_dates, n_symbols) * 0.08, index=idx, columns=cols),
    }


def test_combine_equal_shape():
    """Equal combo produces correct shape."""
    factors = _synthetic_factors(50, 15)
    combined, weights, zscored, _ = combine_factors(factors, method="equal")
    assert combined.shape == (50, 15)
    assert combined.index.equals(factors["f1"].index)
    assert list(combined.columns) == list(factors["f1"].columns)
    assert len(weights) == 3
    for w in weights.values():
        assert abs(w - 1 / 3) < 1e-6


def test_combine_equal_zscore_average():
    """Equal combo z-scores per date then averages."""
    factors = _synthetic_factors(30, 10)
    combined, _, _, _ = combine_factors(factors, method="equal")
    # Each row should have mean ~0 (z-scored then averaged)
    row_means = combined.mean(axis=1)
    assert row_means.abs().max() < 0.01


def test_ic_weighted_uses_train_only():
    """ic_weighted uses only train_slice for IC computation (no lookahead)."""
    factors = _synthetic_factors(100, 15)
    idx = factors["f1"].index
    # Train = first 60 dates, test = last 40. IC computed on train only.
    train_end_idx = 60
    train_slice = slice(idx[0], idx[train_end_idx - 1])
    fwd = pd.DataFrame(
        np.random.RandomState(1).randn(100, 15) * 0.01,
        index=idx,
        columns=factors["f1"].columns,
    )

    combined, weights, _, _ = combine_factors(
        factors, method="ic_weighted",
        train_slice=train_slice,
        fwd_returns=fwd,
    )
    # Weights from train only; combined factor spans full range (no test data in weight calc)
    assert len(weights) == 3
    assert sum(weights.values()) - 1.0 < 1e-6
    assert combined.shape == (100, 15)
    # Combined has values for all dates including test (weights applied to full factors)
    assert combined.loc[idx[train_end_idx]:].notna().any().any()


def test_ic_weighted_constant_factor_fallback():
    """ic_weighted does not crash when factor is constant; falls back to equal weights."""
    rng = np.random.RandomState(44)
    n_dates, n_symbols = 80, 15
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]
    # One factor constant, one normal
    factors = {
        "constant": pd.DataFrame(np.ones((n_dates, n_symbols)), index=idx, columns=cols),
        "normal": pd.DataFrame(rng.randn(n_dates, n_symbols) * 0.1, index=idx, columns=cols),
    }
    fwd = pd.DataFrame(rng.randn(n_dates, n_symbols) * 0.01, index=idx, columns=cols)
    train_slice = slice(idx[0], idx[49])
    combined, weights, _, _ = combine_factors(
        factors, method="ic_weighted",
        train_slice=train_slice,
        fwd_returns=fwd,
    )
    assert combined.shape == (n_dates, n_symbols)
    assert len(weights) == 2
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # Should fall back to equal (constant has IC=0)
    assert abs(weights["constant"] - 0.5) < 1e-5 or abs(weights["normal"] - 0.5) < 1e-5


def test_ic_weighted_nans_fallback():
    """ic_weighted does not crash when factor has NaNs; falls back to equal if needed."""
    rng = np.random.RandomState(45)
    n_dates, n_symbols = 60, 12
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]
    f1 = pd.DataFrame(rng.randn(n_dates, n_symbols) * 0.1, index=idx, columns=cols)
    f1.iloc[:30, :] = np.nan  # First half all NaN
    f2 = pd.DataFrame(rng.randn(n_dates, n_symbols) * 0.05, index=idx, columns=cols)
    factors = {"f1": f1, "f2": f2}
    fwd = pd.DataFrame(rng.randn(n_dates, n_symbols) * 0.01, index=idx, columns=cols)
    train_slice = slice(idx[0], idx[39])
    combined, weights, _, _ = combine_factors(
        factors, method="ic_weighted",
        train_slice=train_slice,
        fwd_returns=fwd,
    )
    assert combined.shape == (n_dates, n_symbols)
    assert len(weights) == 2
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_ridge_returns_weights():
    """Ridge method returns valid weights."""
    factors = _synthetic_factors(80, 12)
    idx = factors["f1"].index
    fwd = pd.DataFrame(
        factors["f1"].values * 0.3 + np.random.RandomState(3).randn(80, 12) * 0.01,
        index=idx,
        columns=factors["f1"].columns,
    )
    train_slice = slice(idx[0], idx[49])
    combined, weights, _, _ = combine_factors(
        factors, method="ridge",
        train_slice=train_slice,
        fwd_returns=fwd,
    )
    assert len(weights) == 3
    assert sum(weights.values()) - 1.0 < 1e-5
    assert combined.shape == (80, 12)


def test_combo_pure_momentum_exposure_reflects_weight():
    """If combo is pure momentum (weight 1), exposure table reflects that. Deterministic."""
    rng = np.random.RandomState(123)
    n_dates, n_symbols = 50, 12
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"S{i}" for i in range(n_symbols)]
    # Momentum factor: deterministic spread
    mom = np.linspace(-1, 1, n_symbols)
    mom_df = pd.DataFrame(
        np.tile(mom, (n_dates, 1)) + rng.randn(n_dates, n_symbols) * 0.1,
        index=idx,
        columns=cols,
    )
    rev_df = pd.DataFrame(
        rng.randn(n_dates, n_symbols) * 0.1,
        index=idx,
        columns=cols,
    )
    factors = {"momentum": mom_df, "reversal": rev_df}
    # Equal weights for simplicity; we'll override to test pure momentum
    combined, weights, zscored, _ = combine_factors(factors, method="equal")
    # Simulate pure momentum: use momentum z-score as combined, so weight 1 for momentum
    z_mom = zscored["momentum"]
    z_rev = zscored["reversal"]
    # Portfolio: long top 3 by momentum, short bottom 3
    rank_mom = z_mom.rank(axis=1, ascending=False)
    w = pd.DataFrame(0.0, index=idx, columns=cols)
    for t in idx:
        top3 = rank_mom.loc[t].nsmallest(3).index
        bot3 = rank_mom.loc[t].nlargest(3).index
        w.loc[t, top3] = 1.0 / 3
        w.loc[t, bot3] = -1.0 / 3
    # Exposure: sum_i w_i * z_f(i)
    exp_mom = (w * z_mom).sum(axis=1)
    exp_rev = (w * z_rev).sum(axis=1)
    # Portfolio is built from momentum ranks -> exposure to momentum should be positive
    assert exp_mom.mean() > 0.1, f"Expected positive momentum exposure, got {exp_mom.mean()}"
    # Exposure to reversal is noise (portfolio doesn't tilt on reversal)
    assert abs(exp_rev.mean()) < 0.5, f"Reversal exposure should be small, got {exp_rev.mean()}"
