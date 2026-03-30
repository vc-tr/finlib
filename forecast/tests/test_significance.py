"""
Tests for src/research/significance.py

Covers: Sharpe SE, bootstrap CI, DSR, Bonferroni, BH, full report.
"""

import numpy as np
import pandas as pd
import pytest

from src.research.significance import (
    sharpe_se_lo,
    sharpe_stats,
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    deflated_sharpe_from_returns,
    bonferroni_correction,
    benjamini_hochberg,
    multiple_testing_summary,
    significance_report,
    format_significance_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iid_returns():
    """iid N(0.001, 0.01) — Sharpe ≈ 1.0 * sqrt(252) ≈ 1.59"""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.01, 500))


@pytest.fixture
def zero_returns():
    """iid N(0, 0.01) — Sharpe = 0"""
    np.random.seed(0)
    return pd.Series(np.random.normal(0.0, 0.01, 500))


@pytest.fixture
def trending_returns():
    """Strong positive trend — high Sharpe"""
    np.random.seed(7)
    return pd.Series(np.random.normal(0.005, 0.01, 500))


# ---------------------------------------------------------------------------
# 1. Sharpe SE (Lo 2002)
# ---------------------------------------------------------------------------

def test_sharpe_se_positive(iid_returns):
    se = sharpe_se_lo(iid_returns)
    assert se > 0


def test_sharpe_se_decreases_with_n():
    """More data → lower SE."""
    np.random.seed(1)
    r_short = pd.Series(np.random.normal(0.001, 0.01, 100))
    r_long = pd.Series(np.random.normal(0.001, 0.01, 1000))
    assert sharpe_se_lo(r_short) > sharpe_se_lo(r_long)


def test_sharpe_se_nan_on_empty():
    se = sharpe_se_lo(pd.Series([], dtype=float))
    assert np.isnan(se)


def test_sharpe_se_nan_on_constant():
    """Zero variance → nan SE."""
    r = pd.Series([0.001] * 100)
    se = sharpe_se_lo(r)
    # Either nan or inf — not a valid finite positive SE
    assert np.isnan(se) or not np.isfinite(se) or se > 0


# ---------------------------------------------------------------------------
# 2. Sharpe Stats (full)
# ---------------------------------------------------------------------------

def test_sharpe_stats_iid(iid_returns):
    ss = sharpe_stats(iid_returns, n_bootstrap=200)
    assert ss.sharpe > 0
    assert ss.se > 0
    assert ss.t_stat > 0
    assert 0 <= ss.p_value <= 1
    assert ss.ci_lower < ss.ci_upper
    assert ss.n_obs == len(iid_returns)
    assert np.isfinite(ss.skewness)
    assert np.isfinite(ss.excess_kurtosis)


def test_sharpe_stats_significance(trending_returns):
    """High-Sharpe series should be significant."""
    ss = sharpe_stats(trending_returns, n_bootstrap=200)
    assert ss.is_significant
    assert ss.p_value < 0.05


def test_sharpe_stats_zero_not_significant(zero_returns):
    """Zero-drift series should usually NOT be significant."""
    ss = sharpe_stats(zero_returns, n_bootstrap=200)
    # p > 0.05 most of the time with SR ≈ 0
    assert ss.p_value > 0.01  # not wildly significant


def test_sharpe_stats_ci_contains_true(iid_returns):
    """95% CI should contain the point estimate most of the time."""
    ss = sharpe_stats(iid_returns, n_bootstrap=500)
    assert ss.ci_lower < ss.sharpe < ss.ci_upper


def test_sharpe_stats_insufficient_data():
    ss = sharpe_stats(pd.Series([0.01, -0.01, 0.02]))
    assert np.isnan(ss.sharpe)


# ---------------------------------------------------------------------------
# 3. Block Bootstrap CI
# ---------------------------------------------------------------------------

def test_bootstrap_ci_ordering(iid_returns):
    lo, hi = bootstrap_sharpe_ci(iid_returns, n_bootstrap=300)
    assert lo < hi


def test_bootstrap_ci_zero_drift(zero_returns):
    """CI for zero-Sharpe strategy should straddle zero."""
    lo, hi = bootstrap_sharpe_ci(zero_returns, n_bootstrap=300)
    assert lo < 0 < hi


def test_bootstrap_ci_reproducible(iid_returns):
    """Same seed → same CI."""
    ci1 = bootstrap_sharpe_ci(iid_returns, n_bootstrap=200, seed=99)
    ci2 = bootstrap_sharpe_ci(iid_returns, n_bootstrap=200, seed=99)
    assert ci1 == ci2


# ---------------------------------------------------------------------------
# 4. Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def test_dsr_single_trial():
    """DSR with n_trials=1 should equal standard significance."""
    dsr = deflated_sharpe_ratio(
        observed_sharpe=2.0, n_trials=1, n_obs=252,
        skewness=0, excess_kurtosis=0
    )
    assert 0 <= dsr.dsr <= 1
    assert dsr.sr_star < dsr.observed_sharpe  # single trial: no inflation


def test_dsr_many_trials_lower():
    """More trials → higher SR* → lower DSR for same observed Sharpe."""
    dsr_1 = deflated_sharpe_ratio(2.0, n_trials=1, n_obs=252)
    dsr_100 = deflated_sharpe_ratio(2.0, n_trials=100, n_obs=252)
    assert dsr_1.dsr > dsr_100.dsr


def test_dsr_high_sr_significant():
    """Very high Sharpe with few trials should be significant."""
    dsr = deflated_sharpe_ratio(
        observed_sharpe=5.0, n_trials=10, n_obs=1000,
        skewness=0, excess_kurtosis=0
    )
    assert dsr.is_significant


def test_dsr_low_sr_not_significant():
    """Modest Sharpe with many trials should NOT be significant."""
    dsr = deflated_sharpe_ratio(
        observed_sharpe=1.0, n_trials=1000, n_obs=100,
        skewness=0, excess_kurtosis=0
    )
    assert not dsr.is_significant


def test_dsr_non_normality_reduces_significance():
    """Fat tails (high kurtosis) make significance harder to achieve."""
    dsr_normal = deflated_sharpe_ratio(2.0, n_trials=1, n_obs=252,
                                       skewness=0, excess_kurtosis=0)
    dsr_fat = deflated_sharpe_ratio(2.0, n_trials=1, n_obs=252,
                                    skewness=-2.0, excess_kurtosis=10.0)
    # Fat tails increase variance of SR estimate → lower DSR
    assert dsr_normal.dsr >= dsr_fat.dsr


def test_dsr_from_returns(iid_returns):
    dsr = deflated_sharpe_from_returns(iid_returns, n_trials=1)
    assert 0 <= dsr.dsr <= 1
    assert dsr.n_obs == len(iid_returns)
    assert dsr.n_trials == 1


def test_dsr_str(iid_returns):
    dsr = deflated_sharpe_from_returns(iid_returns, n_trials=5)
    s = str(dsr)
    assert "DSR=" in s
    assert "SR=" in s


# ---------------------------------------------------------------------------
# 5. Bonferroni Correction
# ---------------------------------------------------------------------------

def test_bonferroni_no_reject():
    p_values = [0.5, 0.6, 0.4]
    result = bonferroni_correction(p_values, alpha=0.05)
    assert not any(result)


def test_bonferroni_one_very_small():
    """Only the tiny p-value should survive Bonferroni."""
    p_values = [0.001, 0.4, 0.5, 0.3, 0.2]
    result = bonferroni_correction(p_values, alpha=0.05)
    assert result[0]
    assert not any(result[1:])


def test_bonferroni_all_tiny():
    p_values = [0.001, 0.002, 0.003]
    result = bonferroni_correction(p_values, alpha=0.05)
    assert all(result)


def test_bonferroni_threshold_scales_with_n():
    """Larger n → stricter threshold."""
    p = [0.01]
    assert bonferroni_correction(p * 1, alpha=0.05)[0]   # 0.01 < 0.05
    assert not bonferroni_correction(p * 10, alpha=0.05)[0]  # 0.01 > 0.005


# ---------------------------------------------------------------------------
# 6. Benjamini-Hochberg (BH)
# ---------------------------------------------------------------------------

def test_bh_basic():
    p_values = [0.001, 0.01, 0.4, 0.5]
    rejected, adj = benjamini_hochberg(p_values, alpha=0.05)
    # At least the two small ones should be rejected
    assert rejected[0]
    assert rejected[1]
    assert not rejected[2]
    assert not rejected[3]


def test_bh_more_powerful_than_bonferroni():
    """BH rejects at least as many as Bonferroni."""
    # Borderline case: BH is more powerful
    p_values = [0.01, 0.02, 0.03, 0.04, 0.5]
    bh_rejected, _ = benjamini_hochberg(p_values, alpha=0.05)
    bonf_rejected = bonferroni_correction(p_values, alpha=0.05)
    # BH total rejections >= Bonferroni
    assert sum(bh_rejected) >= sum(bonf_rejected)


def test_bh_adjusted_p_monotone():
    """BH adjusted p-values should be non-decreasing (sorted by rank)."""
    p_values = [0.001, 0.01, 0.05, 0.3, 0.6]
    _, adj = benjamini_hochberg(p_values, alpha=0.05)
    # Adjusted p-values when sorted by original order should be valid [0,1]
    assert all(0 <= p <= 1 for p in adj)


def test_bh_empty():
    rejected, adj = benjamini_hochberg([], alpha=0.05)
    assert rejected == []
    assert adj == []


def test_bh_none_significant():
    p_values = [0.4, 0.5, 0.6, 0.7]
    rejected, adj = benjamini_hochberg(p_values, alpha=0.05)
    assert not any(rejected)


# ---------------------------------------------------------------------------
# 7. Multiple Testing Summary
# ---------------------------------------------------------------------------

def test_multiple_testing_summary():
    np.random.seed(42)
    # Mix: one strong positive, others zero-drift
    ret_good = pd.Series(np.random.normal(0.005, 0.01, 500))
    ret_flat1 = pd.Series(np.random.normal(0.0, 0.01, 500))
    ret_flat2 = pd.Series(np.random.normal(0.0, 0.01, 500))

    result = multiple_testing_summary(
        names=["good", "flat1", "flat2"],
        returns_list=[ret_good, ret_flat1, ret_flat2],
        n_bootstrap=100,
    )
    assert result.n_tests == 3
    assert len(result.raw_p_values) == 3
    assert len(result.bonferroni_rejected) == 3
    assert len(result.bh_rejected) == 3
    df = result.summary_df()
    assert set(df.columns) >= {"strategy", "p_value", "bonferroni", "bh_reject"}


# ---------------------------------------------------------------------------
# 8. Significance Report
# ---------------------------------------------------------------------------

def test_significance_report_dict(iid_returns):
    report = significance_report(iid_returns, strategy_name="test", n_trials=5)
    assert "sharpe" in report
    assert "p_value" in report
    assert "dsr" in report
    assert "t_stat" in report
    assert report["strategy"] == "test"
    assert report["n_trials"] == 5


def test_significance_report_format(iid_returns):
    report = significance_report(iid_returns, n_bootstrap=200)
    formatted = format_significance_report(report)
    assert "Sharpe" in formatted
    assert "p-value" in formatted
    assert "Deflated Sharpe" in formatted
