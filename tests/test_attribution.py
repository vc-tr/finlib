"""
Tests for src/research/attribution.py

Uses synthetic FF factor data to avoid network calls.
"""

import numpy as np
import pandas as pd
import pytest

from src.research.attribution import (
    AttributionResult,
    FactorLoadings,
    FF5_COLUMNS,
    _parse_ff5_csv,
    _run_ff_regression,
    ff5_attribution,
    attribution_summary_df,
    format_attribution_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ff_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic FF5 factor data with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = {
        "MktRf": rng.normal(0.0005, 0.010, n),
        "SMB": rng.normal(0.0001, 0.005, n),
        "HML": rng.normal(0.0001, 0.005, n),
        "RMW": rng.normal(0.0001, 0.004, n),
        "CMA": rng.normal(0.0001, 0.004, n),
        "Rf": rng.uniform(0.00, 0.0002, n),
    }
    return pd.DataFrame(data, index=dates)


def _make_strategy_returns(
    ff_data: pd.DataFrame,
    alpha: float = 0.0002,
    betas: tuple = (0.8, 0.1, -0.1, 0.05, 0.05),
    seed: int = 0,
) -> pd.Series:
    """
    Synthetic strategy returns = Rf + alpha + factor_exposure + noise.
    Ensures we have a known ground truth for regression checks.
    """
    rng = np.random.default_rng(seed)
    b_mkt, b_smb, b_hml, b_rmw, b_cma = betas
    noise = rng.normal(0, 0.005, len(ff_data))
    returns = (
        ff_data["Rf"].values
        + alpha
        + b_mkt * ff_data["MktRf"].values
        + b_smb * ff_data["SMB"].values
        + b_hml * ff_data["HML"].values
        + b_rmw * ff_data["RMW"].values
        + b_cma * ff_data["CMA"].values
        + noise
    )
    return pd.Series(returns, index=ff_data.index)


# ---------------------------------------------------------------------------
# _parse_ff5_csv
# ---------------------------------------------------------------------------

MINIMAL_FF5_CSV = """\
,

Annual Factors: January-December

 ,Mkt-RF,SMB,HML,RMW,CMA,RF
20200102,-0.70,0.26,-0.29,0.10,-0.09,0.00
20200103,-0.50,-0.03,-0.22,-0.20,-0.19,0.00
20200106,0.36,0.10,0.04,0.01,0.06,0.00

Annual Factors

"""


def test_parse_ff5_csv_returns_dataframe():
    df = _parse_ff5_csv(MINIMAL_FF5_CSV)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FF5_COLUMNS


def test_parse_ff5_csv_date_index():
    df = _parse_ff5_csv(MINIMAL_FF5_CSV)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0].year == 2020


def test_parse_ff5_csv_decimal_conversion():
    """Raw values are in percent; after parse they should be /100."""
    df = _parse_ff5_csv(MINIMAL_FF5_CSV)
    # First row: Mkt-RF = -0.70 percent → -0.0070
    assert abs(df["MktRf"].iloc[0] - (-0.0070)) < 1e-6


def test_parse_ff5_csv_row_count():
    df = _parse_ff5_csv(MINIMAL_FF5_CSV)
    assert len(df) == 3


def test_parse_ff5_csv_missing_header_raises():
    bad_csv = "no,header,here\n20200102,1,2,3,4,5,6\n"
    with pytest.raises(ValueError, match="Mkt-RF"):
        _parse_ff5_csv(bad_csv)


# ---------------------------------------------------------------------------
# _run_ff_regression (unit test — no download needed)
# ---------------------------------------------------------------------------

def test_run_ff_regression_returns_attribution_result():
    ff = _make_ff_data(300)
    ret = _make_strategy_returns(ff, alpha=0.0001)
    excess = ret - ff["Rf"]
    factors = ff[["MktRf", "SMB", "HML", "RMW", "CMA"]]
    result = _run_ff_regression(excess, factors, "test", "FF5", 252)
    assert isinstance(result, AttributionResult)


def test_run_ff_regression_recovers_alpha():
    """OLS should recover the true alpha within 3 annualized percent."""
    ff = _make_ff_data(1000, seed=1)
    true_alpha_daily = 0.0003  # ~7.6% annualized
    # Use seed different from ff_data to avoid RNG correlation
    ret = _make_strategy_returns(ff, alpha=true_alpha_daily, seed=99)
    excess = ret - ff["Rf"]
    factors = ff[["MktRf", "SMB", "HML", "RMW", "CMA"]]
    result = _run_ff_regression(excess, factors, "test", "FF5", 252)
    true_alpha_ann = (1 + true_alpha_daily) ** 252 - 1
    assert abs(result.alpha_annual - true_alpha_ann) < 0.03


def test_run_ff_regression_recovers_market_beta():
    """Market beta should be close to 0.8 (the injected value)."""
    ff = _make_ff_data(1000, seed=2)
    # Use seed different from ff_data to avoid RNG correlation
    ret = _make_strategy_returns(ff, betas=(0.8, 0.0, 0.0, 0.0, 0.0), seed=98)
    excess = ret - ff["Rf"]
    factors = ff[["MktRf", "SMB", "HML", "RMW", "CMA"]]
    result = _run_ff_regression(excess, factors, "test", "FF5", 252)
    mkt_beta = [f.beta for f in result.factors if f.name == "MktRf"][0]
    assert abs(mkt_beta - 0.8) < 0.05


def test_run_ff_regression_r_squared_range():
    ff = _make_ff_data(300)
    ret = _make_strategy_returns(ff)
    excess = ret - ff["Rf"]
    factors = ff[["MktRf", "SMB", "HML", "RMW", "CMA"]]
    result = _run_ff_regression(excess, factors, "test", "FF5", 252)
    assert 0.0 <= result.r_squared <= 1.0
    assert 0.0 <= result.adj_r_squared <= 1.0


def test_run_ff_regression_factor_loadings_count():
    ff = _make_ff_data(200)
    ret = _make_strategy_returns(ff)
    excess = ret - ff["Rf"]
    factors5 = ff[["MktRf", "SMB", "HML", "RMW", "CMA"]]
    factors3 = ff[["MktRf", "SMB", "HML"]]
    r5 = _run_ff_regression(excess, factors5, "test", "FF5", 252)
    r3 = _run_ff_regression(excess, factors3, "test", "FF3", 252)
    assert len(r5.factors) == 5
    assert len(r3.factors) == 3


def test_factor_loadings_is_significant():
    fl = FactorLoadings("MktRf", beta=1.0, se=0.1, t_stat=10.0,
                        p_value=0.001, ci_lower=0.8, ci_upper=1.2)
    assert fl.is_significant is True
    fl_not = FactorLoadings("HML", beta=0.01, se=0.1, t_stat=0.1,
                            p_value=0.92, ci_lower=-0.19, ci_upper=0.21)
    assert fl_not.is_significant is False


# ---------------------------------------------------------------------------
# ff5_attribution (main API — injects mock ff_data)
# ---------------------------------------------------------------------------

def test_ff5_attribution_basic(tmp_path):
    ff = _make_ff_data(400)
    ret = _make_strategy_returns(ff)
    result = ff5_attribution(ret, strategy_name="test_strat", ff_data=ff)
    assert result.strategy == "test_strat"
    assert result.model == "FF5"
    assert result.n_obs >= 30


def test_ff5_attribution_ff3_model():
    ff = _make_ff_data(300)
    ret = _make_strategy_returns(ff)
    result = ff5_attribution(ret, model="FF3", ff_data=ff)
    assert result.model == "FF3"
    assert len(result.factors) == 3
    factor_names = [f.name for f in result.factors]
    assert "RMW" not in factor_names
    assert "CMA" not in factor_names


def test_ff5_attribution_invalid_model():
    ff = _make_ff_data(200)
    ret = _make_strategy_returns(ff)
    with pytest.raises(ValueError, match="model must be"):
        ff5_attribution(ret, model="FF7", ff_data=ff)


def test_ff5_attribution_insufficient_overlap():
    ff = _make_ff_data(200)
    # Strategy on completely different dates
    future_dates = pd.bdate_range("2030-01-01", periods=50)
    ret = pd.Series(np.random.randn(50) * 0.01, index=future_dates)
    with pytest.raises(ValueError, match="overlapping days"):
        ff5_attribution(ret, ff_data=ff)


def test_ff5_attribution_start_end_dates():
    ff = _make_ff_data(400)
    ret = _make_strategy_returns(ff)
    result = ff5_attribution(ret, ff_data=ff)
    # Dates should be parseable strings
    assert result.start_date != ""
    assert result.end_date != ""


def test_ff5_attribution_alpha_sign_positive_strategy():
    """Injected positive alpha → positive attribution alpha."""
    ff = _make_ff_data(1000, seed=5)
    ret = _make_strategy_returns(ff, alpha=0.0005, seed=55)
    result = ff5_attribution(ret, ff_data=ff)
    assert result.alpha_annual > 0


def test_ff5_attribution_alpha_sign_negative_strategy():
    """Injected negative alpha → negative attribution alpha."""
    ff = _make_ff_data(1000, seed=6)
    ret = _make_strategy_returns(ff, alpha=-0.0005, seed=66)
    result = ff5_attribution(ret, ff_data=ff)
    assert result.alpha_annual < 0


def test_ff5_attribution_to_dict():
    ff = _make_ff_data(300)
    ret = _make_strategy_returns(ff)
    result = ff5_attribution(ret, ff_data=ff)
    d = result.to_dict()
    assert isinstance(d, dict)
    for key in (
        "strategy", "alpha_annual", "alpha_t_stat", "alpha_p_value",
        "r_squared", "n_obs", "factors",
    ):
        assert key in d
    assert isinstance(d["factors"], list)
    assert len(d["factors"]) == 5


def test_ff5_attribution_str_repr():
    ff = _make_ff_data(200)
    ret = _make_strategy_returns(ff)
    result = ff5_attribution(ret, strategy_name="golden_cross", ff_data=ff)
    s = str(result)
    assert "golden_cross" in s
    assert "Alpha" in s
    assert "MktRf" in s


# ---------------------------------------------------------------------------
# ff5_attribution_batch
# ---------------------------------------------------------------------------

def test_ff5_attribution_batch_returns_list():
    ff = _make_ff_data(300)
    strategies = {
        "strat_a": _make_strategy_returns(ff, seed=10),
        "strat_b": _make_strategy_returns(ff, alpha=0.0002, seed=11),
    }
    # Patch download to use our synthetic data
    results = []
    for name, ret in strategies.items():
        results.append(ff5_attribution(ret, strategy_name=name, ff_data=ff))
    assert len(results) == 2
    assert all(isinstance(r, AttributionResult) for r in results)


def test_ff5_attribution_batch_names_match():
    ff = _make_ff_data(300)
    strategies = {"alpha_gen": _make_strategy_returns(ff, seed=20)}
    results = [
        ff5_attribution(r, name, ff_data=ff)
        for name, r in strategies.items()
    ]
    assert results[0].strategy == "alpha_gen"


# ---------------------------------------------------------------------------
# attribution_summary_df
# ---------------------------------------------------------------------------

def test_attribution_summary_df_sorted_by_alpha():
    ff = _make_ff_data(300, seed=99)
    # Use seeds >> 99 to avoid seed collision with ff_data
    results = [
        ff5_attribution(
            _make_strategy_returns(ff, alpha=a, seed=100 + i),
            strategy_name=f"s{i}", ff_data=ff,
        )
        for i, a in enumerate([0.0001, -0.0003, 0.0005])
    ]
    df = attribution_summary_df(results)
    assert list(df["alpha_annual"]) == sorted(df["alpha_annual"], reverse=True)


def test_attribution_summary_df_columns():
    ff = _make_ff_data(200, seed=88)
    results = [ff5_attribution(_make_strategy_returns(ff, seed=0),
                               strategy_name="s1", ff_data=ff)]
    df = attribution_summary_df(results)
    for col in ("strategy", "alpha_annual", "alpha_t", "alpha_p", "r2"):
        assert col in df.columns


def test_attribution_summary_df_empty():
    df = attribution_summary_df([])
    assert df.empty


# ---------------------------------------------------------------------------
# format_attribution_report
# ---------------------------------------------------------------------------

def test_format_attribution_report_markdown():
    ff = _make_ff_data(200)
    ret = _make_strategy_returns(ff)
    result = ff5_attribution(ret, strategy_name="test", ff_data=ff)
    report = format_attribution_report(result)
    assert "## Fama-French" in report
    assert "Alpha" in report
    assert "Factor Loadings" in report
    assert "MktRf" in report


def test_format_attribution_report_significant_flag():
    ff = _make_ff_data(1000, seed=77)
    # Very high alpha → likely significant
    ret = _make_strategy_returns(ff, alpha=0.002, seed=77)
    result = ff5_attribution(ret, ff_data=ff)
    report = format_attribution_report(result)
    if result.alpha_significant:
        assert "SIGNIFICANT" in report
    else:
        assert "not significant" in report


def test_format_attribution_report_all_factors_present():
    ff = _make_ff_data(200)
    ret = _make_strategy_returns(ff)
    result = ff5_attribution(ret, ff_data=ff)
    report = format_attribution_report(result)
    for factor in ["MktRf", "SMB", "HML", "RMW", "CMA"]:
        assert factor in report
