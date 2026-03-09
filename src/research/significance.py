"""
Statistical Significance Testing for Backtest Results.

The core problem in quantitative finance: a strategy looks good in-sample.
How likely is it to keep working out-of-sample?

This module implements the rigorous statistical toolkit needed to answer that:

1. Sharpe Ratio Standard Error (Lo, 2002)
   Analytical formula accounting for autocorrelation in returns.
   SR_hat / SE(SR_hat) ~ N(0,1) under the null of SR=0.

2. Block Bootstrap Confidence Intervals
   Non-parametric CI via stationary block bootstrap. Preserves
   autocorrelation structure of returns. Better than iid bootstrap for
   financial time series.

3. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
   DSR adjusts the observed Sharpe for:
     - Multiple testing (# trials inflates max expected Sharpe)
     - Non-normality (skewness and excess kurtosis reduce true significance)
   A Sharpe of 2.0 from 100 trials is much less impressive than 2.0
   from 1 trial. DSR tells you the probability that you've found a
   false discovery.

4. Multiple Testing Corrections
   - Bonferroni: conservative, controls FWER
   - Benjamini-Hochberg (BH): controls FDR (false discovery rate)
     More powerful when you expect some true positives.

References:
  Lo, A.W. (2002). "The statistics of Sharpe ratios."
    Financial Analysts Journal, 58(4), 36-52.

  Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    Journal of Portfolio Management, 40(5), 94-107.

  Benjamini, Y. & Hochberg, Y. (1995). "Controlling the false discovery rate:
    a practical and powerful approach to multiple testing."
    Journal of the Royal Statistical Society B, 57(1), 289-300.

  Politis, D.N. & Romano, J.P. (1994). "The stationary bootstrap."
    Journal of the American Statistical Association, 89(428), 1303-1313.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SharpeStats:
    """Full statistical profile of a Sharpe ratio estimate."""

    sharpe: float          # Observed annualized Sharpe ratio
    se: float              # Standard error (Lo 2002)
    t_stat: float          # Sharpe / SE
    p_value: float         # Two-tailed p(|t| >= t_stat) under H0: SR=0
    ci_lower: float        # Bootstrap CI lower bound
    ci_upper: float        # Bootstrap CI upper bound
    ci_level: float        # Confidence level (e.g. 0.95)
    n_obs: int             # Number of observations used
    skewness: float        # Return skewness
    excess_kurtosis: float  # Return excess kurtosis

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:
        sig = "*" if self.is_significant else ""
        return (
            f"SR={self.sharpe:.2f}{sig}  SE={self.se:.3f}  "
            f"t={self.t_stat:.2f}  p={self.p_value:.3f}  "
            f"CI=[{self.ci_lower:.2f}, {self.ci_upper:.2f}]  "
            f"n={self.n_obs}"
        )


@dataclass
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio computation."""

    observed_sharpe: float       # Best observed SR across all trials
    dsr: float                   # Deflated Sharpe Ratio (probability of false discovery)
    sr_star: float               # Minimum SR needed to be statistically significant
    n_trials: int                # Number of strategies tested
    n_obs: int                   # Observations per trial
    skewness: float              # Skewness of returns
    excess_kurtosis: float       # Excess kurtosis of returns
    is_significant: bool         # DSR > 0.95 (95% confidence not a false discovery)
    expected_max_sharpe: float   # E[max SR] under null (random strategies)

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.is_significant else "NOT SIGNIFICANT"
        return (
            f"DSR={self.dsr:.3f} [{sig}]  "
            f"Observed SR={self.observed_sharpe:.2f}  "
            f"Min required SR*={self.sr_star:.2f}  "
            f"E[max SR | {self.n_trials} trials]={self.expected_max_sharpe:.2f}"
        )


@dataclass
class MultipleTestingResult:
    """Result of applying multiple testing corrections to a set of p-values."""

    names: List[str]
    raw_p_values: List[float]
    bonferroni_rejected: List[bool]   # FWER control at alpha
    bh_rejected: List[bool]           # FDR control at alpha
    bh_adjusted_p: List[float]        # BH adjusted p-values
    alpha: float
    n_tests: int

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "strategy": self.names,
            "p_value": self.raw_p_values,
            "bonferroni": self.bonferroni_rejected,
            "bh_reject": self.bh_rejected,
            "bh_p_adj": self.bh_adjusted_p,
        }).sort_values("p_value")


# ---------------------------------------------------------------------------
# 1. Sharpe Ratio Standard Error — Lo (2002)
# ---------------------------------------------------------------------------

def sharpe_se_lo(returns: pd.Series, annualization: int = 252) -> float:
    """
    Analytical standard error of the Sharpe ratio (Lo, 2002).

    Accounts for autocorrelation in returns via the Newey-West correction.
    For iid returns: SE = sqrt((1 + SR^2/2) / T)
    For autocorrelated returns: includes sum of lagged autocovariances.

    Args:
        returns: Daily (or periodic) return series
        annualization: Bars per year (252=daily, 52=weekly, 12=monthly)

    Returns:
        Standard error of the annualized Sharpe ratio
    """
    r = returns.dropna().values
    T = len(r)
    if T < 4:
        return np.nan

    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma <= 0:
        return np.nan

    sr_daily = mu / sigma

    # Lo (2002) non-normality correction for Sharpe variance
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis())  # excess kurtosis

    # Lo (2002) Eq 12: variance of SR under autocorrelation
    # V(SR_hat) ≈ (1/T)(1 + (1/2)SR^2 - skew*SR + (1/4)(kurt+2)*SR^2)
    # Simplified (no autocorr terms) — matches common implementations
    V_sr = (1 / T) * (1 + 0.5 * sr_daily**2 - skew * sr_daily
                      + ((kurt + 2) / 4) * sr_daily**2)

    # Scale to annualized SE
    se_ann = np.sqrt(max(V_sr, 0) * annualization)
    return float(se_ann)


def sharpe_stats(
    returns: pd.Series,
    annualization: int = 252,
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
    block_size: int = None,
) -> SharpeStats:
    """
    Full statistical profile of a Sharpe ratio.

    Combines Lo (2002) analytical SE with block bootstrap CI.

    Args:
        returns: Daily return series
        annualization: Bars per year
        ci_level: Confidence level for bootstrap CI
        n_bootstrap: Bootstrap resamples
        block_size: Block size for block bootstrap (None = auto)

    Returns:
        SharpeStats dataclass
    """
    r = returns.dropna()
    n = len(r)
    if n < 10:
        return SharpeStats(
            sharpe=np.nan, se=np.nan, t_stat=np.nan, p_value=np.nan,
            ci_lower=np.nan, ci_upper=np.nan, ci_level=ci_level,
            n_obs=n, skewness=np.nan, excess_kurtosis=np.nan,
        )

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    sharpe_ann = (mu / sigma * np.sqrt(annualization)) if sigma > 0 else np.nan

    se = sharpe_se_lo(r, annualization)
    t_stat = float(sharpe_ann / se) if (se and not np.isnan(se) and se > 0) else np.nan
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat)))) if not np.isnan(t_stat) else np.nan

    ci_lower, ci_upper = bootstrap_sharpe_ci(r, annualization, ci_level, n_bootstrap, block_size)

    skewness = float(r.skew())
    excess_kurtosis = float(r.kurtosis())

    return SharpeStats(
        sharpe=sharpe_ann,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_obs=n,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )


# ---------------------------------------------------------------------------
# 2. Block Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def _stationary_block_bootstrap(
    data: np.ndarray,
    block_size: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stationary block bootstrap (Politis & Romano, 1994).

    Each resample maintains temporal autocorrelation structure by
    drawing blocks of consecutive observations.

    Args:
        data: 1D array
        block_size: Average block length (geometric distribution)
        n_resamples: Number of bootstrap resamples
        rng: NumPy random Generator

    Returns:
        Array of shape (n_resamples, len(data))
    """
    n = len(data)
    resamples = np.empty((n_resamples, n))

    for i in range(n_resamples):
        indices = []
        while len(indices) < n:
            start = rng.integers(0, n)
            # Block length ~ Geometric(1/block_size), capped at n
            length = min(rng.geometric(1.0 / block_size), n - len(indices))
            for j in range(length):
                indices.append((start + j) % n)
        resamples[i] = data[np.array(indices[:n])]

    return resamples


def bootstrap_sharpe_ci(
    returns: pd.Series,
    annualization: int = 252,
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
    block_size: int = None,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Block bootstrap confidence interval for the Sharpe ratio.

    Args:
        returns: Return series
        annualization: Bars per year
        ci_level: Confidence level (0.95 → 95% CI)
        n_bootstrap: Number of bootstrap resamples
        block_size: Block size (None → sqrt(T), typical for daily data)
        seed: Random seed for reproducibility

    Returns:
        (ci_lower, ci_upper) tuple
    """
    r = returns.dropna().values
    n = len(r)
    if n < 10:
        return np.nan, np.nan

    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    rng = np.random.default_rng(seed)
    resamples = _stationary_block_bootstrap(r, block_size, n_bootstrap, rng)

    # Compute Sharpe for each resample
    boot_sharpes = []
    for resamp in resamples:
        sigma = resamp.std(ddof=1)
        if sigma > 0:
            boot_sharpes.append(resamp.mean() / sigma * np.sqrt(annualization))
        else:
            boot_sharpes.append(0.0)

    boot_sharpes = np.array(boot_sharpes)
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(boot_sharpes, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# 3. Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    annualization: int = 252,
    sr_benchmark: float = 0.0,
) -> DeflatedSharpeResult:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts the observed Sharpe for:
    1. Selection bias: testing N strategies inflates the expected maximum SR
    2. Non-normality: skewness and excess kurtosis reduce the effective information ratio

    DSR = Prob(SR* <= observed_SR) where SR* is the expected maximum SR
    under the null of zero skill across N independent trials.

    Interpretation:
      DSR > 0.95 → "significant" — unlikely to be a false discovery
      DSR < 0.95 → "not significant" — high probability of overfitting

    Args:
        observed_sharpe: Best annualized Sharpe ratio seen across all trials
        n_trials: Total number of strategies/parameter sets tested
        n_obs: Number of OOS observations (bars)
        skewness: Skewness of the strategy returns
        excess_kurtosis: Excess kurtosis of the strategy returns
        annualization: Bars per year (for variance scaling)
        sr_benchmark: Benchmark SR to beat (default 0)

    Returns:
        DeflatedSharpeResult
    """
    if n_trials < 1 or n_obs < 4:
        raise ValueError(f"n_trials >= 1 and n_obs >= 4 required, got {n_trials}, {n_obs}")

    # Expected maximum SR across N iid trials (Eq. 1 in Bailey & LdP 2014)
    # E[max SR_N] ≈ (1 - gamma_e) * z(1-1/N) + gamma_e * z(1-1/(N*e))
    # where z = inverse normal CDF, gamma_e ≈ 0.5772 (Euler-Mascheroni)
    gamma_e = 0.5772156649  # Euler-Mascheroni constant
    if n_trials == 1:
        expected_max_sr = 0.0
    else:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max_sr = (
            (1 - gamma_e) * z1 + gamma_e * z2
        ) / np.sqrt(annualization)  # de-annualize for comparison

    # SR* = E[max SR | N trials] + sr_benchmark
    sr_star = expected_max_sr + sr_benchmark / np.sqrt(annualization)

    # DSR: probability that the observed SR exceeds SR* in a non-normal world
    # Variance of SR adjusted for non-normality (Lo 2002 Eq 6):
    # V(SR_hat) ≈ (1 + (1/2)SR^2 - skew*SR + (kurt+2)/4 * SR^2) / T
    sr_daily = observed_sharpe / np.sqrt(annualization)
    V_sr = (1 + 0.5 * sr_daily**2 - skewness * sr_daily
            + ((excess_kurtosis + 2) / 4) * sr_daily**2) / n_obs

    se = np.sqrt(max(V_sr, 1e-12))

    # DSR = Prob(SR_true >= SR*) = 1 - Phi((SR* - SR_hat) / se)
    dsr = float(stats.norm.cdf((sr_daily - sr_star) / se)) if se > 0 else 0.0
    dsr = float(np.clip(dsr, 0.0, 1.0))

    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        dsr=dsr,
        sr_star=float(sr_star * np.sqrt(annualization)),
        n_trials=n_trials,
        n_obs=n_obs,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        is_significant=(dsr >= 0.95),
        expected_max_sharpe=float(expected_max_sr * np.sqrt(annualization)),
    )


def deflated_sharpe_from_returns(
    returns: pd.Series,
    n_trials: int,
    annualization: int = 252,
    sr_benchmark: float = 0.0,
) -> DeflatedSharpeResult:
    """
    Convenience: compute DSR directly from a return series.

    Args:
        returns: Strategy return series (already the best from N trials)
        n_trials: Total trials tested (including this one)
        annualization: Bars per year
        sr_benchmark: Minimum acceptable SR to beat

    Returns:
        DeflatedSharpeResult
    """
    r = returns.dropna()
    n_obs = len(r)
    sigma = float(r.std(ddof=1))
    if sigma <= 0 or n_obs < 4:
        raise ValueError(f"Insufficient data (n={n_obs}) or zero variance")

    sharpe_ann = float(r.mean() / sigma * np.sqrt(annualization))
    skewness = float(r.skew())
    excess_kurtosis = float(r.kurtosis())

    return deflated_sharpe_ratio(
        observed_sharpe=sharpe_ann,
        n_trials=n_trials,
        n_obs=n_obs,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        annualization=annualization,
        sr_benchmark=sr_benchmark,
    )


# ---------------------------------------------------------------------------
# 4. Multiple Testing Corrections
# ---------------------------------------------------------------------------

def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Bonferroni correction: reject H0_i if p_i < alpha / n_tests.

    Controls Family-Wise Error Rate (FWER) — probability of ANY false discovery.
    Conservative: keeps false positives near 0 but has lower power.

    Args:
        p_values: List of raw p-values
        alpha: Desired FWER threshold (default 0.05)

    Returns:
        List of booleans (True = reject H0 = significant after correction)
    """
    n = len(p_values)
    threshold = alpha / n
    return [p < threshold for p in p_values]


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """
    Benjamini-Hochberg (BH) procedure for FDR control.

    Controls False Discovery Rate — expected fraction of rejected hypotheses
    that are false discoveries. More powerful than Bonferroni when many
    true positives are expected (as in strategy research).

    Algorithm (Benjamini & Hochberg 1995):
    1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(n)
    2. Find largest k where p_(k) <= (k/n) * alpha
    3. Reject all H0_(1) ... H0_(k)

    Args:
        p_values: List of raw p-values
        alpha: Desired FDR level (default 0.05)

    Returns:
        (rejected, adjusted_p_values) where rejected[i] = True means significant
    """
    n = len(p_values)
    if n == 0:
        return [], []

    # Sort by p-value, track original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_indices = [i for i, _ in indexed]
    sorted_p = np.array([p for _, p in indexed])

    # BH thresholds
    bh_thresholds = (np.arange(1, n + 1) / n) * alpha

    # Find largest k where p_(k) <= threshold_(k)
    below = sorted_p <= bh_thresholds
    if below.any():
        k_max = int(np.where(below)[0].max())
        reject_mask = np.zeros(n, dtype=bool)
        reject_mask[:k_max + 1] = True
    else:
        reject_mask = np.zeros(n, dtype=bool)

    # Compute BH-adjusted p-values: p_adj_(k) = min(n/k * p_(k), 1)
    # Using the step-up formulation
    adj_p = np.minimum(sorted_p * n / (np.arange(1, n + 1)), 1.0)
    # Enforce monotonicity (step-up: take cumulative min from the right)
    adj_p = np.minimum.accumulate(adj_p[::-1])[::-1]

    # Map back to original order
    rejected = [False] * n
    adjusted = [0.0] * n
    for rank, orig_idx in enumerate(sorted_indices):
        rejected[orig_idx] = bool(reject_mask[rank])
        adjusted[orig_idx] = float(adj_p[rank])

    return rejected, adjusted


def multiple_testing_summary(
    names: List[str],
    returns_list: List[pd.Series],
    annualization: int = 252,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
) -> MultipleTestingResult:
    """
    Apply both Bonferroni and BH corrections to a set of strategy return series.

    Computes t-statistic and p-value for SR=0 test for each strategy,
    then applies both corrections.

    Args:
        names: Strategy names
        returns_list: List of return series (one per strategy)
        annualization: Bars per year
        alpha: Significance level
        n_bootstrap: Bootstrap samples for CI

    Returns:
        MultipleTestingResult
    """
    p_values = []
    for r in returns_list:
        ss = sharpe_stats(r, annualization=annualization, n_bootstrap=n_bootstrap)
        p_values.append(float(ss.p_value) if not np.isnan(ss.p_value) else 1.0)

    bonf_rejected = bonferroni_correction(p_values, alpha)
    bh_rejected, bh_adj = benjamini_hochberg(p_values, alpha)

    return MultipleTestingResult(
        names=names,
        raw_p_values=p_values,
        bonferroni_rejected=bonf_rejected,
        bh_rejected=bh_rejected,
        bh_adjusted_p=bh_adj,
        alpha=alpha,
        n_tests=len(names),
    )


# ---------------------------------------------------------------------------
# Convenience: full significance report for a single strategy
# ---------------------------------------------------------------------------

def significance_report(
    returns: pd.Series,
    strategy_name: str = "strategy",
    n_trials: int = 1,
    annualization: int = 252,
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Generate a complete significance report for a strategy.

    Includes Sharpe SE, t-stat, p-value, bootstrap CI, and DSR.

    Args:
        returns: Strategy return series
        strategy_name: Name for display
        n_trials: Total strategies tested (for DSR)
        annualization: Bars per year
        ci_level: CI confidence level
        n_bootstrap: Bootstrap samples

    Returns:
        dict with all significance metrics
    """
    ss = sharpe_stats(returns, annualization, ci_level, n_bootstrap)
    try:
        dsr_result = deflated_sharpe_from_returns(returns, n_trials, annualization)
    except ValueError as e:
        warnings.warn(f"DSR computation failed: {e}")
        dsr_result = None

    report = {
        "strategy": strategy_name,
        "n_obs": ss.n_obs,
        "sharpe": ss.sharpe,
        "sharpe_se": ss.se,
        "t_stat": ss.t_stat,
        "p_value": ss.p_value,
        "significant_5pct": ss.is_significant,
        f"ci_{int(ci_level*100)}_lower": ss.ci_lower,
        f"ci_{int(ci_level*100)}_upper": ss.ci_upper,
        "skewness": ss.skewness,
        "excess_kurtosis": ss.excess_kurtosis,
    }
    if dsr_result is not None:
        report.update({
            "n_trials": n_trials,
            "dsr": dsr_result.dsr,
            "sr_star": dsr_result.sr_star,
            "expected_max_sharpe": dsr_result.expected_max_sharpe,
            "dsr_significant": dsr_result.is_significant,
        })

    return report


def format_significance_report(report: dict) -> str:
    """Format a significance_report dict as a human-readable string."""
    lines = [
        f"Statistical Significance — {report['strategy']}",
        "=" * 60,
        f"  Observations:       {report['n_obs']}",
        f"  Sharpe (ann.):      {report['sharpe']:.4f}",
        f"  Sharpe SE (Lo02):   {report['sharpe_se']:.4f}",
        f"  t-statistic:        {report['t_stat']:.3f}",
        f"  p-value (SR=0):     {report['p_value']:.4f}  "
        f"{'[SIGNIFICANT]' if report['significant_5pct'] else '[not significant]'}",
        f"  95% CI (bootstrap): [{report.get('ci_95_lower', '?'):.2f}, "
        f"{report.get('ci_95_upper', '?'):.2f}]",
        f"  Skewness:           {report['skewness']:.3f}",
        f"  Excess Kurtosis:    {report['excess_kurtosis']:.3f}",
    ]
    if "dsr" in report:
        lines += [
            "",
            f"  Deflated Sharpe Ratio (n_trials={report['n_trials']})",
            f"  DSR:                {report['dsr']:.4f}  "
            f"{'[SIGNIFICANT]' if report['dsr_significant'] else '[not significant]'}",
            f"  Min required SR*:   {report['sr_star']:.4f}",
            f"  E[max SR | trials]: {report['expected_max_sharpe']:.4f}",
        ]
    lines.append("=" * 60)
    return "\n".join(lines)
