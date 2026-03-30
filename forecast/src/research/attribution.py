"""
Fama-French 5-Factor Attribution — Fama & French (1993, 2015).

Regresses strategy excess returns against the five Fama-French factors:
  R_t - Rf_t = α + β₁·MktRf_t + β₂·SMB_t + β₃·HML_t + β₄·RMW_t + β₅·CMA_t + ε_t

Factor definitions:
  MktRf: Market excess return (Mkt - Rf)
  SMB:   Small Minus Big (size premium)
  HML:   High Minus Low (value premium)
  RMW:   Robust Minus Weak (profitability premium)
  CMA:   Conservative Minus Aggressive (investment premium)
  Rf:    Risk-free rate (1-month T-bill)

The intercept α ("alpha") is the strategy's return unexplained by common
risk factors — the measure of true skill. A t-stat > 2 with p < 0.05 is
the industry standard for claiming factor-adjusted outperformance.

Standard errors use Newey-West HAC correction (statsmodels OLS +
`cov_type='HAC'`) to account for autocorrelation in residuals — the right
approach for
time-series regressions on daily strategy returns.

Data source: Kenneth French Data Library
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  Factors are published in CSV.zip format, updated monthly.

References:
  Fama, E.F. & French, K.R. (1993). "Common risk factors in the returns on
    stocks and bonds." Journal of Financial Economics, 33(1), 3-56.

  Fama, E.F. & French, K.R. (2015). "A five-factor asset pricing model."
    Journal of Financial Economics, 116(1), 1-22.

  Newey, W.K. & West, K.D. (1987). "A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix."
    Econometrica, 55(3), 703-708.
"""

from __future__ import annotations

import io
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

FF5_COLUMNS = ["MktRf", "SMB", "HML", "RMW", "CMA", "Rf"]
FF3_COLUMNS = ["MktRf", "SMB", "HML", "Rf"]

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "quant_lab" / "ff_factors"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_ff5_daily(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """
    Download (or load from cache) the Fama-French 5-factor daily data.

    Returns a DataFrame with columns [MktRf, SMB, HML, RMW, CMA, Rf]
    and DatetimeIndex. Values are in DECIMAL (divided by 100 from raw).

    Args:
        cache_dir: Local directory for caching the CSV

    Returns:
        pd.DataFrame with factor returns in decimal form

    Raises:
        RuntimeError: If download fails and no cached data exists
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "F-F_Research_Data_5_Factors_2x3_daily.csv"

    if cache_path.exists():
        return _parse_ff5_csv(cache_path.read_text(encoding="latin-1"))

    try:
        with urlopen(FF5_DAILY_URL, timeout=30) as resp:
            raw_bytes = resp.read()
    except URLError as e:
        raise RuntimeError(
            f"Failed to download FF5 data from Kenneth French library: {e}\n"
            f"Check your internet connection or supply pre-downloaded "
            f"data at:\n  {cache_path}"
        ) from e

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
        csv_name = next(
            n for n in z.namelist()
            if n.endswith(".CSV") or n.endswith(".csv")
        )
        csv_text = z.read(csv_name).decode("latin-1")

    cache_path.write_text(csv_text, encoding="utf-8")
    return _parse_ff5_csv(csv_text)


def _parse_ff5_csv(csv_text: str) -> pd.DataFrame:
    """
    Parse the Kenneth French CSV format.

    The file has descriptive text at the top and bottom. The daily data
    section starts with a header line containing 'Mkt-RF' and ends before
    the next section or footer.
    """
    lines = csv_text.splitlines()

    # Find the start of the daily data (header line with Mkt-RF)
    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line or "MKT-RF" in line.upper():
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find Mkt-RF header in FF5 CSV")

    # Find end of data: blank line after data starts, or non-numeric dates
    data_lines = [lines[header_idx]]  # header
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Data rows start with 8-digit date (YYYYMMDD)
        parts = stripped.split(",")
        first = parts[0].strip()
        if len(parts) >= 6 and first.isdigit() and len(first) == 8:
            data_lines.append(line)
        else:
            # Stop when we hit non-data content
            if data_lines and len(data_lines) > 1:
                break

    raw = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        index_col=0,
        skipinitialspace=True,
    )

    # Normalize column names
    raw.columns = [c.strip().replace("-", "") for c in raw.columns]
    # MktRF → MktRf
    rename = {"MktRF": "MktRf", "RF": "Rf"}
    raw = raw.rename(columns=rename)

    # Parse dates (YYYYMMDD format)
    raw.index = pd.to_datetime(raw.index.astype(str), format="%Y%m%d")
    raw.index.name = "Date"

    # Convert from percent to decimal
    raw = raw.apply(pd.to_numeric, errors="coerce") / 100.0

    return raw[FF5_COLUMNS].dropna()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FactorLoadings:
    """Single factor's regression results."""
    name: str
    beta: float
    se: float          # Newey-West HAC standard error
    t_stat: float
    p_value: float
    ci_lower: float    # 95% CI lower
    ci_upper: float    # 95% CI upper

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


@dataclass
class AttributionResult:
    """Full FF5 factor attribution result for a strategy."""
    strategy: str
    alpha_daily: float       # Daily intercept
    alpha_annual: float      # Annualized alpha (compounded)
    alpha_se: float          # Newey-West SE of alpha
    alpha_t_stat: float      # t-stat: alpha / SE
    alpha_p_value: float     # p-value (two-tailed)
    alpha_significant: bool  # p < 0.05
    r_squared: float         # OLS R²
    adj_r_squared: float     # Adjusted R²
    n_obs: int
    factors: List[FactorLoadings]  # One per factor in the model
    model: str = "FF5"       # "FF3" or "FF5"
    start_date: str = ""
    end_date: str = ""
    residual_skew: float = 0.0
    residual_kurtosis: float = 0.0
    factor_data_source: str = "Kenneth French Data Library"

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.alpha_significant else "not significant"
        lines = [
            f"FF5 Attribution — {self.strategy}",
            f"  Period: {self.start_date} to {self.end_date}"
            f"  (n={self.n_obs})",
            f"  Alpha (ann.): {self.alpha_annual:.4f}  "
            f"t={self.alpha_t_stat:.2f}  p={self.alpha_p_value:.4f}  [{sig}]",
            f"  R²={self.r_squared:.4f}  Adj-R²={self.adj_r_squared:.4f}",
            "",
            f"  {'Factor':<8} {'Beta':>8} {'SE':>8} {'t':>7} {'p':>8}",
            f"  {'-'*45}",
        ]
        for f in self.factors:
            sig_star = "*" if f.is_significant else " "
            lines.append(
                f"  {f.name:<8} {f.beta:>8.4f} {f.se:>8.4f} "
                f"{f.t_stat:>7.2f} {f.p_value:>8.4f}{sig_star}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable dict for JSON output."""
        return {
            "strategy": self.strategy,
            "model": self.model,
            "alpha_annual": self.alpha_annual,
            "alpha_daily": self.alpha_daily,
            "alpha_se": self.alpha_se,
            "alpha_t_stat": self.alpha_t_stat,
            "alpha_p_value": self.alpha_p_value,
            "alpha_significant": self.alpha_significant,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "n_obs": self.n_obs,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "residual_skew": self.residual_skew,
            "residual_kurtosis": self.residual_kurtosis,
            "factors": [
                {
                    "name": f.name,
                    "beta": f.beta,
                    "se": f.se,
                    "t_stat": f.t_stat,
                    "p_value": f.p_value,
                    "significant": f.is_significant,
                }
                for f in self.factors
            ],
        }


# ---------------------------------------------------------------------------
# Core regression
# ---------------------------------------------------------------------------

def _run_ff_regression(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    strategy_name: str,
    model: str,
    annualization: int,
) -> AttributionResult:
    """
    OLS regression of excess_returns on factors with Newey-West HAC SEs.


    Args:
        excess_returns: Strategy excess returns (already Rf-adjusted), daily
        factors: Factor returns DataFrame (columns = factor names,
            daily decimal)
        strategy_name: Name for reporting
        model: "FF3" or "FF5"
        annualization: 252 for daily

    Returns:
        AttributionResult
    """
    import statsmodels.api as sm

    y = excess_returns.values
    X = sm.add_constant(factors.values)  # [1, MktRf, SMB, HML, ...]

    ols = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(np.sqrt(len(y)))},
    )

    alpha_daily = float(ols.params[0])
    alpha_annual = float((1 + alpha_daily) ** annualization - 1)
    alpha_se = float(ols.bse[0])
    alpha_t = float(ols.tvalues[0])
    alpha_p = float(ols.pvalues[0])

    factor_names = list(factors.columns)
    factor_loadings = []
    for i, name in enumerate(factor_names, start=1):
        beta = float(ols.params[i])
        se = float(ols.bse[i])
        t = float(ols.tvalues[i])
        p = float(ols.pvalues[i])
        ci = ols.conf_int(alpha=0.05)
        factor_loadings.append(FactorLoadings(
            name=name,
            beta=beta,
            se=se,
            t_stat=t,
            p_value=p,
            ci_lower=float(ci[i, 0]),
            ci_upper=float(ci[i, 1]),
        ))

    resid = pd.Series(ols.resid)
    return AttributionResult(
        strategy=strategy_name,
        alpha_daily=alpha_daily,
        alpha_annual=alpha_annual,
        alpha_se=alpha_se,
        alpha_t_stat=alpha_t,
        alpha_p_value=alpha_p,
        alpha_significant=(alpha_p < 0.05),
        r_squared=float(ols.rsquared),
        adj_r_squared=float(ols.rsquared_adj),
        n_obs=len(y),
        factors=factor_loadings,
        model=model,
        start_date=str(factors.index[0].date()),
        end_date=str(factors.index[-1].date()),
        residual_skew=float(resid.skew()),
        residual_kurtosis=float(resid.kurtosis()),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ff5_attribution(
    strategy_returns: pd.Series,
    strategy_name: str = "strategy",
    model: str = "FF5",
    annualization: int = 252,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    ff_data: Optional[pd.DataFrame] = None,
) -> AttributionResult:
    """
    Fama-French 5-factor attribution for a strategy.

    Aligns strategy returns with FF factor data, subtracts Rf from strategy
    returns to get excess returns, then runs OLS with HAC standard errors.

    Args:
        strategy_returns: Daily strategy return series (NOT excess returns)
        strategy_name: Name for display and reporting
        model: "FF5" (default) or "FF3" (3-factor subset: MktRf, SMB, HML)
        annualization: 252 for daily
        cache_dir: Where to cache Kenneth French data
        ff_data: Pre-loaded FF data (skips download; useful in tests)

    Returns:
        AttributionResult with alpha, factor betas, t-stats, p-values, R²

    Raises:
        RuntimeError: If FF data cannot be downloaded or loaded
        ValueError: If there is insufficient overlapping data
    """
    if model not in ("FF3", "FF5"):
        raise ValueError(f"model must be 'FF3' or 'FF5', got '{model}'")

    factor_cols = FF3_COLUMNS[:3] if model == "FF3" else FF5_COLUMNS[:5]

    # Load FF data
    if ff_data is None:
        ff_data = download_ff5_daily(cache_dir)

    # Align on common dates
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    common_idx = strategy_returns.index.intersection(ff_data.index)

    if len(common_idx) < 30:
        start = strategy_returns.index[0].date()
        end = strategy_returns.index[-1].date()
        raise ValueError(
            f"Only {len(common_idx)} overlapping days between strategy "
            f"and FF data. Need at least 30. Strategy range: "
            f"{start} to {end}"
        )

    strat = strategy_returns.reindex(common_idx)
    rf = ff_data["Rf"].reindex(common_idx)
    factors = ff_data[factor_cols].reindex(common_idx)

    # Excess returns = strategy return - risk-free rate
    excess_returns = strat - rf

    # Drop NaN (gaps in either series)
    valid = excess_returns.notna() & factors.notna().all(axis=1)
    excess_returns = excess_returns[valid]
    factors = factors[valid]

    if len(excess_returns) < 30:
        raise ValueError(
            f"Only {len(excess_returns)} valid observations after dropping NaN"
        )

    return _run_ff_regression(
        excess_returns, factors, strategy_name, model, annualization
    )


def ff5_attribution_batch(
    returns_dict: dict,
    model: str = "FF5",
    annualization: int = 252,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> List[AttributionResult]:
    """
    Run FF5 attribution on multiple strategies (downloads FF data once).

    Args:
        returns_dict: {strategy_name: pd.Series of returns}
        model: "FF3" or "FF5"
        annualization: 252 for daily
        cache_dir: FF data cache directory

    Returns:
        List of AttributionResult, one per strategy
    """
    # Download once, share across strategies
    try:
        ff_data = download_ff5_daily(cache_dir)
    except RuntimeError as e:
        warnings.warn(f"FF data unavailable: {e}")
        return []

    results = []
    for name, returns in returns_dict.items():
        try:
            result = ff5_attribution(
                returns, name, model, annualization,
                ff_data=ff_data,
            )
            results.append(result)
        except Exception as e:
            warnings.warn(f"Attribution failed for {name}: {e}")

    return results


def attribution_summary_df(results: List[AttributionResult]) -> pd.DataFrame:
    """
    Build a summary DataFrame from a list of AttributionResult.

    Columns: strategy, alpha_annual, alpha_t_stat, alpha_p_value,
             alpha_significant, r_squared, + beta per factor.
    Sorted by alpha_annual descending.
    """
    rows = []
    for r in results:
        row = {
            "strategy": r.strategy,
            "model": r.model,
            "alpha_annual": r.alpha_annual,
            "alpha_t": r.alpha_t_stat,
            "alpha_p": r.alpha_p_value,
            "alpha_sig": r.alpha_significant,
            "r2": r.r_squared,
            "n_obs": r.n_obs,
        }
        for f in r.factors:
            row[f"beta_{f.name}"] = f.beta
            row[f"t_{f.name}"] = f.t_stat
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            "alpha_annual", ascending=False
        ).reset_index(drop=True)
    return df


def format_attribution_report(result: AttributionResult) -> str:
    """Format an AttributionResult as Markdown for REPORT.md."""
    sig = "**SIGNIFICANT**" if result.alpha_significant else "not significant"
    lines = [
        f"## Fama-French {result.model} Factor Attribution",
        "",
        f"> Period: {result.start_date} to {result.end_date}"
        f"  ·  n={result.n_obs}"
        f"  ·  Source: Kenneth French Data Library",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Alpha (annualized) | {result.alpha_annual:.4f}"
        f" ({result.alpha_annual:.2%}) |",
        f"| Alpha t-statistic | {result.alpha_t_stat:.3f} |",
        f"| Alpha p-value | {result.alpha_p_value:.4f} |",
        f"| Alpha significant (5%) | {sig} |",
        f"| R² | {result.r_squared:.4f} |",
        f"| Adjusted R² | {result.adj_r_squared:.4f} |",
        f"| Residual skewness | {result.residual_skew:.3f} |",
        f"| Residual excess kurtosis | {result.residual_kurtosis:.3f} |",
        "",
        "**Factor Loadings** (Newey-West HAC standard errors):",
        "",
        "| Factor | Beta | SE | t-stat | p-value | Sig |",
        "|--------|------|----|--------|---------|-----|",
    ]
    for f in result.factors:
        star = "✓" if f.is_significant else ""
        lines.append(
            f"| {f.name} | {f.beta:.4f} | {f.se:.4f} | "
            f"{f.t_stat:.2f} | {f.p_value:.4f} | {star} |"
        )
    lines.append("")
    return "\n".join(lines)
