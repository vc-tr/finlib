"""
Factor research metrics: forward returns, IC/IR, decay.
"""

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd


def forward_returns(
    prices_wide: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Compute forward returns for each horizon.

    Forward return at date t for horizon h: (price[t+h] / price[t]) - 1.

    Args:
        prices_wide: DataFrame (date x symbol) of close prices
        horizons: List of forward horizons in bars (default [1, 5, 21])

    Returns:
        Dict mapping horizon -> DataFrame (date x symbol) of forward returns
    """
    if horizons is None:
        horizons = [1, 5, 21]
    out: Dict[int, pd.DataFrame] = {}
    for h in horizons:
        fwd = prices_wide.shift(-h) / prices_wide - 1
        out[h] = fwd
    return out


def cross_sectional_ic(
    factor_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.Series:
    """
    Compute cross-sectional information coefficient over time.

    At each date, correlation between factor values and forward returns across symbols.

    Robustness:
    - Drop NaNs in either series
    - Require >= 5 symbols that date
    - If variance is 0 (constant factor or returns), return NaN for that date

    Args:
        factor_df: DataFrame (date x symbol) of factor values
        fwd_ret_df: DataFrame (date x symbol) of forward returns (same horizon)
        method: "spearman" (default) or "pearson"

    Returns:
        Series of IC per date (index=date)
    """
    common_idx = factor_df.index.intersection(fwd_ret_df.index)
    common_cols = factor_df.columns.intersection(fwd_ret_df.columns)
    f = factor_df.reindex(index=common_idx, columns=common_cols)
    r = fwd_ret_df.reindex(index=common_idx, columns=common_cols)

    ic_list = []
    for t in common_idx:
        x = f.loc[t].dropna()
        y = r.loc[t].reindex(x.index).dropna()
        valid = x.index.intersection(y.index)
        if len(valid) < 5:
            ic_list.append((t, np.nan))
            continue
        xv = x.loc[valid].values.astype(float)
        yv = y.loc[valid].values.astype(float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        if mask.sum() < 5:
            ic_list.append((t, np.nan))
            continue
        xv, yv = xv[mask], yv[mask]
        if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
            ic_list.append((t, np.nan))
            continue
        if method == "spearman":
            ic = np.corrcoef(
                pd.Series(xv).rank().values,
                pd.Series(yv).rank().values,
            )[0, 1]
        else:
            ic = np.corrcoef(xv, yv)[0, 1]
        ic_list.append((t, ic if np.isfinite(ic) else np.nan))

    return pd.Series(dict(ic_list)).sort_index()


def information_coefficient(
    factor_df: pd.DataFrame,
    fwd_ret_df: pd.DataFrame,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.Series:
    """Alias for cross_sectional_ic (backward compatibility)."""
    return cross_sectional_ic(factor_df, fwd_ret_df, method=method)


def summarize_ic(ic_series: pd.Series) -> Dict[str, float]:
    """
    Summarize IC time series.

    Handles n=0 gracefully (all NaN or empty).

    Returns:
        dict with mean_ic, std_ic, ir (information ratio), t_stat, n
    """
    ic = ic_series.dropna()
    n = len(ic)
    if n == 0:
        return {"mean_ic": np.nan, "std_ic": np.nan, "ir": np.nan, "t_stat": np.nan, "n": 0}
    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    if std_ic > 1e-10:
        ir = mean_ic / std_ic
        t_stat = mean_ic / (std_ic / np.sqrt(n)) if n > 1 else np.nan
    else:
        ir = 0.0 if mean_ic == 0 else np.nan
        t_stat = np.nan  # n==1 or zero variance
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ir": ir,
        "t_stat": t_stat,
        "n": n,
    }
