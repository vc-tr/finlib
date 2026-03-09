"""
Risk estimation for factor portfolios.

Rolling beta via covariance/variance (no new deps).
"""

import numpy as np
import pandas as pd


def estimate_beta(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """
    Rolling beta of each asset vs market.

    beta_i = cov(r_i, r_m) / var(r_m) over rolling window.

    Args:
        asset_returns: Wide DataFrame (date x symbol)
        market_returns: Market return series (e.g. SPY)
        window: Rolling window in bars

    Returns:
        DataFrame (date x symbol) of rolling betas
    """
    common_idx = asset_returns.index.intersection(market_returns.index)
    ar = asset_returns.reindex(common_idx).fillna(0)
    mr = market_returns.reindex(common_idx).fillna(0)

    var_m = mr.rolling(window).var()
    var_m = var_m.replace(0, np.nan)
    betas = pd.DataFrame(index=ar.index, columns=ar.columns, dtype=float)

    for col in ar.columns:
        cov = ar[col].rolling(window).cov(mr)
        betas[col] = cov / var_m

    return betas.fillna(0)


def rolling_portfolio_beta(
    port_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Rolling beta of portfolio vs market: beta_p(t) = cov(r_p, r_m) / var(r_m) over window.

    Args:
        port_returns: Portfolio return series
        market_returns: Market return series (e.g. SPY close-to-close)
        window: Rolling window in bars

    Returns:
        Series of rolling portfolio beta
    """
    common_idx = port_returns.index.intersection(market_returns.index)
    pr = port_returns.reindex(common_idx).fillna(0)
    mr = market_returns.reindex(common_idx).fillna(0)

    var_m = mr.rolling(window).var().replace(0, np.nan)
    cov_pm = pr.rolling(window).cov(mr)
    beta_p = cov_pm / var_m
    return beta_p.fillna(0)
