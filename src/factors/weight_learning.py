"""
Combo weight learning: IC-based, ridge, and Sharpe-optimal.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .research import cross_sectional_ic, summarize_ic


def learn_weights_ic(
    factor_values_train: Dict[str, pd.DataFrame],
    fwd_returns_train: Dict[int, pd.DataFrame],
    horizons: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Weight factors by IC IR (mean/std) across horizons. Nan-safe, fallback to equal if unstable.

    Args:
        factor_values_train: {factor_name: DataFrame (date x symbol)}
        fwd_returns_train: {horizon: DataFrame (date x symbol)} forward returns
        horizons: Which horizons to use (default [1, 5, 21])

    Returns:
        {factor_name: weight} summing to 1
    """
    if horizons is None:
        horizons = [1, 5, 21]
    names = list(factor_values_train.keys())
    if not names:
        return {}

    irs = []
    for n in names:
        fac = factor_values_train[n]
        ir_list = []
        for h in horizons:
            if h not in fwd_returns_train:
                continue
            fwd = fwd_returns_train[h]
            ic_s = cross_sectional_ic(fac, fwd, method="spearman")
            s = summarize_ic(ic_s)
            ir_val = s.get("ir")
            if ir_val is not None and np.isfinite(ir_val) and ir_val > 0:
                ir_list.append(ir_val)
        if ir_list:
            irs.append(np.mean(ir_list))
        else:
            irs.append(0.0)

    irs_arr = np.array(irs, dtype=float)
    irs_arr = np.maximum(irs_arr, 0.0)
    if irs_arr.sum() < 1e-10 or not np.any(np.isfinite(irs_arr)):
        return {n: 1.0 / len(names) for n in names}
    weights = irs_arr / irs_arr.sum()
    return {n: float(w) for n, w in zip(names, weights)}


def learn_weights_ridge(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
) -> np.ndarray:
    """
    Ridge regression: w = (X'X + l2*I)^{-1} X'y. Non-negative, sum to 1.

    Args:
        X: (n_samples, n_features) factor values
        y: (n_samples,) forward returns
        l2: L2 regularization

    Returns:
        weights (n_features,) non-negative, sum to 1
    """
    n_features = X.shape[1]
    XtX = X.T @ X + l2 * np.eye(n_features)
    Xty = X.T @ y
    try:
        coef = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.ones(n_features) / n_features
    coef = np.maximum(coef, 0.0)
    if coef.sum() < 1e-10:
        return np.ones(n_features) / n_features
    return coef / coef.sum()


def learn_weights_sharpe(
    train_portfolio_returns_by_factor: Dict[str, np.ndarray],
    l2: float = 0.5,
) -> Dict[str, float]:
    """
    Weights maximizing mean/vol (Sharpe) with L2 penalty. Sum to 1, clip to [-1,1] then renormalize.

    Args:
        train_portfolio_returns_by_factor: {factor_name: (T,) array} of standalone portfolio returns
        l2: L2 penalty on ||w||^2

    Returns:
        {factor_name: weight} summing to 1
    """
    names = list(train_portfolio_returns_by_factor.keys())
    if not names:
        return {}

    # Stack: R is (T, n_factors)
    R = np.column_stack([train_portfolio_returns_by_factor[n] for n in names])
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    if R.shape[0] < 10:
        return {n: 1.0 / len(names) for n in names}

    m = np.mean(R, axis=0)
    C = np.cov(R.T)
    if C.ndim == 0:
        C = np.array([[C]])
    n_f = len(names)
    C = C + l2 * np.eye(n_f)

    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return {n: 1.0 / len(names) for n in names}

    w = C_inv @ m
    ones = np.ones(n_f)
    denom = ones @ C_inv @ m
    if np.abs(denom) < 1e-12:
        return {n: 1.0 / len(names) for n in names}
    w = w / denom
    w = np.clip(w, -1.0, 1.0)
    if np.abs(w.sum()) < 1e-12:
        return {n: 1.0 / len(names) for n in names}
    w = w / w.sum()
    return {n: float(w[i]) for i, n in enumerate(names)}
