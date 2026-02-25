"""
Multi-factor ensemble: combine factors into a single composite score.
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional


def combine_factors(
    factors: dict[str, pd.DataFrame],
    method: Literal["equal", "ic_weighted", "ridge"] = "equal",
    train_slice: Optional[slice] = None,
    fwd_returns: Optional[pd.DataFrame] = None,
    ridge_alpha: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Combine multiple factors into a single composite factor.

    Args:
        factors: dict[factor_name, DataFrame] with index=date, columns=symbol
        method: "equal" | "ic_weighted" | "ridge"
        train_slice: For ic_weighted/ridge, slice of index to use for training (e.g. slice(None, "2022-12-31"))
        fwd_returns: For ic_weighted/ridge, forward returns (date x symbol). Required for ic_weighted and ridge.
        ridge_alpha: L2 regularization strength for ridge (default 1.0)

    Returns:
        (combined_factor_df, weights_dict)
        - combined_factor_df: index=date, columns=symbol, values=composite factor
        - weights_dict: {"factor_name": weight, ...} for reporting
    """
    if not factors:
        return pd.DataFrame(), {}

    # Align all factors to common index/columns
    names = list(factors.keys())
    dfs = [factors[n] for n in names]
    common_idx = dfs[0].index
    common_cols = dfs[0].columns
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)

    aligned = {}
    for n, df in zip(names, dfs):
        aligned[n] = df.reindex(index=common_idx, columns=common_cols)

    if method == "equal":
        # Z-score each factor cross-sectionally per date, then average
        zscored = {}
        for n, df in aligned.items():
            z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
            z = z.fillna(0)
            zscored[n] = z
        combined = sum(zscored.values()) / len(zscored)
        weights = {n: 1.0 / len(names) for n in names}
        return combined, weights

    if method == "ic_weighted":
        if train_slice is None or fwd_returns is None:
            raise ValueError("ic_weighted requires train_slice and fwd_returns")
        # Compute mean IC per factor on train_slice only (no lookahead)
        train_idx = _slice_to_index(common_idx, train_slice)
        if len(train_idx) < 10:
            # Fallback to equal
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}

        # Compute IC per factor: per-date cross-sectional corr, then nan-safe mean
        ics = []
        for n, df in aligned.items():
            fac = df.loc[train_idx].reindex(columns=fwd_returns.columns)
            fwd = fwd_returns.loc[train_idx].reindex(index=fac.index, columns=fac.columns)
            ic_per_date = []
            for dt in fac.index:
                if dt not in fwd.index:
                    continue
                x = fac.loc[dt].dropna()
                y = fwd.loc[dt].reindex(x.index).dropna()
                valid = x.index.intersection(y.index)
                if len(valid) < 2:
                    ic_per_date.append(0.0)
                    continue
                xv = x.loc[valid].values
                yv = y.loc[valid].values
                # Drop rows with NaN in either series
                mask = np.isfinite(xv) & np.isfinite(yv)
                if mask.sum() < 2:
                    ic_per_date.append(0.0)
                    continue
                xv, yv = xv[mask], yv[mask]
                if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
                    ic_per_date.append(0.0)
                    continue
                ic = np.corrcoef(xv, yv)[0, 1]
                ic_per_date.append(ic if np.isfinite(ic) else 0.0)
            mean_ic = float(np.nanmean(ic_per_date)) if ic_per_date else 0.0
            ics.append(mean_ic if np.isfinite(mean_ic) else 0.0)

        # Weight by |IC| (or IC if positive), avoid negative/zero
        ics_arr = np.array(ics)
        ics_arr = np.where(np.isfinite(ics_arr) & (ics_arr > 0), ics_arr, 0.0)
        if ics_arr.sum() < 1e-10 or not np.any(np.isfinite(ics_arr)):
            weights = {n: 1.0 / len(names) for n in names}
        else:
            weights = {n: float(w) for n, w in zip(names, ics_arr / ics_arr.sum())}

        zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
        combined = sum(weights[n] * zscored[n] for n in names)
        return combined, weights

    if method == "ridge":
        if train_slice is None or fwd_returns is None:
            raise ValueError("ridge requires train_slice and fwd_returns")
        train_idx = _slice_to_index(common_idx, train_slice)
        if len(train_idx) < 30:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}

        # Stack: rows = (date, symbol), cols = factor values, y = fwd return
        X_list = []
        y_list = []
        for dt in train_idx:
            if dt not in fwd_returns.index:
                continue
            for sym in common_cols:
                if sym not in fwd_returns.columns:
                    continue
                fwd_val = fwd_returns.loc[dt, sym]
                if not np.isfinite(fwd_val):
                    continue
                row = [aligned[n].loc[dt, sym] for n in names]
                if any(not np.isfinite(v) for v in row):
                    continue
                X_list.append(row)
                y_list.append(fwd_val)

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)
        if len(X) < 20:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}

        # Ridge: (X'X + alpha*I)^{-1} X' y
        XtX = X.T @ X + ridge_alpha * np.eye(len(names))
        Xty = X.T @ y
        try:
            coef = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            coef = np.ones(len(names)) / len(names)
        # Normalize to sum to 1 (for interpretability)
        coef = np.maximum(coef, 0)
        if coef.sum() < 1e-10:
            coef = np.ones(len(names)) / len(names)
        else:
            coef = coef / coef.sum()
        weights = {n: float(c) for n, c in zip(names, coef)}

        zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
        combined = sum(weights[n] * zscored[n] for n in names)
        return combined, weights

    raise ValueError(f"Unknown method: {method}")


def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score cross-sectionally per date."""
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
    return z.fillna(0)


def _slice_to_index(idx: pd.Index, s: slice) -> pd.Index:
    """Convert slice to index subset (supports datetime start/stop)."""
    if s.start is None and s.stop is None:
        return idx
    start, stop = s.start, s.stop
    if start is not None and not isinstance(start, (int, type(None))):
        start = pd.Timestamp(start)
    if stop is not None and not isinstance(stop, (int, type(None))):
        stop = pd.Timestamp(stop)
    sub = idx
    if start is not None:
        sub = sub[sub >= start]
    if stop is not None:
        sub = sub[sub <= stop]
    return sub
