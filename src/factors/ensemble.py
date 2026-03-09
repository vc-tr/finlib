"""
Multi-factor ensemble: combine factors into a single composite score.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Literal, Optional

from .portfolio import weights_at_rebalance
from .research import cross_sectional_ic, forward_returns, summarize_ic
from .weight_learning import apply_shrinkage, learn_weights_ic, learn_weights_ridge, learn_weights_sharpe


def combine_factors(
    factors: dict[str, pd.DataFrame],
    method: Literal["equal", "ic_weighted", "ridge", "sharpe_opt", "auto", "auto_robust"] = "equal",
    train_slice: Optional[slice] = None,
    fwd_returns: Optional[pd.DataFrame] = None,
    fwd_returns_dict: Optional[Dict[int, pd.DataFrame]] = None,
    prices: Optional[pd.DataFrame] = None,
    top_k: int = 10,
    bottom_k: int = 10,
    rebalance: str = "M",
    ridge_alpha: float = 1.0,
    sharpe_l2: float = 0.5,
    cost_bps: float = 4.0,
    auto_metric: Literal["val_sharpe", "val_ic_ir"] = "val_ic_ir",
    val_split: float = 0.3,
    shrinkage: float = 0.5,
    val_ic_horizon: int = 1,
) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame], dict[str, Any]]:
    """
    Combine multiple factors into a single composite factor.

    Args:
        factors: dict[factor_name, DataFrame] with index=date, columns=symbol
        method: "equal" | "ic_weighted" | "ridge" | "sharpe_opt" | "auto"
        train_slice: For train-based methods, slice of index for training
        fwd_returns: Forward returns h=1 (date x symbol). For ridge.
        fwd_returns_dict: {horizon: DataFrame} for ic_weighted (default from prices if available)
        prices: For sharpe_opt/auto, prices (date x symbol)
        top_k, bottom_k, rebalance: For sharpe_opt/auto portfolio construction
        ridge_alpha: L2 for ridge (default 1.0)
        sharpe_l2: L2 for sharpe_opt (default 0.5)
        cost_bps: Cost in bps for auto train Sharpe (default 4.0)

    Returns:
        (combined_factor_df, weights_dict, zscored_dict, meta)
        - meta: {"selected_method": str} for auto; else {}
    """
    meta: dict[str, Any] = {}
    if not factors:
        return pd.DataFrame(), {}, {}, meta

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
        return combined, weights, zscored, meta

    if method == "ic_weighted":
        if train_slice is None:
            raise ValueError("ic_weighted requires train_slice")
        train_idx = _slice_to_index(common_idx, train_slice)
        if len(train_idx) < 10:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta

        # Build fwd_returns_dict for learn_weights_ic (IC IR by horizon)
        if fwd_returns_dict is None and prices is not None:
            fwd_returns_dict = forward_returns(prices, horizons=[1, 5, 21])
        elif fwd_returns_dict is None and fwd_returns is not None:
            fwd_returns_dict = {1: fwd_returns}

        if not fwd_returns_dict:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta

        fac_train = {n: df.loc[train_idx] for n, df in aligned.items()}
        fwd_train = {h: fwd.reindex(index=train_idx) for h, fwd in fwd_returns_dict.items()}
        weights = learn_weights_ic(fac_train, fwd_train, horizons=[1, 5, 21])
        zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
        combined = sum(weights[n] * zscored[n] for n in names)
        return combined, weights, zscored, meta

    if method == "ridge":
        if train_slice is None or fwd_returns is None:
            raise ValueError("ridge requires train_slice and fwd_returns")
        train_idx = _slice_to_index(common_idx, train_slice)
        if len(train_idx) < 30:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta

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
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta

        coef = learn_weights_ridge(X, y, l2=ridge_alpha)
        weights = {n: float(c) for n, c in zip(names, coef)}
        zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
        combined = sum(weights[n] * zscored[n] for n in names)
        return combined, weights, zscored, meta

    if method == "sharpe_opt":
        if train_slice is None or prices is None:
            raise ValueError("sharpe_opt requires train_slice and prices")
        train_idx = _slice_to_index(common_idx, train_slice)
        if len(train_idx) < 30:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta

        returns_by_factor = _train_portfolio_returns_per_factor(
            aligned, names, train_idx, prices, top_k, bottom_k, rebalance,
        )
        if not returns_by_factor:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta
        weights = learn_weights_sharpe(returns_by_factor, l2=sharpe_l2)
        zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
        combined = sum(weights[n] * zscored[n] for n in names)
        return combined, weights, zscored, meta

    if method == "auto":
        if train_slice is None or prices is None:
            raise ValueError("auto requires train_slice and prices")
        train_idx = _slice_to_index(common_idx, train_slice)
        if len(train_idx) < 30:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta

        if fwd_returns_dict is None:
            fwd_returns_dict = forward_returns(prices, horizons=[1, 5, 21])
        if fwd_returns is None:
            fwd_returns = fwd_returns_dict.get(1, prices.pct_change().shift(-1))

        candidates = ["equal", "ic_weighted", "ridge", "sharpe_opt"]
        best_sharpe = -np.inf
        best_weights = {n: 1.0 / len(names) for n in names}
        best_method = "equal"

        for cand in candidates:
            try:
                if cand == "equal":
                    c, w, z = _combine_one(aligned, names, "equal", None)
                elif cand == "ic_weighted":
                    c, w, z = _combine_one(
                        aligned, names, "ic_weighted", train_idx,
                        fwd_returns_dict=fwd_returns_dict,
                    )
                elif cand == "ridge":
                    c, w, z = _combine_one(
                        aligned, names, "ridge", train_idx,
                        fwd_returns=fwd_returns,
                        ridge_alpha=ridge_alpha,
                    )
                else:
                    c, w, z = _combine_one(
                        aligned, names, "sharpe_opt", train_idx,
                        prices=prices, top_k=top_k, bottom_k=bottom_k,
                        rebalance=rebalance, sharpe_l2=sharpe_l2,
                    )
                train_ret = _train_portfolio_returns(
                    c, train_idx, prices, top_k, bottom_k, rebalance, cost_bps,
                )
                sharpe = _sharpe(train_ret)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = w
                    best_method = cand
            except Exception:
                continue

        meta["selected_method"] = best_method
        zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
        combined = sum(best_weights[n] * zscored[n] for n in names)
        return combined, best_weights, zscored, meta

    if method == "auto_robust":
        if train_slice is None or prices is None:
            raise ValueError("auto_robust requires train_slice and prices")
        train_idx = _slice_to_index(common_idx, train_slice)
        if len(train_idx) < 30:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta

        if fwd_returns_dict is None:
            fwd_returns_dict = forward_returns(prices, horizons=[1, 5, 21])
        if fwd_returns is None:
            fwd_returns = fwd_returns_dict.get(1, prices.pct_change().shift(-1))

        # Chronological split: train_sub = first (1-val_split), val_sub = last val_split
        n_train = len(train_idx)
        n_val = max(5, int(n_train * val_split))
        n_train_sub = n_train - n_val
        if n_train_sub < 20:
            zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
            combined = sum(zscored.values()) / len(zscored)
            return combined, {n: 1.0 / len(names) for n in names}, zscored, meta

        train_sub_idx = train_idx[:n_train_sub]
        val_sub_idx = train_idx[-n_val:]

        candidates = ["equal", "ic_weighted", "ridge", "sharpe_opt"]
        best_score = -np.inf
        best_weights_raw = {n: 1.0 / len(names) for n in names}
        best_method = "equal"

        for cand in candidates:
            try:
                if cand == "equal":
                    c, w, z = _combine_one(aligned, names, "equal", None)
                elif cand == "ic_weighted":
                    c, w, z = _combine_one(
                        aligned, names, "ic_weighted", train_sub_idx,
                        fwd_returns_dict=fwd_returns_dict,
                    )
                elif cand == "ridge":
                    c, w, z = _combine_one(
                        aligned, names, "ridge", train_sub_idx,
                        fwd_returns=fwd_returns,
                        ridge_alpha=ridge_alpha,
                    )
                else:
                    c, w, z = _combine_one(
                        aligned, names, "sharpe_opt", train_sub_idx,
                        prices=prices, top_k=top_k, bottom_k=bottom_k,
                        rebalance=rebalance, sharpe_l2=sharpe_l2,
                    )

                if auto_metric == "val_sharpe":
                    val_ret = _train_portfolio_returns(
                        c, val_sub_idx, prices, top_k, bottom_k, rebalance, cost_bps,
                    )
                    score = _sharpe(val_ret)
                else:
                    fwd_val = fwd_returns_dict.get(val_ic_horizon, fwd_returns)
                    fwd_val = fwd_val.reindex(val_sub_idx)
                    c_val = c.reindex(val_sub_idx).dropna(how="all")
                    common_val = c_val.index.intersection(fwd_val.index)
                    if len(common_val) < 5:
                        score = -np.inf
                    else:
                        ic_s = cross_sectional_ic(
                            c.reindex(common_val),
                            fwd_val.reindex(common_val),
                            method="spearman",
                        )
                        s = summarize_ic(ic_s)
                        ir = s.get("ir")
                        score = ir if ir is not None and np.isfinite(ir) else -np.inf

                if score > best_score:
                    best_score = score
                    best_weights_raw = w
                    best_method = cand
            except Exception:
                continue

        # Refit on full train window using best method
        try:
            if best_method == "equal":
                _, best_weights_raw, _ = _combine_one(aligned, names, "equal", None)
            elif best_method == "ic_weighted":
                _, best_weights_raw, _ = _combine_one(
                    aligned, names, "ic_weighted", train_idx,
                    fwd_returns_dict=fwd_returns_dict,
                )
            elif best_method == "ridge":
                _, best_weights_raw, _ = _combine_one(
                    aligned, names, "ridge", train_idx,
                    fwd_returns=fwd_returns,
                    ridge_alpha=ridge_alpha,
                )
            else:
                _, best_weights_raw, _ = _combine_one(
                    aligned, names, "sharpe_opt", train_idx,
                    prices=prices, top_k=top_k, bottom_k=bottom_k,
                    rebalance=rebalance, sharpe_l2=sharpe_l2,
                )
        except Exception:
            pass

        weights_final = apply_shrinkage(best_weights_raw, shrinkage)
        meta["selected_method"] = best_method
        meta["val_score"] = float(best_score)
        meta["weights_before_shrink"] = {k: float(v) for k, v in best_weights_raw.items()}
        meta["weights_final"] = {k: float(v) for k, v in weights_final.items()}

        zscored = {n: _zscore_cs(df) for n, df in aligned.items()}
        combined = sum(weights_final[n] * zscored[n] for n in names)
        return combined, weights_final, zscored, meta

    raise ValueError(f"Unknown method: {method}")


def _train_portfolio_returns_per_factor(
    aligned: dict[str, pd.DataFrame],
    names: list[str],
    train_idx: pd.Index,
    prices: pd.DataFrame,
    top_k: int,
    bottom_k: int,
    rebalance: str,
) -> dict[str, np.ndarray]:
    """Compute standalone portfolio return series per factor on train window."""
    from .portfolio import apply_rebalance_costs

    out = {}
    prices_train = prices.reindex(index=train_idx).ffill().bfill()
    common_idx = None
    for n in names:
        fac = aligned[n].loc[train_idx]
        w = weights_at_rebalance(fac, rebalance=rebalance, top_k=top_k, bottom_k=bottom_k)
        w = w.reindex(prices_train.index).ffill().fillna(0)
        ret = prices_train.pct_change()
        w_held = w.shift(1).fillna(0)
        syms = w_held.columns.intersection(ret.columns)
        w_held = w_held.reindex(columns=syms).fillna(0)
        ret = ret.reindex(columns=syms).fillna(0)
        port_ret = (w_held * ret).sum(axis=1)
        port_ret = apply_rebalance_costs(port_ret, w, cost_bps=4.0)
        out[n] = port_ret
        if common_idx is None:
            common_idx = port_ret.dropna().index
        else:
            common_idx = common_idx.intersection(port_ret.dropna().index)
    if common_idx is None or len(common_idx) < 5:
        return {}
    return {n: out[n].reindex(common_idx).fillna(0).values for n in names}


def _train_portfolio_returns(
    combined: pd.DataFrame,
    train_idx: pd.Index,
    prices: pd.DataFrame,
    top_k: int,
    bottom_k: int,
    rebalance: str,
    cost_bps: float,
) -> pd.Series:
    """Portfolio returns on train window for combined factor, net of costs."""
    from .portfolio import apply_rebalance_costs

    fac = combined.loc[train_idx]
    w = weights_at_rebalance(fac, rebalance=rebalance, top_k=top_k, bottom_k=bottom_k)
    prices_train = prices.reindex(index=train_idx).ffill().bfill()
    w = w.reindex(prices_train.index).ffill().fillna(0)
    ret = prices_train.pct_change()
    w_held = w.shift(1).fillna(0)
    common = w_held.index.intersection(ret.index).intersection(prices_train.columns)
    w_held = w_held.reindex(columns=common).fillna(0)
    ret = ret.reindex(columns=common).fillna(0)
    port_ret = (w_held * ret).sum(axis=1)
    return apply_rebalance_costs(port_ret, w, cost_bps=cost_bps)


def _sharpe(returns: pd.Series, ann: float = 252.0) -> float:
    """Annualized Sharpe. Returns -inf if empty or zero vol."""
    r = returns.dropna()
    if len(r) < 5:
        return -np.inf
    vol = r.std()
    if vol < 1e-12:
        return -np.inf
    return float(r.mean() / vol * np.sqrt(ann))


def _combine_one(
    aligned: dict[str, pd.DataFrame],
    names: list[str],
    method: str,
    train_idx: Optional[pd.Index],
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict, dict]:
    """Call combine_factors for one method, return (combined, weights, zscored) without meta."""
    ts = slice(train_idx[0], train_idx[-1]) if train_idx is not None and len(train_idx) > 0 else None
    c, w, z, _ = combine_factors(aligned, method=method, train_slice=ts, **kwargs)
    return c, w, z


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
