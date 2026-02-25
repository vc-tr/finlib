"""
Strategy adapter for paper trading replay.

Precomputes target portfolio weights on rebalance dates for factor strategies.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.factors import compute_factor, compute_factors
from src.factors.factors import get_prices_wide
from src.factors.ensemble import combine_factors
from src.factors.portfolio import rebalance_dates, weights_at_rebalance


def get_factor_target_weights(
    df_by_symbol: Dict[str, pd.DataFrame],
    factor: str,
    combo_list: Optional[List[str]] = None,
    combo_method: str = "equal",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    rebalance: str = "M",
    top_k: int = 10,
    bottom_k: int = 10,
    auto_metric: str = "val_ic_ir",
    val_split: float = 0.3,
    shrinkage: float = 0.5,
) -> pd.DataFrame:
    """
    Precompute target weights on rebalance dates for factor strategy.

    Args:
        df_by_symbol: {symbol: OHLCV DataFrame}
        factor: "momentum_12_1" | "reversal_5d" | "lowvol_20d" | "combo"
        combo_list: For combo, list of factor names
        combo_method: "equal" | "ic_weighted" | "ridge" | "sharpe_opt" | "auto" | "auto_robust"
        start, end: Date range subset (None = full)
        rebalance: "D" | "W" | "M"
        top_k, bottom_k: For cross-sectional rank
        auto_metric, val_split, shrinkage: For auto_robust

    Returns:
        DataFrame (date x symbol) of target weights on rebalance dates, ffill between.
    """
    if combo_list is None:
        factor_df = compute_factor(df_by_symbol, factor)
    else:
        factors_dict = compute_factors(df_by_symbol, combo_list)
        prices = get_prices_wide(df_by_symbol)
        fwd_returns = prices.pct_change().shift(-1)
        from src.factors.research import forward_returns

        fwd_returns_dict = forward_returns(prices, horizons=[1, 5, 21])
        cost_bps = 4.0
        combo_kw = dict(
            fwd_returns=fwd_returns,
            fwd_returns_dict=fwd_returns_dict,
            prices=prices,
            top_k=top_k,
            bottom_k=bottom_k,
            rebalance=rebalance,
            cost_bps=cost_bps,
        )
        if combo_method == "auto_robust":
            combo_kw["auto_metric"] = auto_metric
            combo_kw["val_split"] = val_split
            combo_kw["shrinkage"] = shrinkage

        idx = prices.index
        if start is not None:
            idx = idx[idx >= start]
        if end is not None:
            idx = idx[idx <= end]

        if combo_method == "equal" or len(idx) < 30:
            combined, _, _, _ = combine_factors(factors_dict, method="equal")
            factor_df = combined
        else:
            n = max(30, int(len(idx) * 0.7))
            train_slice = slice(idx[0], idx[min(n - 1, len(idx) - 1)])
            combined, _, _, _ = combine_factors(
                factors_dict,
                method=combo_method,
                train_slice=train_slice,
                **combo_kw,
            )
            factor_df = combined

    if factor_df.empty:
        return pd.DataFrame()

    weights_df = weights_at_rebalance(
        factor_df,
        rebalance=rebalance,
        top_k=top_k,
        bottom_k=bottom_k,
        method="zscore",
        long_short=True,
        gross_leverage=1.0,
        max_weight=0.1,
    )

    if start is not None or end is not None:
        s = start if start is not None else weights_df.index.min()
        e = end if end is not None else weights_df.index.max()
        weights_df = weights_df.loc[s:e]

    return weights_df
