"""
Cross-sectional ranking: factor values -> portfolio weights.
"""

import pandas as pd


def cross_sectional_rank(
    factor_df: pd.DataFrame,
    top_k: int = 10,
    bottom_k: int = 10,
    method: str = "zscore",
    long_short: bool = True,
    gross_leverage: float = 1.0,
    max_weight: float = 0.1,
) -> pd.DataFrame:
    """
    Convert factor values to portfolio weights per date.

    Args:
        factor_df: index=date, columns=symbol, values=factor
        top_k: Number of long positions (highest factor)
        bottom_k: Number of short positions (lowest factor)
        method: "zscore" (rank by z-score) or "quantile"
        long_short: If True, long top + short bottom; else long top only
        gross_leverage: Sum of |weights| (1.0 = dollar neutral if long_short)
        max_weight: Cap |weight| per symbol

    Returns:
        DataFrame index=date, columns=symbol, values=weight
    """
    weights = pd.DataFrame(0.0, index=factor_df.index, columns=factor_df.columns)

    for dt in factor_df.index:
        row = factor_df.loc[dt].dropna()
        if len(row) < top_k + (bottom_k if long_short else 0):
            continue

        if method == "zscore":
            mu, sig = row.mean(), row.std()
            if sig < 1e-10:
                scores = pd.Series(0.0, index=row.index)
            else:
                scores = (row - mu) / sig
        else:
            scores = row.rank(pct=True)

        # Long top_k, short bottom_k
        top = scores.nlargest(top_k)
        for sym in top.index:
            weights.loc[dt, sym] = 1.0 / top_k

        if long_short and bottom_k > 0:
            bot = scores.nsmallest(bottom_k)
            for sym in bot.index:
                weights.loc[dt, sym] = -1.0 / bottom_k

    # Scale to gross_leverage
    gross = weights.abs().sum(axis=1)
    gross = gross.replace(0, 1)
    weights = weights.div(gross, axis=0) * gross_leverage

    # Cap max weight
    weights = weights.clip(-max_weight, max_weight)
    gross = weights.abs().sum(axis=1)
    gross = gross.replace(0, 1)
    weights = weights.div(gross, axis=0) * gross_leverage

    return weights
