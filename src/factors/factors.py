"""
Cross-sectional factor computation.

Input: df_by_symbol[str, DataFrame] with 'close' column.
Output: DataFrame index=date, columns=symbol, values=factor.
"""

import numpy as np
import pandas as pd
from typing import Dict


def get_prices_wide(df_by_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack close prices into wide DataFrame (date x symbol)."""
    closes = {}
    for sym, df in df_by_symbol.items():
        if "close" in df.columns:
            closes[sym] = df["close"]
        else:
            closes[sym] = df.iloc[:, 0]
    return pd.DataFrame(closes)


def _align_prices_and_returns(df_by_symbol: Dict[str, pd.DataFrame]) -> tuple:
    """Stack close prices, return (prices_df, returns_df) wide (date x symbol)."""
    price_df = get_prices_wide(df_by_symbol)
    returns_df = price_df.pct_change()
    return price_df, returns_df


def compute_factor(
    df_by_symbol: Dict[str, pd.DataFrame],
    factor: str,
) -> pd.DataFrame:
    """
    Compute factor values per date per symbol.

    Args:
        df_by_symbol: {symbol: OHLCV DataFrame}
        factor: "momentum_12_1" | "reversal_5d" | "lowvol_20d"

    Returns:
        DataFrame index=date, columns=symbol, values=factor (higher = more attractive)
    """
    _, returns = _align_prices_and_returns(df_by_symbol)
    if returns.empty:
        return pd.DataFrame()

    if factor == "momentum_12_1":
        # 12-1 momentum: return from t-252 to t-21 (skip last 21d)
        # (price[t-21] / price[t-252]) - 1, shifted so at t we use value available at t-1
        prices, _ = _align_prices_and_returns(df_by_symbol)
        mom = prices / prices.shift(252) - 1
        mom = mom.shift(21)  # at t, use return that ended at t-21
        return mom

    if factor == "reversal_5d":
        # 5d return * -1 (short-term reversal: recent losers bounce)
        rev = -returns.rolling(5).apply(
            lambda x: (1 + x).prod() - 1 if len(x) == 5 else np.nan,
            raw=False,
        )
        return rev

    if factor == "lowvol_20d":
        # Rolling 20d vol * -1 (low vol = high factor)
        vol = returns.rolling(20).std()
        return -vol

    raise ValueError(f"Unknown factor: {factor}")
