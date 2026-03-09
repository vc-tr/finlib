# src/pipeline/pipeline.py
import pandas as pd
from typing import Optional


def reindex_and_backfill(df: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
    """
    1. Build a complete DatetimeIndex from min→max timestamp
    2. Reindex df to that index
    3. Forward-fill OHLC columns
    4. Fill missing volume as zero
    5. Drop any leading NaNs (before first real bar)

    Args:
        df: OHLCV DataFrame with datetime index
        freq: Bar frequency ("1min", "5min", etc.). Default "1min".
    """
    # 1. full index
    start, end = df.index.min(), df.index.max()
    bar_freq = freq or "1min"
    full_idx = pd.date_range(start, end, freq=bar_freq)

    # 2. reindex
    df_full = df.reindex(full_idx)

    # After reindexing, filter for regular hours (09:30–16:00, Monday–Friday), will uncommented later
    # df_full = df_full.between_time("09:30", "16:00")
    # df_full = df_full[df_full.index.dayofweek < 5]  # 0=Monday, 4=Friday

    # 3. ffill OHLC
    for col in ["open", "high", "low", "close"]:
        df_full[col] = df_full[col].ffill()

    # 4. zero-fill volume
    df_full["volume"] = df_full["volume"].fillna(0)

    # 4. drop head NaNs if any
    df_clean = df_full.dropna()

    # restore name
    df_clean.index.name = df.index.name

    return df_clean
