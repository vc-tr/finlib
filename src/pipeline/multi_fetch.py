# src/pipeline/multi_fetch.py
"""
Multi-symbol OHLCV fetch with caching and date alignment.
"""
import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _sanitize(s: str) -> str:
    """Make string safe for filenames."""
    return re.sub(r"[^\w\-.]", "_", str(s))


def _cache_path(symbol: str, period: str, interval: str) -> Path:
    return CACHE_DIR / f"{_sanitize(symbol)}_{_sanitize(period)}_{_sanitize(interval)}.parquet"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, rename to lowercase OHLCV."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close",
        "Volume": "volume"
    })
    cols = [c for c in OHLCV_COLS if c in df.columns]
    return df[cols].copy()


def _load_cache(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, period, interval)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return _normalize_df(df)
    except Exception as e:
        logger.warning("Cache read failed for %s: %s", symbol, e)
        return None


def _save_cache(symbol: str, period: str, interval: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, period, interval)
    try:
        df.to_parquet(path, index=True)
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", symbol, e)


def _fetch_one(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    """Fetch single symbol via yfinance."""
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        return _normalize_df(df)
    except Exception as e:
        logger.warning("Fetch failed for %s: %s", symbol, e)
        return None


def _align_dates(
    data: dict[str, pd.DataFrame],
    mode: Literal["intersection", "union"],
) -> dict[str, pd.DataFrame]:
    """Align dates across symbols. Intersection = common dates only; union = union with ffill."""
    if not data:
        return data

    if mode == "intersection":
        common = None
        for df in data.values():
            if df is not None and not df.empty:
                idx = set(df.index)
                common = idx if common is None else common & idx
        if common is None:
            return data
        common_sorted = sorted(common)
        return {s: df.loc[df.index.isin(common_sorted)].reindex(common_sorted) for s, df in data.items()}

    # union: reindex to union of all dates, ffill OHLC, zero-fill volume
    all_dates = set()
    for df in data.values():
        if df is not None and not df.empty:
            all_dates.update(df.index)
    union_sorted = sorted(all_dates)

    out = {}
    for s, df in data.items():
        if df is None or df.empty:
            out[s] = df
            continue
        reindexed = df.reindex(union_sorted)
        for col in ["open", "high", "low", "close"]:
            if col in reindexed.columns:
                reindexed[col] = reindexed[col].ffill()
        if "volume" in reindexed.columns:
            reindexed["volume"] = reindexed["volume"].fillna(0)
        out[s] = reindexed

    return out


def fetch_many(
    symbols: list[str],
    period: str,
    interval: str,
    use_cache: bool = True,
    date_align: Literal["intersection", "union"] = "intersection",
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple symbols efficiently with optional caching.

    Uses yfinance batch download when possible; otherwise falls back to
    per-symbol fetch with caching. Failed symbols are skipped and logged.

    Args:
        symbols: List of ticker symbols.
        period: yfinance period (e.g. "5d", "1mo").
        interval: Bar interval (e.g. "1d", "1m").
        use_cache: If True, read/write parquet cache under data/cache/.
        date_align: "intersection" (default) = common dates only;
                    "union" = union of dates with forward-fill.

    Returns:
        dict[symbol, DataFrame] with OHLCV columns, datetime index.
    """
    symbols = [s.strip().upper() for s in symbols if s and str(s).strip()]
    if not symbols:
        return {}

    result: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    if use_cache:
        for s in symbols:
            cached = _load_cache(s, period, interval)
            if cached is not None and not cached.empty:
                result[s] = cached
            else:
                to_fetch.append(s)
    else:
        to_fetch = list(symbols)

    if to_fetch:
        try:
            df_batch = yf.download(
                to_fetch,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=False,
                threads=True,
                group_by="ticker",
            )
            if df_batch is None or df_batch.empty:
                for s in to_fetch:
                    single = _fetch_one(s, period, interval)
                    if single is not None and not single.empty:
                        result[s] = single
                        if use_cache:
                            _save_cache(s, period, interval, single)
                    else:
                        logger.warning("Skipped symbol (no data): %s", s)
            elif len(to_fetch) == 1:
                df = _normalize_df(df_batch)
                if not df.empty:
                    result[to_fetch[0]] = df
                    if use_cache:
                        _save_cache(to_fetch[0], period, interval, df)
                else:
                    logger.warning("Skipped symbol (empty): %s", to_fetch[0])
            else:
                for s in to_fetch:
                    if isinstance(df_batch.columns, pd.MultiIndex):
                        ticker_cols = df_batch.columns.get_level_values(0)
                        if s in ticker_cols:
                            sub = df_batch[s].copy()
                        else:
                            sub = None
                    else:
                        sub = df_batch
                    if sub is not None and not sub.empty:
                        df = _normalize_df(sub)
                        if not df.empty:
                            result[s] = df
                            if use_cache:
                                _save_cache(s, period, interval, df)
                        else:
                            logger.warning("Skipped symbol (empty): %s", s)
                    else:
                        single = _fetch_one(s, period, interval)
                        if single is not None and not single.empty:
                            result[s] = single
                            if use_cache:
                                _save_cache(s, period, interval, single)
                        else:
                            logger.warning("Skipped symbol (no data): %s", s)
        except Exception as e:
            logger.warning("Batch download failed, falling back to per-symbol: %s", e)
            for s in to_fetch:
                single = _fetch_one(s, period, interval)
                if single is not None and not single.empty:
                    result[s] = single
                    if use_cache:
                        _save_cache(s, period, interval, single)
                else:
                    logger.warning("Skipped symbol: %s", s)

    return _align_dates(result, date_align)
