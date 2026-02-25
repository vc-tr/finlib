"""
Shared I/O and parsing utilities for scripts and pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd


# Period caps for minute data (Yahoo: 1m ~7d, 5m ~60d)
INTERVAL_PERIOD_CAP = {"1m": "7d", "5m": "60d", "1h": "730d"}


def parse_period_days(period: str) -> int:
    """
    Convert period string (yfinance format) to approximate trading days.

    Supports: 7d, 30d, 1y, 2y, 1mo, 1wk, etc.
    """
    p = period.lower().strip()
    if p.endswith("d"):
        return int(p[:-1])
    if p.endswith("wk"):
        return int(p[:-2]) * 5
    if p.endswith("mo"):
        return int(p[:-2]) * 21
    if p.endswith("y"):
        return int(p[:-1]) * 252
    return 252


def timestamp_for_run() -> str:
    """Return YYYYMMDD_HHMMSS for run directory prefixes."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_dir(base: str = "output/runs", suffix: str = "") -> Path:
    """
    Create output directory: base/suffix, with parents. Returns resolved Path.
    suffix should include timestamp, e.g. "20240225_120000_factors_momentum_M".
    """
    out = Path(base) / suffix if suffix else Path(base)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


class OHLCVFetcher(Protocol):
    """Protocol for fetchers with fetch_ohlcv(symbol, interval, period=...)."""

    def fetch_ohlcv(self, symbol: str, interval: str, period: str = "5d") -> pd.DataFrame:
        ...


def fetch_universe_ohlcv(
    symbols: list[str],
    interval: str,
    period: str,
    fetcher: OHLCVFetcher,
    min_bars: int = 30,
    warn_fn: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for each symbol, return df_by_symbol.

    Skips symbols that fail or have fewer than min_bars rows.
    """
    warn = warn_fn or (lambda msg: None)
    df_by_symbol = {}
    for sym in symbols:
        try:
            df = fetcher.fetch_ohlcv(sym, interval, period=period)
            df = df.dropna()
            if len(df) >= min_bars:
                df_by_symbol[sym] = df
        except Exception as e:
            warn(f"  [WARN] Skip {sym}: {e}")
    return df_by_symbol


def cap_period_for_interval(
    interval: str,
    period: str,
    period_override: bool = False,
) -> str:
    """
    Cap period for minute/intraday intervals per Yahoo limits.

    Returns capped period string, or original if no cap applies.
    """
    if period_override or interval not in INTERVAL_PERIOD_CAP:
        return period
    cap = INTERVAL_PERIOD_CAP[interval]
    cap_days = parse_period_days(cap)
    req_days = parse_period_days(period)
    if req_days > cap_days:
        return cap
    return period
