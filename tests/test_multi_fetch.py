# tests/test_multi_fetch.py
"""Tests for multi_fetch with fake cache files."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.multi_fetch import (
    CACHE_DIR,
    _align_dates,
    _cache_path,
    _load_cache,
    _normalize_df,
    _save_cache,
    fetch_many,
)


@pytest.fixture
def tmp_cache(tmp_path):
    """Use temp dir for cache to avoid polluting data/cache."""
    import src.pipeline.multi_fetch as mf

    orig = mf.CACHE_DIR
    mf.CACHE_DIR = tmp_path
    yield tmp_path
    mf.CACHE_DIR = orig


def _make_ohlcv(dates, prefix=""):
    """Build minimal OHLCV DataFrame."""
    n = len(dates)
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000] * n,
        },
        index=pd.to_datetime(dates),
    )


def test_normalize_df():
    df = pd.DataFrame(
        {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
        index=pd.to_datetime(["2025-01-01"]),
    )
    out = _normalize_df(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["open"].iloc[0] == 1


def test_cache_path():
    p = _cache_path("SPY", "5d", "1d")
    assert "SPY" in str(p)
    assert "5d" in str(p)
    assert "1d" in str(p)
    assert p.suffix == ".parquet"


def test_save_and_load_cache(tmp_cache, monkeypatch):
    import src.pipeline.multi_fetch as mf

    monkeypatch.setattr(mf, "CACHE_DIR", tmp_cache)

    df = _make_ohlcv(["2025-01-01", "2025-01-02"])
    _save_cache("FAKE", "5d", "1d", df)

    path = tmp_cache / "FAKE_5d_1d.parquet"
    assert path.exists()

    loaded = _load_cache("FAKE", "5d", "1d")
    assert loaded is not None
    assert len(loaded) == 2
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]


def test_fetch_many_uses_cache(tmp_cache, monkeypatch):
    import src.pipeline.multi_fetch as mf

    monkeypatch.setattr(mf, "CACHE_DIR", tmp_cache)

    # Pre-create fake cache for SPY
    df_spy = _make_ohlcv(["2025-01-01", "2025-01-02", "2025-01-03"])
    _save_cache("SPY", "5d", "1d", df_spy)

    # Mock yfinance to ensure we don't call it for cached symbol
    download_calls = []

    def fake_download(*args, **kwargs):
        download_calls.append(1)
        return pd.DataFrame()

    monkeypatch.setattr("src.pipeline.multi_fetch.yf.download", fake_download)

    result = fetch_many(["SPY"], "5d", "1d", use_cache=True)

    assert "SPY" in result
    assert len(result["SPY"]) == 3
    assert len(download_calls) == 0


def test_fetch_many_skips_failed_symbol(monkeypatch):
    def fake_download(symbols, **kwargs):
        if isinstance(symbols, list):
            symbols = symbols[0] if len(symbols) == 1 else symbols
        if symbols == "INVALID" or (isinstance(symbols, list) and "INVALID" in symbols):
            raise ValueError("Invalid ticker")
        return _make_ohlcv(["2025-01-01"])

    monkeypatch.setattr("src.pipeline.multi_fetch.yf.download", fake_download)

    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr("src.pipeline.multi_fetch.CACHE_DIR", Path(d))
        result = fetch_many(["SPY", "INVALID"], "5d", "1d", use_cache=False)

    assert "SPY" in result
    assert "INVALID" not in result


def test_align_dates_intersection():
    a = _make_ohlcv(["2025-01-01", "2025-01-02", "2025-01-03"])
    b = _make_ohlcv(["2025-01-02", "2025-01-03"])  # subset

    out = _align_dates({"A": a, "B": b}, "intersection")

    assert len(out["A"]) == 2
    assert len(out["B"]) == 2
    assert list(out["A"].index) == list(out["B"].index)


def test_align_dates_union():
    a = _make_ohlcv(["2025-01-01", "2025-01-02"])
    b = _make_ohlcv(["2025-01-02", "2025-01-03"])

    out = _align_dates({"A": a, "B": b}, "union")

    assert len(out["A"]) == 3
    assert len(out["B"]) == 3
    # A should have ffill for 2025-01-03
    assert pd.notna(out["A"].loc[pd.Timestamp("2025-01-03"), "close"])


def test_fetch_many_empty_symbols():
    result = fetch_many([], "5d", "1d")
    assert result == {}


def test_fetch_many_date_align_union(tmp_cache, monkeypatch):
    import src.pipeline.multi_fetch as mf

    monkeypatch.setattr(mf, "CACHE_DIR", tmp_cache)

    df_a = _make_ohlcv(["2025-01-01", "2025-01-02"])
    df_b = _make_ohlcv(["2025-01-02", "2025-01-03"])
    _save_cache("A", "5d", "1d", df_a)
    _save_cache("B", "5d", "1d", df_b)

    result = fetch_many(["A", "B"], "5d", "1d", use_cache=True, date_align="union")

    assert len(result["A"]) == 3
    assert len(result["B"]) == 3
