"""Pytest fixtures for quant-forecast tests."""

import numpy as np
import pandas as pd
import pytest


def _date_index(n: int, start: str = "2020-01-01", freq: str = "B") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq)


@pytest.fixture
def rising_prices() -> pd.Series:
    """Deterministic rising close prices: 100, 101, 102, ..."""
    n = 20
    return pd.Series(100 + np.arange(n), index=_date_index(n))


@pytest.fixture
def oscillating_prices() -> pd.Series:
    """Deterministic oscillating: 100, 101, 100, 101, ..."""
    n = 20
    arr = 100 + np.array([1, -1] * (n // 2) + ([1] if n % 2 else []))[:n]
    return pd.Series(arr, index=_date_index(n))


@pytest.fixture
def constant_prices() -> pd.Series:
    """Constant close prices."""
    n = 20
    return pd.Series(np.full(n, 100.0), index=_date_index(n))


@pytest.fixture
def ohlcv_rising() -> pd.DataFrame:
    """OHLCV DataFrame with rising prices (deterministic)."""
    n = 20
    idx = _date_index(n)
    close = 100 + np.arange(n)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
