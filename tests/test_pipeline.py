# tests/test_pipeline.py
import pandas as pd
from datetime import datetime, timedelta
import pytest

from src.pipeline.pipeline import reindex_and_backfill

@pytest.fixture
def sample_df():
    # Create a DataFrame with a missing minute between 09:30 and 09:32
    idx = [
        datetime(2025, 6, 20, 9, 30),
        datetime(2025, 6, 20, 9, 32),
    ]
    df = pd.DataFrame({
        "open": [100, 102],
        "high": [101, 103],
        "low":  [99,  101],
        "close":[100.5, 102.5],
        "volume":[10,    15],
    }, index=pd.DatetimeIndex(idx))
    return df

def test_reindex_inserts_missing_minute(sample_df):
    out = reindex_and_backfill(sample_df)
    # Should have 3 rows: 09:30, 09:31, 09:32
    assert len(out) == 3
    expected_idx = pd.date_range(
        sample_df.index.min(), sample_df.index.max(), freq="1min"
    )
    assert (out.index == expected_idx).all()

def test_backfill_values(sample_df):
    out = reindex_and_backfill(sample_df)
    # Row at 09:31 should carry forward 09:30 values for OHLC, zero volume
    row = out.loc["2025-06-20 09:31"]
    assert row["open"] == 100
    assert row["high"] == 101
    assert row["low"]  == 99
    assert row["close"]== 100.5
    assert row["volume"] == 0

def test_preserves_original_rows(sample_df):
    out = reindex_and_backfill(sample_df)
    # 09:30 and 09:32 should remain exactly the same
    assert out.loc["2025-06-20 09:30", "open"] == 100
    assert out.loc["2025-06-20 09:32", "close"] == 102.5