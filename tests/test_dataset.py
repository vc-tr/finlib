"""
Tests for time series cross-validation scheduler.

Tests that block cross-validation prevents lookahead leakage by generating
synthetic sequences and asserting that no test index ever precedes any 
train index in time.
"""

import pytest
import numpy as np
from typing import List, Tuple

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.pipeline.scheduler import (
    block_time_series_split,
    rolling_window_split,
    validate_time_series_split,
    get_split_info
)


class TestBlockTimeSeriesSplit:
    """Test suite for block time series cross-validation."""
    
    def test_no_lookahead_basic(self):
        """Test that no validation index precedes any training index."""
        n_samples = 100
        n_splits = 5
        
        splits = block_time_series_split(n_samples, n_splits, train_size=20, val_size=10)
        
        for i, (train_idx, val_idx) in enumerate(splits):
            # Assert no training index >= any validation index
            assert train_idx.max() < val_idx.min(), (
                f"Lookahead detected in split {i}: "
                f"train max {train_idx.max()} >= val min {val_idx.min()}"
            )
            
            # Assert indices are sorted (time order preserved)
            assert np.all(train_idx[:-1] <= train_idx[1:]), f"Train indices not sorted in split {i}"
            assert np.all(val_idx[:-1] <= val_idx[1:]), f"Val indices not sorted in split {i}"
    
    def test_no_lookahead_with_gap(self):
        """Test lookahead prevention with gap between train and validation."""
        n_samples = 200
        n_splits = 4
        gap = 5
        
        splits = block_time_series_split(n_samples, n_splits, train_size=30, val_size=15, gap=gap)
        
        for i, (train_idx, val_idx) in enumerate(splits):
            # With gap, ensure sufficient separation
            assert train_idx.max() + gap < val_idx.min(), (
                f"Insufficient gap in split {i}: "
                f"train max {train_idx.max()} + gap {gap} >= val min {val_idx.min()}"
            )
    
    def test_expanding_window(self):
        """Test expanding window mode (train_size=None)."""
        n_samples = 150
        n_splits = 3
        
        splits = block_time_series_split(n_samples, n_splits, train_size=None, val_size=20)
        
        # In expanding window, each split should have larger training set
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        
        # Should be non-decreasing (expanding)
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1], (
                f"Training window not expanding: split {i-1} size {train_sizes[i-1]} "
                f"> split {i} size {train_sizes[i]}"
            )
        
        # Validate no lookahead
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min()
    
    def test_rolling_window(self):
        """Test rolling window mode (fixed train_size)."""
        n_samples = 120
        n_splits = 3
        train_size = 30
        
        splits = block_time_series_split(n_samples, n_splits, train_size=train_size, val_size=15)
        
        # All training sets should have same size
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert all(size == train_size for size in train_sizes)
        
        # Validate no lookahead
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min()
    
    def test_synthetic_time_series_data(self):
        """Test with synthetic time series to ensure temporal order."""
        # Create synthetic time series with trend
        n_samples = 200
        timestamps = np.arange(n_samples)
        values = np.cumsum(np.random.randn(n_samples)) + 0.1 * timestamps  # Trend + random walk
        
        # Create splits
        splits = block_time_series_split(n_samples, n_splits=4, train_size=40, val_size=20)
        
        for i, (train_idx, val_idx) in enumerate(splits):
            # Extract actual timestamp values
            train_times = timestamps[train_idx]
            val_times = timestamps[val_idx]
            
            # Assert all training timestamps come before validation timestamps
            assert train_times.max() < val_times.min(), (
                f"Temporal violation in split {i}: "
                f"latest train time {train_times.max()} >= earliest val time {val_times.min()}"
            )
            
            # Assert training data is from past, validation from future
            train_values = values[train_idx]
            val_values = values[val_idx]
            
            # This ensures we're not using future information to predict the past
            assert len(train_values) > 0 and len(val_values) > 0
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            block_time_series_split(n_samples=10, n_splits=5, train_size=50, val_size=10)
        
        # Test invalid parameters
        with pytest.raises(ValueError, match="n_samples must be positive"):
            block_time_series_split(n_samples=0, n_splits=3)
            
        with pytest.raises(ValueError, match="n_splits must be positive"):
            block_time_series_split(n_samples=100, n_splits=0)
            
        with pytest.raises(ValueError, match="gap must be non-negative"):
            block_time_series_split(n_samples=100, n_splits=3, gap=-1)
    
    def test_validate_function(self):
        """Test the validation helper function."""
        # Valid split (no lookahead)
        train_idx = np.array([0, 1, 2, 3, 4])
        val_idx = np.array([5, 6, 7])
        assert validate_time_series_split(train_idx, val_idx) == True
        
        # Invalid split (lookahead)
        train_idx = np.array([0, 1, 2, 5, 6])  # Index 5,6 overlap with val
        val_idx = np.array([4, 5, 6])
        assert validate_time_series_split(train_idx, val_idx) == False
        
        # With gap
        train_idx = np.array([0, 1, 2])
        val_idx = np.array([5, 6, 7])  # Gap of 2 (indices 3,4 missing: max_train=2, min_val=5, gap=5-2-1=2)
        assert validate_time_series_split(train_idx, val_idx, gap=2) == True
        assert validate_time_series_split(train_idx, val_idx, gap=3) == False
    
    def test_split_info(self):
        """Test split information utility function."""
        splits = block_time_series_split(n_samples=100, n_splits=3, train_size=25, val_size=15)
        info = get_split_info(splits)
        
        assert info["n_splits"] == 3
        assert len(info["train_sizes"]) == 3
        assert len(info["val_sizes"]) == 3
        assert all(size == 25 for size in info["train_sizes"])
        assert all(size == 15 for size in info["val_sizes"])
        assert info["avg_train_size"] == 25.0
        assert info["avg_val_size"] == 15.0
        
        # Test empty splits
        info_empty = get_split_info([])
        assert info_empty["n_splits"] == 0


class TestRollingWindowSplit:
    """Test suite for rolling window cross-validation."""
    
    def test_rolling_window_basic(self):
        """Test basic rolling window functionality."""
        n_samples = 50
        window_size = 10
        val_size = 5
        
        splits = list(rolling_window_split(n_samples, window_size, val_size=val_size))
        
        for train_idx, val_idx in splits:
            # Check sizes
            assert len(train_idx) == window_size
            assert len(val_idx) == val_size
            
            # Check no lookahead
            assert train_idx.max() < val_idx.min()
            
            # Check continuity (validation immediately follows training)
            assert val_idx.min() == train_idx.max() + 1
    
    def test_rolling_window_step_size(self):
        """Test rolling window with different step sizes."""
        n_samples = 40
        window_size = 8
        step_size = 3
        val_size = 4
        
        splits = list(rolling_window_split(
            n_samples, window_size, step_size=step_size, val_size=val_size
        ))
        
        # Check that windows advance by step_size
        for i in range(1, len(splits)):
            prev_train_start = splits[i-1][0][0]
            curr_train_start = splits[i][0][0]
            assert curr_train_start == prev_train_start + step_size
    
    def test_rolling_window_no_lookahead(self):
        """Comprehensive lookahead test for rolling window."""
        n_samples = 100
        window_size = 15
        
        splits = list(rolling_window_split(n_samples, window_size, step_size=5, val_size=8))
        
        for i, (train_idx, val_idx) in enumerate(splits):
            # Strict temporal ordering
            assert np.all(train_idx < val_idx.min()), (
                f"Lookahead in rolling window split {i}: "
                f"train indices {train_idx} overlap with val start {val_idx.min()}"
            )
            
            # Validate using our validation function
            assert validate_time_series_split(train_idx, val_idx)


class TestRealWorldScenarios:
    """Test realistic scenarios for financial time series."""
    
    def test_daily_stock_data_simulation(self):
        """Simulate daily stock data validation scenario."""
        # Simulate 2 years of daily data (500 trading days)
        n_samples = 500
        
        # 5-fold cross-validation with 60-day training windows
        # and 20-day validation periods
        splits = block_time_series_split(
            n_samples=n_samples,
            n_splits=5,
            train_size=60,
            val_size=20,
            gap=1  # 1-day gap to prevent immediate dependency
        )
        
        # Validate all splits
        for i, (train_idx, val_idx) in enumerate(splits):
            # Check no lookahead with gap
            assert validate_time_series_split(train_idx, val_idx, gap=1)
            
            # Ensure reasonable sizes
            assert len(train_idx) == 60, f"Split {i} train size: {len(train_idx)}"
            assert len(val_idx) == 20, f"Split {i} val size: {len(val_idx)}"
    
    def test_minute_data_simulation(self):
        """Simulate high-frequency minute data scenario."""
        # Simulate 1 week of minute data (5 days * 6.5 hours * 60 minutes)
        n_samples = 1950
        
        # Rolling window with 30-minute training and 5-minute validation
        window_size = 30
        val_size = 5
        step_size = 10  # Advance by 10 minutes each time
        
        splits = list(rolling_window_split(
            n_samples, window_size, step_size=step_size, val_size=val_size
        ))
        
        # Should have many splits for high-frequency data
        assert len(splits) > 100
        
        # Validate no lookahead in any split
        for train_idx, val_idx in splits:
            assert validate_time_series_split(train_idx, val_idx)
    
    def test_expanding_window_backtest(self):
        """Test expanding window like a typical backtest scenario."""
        # Simulate 3 years of data
        n_samples = 1000
        
        # Start with 6 months minimum, expand by 3 months each split
        splits = block_time_series_split(
            n_samples=n_samples,
            n_splits=6,
            train_size=None,  # Expanding window
            val_size=60,  # 2 months validation
            gap=5  # 1 week gap
        )
        
        # Verify expanding nature
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1], "Training window should expand"
        
        # Verify no lookahead
        for train_idx, val_idx in splits:
            assert validate_time_series_split(train_idx, val_idx, gap=5)


if __name__ == "__main__":
    # Run basic tests when script is executed directly
    test_case = TestBlockTimeSeriesSplit()
    test_case.test_no_lookahead_basic()
    test_case.test_synthetic_time_series_data()
    print("✅ All basic time series cross-validation tests passed!")
    
    # Test with real scenario
    real_test = TestRealWorldScenarios()
    real_test.test_daily_stock_data_simulation()
    print("✅ Real-world scenario tests passed!")
    
    print("🎉 Time series cross-validation implementation validated!") 