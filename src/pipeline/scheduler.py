"""
Time Series Cross-Validation Scheduler

Tasks:
	1.	Design a block cross-validation splitter that prevents lookahead leakage (e.g., rolling-window with no overlap).
	2.	Implement it in src/pipeline/scheduler.py as a function that, given N samples and k folds, returns train/val indices.
	3.	In tests/test_dataset.py, generate a synthetic sequence and assert that no test index ever precedes any train index in time.
	4.	Update dataset.py to accept a splitter argument and integrate your function.
"""

import numpy as np
from typing import List, Tuple, Generator, Optional
import warnings


def block_time_series_split(
    n_samples: int, 
    n_splits: int,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    gap: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create block cross-validation splits for time series data that prevent lookahead leakage.
    
    This function implements a rolling window approach where:
    - Training data always precedes validation data in time
    - No overlap between consecutive validation sets
    - Optional gap between train and validation to prevent immediate temporal dependency
    
    Args:
        n_samples: Total number of samples in the dataset
        n_splits: Number of cross-validation splits to create
        train_size: Size of training window. If None, uses expanding window
        val_size: Size of validation window. If None, uses remaining data after train
        gap: Number of samples to skip between train and validation sets (default: 0)
        
    Returns:
        List of (train_indices, val_indices) tuples
        
    Raises:
        ValueError: If parameters result in insufficient data for splits
        
    Example:
        >>> n_samples = 100
        >>> splits = block_time_series_split(n_samples, n_splits=3, train_size=30, val_size=10)
        >>> for i, (train_idx, val_idx) in enumerate(splits):
        ...     print(f"Split {i}: Train [{train_idx[0]}:{train_idx[-1]}], Val [{val_idx[0]}:{val_idx[-1]}]")
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
        
    if gap < 0:
        raise ValueError("gap must be non-negative")
    
    # If val_size not specified, calculate a reasonable default
    if val_size is None:
        # Reserve enough data for all validation sets plus gaps
        min_data_needed = n_splits * (1 + gap)  # minimum 1 sample per val set + gaps
        if train_size is None:
            val_size = max(1, (n_samples - min_data_needed) // (n_splits * 2))
        else:
            remaining_after_train = n_samples - train_size - gap
            val_size = max(1, remaining_after_train // n_splits)
    
    # If train_size not specified, use expanding window
    if train_size is None:
        use_expanding_window = True
        min_train_size = max(1, n_samples // (n_splits + 2))  # Reasonable minimum
    else:
        use_expanding_window = False
        min_train_size = train_size
    
    # Validate we have enough data
    min_required = min_train_size + gap + val_size * n_splits
    if min_required > n_samples:
        raise ValueError(
            f"Insufficient data: need at least {min_required} samples "
            f"but got {n_samples}. Try reducing n_splits, val_size, or gap."
        )
    
    splits = []
    
    for i in range(n_splits):
        # Calculate validation start position
        if use_expanding_window:
            # Expanding window: train size grows with each split
            val_start = min_train_size + gap + (val_size * i)
        else:
            # Rolling window: fixed train size
            val_start = train_size + gap + (val_size * i)
        
        val_end = val_start + val_size
        
        # Check if we have enough data for this split
        if val_end > n_samples:
            warnings.warn(
                f"Split {i} exceeds data bounds. Stopping at {i} splits instead of {n_splits}."
            )
            break
            
        # Calculate training indices
        if use_expanding_window:
            train_start = 0
            train_end = val_start - gap
        else:
            train_start = val_start - gap - train_size
            train_end = val_start - gap
            
        # Ensure train_start is not negative
        train_start = max(0, train_start)
        
        # Create index arrays
        train_indices = np.arange(train_start, train_end)
        val_indices = np.arange(val_start, val_end)
        
        # Validate no lookahead
        if len(train_indices) > 0 and len(val_indices) > 0:
            if train_indices.max() >= val_indices.min() - gap:
                raise ValueError(
                    f"Lookahead detected in split {i}: "
                    f"train max index {train_indices.max()} >= "
                    f"val min index {val_indices.min()} - gap {gap}"
                )
        
        splits.append((train_indices, val_indices))
    
    if len(splits) == 0:
        raise ValueError("No valid splits could be created with the given parameters")
    
    return splits


def rolling_window_split(
    n_samples: int,
    window_size: int,
    step_size: int = 1,
    val_size: int = 1
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate rolling window splits for time series data.
    
    This creates a sliding window of training data with validation data
    immediately following each training window.
    
    Args:
        n_samples: Total number of samples
        window_size: Size of the training window
        step_size: Step size for sliding the window (default: 1)
        val_size: Size of validation set after each window (default: 1)
        
    Yields:
        Tuple of (train_indices, val_indices)
        
    Example:
        >>> for train_idx, val_idx in rolling_window_split(100, window_size=20, step_size=5):
        ...     print(f"Train: [{train_idx[0]}:{train_idx[-1]}], Val: [{val_idx[0]}:{val_idx[-1]}]")
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if val_size <= 0:
        raise ValueError("val_size must be positive")
        
    start = 0
    while start + window_size + val_size <= n_samples:
        train_end = start + window_size
        val_start = train_end
        val_end = val_start + val_size
        
        train_indices = np.arange(start, train_end)
        val_indices = np.arange(val_start, val_end)
        
        yield train_indices, val_indices
        
        start += step_size


def validate_time_series_split(
    train_indices: np.ndarray, 
    val_indices: np.ndarray,
    gap: int = 0
) -> bool:
    """
    Validate that a train/validation split prevents lookahead leakage.
    
    Args:
        train_indices: Training data indices
        val_indices: Validation data indices  
        gap: Expected gap between train and validation
        
    Returns:
        bool: True if split is valid (no lookahead), False otherwise
    """
    if len(train_indices) == 0 or len(val_indices) == 0:
        return True
        
    # Check that all training indices come before validation indices (accounting for gap)
    max_train_idx = train_indices.max()
    min_val_idx = val_indices.min()
    
    return max_train_idx + gap < min_val_idx


def get_split_info(splits: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    Get information about the cross-validation splits.
    
    Args:
        splits: List of (train_indices, val_indices) tuples
        
    Returns:
        dict: Summary information about the splits
    """
    if not splits:
        return {"n_splits": 0}
        
    train_sizes = [len(train_idx) for train_idx, _ in splits]
    val_sizes = [len(val_idx) for _, val_idx in splits]
    
    gaps = []
    for train_idx, val_idx in splits:
        if len(train_idx) > 0 and len(val_idx) > 0:
            gap = val_idx.min() - train_idx.max() - 1
            gaps.append(gap)
    
    return {
        "n_splits": len(splits),
        "train_sizes": train_sizes,
        "val_sizes": val_sizes,
        "gaps": gaps,
        "avg_train_size": np.mean(train_sizes),
        "avg_val_size": np.mean(val_sizes),
        "avg_gap": np.mean(gaps) if gaps else 0
    }


# Example usage and demonstration
if __name__ == "__main__":
    """
    Demonstration of time series cross-validation for financial data.
    """
    print("🕒 Time Series Cross-Validation Scheduler Demo")
    print("=" * 50)
    
    # Example 1: Block cross-validation with fixed window
    print("\n📊 Example 1: Block Cross-Validation (Fixed Window)")
    print("-" * 45)
    
    n_samples = 200  # 200 days of data
    splits = block_time_series_split(
        n_samples=n_samples,
        n_splits=4,
        train_size=60,  # 60-day training window
        val_size=20,    # 20-day validation window
        gap=1           # 1-day gap between train and val
    )
    
    print(f"Total samples: {n_samples}")
    print(f"Number of splits: {len(splits)}")
    
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"Split {i+1}: Train [{train_idx[0]:3d}:{train_idx[-1]:3d}] → Val [{val_idx[0]:3d}:{val_idx[-1]:3d}]")
    
    # Example 2: Expanding window backtest
    print("\n📈 Example 2: Expanding Window Backtest")
    print("-" * 38)
    
    splits_expanding = block_time_series_split(
        n_samples=300,
        n_splits=5,
        train_size=None,  # Expanding window
        val_size=30,
        gap=2
    )
    
    for i, (train_idx, val_idx) in enumerate(splits_expanding):
        print(f"Split {i+1}: Train [{train_idx[0]:3d}:{train_idx[-1]:3d}] ({len(train_idx):3d} samples) → Val [{val_idx[0]:3d}:{val_idx[-1]:3d}]")
    
    # Example 3: Rolling window for high-frequency data
    print("\n⚡ Example 3: Rolling Window (High-Frequency)")
    print("-" * 42)
    
    rolling_splits = list(rolling_window_split(
        n_samples=100,
        window_size=20,
        step_size=5,
        val_size=10
    ))
    
    print(f"Rolling window: {len(rolling_splits)} splits generated")
    for i, (train_idx, val_idx) in enumerate(rolling_splits[:5]):  # Show first 5
        print(f"Split {i+1}: Train [{train_idx[0]:2d}:{train_idx[-1]:2d}] → Val [{val_idx[0]:2d}:{val_idx[-1]:2d}]")
    if len(rolling_splits) > 5:
        print(f"... and {len(rolling_splits) - 5} more splits")
    
    # Validation check
    print("\n✅ Validation Check: No Lookahead")
    print("-" * 32)
    
    all_valid = True
    for i, (train_idx, val_idx) in enumerate(splits):
        is_valid = validate_time_series_split(train_idx, val_idx, gap=1)
        if not is_valid:
            print(f"❌ Split {i+1}: LOOKAHEAD DETECTED!")
            all_valid = False
        else:
            print(f"✅ Split {i+1}: Valid (no lookahead)")
    
    if all_valid:
        print("\n🎉 All splits are valid! No temporal leakage detected.")
    
    # Split information
    print("\n📋 Split Information Summary")
    print("-" * 28)
    info = get_split_info(splits)
    print(f"Number of splits: {info['n_splits']}")
    print(f"Average train size: {info['avg_train_size']:.1f}")
    print(f"Average val size: {info['avg_val_size']:.1f}")
    print(f"Average gap: {info['avg_gap']:.1f}")
    
    print("\n💡 Usage Tips:")
    print("- Use expanding window for backtesting (train_size=None)")
    print("- Use fixed window for walk-forward validation")  
    print("- Add gap to prevent immediate temporal dependencies")
    print("- Validate splits to ensure no lookahead leakage")
    print("\n🚀 Ready to use for time series model validation!")