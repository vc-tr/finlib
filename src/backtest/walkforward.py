"""
Walk-forward evaluation for out-of-sample backtesting.

Rolling windows: train_start/train_end (calibration) and test_start/test_end (OOS).
For non-ML strategies: train window used for optional parameter calibration;
test window always runs out-of-sample.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

from .engine import Backtester, BacktestResult


@dataclass
class WalkForwardFold:
    """Single fold: train and test date ranges."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: Optional[BacktestResult] = None


def generate_folds(
    index: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
) -> List[WalkForwardFold]:
    """
    Generate rolling walk-forward folds.

    Args:
        index: Datetime index of the data
        train_days: Length of train/calibration window in days
        test_days: Length of test window in days
        step_days: Step between folds (default: test_days)

    Returns:
        List of WalkForwardFold with date boundaries
    """
    if step_days is None:
        step_days = test_days

    dates = index.sort_values().unique()
    if len(dates) < train_days + test_days:
        return []

    folds: List[WalkForwardFold] = []
    i = 0
    while i + train_days + test_days <= len(dates):
        train_end_idx = i + train_days
        test_end_idx = train_end_idx + test_days
        train_start = pd.Timestamp(dates[i])
        train_end = pd.Timestamp(dates[train_end_idx - 1])
        test_start = pd.Timestamp(dates[train_end_idx])
        test_end = pd.Timestamp(dates[test_end_idx - 1])

        folds.append(
            WalkForwardFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        i += step_days

    return folds


def run_walkforward(
    prices: pd.Series,
    strategy_fn: Callable[[pd.Series], Tuple[pd.Series, pd.Series]],
    train_days: int = 252,
    test_days: int = 63,
    step_days: Optional[int] = None,
    backtester: Optional[Backtester] = None,
) -> Tuple[List[WalkForwardFold], BacktestResult]:
    """
    Run walk-forward backtest.

    Args:
        prices: Close price series
        strategy_fn: Callable(prices) -> (signals, strategy_returns)
        train_days: Train window length
        test_days: Test window length
        step_days: Step between folds
        backtester: Backtester instance (default: new Backtester())

    Returns:
        (list of folds with results, aggregated OOS result)
    """
    if backtester is None:
        backtester = Backtester(annualization_factor=252)

    folds = generate_folds(prices.index, train_days, test_days, step_days)
    all_oos_returns: List[pd.Series] = []

    for fold in folds:
        # For non-ML: use history up to test_end; no lookahead
        hist_prices = prices.loc[:fold.test_end]
        if len(hist_prices) < 2:
            continue
        signals, _ = strategy_fn(hist_prices)

        # Run backtest on full history, extract test-period returns
        full_result = backtester.run_from_signals(hist_prices, signals)
        test_returns = full_result.returns.loc[fold.test_start:fold.test_end].dropna()
        if len(test_returns) < 2:
            continue
        result = backtester.run(test_returns)
        fold.result = result
        all_oos_returns.append(test_returns)

    # Aggregate OOS returns
    if all_oos_returns:
        agg_returns = pd.concat(all_oos_returns).sort_index()
        agg_returns = agg_returns[~agg_returns.index.duplicated(keep="first")]
        agg_result = backtester.run(agg_returns)
    else:
        agg_result = BacktestResult(
            returns=pd.Series(dtype=float),
            cumulative_returns=pd.Series(dtype=float),
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_return=0.0,
            n_trades=0,
            win_rate=0.0,
        )

    return folds, agg_result
