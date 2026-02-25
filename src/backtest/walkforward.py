"""
Walk-forward evaluation for out-of-sample backtesting.

Rolling windows: train_start/train_end (calibration) and test_start/test_end (OOS).
For non-ML strategies: train window used for optional parameter calibration;
test window always runs out-of-sample. No lookahead.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import Backtester, BacktestResult
from .execution import ExecutionConfig, throttle_positions


@dataclass
class WalkForwardFold:
    """Single fold: train and test date ranges."""

    fold_idx: int
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
    max_folds: Optional[int] = None,
    embargo_days: int = 0,
) -> List[WalkForwardFold]:
    """
    Generate rolling walk-forward folds (chronological only).

    Args:
        index: Datetime index of the data
        train_days: Length of train/calibration window in days
        test_days: Length of test window in days
        step_days: Step between folds (default: test_days)
        max_folds: Max number of folds to return (None = all)
        embargo_days: Days between train_end and test_start (avoids lookahead for IC/fwd returns)

    Returns:
        List of WalkForwardFold with date boundaries
    """
    if step_days is None:
        step_days = test_days

    dates = index.sort_values().unique()
    if len(dates) < train_days + embargo_days + test_days:
        return []

    folds: List[WalkForwardFold] = []
    i = 0
    fold_idx = 0
    while i + train_days + embargo_days + test_days <= len(dates):
        if max_folds is not None and fold_idx >= max_folds:
            break
        train_end_idx = i + train_days
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = test_start_idx + test_days
        train_start = pd.Timestamp(dates[i])
        train_end = pd.Timestamp(dates[train_end_idx - 1])
        test_start = pd.Timestamp(dates[test_start_idx])
        test_end = pd.Timestamp(dates[test_end_idx - 1])

        folds.append(
            WalkForwardFold(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_idx += 1
        i += step_days

    return folds


def run_walkforward(
    df: pd.DataFrame,
    strategy_factory: Callable[[dict], Any],
    backtest_factory: Callable[[dict], Backtester],
    folds: int,
    train_days: int,
    test_days: int,
    step_days: Optional[int] = None,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run walk-forward evaluation across rolling folds.

    Chronological splits only. No future data in train. Backtest on test window ONLY.

    Args:
        df: OHLCV DataFrame with 'close' column
        strategy_factory: config -> Strategy (has backtest_returns(prices) -> (positions, strategy_returns))
        backtest_factory: config -> Backtester
        folds: Number of folds to run
        train_days: Train window length (days)
        test_days: Test window length (days)
        step_days: Step between folds (default: test_days)
        config: Strategy/backtest config (passed to factories)

    Returns:
        dict with:
          - per_fold: list of {fold_idx, train_start, train_end, test_start, test_end, metrics}
          - aggregated: {mean_sharpe, median_sharpe, mean_return, worst_drawdown, ...}
    """
    config = config or {}
    if step_days is None:
        step_days = test_days

    close = df["close"] if isinstance(df, pd.DataFrame) else df
    if isinstance(close, pd.DataFrame):
        close = close["close"] if "close" in close.columns else close.iloc[:, 0]
    index = close.index

    fold_list = generate_folds(index, train_days, test_days, step_days, max_folds=folds)
    strategy = strategy_factory(config)
    backtester = backtest_factory(config)

    exec_cfg = ExecutionConfig(
        fee_bps=config.get("fee_bps", 1.0),
        slippage_bps=config.get("slippage_bps", 2.0),
        spread_bps=config.get("spread_bps", 1.0),
        execution_delay_bars=config.get("delay_bars", 1),
    )

    decision_interval = config.get("decision_interval_bars", 1)

    per_fold: List[Dict[str, Any]] = []
    all_oos_returns: List[pd.Series] = []
    sharpes: List[float] = []
    returns_list: List[float] = []
    drawdowns: List[float] = []

    for fold in fold_list:
        hist_close = close.loc[: fold.test_end]
        if len(hist_close) < 2:
            continue

        positions, _ = strategy.backtest_returns(hist_close)
        positions = throttle_positions(positions, decision_interval)

        result = backtester.run_from_signals(hist_close, positions, execution_config=exec_cfg)
        test_returns = result.returns.loc[fold.test_start : fold.test_end].dropna()
        if len(test_returns) < 2:
            continue

        fold_result = backtester.run(test_returns)
        fold.result = fold_result

        per_fold.append(
            {
                "fold_idx": fold.fold_idx,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "sharpe": fold_result.sharpe_ratio,
                "total_return": fold_result.total_return,
                "max_drawdown": fold_result.max_drawdown,
                "n_trades": fold_result.n_trades,
                "win_rate": fold_result.win_rate,
            }
        )
        all_oos_returns.append(test_returns)
        sharpes.append(fold_result.sharpe_ratio)
        returns_list.append(fold_result.total_return)
        drawdowns.append(fold_result.max_drawdown)

    # Aggregated metrics
    if all_oos_returns:
        agg_returns = pd.concat(all_oos_returns).sort_index()
        agg_returns = agg_returns[~agg_returns.index.duplicated(keep="first")]
        agg_result = backtester.run(agg_returns)
        worst_dd = max(drawdowns) if drawdowns else 0.0
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
        worst_dd = 0.0

    aggregated = {
        "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
        "median_sharpe": float(np.median(sharpes)) if sharpes else 0.0,
        "mean_return": float(np.mean(returns_list)) if returns_list else 0.0,
        "worst_drawdown": float(worst_dd),
        "agg_sharpe": agg_result.sharpe_ratio,
        "agg_total_return": agg_result.total_return,
        "agg_max_drawdown": agg_result.max_drawdown,
        "agg_n_trades": agg_result.n_trades,
        "n_folds": len(per_fold),
    }

    return {"per_fold": per_fold, "aggregated": aggregated}


# Legacy API (backward compat)
def run_walkforward_legacy(
    prices: pd.Series,
    strategy_fn: Callable[[pd.Series], Tuple[pd.Series, pd.Series]],
    train_days: int = 252,
    test_days: int = 63,
    step_days: Optional[int] = None,
    backtester: Optional[Backtester] = None,
) -> Tuple[List[WalkForwardFold], BacktestResult]:
    """
    Legacy walk-forward: strategy_fn(prices) -> (signals, strategy_returns).
    """
    if backtester is None:
        backtester = Backtester(annualization_factor=252)

    fold_list = generate_folds(prices.index, train_days, test_days, step_days)
    all_oos_returns: List[pd.Series] = []

    for fold in fold_list:
        hist_prices = prices.loc[: fold.test_end]
        if len(hist_prices) < 2:
            continue
        signals, _ = strategy_fn(hist_prices)
        full_result = backtester.run_from_signals(hist_prices, signals)
        test_returns = full_result.returns.loc[fold.test_start : fold.test_end].dropna()
        if len(test_returns) < 2:
            continue
        result = backtester.run(test_returns)
        fold.result = result
        all_oos_returns.append(test_returns)

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

    return fold_list, agg_result
