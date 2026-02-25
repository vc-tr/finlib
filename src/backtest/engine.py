"""
Unified backtesting engine for quantitative strategies.

Supports both rule-based strategies (signals) and model-based forecasts.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Callable, Union

from .execution import ExecutionConfig, apply_execution_realism


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    returns: pd.Series
    cumulative_returns: pd.Series
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    n_trades: int
    win_rate: float


class Backtester:
    """
    Backtest strategy returns.
    
    Assumes strategy_returns = signals * asset_returns (or direct strategy returns).
    """

    def __init__(
        self,
        initial_capital: float = 1.0,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252,
    ):
        """
        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe
            annualization_factor: 252 for daily, 52 for weekly
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def run(
        self,
        strategy_returns: pd.Series,
        align_prices: Optional[pd.Series] = None,
        n_trades_override: Optional[int] = None,
    ) -> BacktestResult:
        """
        Run backtest on strategy returns.

        Args:
            strategy_returns: Period returns of the strategy
            align_prices: Optional price series to align indices

        Returns:
            BacktestResult with metrics
        """
        if align_prices is not None:
            strategy_returns = strategy_returns.reindex(align_prices.index).ffill().bfill().fillna(0)

        strategy_returns = strategy_returns.dropna()
        if len(strategy_returns) == 0:
            return BacktestResult(
                returns=pd.Series(dtype=float),
                cumulative_returns=pd.Series(dtype=float),
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_return=0.0,
                n_trades=0,
                win_rate=0.0,
            )

        cum = (1 + strategy_returns).cumprod()
        total_return = cum.iloc[-1] - 1.0

        # Sharpe ratio (annualized)
        excess = strategy_returns - self.risk_free_rate / self.annualization_factor
        sharpe = (
            np.sqrt(self.annualization_factor) * excess.mean() / excess.std()
            if excess.std() > 1e-10
            else 0.0
        )

        # Max drawdown
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Trade count: use override if provided, else count periods with nonzero returns
        n_trades = (
            int(n_trades_override)
            if n_trades_override is not None
            else (strategy_returns.abs() > 1e-8).sum()
        )
        # Win rate = fraction of bars with positive return (among bars with nonzero return)
        active = (strategy_returns.abs() > 1e-8)
        wins = (strategy_returns > 0).sum()
        win_rate = wins / active.sum() if active.sum() > 0 else 0.0

        return BacktestResult(
            returns=strategy_returns,
            cumulative_returns=cum,
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_drawdown),
            total_return=float(total_return),
            n_trades=int(n_trades),
            win_rate=float(win_rate),
        )

    def run_from_signals(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        signals: pd.Series,
        execution_config: Optional[ExecutionConfig] = None,
    ) -> BacktestResult:
        """
        Backtest from price series and signals (-1, 0, 1).

        No lookahead: signal at close t executes at bar t+1.

        Args:
            prices: Close prices (Series) or OHLCV DataFrame with 'close' column
            signals: Target position in {-1, 0, 1}
            execution_config: Optional fees, slippage, timing. If None, raw execution.
        """
        if isinstance(prices, pd.DataFrame):
            prices = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
        if execution_config is not None:
            strategy_returns, pos = apply_execution_realism(prices, signals, execution_config)
        else:
            returns = prices.pct_change()
            pos = signals.shift(1).fillna(0)
            strategy_returns = pos * returns
        # Trade count = position changes only (enter/exit), not every bar with exposure
        n_trades = int((pos.diff().abs() > 1e-8).sum())
        return self.run(strategy_returns, align_prices=prices, n_trades_override=n_trades)
