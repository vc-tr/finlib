"""
Unified backtesting engine for quantitative strategies.

Supports both rule-based strategies (signals) and model-based forecasts.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Callable


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

        # Trade count (simplified: count sign changes in returns)
        n_trades = (strategy_returns.abs() > 1e-8).sum()
        wins = (strategy_returns > 0).sum()
        win_rate = wins / n_trades if n_trades > 0 else 0.0

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
        prices: pd.Series,
        signals: pd.Series,
    ) -> BacktestResult:
        """Backtest from price series and signals (-1, 0, 1)."""
        returns = prices.pct_change()
        strategy_returns = signals.shift(1).fillna(0) * returns
        return self.run(strategy_returns, align_prices=prices)
