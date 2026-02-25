"""Backtesting engine for strategies and models."""

from .engine import Backtester, BacktestResult
from .execution import ExecutionConfig, apply_execution_realism

__all__ = ["Backtester", "BacktestResult", "ExecutionConfig", "apply_execution_realism"]
