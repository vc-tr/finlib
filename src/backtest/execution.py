"""
Execution realism layer for backtesting.

Provides fee models, slippage models, and execution timing to avoid lookahead.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionConfig:
    """Configuration for execution realism."""

    fee_bps: float = 5.0  # Fee in basis points per trade (round-trip)
    slippage_bps: float = 5.0  # Slippage in bps per trade
    slippage_vol_scalar: float = 0.0  # Extra slippage = scalar * vol (0 = disabled)
    fill_at_next_open: bool = True  # True: signal at close t → fill at open t+1
    # False: signal at close t → fill at close t+1 (same as fill_at_next_open for daily)


def apply_execution_realism(
    prices: pd.Series,
    signals: pd.Series,
    config: ExecutionConfig,
) -> pd.Series:
    """
    Apply fees, slippage, and execution timing to convert signals into strategy returns.

    No lookahead: signal computed from bar t data executes at bar t+1.

    Args:
        prices: Close prices
        signals: Target position in {-1, 0, 1}
        config: Execution parameters

    Returns:
        Strategy returns series (aligned to prices index)
    """
    returns = prices.pct_change()

    # Execution timing: signal at close t → position held during bar t+1
    # So we use signals.shift(1) for the position during each bar
    pos = signals.shift(1).fillna(0)

    # Raw strategy returns: position * asset return
    strategy_returns = pos * returns

    # Slippage: proportional to |position change| and volatility
    pos_change = pos.diff().abs()
    slippage_pct = (config.slippage_bps / 10_000) * pos_change

    if config.slippage_vol_scalar > 0:
        vol = returns.rolling(20).std().fillna(returns.std())
        slippage_pct = slippage_pct + config.slippage_vol_scalar * vol * pos_change

    # Fees: round-trip when position changes
    fee_pct = (config.fee_bps / 10_000) * pos_change

    # Subtract costs from strategy returns
    strategy_returns = strategy_returns - slippage_pct - fee_pct

    return strategy_returns.reindex(prices.index).fillna(0)


def compute_turnover(signals: pd.Series) -> pd.Series:
    """
    Compute daily turnover as |position change|.

    Args:
        signals: Position series in {-1, 0, 1} or weights

    Returns:
        Daily turnover series
    """
    return signals.diff().abs().fillna(0)


def compute_turnover_annualized(signals: pd.Series, annualization_factor: float = 252) -> float:
    """Annualized average daily turnover."""
    turnover = compute_turnover(signals)
    return float(turnover.mean() * annualization_factor)
