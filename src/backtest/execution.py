"""
Execution realism layer for backtesting.

Provides fee models, slippage, spread, execution delay, and trades dataframe.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ExecutionConfig:
    """Configuration for execution realism."""

    fee_bps: float = 1.0  # Fee in basis points per trade (one-way)
    slippage_bps: float = 2.0  # Slippage in bps per trade
    spread_bps: float = 1.0  # Bid-ask spread proxy in bps
    slippage_vol_scalar: float = 0.0  # Extra slippage = scalar * vol (0 = disabled)
    execution_delay_bars: int = 1  # Signal at t → fill at t + delay
    fill_price_mode: str = "next_open"  # "next_open" or "next_close"
    fill_at_next_open: bool = True  # Legacy alias for fill_price_mode


def apply_execution_realism(
    prices: pd.Series,
    signals: pd.Series,
    config: ExecutionConfig,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply fees, slippage, spread, and execution timing. Costs only on position changes.

    No lookahead: signal at close t executes at bar t + execution_delay_bars.

    Args:
        prices: Close prices
        signals: Target position in {-1, 0, 1}
        config: Execution parameters

    Returns:
        (strategy_returns, positions) — positions used for trade count
    """
    returns = prices.pct_change()
    delay = config.execution_delay_bars

    # Position: signal at t → held during bar t+delay
    pos = signals.shift(delay).fillna(0)

    # Raw strategy returns: position * asset return
    strategy_returns = pos * returns

    # Costs ONLY on position changes (trades)
    pos_change = pos.diff().abs()
    cost_bps = config.fee_bps + config.slippage_bps + config.spread_bps
    cost_pct = (cost_bps / 10_000) * pos_change

    if config.slippage_vol_scalar > 0:
        vol = returns.rolling(20).std().fillna(returns.std())
        cost_pct = cost_pct + config.slippage_vol_scalar * vol * pos_change

    strategy_returns = strategy_returns - cost_pct

    return strategy_returns.reindex(prices.index).fillna(0), pos.reindex(prices.index).fillna(0)


def build_trades_dataframe(
    prices: pd.Series,
    signals: pd.Series,
    config: ExecutionConfig,
) -> pd.DataFrame:
    """
    Build detailed trades dataframe: timestamp, side, weight, fill_price, fee, slippage_cost, pnl_contrib.
    """
    delay = config.execution_delay_bars
    pos = signals.shift(delay).fillna(0)
    pos_change = pos.diff()

    rows = []
    for i in range(1, len(prices)):
        chg = pos_change.iloc[i]
        if abs(chg) < 0.5:
            continue
        ts = prices.index[i]
        fill_price = prices.iloc[i]
        side = "buy" if chg > 0 else "sell"
        weight = abs(chg)
        fee_pct = (config.fee_bps / 10_000) * weight
        slip_pct = (config.slippage_bps / 10_000) * weight
        spread_pct = (config.spread_bps / 10_000) * weight
        total_cost_pct = fee_pct + slip_pct + spread_pct
        pnl_contrib = -total_cost_pct  # cost reduces PnL

        rows.append({
            "timestamp": ts,
            "side": side,
            "weight": weight,
            "fill_price": fill_price,
            "fee": fee_pct,
            "slippage_cost": slip_pct,
            "spread_cost": spread_pct,
            "pnl_contrib": pnl_contrib,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp", "side", "weight", "fill_price", "fee", "slippage_cost", "spread_cost", "pnl_contrib"]
    )


def compute_turnover(signals: pd.Series) -> pd.Series:
    """Compute turnover as |position change|."""
    return signals.diff().abs().fillna(0)


def compute_turnover_annualized(signals: pd.Series, annualization_factor: float = 252) -> float:
    """Annualized average turnover."""
    turnover = compute_turnover(signals)
    return float(turnover.mean() * annualization_factor)
