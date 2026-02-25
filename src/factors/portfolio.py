"""
Portfolio construction from cross-sectional weights.

Resamples weights to rebalance frequency, applies execution delay.
"""

import pandas as pd
from typing import Union


def _resample_weights_to_rebalance(weights: pd.DataFrame, rebalance: str) -> pd.DataFrame:
    """Forward-fill weights; only update on rebalance dates."""
    if rebalance == "D":
        return weights
    if rebalance == "W":
        freq = "W-FRI"
    elif rebalance == "M":
        freq = "ME"
    else:
        freq = "ME"

    # Take last weight of each period, then ffill to next period
    rb = weights.resample(freq).last()
    return rb.reindex(weights.index).ffill().fillna(0)


def build_portfolio(
    weights: pd.DataFrame,
    prices: Union[pd.DataFrame, pd.Series],
    rebalance: str = "M",
    long_short: bool = True,
    gross_leverage: float = 1.0,
    max_weight: float = 0.1,
    execution_delay: int = 1,
) -> pd.Series:
    """
    Build portfolio returns from weights and prices.

    Weights at t (after resample) apply to returns at t+execution_delay (no lookahead).

    Args:
        weights: index=date, columns=symbol, values=weight (from cross_sectional_rank)
        prices: Wide DataFrame (date x symbol) or dict of Series
        rebalance: "D"|"W"|"M"
        long_short: Unused (weights already signed)
        gross_leverage: Unused (weights already scaled)
        max_weight: Unused (weights already capped)
        execution_delay: Bars between signal and fill (default 1)

    Returns:
        Series of portfolio returns (index=date)
    """
    if isinstance(prices, pd.Series):
        returns = prices.pct_change().to_frame()
    else:
        returns = prices.pct_change()

    w = _resample_weights_to_rebalance(weights, rebalance)
    common_idx = w.index.intersection(returns.index)
    w = w.reindex(common_idx).ffill().fillna(0)
    r = returns.reindex(common_idx).fillna(0)

    # Align columns (use common symbols only)
    syms = w.columns.intersection(r.columns)
    w = w.reindex(columns=syms).fillna(0)
    r = r.reindex(columns=syms).fillna(0)

    w_held = w.shift(execution_delay).fillna(0)
    port_ret = (w_held * r).sum(axis=1)
    return port_ret


def apply_rebalance_costs(
    port_returns: pd.Series,
    weights: pd.DataFrame,
    cost_bps: float = 4.0,
) -> pd.Series:
    """
    Subtract transaction costs from portfolio returns. Costs only on weight changes.

    cost_bps: one-way cost in bps (fee + slippage + spread)
    """
    w = weights.fillna(0)
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    cost_pct = (cost_bps / 10_000) * turnover
    common = port_returns.index.intersection(cost_pct.index)
    return port_returns.reindex(common).fillna(0) - cost_pct.reindex(common).fillna(0)
