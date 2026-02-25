"""
Portfolio construction from cross-sectional weights.

Resamples weights to rebalance frequency, applies execution delay,
beta-neutralization, and constraints (max_gross, max_net).

True rebalance: compute target weights ONLY on rebalance dates,
forward-fill between rebalances. Turnover occurs only at rebalance timestamps.
"""

import pandas as pd
from typing import Optional, Tuple, Union


def rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    """
    Return DatetimeIndex of rebalance points for a given index.

    Args:
        index: Full date index (e.g. daily)
        rebalance: "D" (daily), "W" (week-end), "M" (month-end)

    Returns:
        DatetimeIndex of dates when rebalancing occurs
    """
    if rebalance == "D":
        return index
    if rebalance == "W":
        freq = "W-FRI"
    elif rebalance == "M":
        freq = "ME"
    else:
        freq = "ME"
    # Last date of each period
    rb = index.to_series().resample(freq).last().dropna()
    return rb.index.intersection(index)


def apply_constraints(
    weights: pd.DataFrame,
    max_gross: Optional[float] = None,
    max_net: Optional[float] = None,
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """
    Apply max_gross and max_net caps. Re-normalize to preserve gross_leverage when possible.

    Args:
        weights: Raw weights (date x symbol)
        max_gross: Cap on sum(|w|) per date (None = no cap)
        max_net: Cap on |sum(w)| per date (None = no cap)
        gross_leverage: Target gross after scaling (used when capping)
    """
    w = weights.copy()
    gross = w.abs().sum(axis=1)
    net = w.sum(axis=1)

    if max_gross is not None:
        scale = (max_gross / gross).clip(upper=1.0)
        scale = scale.replace(0, 1)
        w = w.mul(scale, axis=0)

    if max_net is not None:
        net = w.sum(axis=1)
        scale = (max_net / net.abs()).clip(upper=1.0)
        scale = scale.replace(0, 1)
        w = w.mul(scale, axis=0)

    gross = w.abs().sum(axis=1).replace(0, 1)
    w = w.div(gross, axis=0) * gross_leverage
    return w


def apply_beta_neutral(
    weights: pd.DataFrame,
    betas: pd.DataFrame,
    market_symbol: str = "SPY",
    method: str = "hedge",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Neutralize portfolio beta.

    Option A (hedge): Add w_market = -beta_p to weights. Returns (weights, beta_before, beta_after).
    Option B (demean): Project out beta vector (not implemented; use hedge).

    Returns:
        (weights with hedge, portfolio_beta_before, portfolio_beta_after)
    """
    common_idx = weights.index.intersection(betas.index)
    syms = weights.columns.intersection(betas.columns)
    w = weights.reindex(common_idx).ffill().fillna(0)
    b = betas.reindex(common_idx).ffill().fillna(0)
    w = w.reindex(columns=syms).fillna(0)
    b = b.reindex(columns=syms).fillna(0)

    beta_p = (w * b).sum(axis=1)

    if method == "hedge":
        hedge = -beta_p
        if market_symbol not in w.columns:
            w = w.reindex(columns=list(w.columns) + [market_symbol]).fillna(0)
        w[market_symbol] = w[market_symbol].add(hedge, fill_value=0)
        beta_after = beta_p + hedge
        return w, beta_p, beta_after
    raise ValueError(f"Unknown method: {method}")


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


def weights_at_rebalance(
    factor_df: pd.DataFrame,
    rebalance: str,
    top_k: int = 10,
    bottom_k: int = 10,
    method: str = "zscore",
    long_short: bool = True,
    gross_leverage: float = 1.0,
    max_weight: float = 0.1,
) -> pd.DataFrame:
    """
    Compute target weights ONLY on rebalance dates, forward-fill between rebalances.

    True rebalance: rank only on rebalance dates; hold constant between.
    Trades/turnover occur only at rebalance timestamps.

    Args:
        factor_df: Daily factor values (date x symbol)
        rebalance: "D"|"W"|"M"
        top_k, bottom_k, method, long_short, gross_leverage, max_weight: passed to cross_sectional_rank

    Returns:
        Weights (date x symbol), full index, 0 before first rebalance, ffill between.
    """
    from src.factors.ranking import cross_sectional_rank

    rb_dates = rebalance_dates(factor_df.index, rebalance)
    common = factor_df.index.intersection(rb_dates)
    if len(common) == 0:
        return pd.DataFrame(0.0, index=factor_df.index, columns=factor_df.columns)

    factor_rb = factor_df.loc[common]
    weights_rb = cross_sectional_rank(
        factor_rb,
        top_k=top_k,
        bottom_k=bottom_k,
        method=method,
        long_short=long_short,
        gross_leverage=gross_leverage,
        max_weight=max_weight,
    )
    return weights_rb.reindex(factor_df.index).ffill().fillna(0)


def build_portfolio(
    weights: pd.DataFrame,
    prices: Union[pd.DataFrame, pd.Series],
    rebalance: str = "M",
    long_short: bool = True,
    gross_leverage: float = 1.0,
    max_weight: float = 0.1,
    execution_delay: int = 1,
    max_gross: Optional[float] = None,
    max_net: Optional[float] = None,
    beta_neutral: bool = False,
    betas: Optional[pd.DataFrame] = None,
    market_symbol: str = "SPY",
) -> Union[pd.Series, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    """
    Build portfolio returns from weights and prices.

    Weights at t (after resample) apply to returns at t+execution_delay (no lookahead).

    Args:
        weights: index=date, columns=symbol, values=weight (from cross_sectional_rank)
        prices: Wide DataFrame (date x symbol)
        rebalance: "D"|"W"|"M"
        max_gross: Cap sum(|w|) per rebalance (None = no cap)
        max_net: Cap |sum(w)| per rebalance (None = no cap)
        beta_neutral: If True, add market hedge (requires betas, market in prices)
        betas: Rolling betas (date x symbol) for beta-neutral
        market_symbol: Symbol for hedge (default SPY)

    Returns:
        If beta_neutral: (port_ret, beta_before, beta_after, hedge_weight)
        Else: port_ret
    """
    if isinstance(prices, pd.Series):
        returns = prices.pct_change().to_frame()
    else:
        returns = prices.pct_change()

    w = weights.copy()
    if max_gross is not None or max_net is not None:
        w = apply_constraints(w, max_gross=max_gross, max_net=max_net, gross_leverage=gross_leverage)
    if beta_neutral and betas is not None:
        w, beta_before, beta_after = apply_beta_neutral(w, betas, market_symbol=market_symbol)
    w = _resample_weights_to_rebalance(w, rebalance)
    common_idx = w.index.intersection(returns.index)
    w = w.reindex(common_idx).ffill().fillna(0)
    r = returns.reindex(common_idx).fillna(0)

    # Align columns (use common symbols only)
    syms = w.columns.intersection(r.columns)
    w = w.reindex(columns=syms).fillna(0)
    r = r.reindex(columns=syms).fillna(0)

    w_held = w.shift(execution_delay).fillna(0)
    port_ret = (w_held * r).sum(axis=1)
    if beta_neutral and betas is not None:
        hedge = w[market_symbol] if market_symbol in w.columns else pd.Series(0.0, index=w.index)
        hedge_held = hedge.shift(execution_delay).reindex(port_ret.index).fillna(0)
        return port_ret, beta_before, beta_after, hedge_held
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
