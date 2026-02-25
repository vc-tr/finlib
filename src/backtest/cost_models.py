"""
Transaction cost models for backtesting.

CostModel.estimate_costs(trades_df, ohlcv_by_symbol, config) -> trades_df with
spread_cost, slippage_cost, impact_cost, total_cost.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


def build_trades_from_weights(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    portfolio_value: float = 1.0,
) -> pd.DataFrame:
    """
    Build trades DataFrame from weight changes.

    Args:
        weights: (date x symbol) weights held
        prices: (date x symbol) close prices
        portfolio_value: AUM for notional calculation

    Returns:
        DataFrame with columns: timestamp, symbol, trade_weight, side, fill_price,
        trade_notional, fee_bps, slippage_bps, spread_bps (placeholders for cost model)
    """
    w = weights.fillna(0)
    chg = w.diff()
    common_idx = w.index.intersection(prices.index)
    common_cols = w.columns.intersection(prices.columns)
    chg = chg.reindex(common_idx).reindex(columns=common_cols).fillna(0)
    prices_aligned = prices.reindex(common_idx).reindex(columns=common_cols).ffill()

    rows = []
    for ts in chg.index:
        for sym in chg.columns:
            dw = chg.loc[ts, sym]
            if abs(dw) < 1e-10:
                continue
            price = prices_aligned.loc[ts, sym]
            if pd.isna(price) or price <= 0:
                continue
            trade_weight = abs(dw)
            side = "buy" if dw > 0 else "sell"
            trade_notional = trade_weight * portfolio_value
            rows.append({
                "timestamp": ts,
                "symbol": sym,
                "trade_weight": trade_weight,
                "side": side,
                "fill_price": float(price),
                "trade_notional": trade_notional,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "trade_weight", "side", "fill_price", "trade_notional",
            "spread_cost", "slippage_cost", "impact_cost", "total_cost",
        ])
    return pd.DataFrame(rows)


def _get_col(df: pd.DataFrame, candidates: list) -> Optional[pd.Series]:
    """Get first matching column (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return df[cols_lower[cand.lower()]]
    return None


def _compute_adv(ohlcv_by_symbol: Dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:
    """Compute ADV (avg daily dollar volume) per symbol. Returns (date x symbol)."""
    adv_by = {}
    for sym, df in ohlcv_by_symbol.items():
        close = _get_col(df, ["close", "Close"])
        vol = _get_col(df, ["volume", "Volume"])
        if close is None or vol is None:
            continue
        vol = vol.fillna(0)
        dv = close * vol
        adv = dv.rolling(window).mean()
        adv_by[sym] = adv
    if not adv_by:
        return pd.DataFrame()
    return pd.DataFrame(adv_by)


def _compute_atr_pct(ohlcv_by_symbol: Dict[str, pd.DataFrame], window: int = 14) -> pd.DataFrame:
    """ATR/close as spread proxy. Returns (date x symbol)."""
    out = {}
    for sym, df in ohlcv_by_symbol.items():
        h = _get_col(df, ["high", "High"])
        l = _get_col(df, ["low", "Low"])
        c = _get_col(df, ["close", "Close"])
        if h is None or l is None or c is None:
            continue
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        atr_pct = (atr / c).replace(0, np.nan)
        out[sym] = atr_pct
    return pd.DataFrame(out) if out else pd.DataFrame()


class CostModel(ABC):
    """Interface for transaction cost models."""

    @abstractmethod
    def estimate_costs(
        self,
        trades_df: pd.DataFrame,
        ohlcv_by_symbol: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Add spread_cost, slippage_cost, impact_cost, total_cost to trades_df.

        Returns:
            trades_df with added cost columns (as decimal, e.g. 0.0004 = 4 bps of portfolio)
        """
        pass


class FixedBpsCostModel(CostModel):
    """Fixed bps per trade: fee + slippage + spread (current behavior)."""

    def estimate_costs(
        self,
        trades_df: pd.DataFrame,
        ohlcv_by_symbol: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        fee_bps = config.get("fee_bps", 1.0)
        slippage_bps = config.get("slippage_bps", 2.0)
        spread_bps = config.get("spread_bps", 1.0)
        total_bps = fee_bps + slippage_bps + spread_bps

        df = trades_df.copy()
        w = df["trade_weight"]
        fee_pct = (fee_bps / 10_000) * w
        slip_pct = (slippage_bps / 10_000) * w
        spread_pct = (spread_bps / 10_000) * w
        df["spread_cost"] = spread_pct
        df["slippage_cost"] = slip_pct
        df["impact_cost"] = 0.0
        df["total_cost"] = fee_pct + slip_pct + spread_pct
        return df


class LiquidityAwareCostModel(CostModel):
    """
    Liquidity-aware costs: ADV-based impact, optional ATR-based spread.

    impact_cost_bps = impact_k * (trade_notional / ADV)^impact_alpha, clamped to max_impact_bps
    spread: spread_k * (ATR/close) or fixed spread_bps
    """

    def estimate_costs(
        self,
        trades_df: pd.DataFrame,
        ohlcv_by_symbol: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        fee_bps = config.get("fee_bps", 1.0)
        slippage_bps = config.get("slippage_bps", 2.0)
        spread_bps = config.get("spread_bps", 1.0)
        impact_k = config.get("impact_k", 10.0)
        impact_alpha = config.get("impact_alpha", 0.5)
        max_impact_bps = config.get("max_impact_bps", 50.0)
        adv_window = config.get("adv_window", 20)
        spread_k = config.get("spread_k", 0.0)  # if > 0, use ATR-based spread
        atr_window = config.get("atr_window", 14)

        df = trades_df.copy()
        if len(df) == 0:
            df["spread_cost"] = pd.Series(dtype=float)
            df["slippage_cost"] = pd.Series(dtype=float)
            df["impact_cost"] = pd.Series(dtype=float)
            df["total_cost"] = pd.Series(dtype=float)
            return df

        adv = _compute_adv(ohlcv_by_symbol, window=adv_window)
        atr_pct = _compute_atr_pct(ohlcv_by_symbol, window=atr_window) if spread_k > 0 else None

        impact_list = []
        spread_list = []
        for _, row in df.iterrows():
            ts, sym, trade_weight, fill_price, trade_notional = (
                row["timestamp"], row["symbol"], row["trade_weight"],
                row["fill_price"], row["trade_notional"],
            )
            adv_val = np.nan
            if sym in adv.columns and ts in adv.index:
                adv_val = adv.loc[ts, sym]
            if pd.isna(adv_val) or adv_val <= 0:
                impact_bps = max_impact_bps
            else:
                participation = trade_notional / adv_val
                impact_bps = impact_k * (participation ** impact_alpha)
                impact_bps = min(impact_bps, max_impact_bps)
            impact_pct = (impact_bps / 10_000) * trade_weight
            impact_list.append(impact_pct)

            if spread_k > 0 and atr_pct is not None and sym in atr_pct.columns and ts in atr_pct.index:
                atr_val = atr_pct.loc[ts, sym]
                spread_bps_val = spread_k * (atr_val * 10_000) if not pd.isna(atr_val) else spread_bps
            else:
                spread_bps_val = spread_bps
            spread_pct = (spread_bps_val / 10_000) * trade_weight
            spread_list.append(spread_pct)

        df["impact_cost"] = impact_list
        df["spread_cost"] = spread_list
        df["slippage_cost"] = (slippage_bps / 10_000) * df["trade_weight"]
        fee_pct = (fee_bps / 10_000) * df["trade_weight"]
        df["total_cost"] = fee_pct + df["slippage_cost"] + df["spread_cost"] + df["impact_cost"]
        return df


def apply_costs_from_trades(
    port_returns: pd.Series,
    trades_df: pd.DataFrame,
) -> pd.Series:
    """
    Subtract total costs from portfolio returns by date.

    trades_df must have timestamp, total_cost. Costs are summed per date and subtracted.
    """
    if len(trades_df) == 0:
        return port_returns
    cost_by_date = trades_df.groupby("timestamp")["total_cost"].sum()
    cost_series = cost_by_date.reindex(port_returns.index).fillna(0)
    return port_returns.reindex(cost_series.index).fillna(0) - cost_series


def compute_capacity_report(
    trades_df: pd.DataFrame,
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    adv_window: int = 20,
    target_impact_bps: float = 10.0,
) -> Dict[str, Any]:
    """
    Compute capacity report: per-symbol ADV, trade notional, impact distribution, capacity.

    Returns dict for capacity_report.json.
    """
    adv = _compute_adv(ohlcv_by_symbol, window=adv_window)
    impact_k = config.get("impact_k", 10.0)
    impact_alpha = config.get("impact_alpha", 0.5)

    per_sym = {}
    for sym in set(adv.columns) | set(ohlcv_by_symbol.keys()):
        adv_mean = 0.0
        if sym in adv.columns:
            adv_series = adv[sym].dropna()
            adv_mean = float(adv_series.mean()) if len(adv_series) > 0 else 0.0
        per_sym[sym] = {"avg_adv": adv_mean}

    if len(trades_df) == 0:
        return {
            "per_symbol_adv": per_sym,
            "avg_trade_notional": 0.0,
            "impact_bps": {"min": 0, "median": 0, "max": 0},
            "capacity_notional_at_target_bps": 0.0,
            "target_impact_bps": target_impact_bps,
        }

    avg_notional = float(trades_df["trade_notional"].mean())
    impact_bps_list = []
    for _, row in trades_df.iterrows():
        sym, trade_notional = row["symbol"], row["trade_notional"]
        adv_val = adv[sym].mean() if sym in adv.columns else np.nan
        if pd.isna(adv_val) or adv_val <= 0:
            impact_bps_list.append(50.0)
            continue
        participation = trade_notional / adv_val
        impact_bps = impact_k * (participation ** impact_alpha)
        impact_bps_list.append(min(impact_bps, 50.0))
    impact_bps_arr = np.array(impact_bps_list)

    capacity_notional = 0.0
    if impact_k > 0 and impact_alpha > 0:
        participation_at_target = (target_impact_bps / impact_k) ** (1 / impact_alpha)
        adv_vals = adv.replace(0, np.nan).stack().dropna()
        adv_median = float(adv_vals.median()) if len(adv_vals) > 0 else 0.0
        capacity_notional = participation_at_target * adv_median if adv_median > 0 else 0.0

    return {
        "per_symbol_adv": {k: {"avg_adv": float(v["avg_adv"])} for k, v in per_sym.items()},
        "avg_trade_notional": avg_notional,
        "impact_bps": {
            "min": float(np.min(impact_bps_arr)) if len(impact_bps_arr) > 0 else 0,
            "median": float(np.median(impact_bps_arr)) if len(impact_bps_arr) > 0 else 0,
            "max": float(np.max(impact_bps_arr)) if len(impact_bps_arr) > 0 else 0,
        },
        "capacity_notional_at_target_bps": capacity_notional,
        "target_impact_bps": target_impact_bps,
    }
