"""
Regime analysis for backtested strategies.

Identifies market regimes from price data, then measures conditional
strategy performance within each regime. Supports:

1. Volatility regimes — rolling realized vol ranked into quintiles.
   Low-vol environments tend to favour trend-following; high-vol
   environments often favor mean-reversion or risk-off strategies.

2. Trend regimes — Hurst exponent via R/S analysis (Hurst 1951).
   H > 0.55 → trending; H < 0.45 → mean-reverting; otherwise random walk.

3. Drawdown analysis — top-N drawdowns with start, trough, recovery,
   duration, and depth. A standard part of institutional performance review.

References:
  Hurst, H.E. (1951). "Long-term storage capacity of reservoirs."
    Transactions of the American Society of Civil Engineers, 116, 770-799.

  Mandelbrot, B.B. & Wallis, J.R. (1969). "Robustness of the rescaled
    range R/S in the measurement of noncyclical long-run statistical
    dependence." Water Resources Research, 5(5), 967-988.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def volatility_regimes(
    prices: pd.Series,
    window: int = 21,
    n_regimes: int = 5,
) -> pd.Series:
    """
    Classify each day into a volatility regime (1 = lowest, n = highest).

    Uses rolling realized volatility (annualized), then ranks into quantile
    buckets on an expanding basis so there is no lookahead.

    Args:
        prices: Daily close prices with DatetimeIndex
        window: Rolling window for realized vol (default 21 trading days)
        n_regimes: Number of regimes / quantile buckets (default 5)

    Returns:
        pd.Series of integers 1..n_regimes (NaN where not enough history)
    """
    returns = prices.pct_change()
    roll_vol = returns.rolling(window).std() * np.sqrt(252)

    # Expanding quantile labels to avoid lookahead
    labels = pd.Series(np.nan, index=prices.index)
    min_periods = window * 2  # need 2x window to form stable quantiles
    for i in range(min_periods, len(roll_vol)):
        snap = roll_vol.iloc[: i + 1].dropna()
        if len(snap) < n_regimes:
            continue
        quantiles = [snap.quantile(k / n_regimes) for k in range(1, n_regimes)]
        val = roll_vol.iloc[i]
        if pd.isna(val):
            continue
        bucket = 1
        for q in quantiles:
            if val > q:
                bucket += 1
        labels.iloc[i] = bucket

    labels.name = "vol_regime"
    return labels


def hurst_regime(
    prices: pd.Series,
    window: int = 126,
    high_threshold: float = 0.55,
    low_threshold: float = 0.45,
) -> pd.Series:
    """
    Classify each day into a trend regime using the Hurst exponent.

    Regimes:
        1 — trending (H > high_threshold)
        0 — random walk (low_threshold <= H <= high_threshold)
       -1 — mean-reverting (H < low_threshold)

    Args:
        prices: Daily close prices
        window: Lookback window for R/S analysis (default 126 = 6 months)
        high_threshold: H above this → trending (default 0.55)
        low_threshold: H below this → mean-reverting (default 0.45)

    Returns:
        pd.Series of {-1, 0, 1} regime labels
    """
    labels = pd.Series(np.nan, index=prices.index, dtype=float)
    log_p = np.log(prices.values)

    for i in range(window, len(prices)):
        chunk = log_p[i - window: i]
        h = _hurst_rs(chunk)
        if h > high_threshold:
            labels.iloc[i] = 1.0
        elif h < low_threshold:
            labels.iloc[i] = -1.0
        else:
            labels.iloc[i] = 0.0

    labels.name = "hurst_regime"
    return labels


def _hurst_rs(log_prices: np.ndarray, min_n: int = 10) -> float:
    """
    Estimate Hurst exponent via R/S (rescaled range) analysis.

    Fits log(R/S) ~ H * log(n) over sub-period lengths n.
    Returns H in [0, 1]; 0.5 means random walk.
    """
    returns = np.diff(log_prices)
    n_total = len(returns)
    if n_total < min_n * 2:
        return 0.5

    ns = []
    rs_vals = []
    n = min_n
    while n <= n_total // 2:
        n_blocks = n_total // n
        block_rs = []
        for b in range(n_blocks):
            seg = returns[b * n: (b + 1) * n]
            mean_seg = np.mean(seg)
            devs = np.cumsum(seg - mean_seg)
            r = devs.max() - devs.min()
            s = np.std(seg, ddof=1) if len(seg) > 1 else 1.0
            if s > 0:
                block_rs.append(r / s)
        if block_rs:
            ns.append(n)
            rs_vals.append(np.mean(block_rs))
        n = int(n * 1.5) + 1

    if len(ns) < 2:
        return 0.5

    log_n = np.log(ns)
    log_rs = np.log(rs_vals)
    slope = np.polyfit(log_n, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Conditional performance by regime
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    """Performance statistics for a single regime."""
    regime: str
    n_days: int
    mean_daily_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    hit_rate: float      # fraction of days with positive return
    max_drawdown: float


def conditional_performance(
    strategy_returns: pd.Series,
    regime_labels: pd.Series,
    regime_names: Optional[Dict] = None,
    annualization: int = 252,
    min_obs: int = 21,
) -> Dict[str, RegimeStats]:
    """
    Compute conditional strategy performance within each regime.

    Args:
        strategy_returns: Daily strategy returns
        regime_labels: Regime classification (same index)
        regime_names: Optional dict mapping regime value → display name
        annualization: 252 for daily
        min_obs: Minimum observations to compute stats (else skipped)

    Returns:
        Dict mapping regime value (as str) to RegimeStats
    """
    aligned = strategy_returns.align(regime_labels, join="inner")
    ret = aligned[0].dropna()
    labels = aligned[1].reindex(ret.index)

    results = {}
    for regime_val in sorted(labels.dropna().unique()):
        mask = labels == regime_val
        seg = ret[mask].dropna()
        if len(seg) < min_obs:
            continue

        name_key = str(regime_val)
        if regime_names and regime_val in regime_names:
            display = regime_names[regime_val]
        else:
            display = name_key

        ann_ret = float((1 + seg.mean()) ** annualization - 1)
        ann_vol = float(seg.std() * np.sqrt(annualization))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        max_dd = float(_max_drawdown(seg))
        hit = float((seg > 0).mean())

        results[display] = RegimeStats(
            regime=display,
            n_days=len(seg),
            mean_daily_return=float(seg.mean()),
            annualized_return=ann_ret,
            annualized_vol=ann_vol,
            sharpe=sharpe,
            hit_rate=hit,
            max_drawdown=max_dd,
        )
    return results


def _max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from a returns series."""
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max
    return float(dd.min())


# ---------------------------------------------------------------------------
# Drawdown analysis
# ---------------------------------------------------------------------------

@dataclass
class DrawdownEvent:
    """A single drawdown episode."""
    rank: int
    peak_date: str
    trough_date: str
    recovery_date: str       # "" if not yet recovered
    depth: float             # max decline from peak (negative)
    duration_days: int       # peak → trough
    recovery_days: int       # trough → recovery (0 if not recovered)
    underwater_days: int     # total days from peak to recovery


def top_drawdowns(
    strategy_returns: pd.Series,
    n: int = 5,
) -> List[DrawdownEvent]:
    """
    Identify and characterize the top-N worst drawdown events.

    Args:
        strategy_returns: Daily strategy returns
        n: Number of worst drawdowns to return

    Returns:
        List of DrawdownEvent sorted by depth (worst first)
    """
    wealth = (1 + strategy_returns.fillna(0)).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max

    events = _extract_drawdown_events(wealth, dd)
    events.sort(key=lambda e: e.depth)  # most negative first
    return events[:n]


def _extract_drawdown_events(
    wealth: pd.Series,
    drawdown: pd.Series,
) -> List[DrawdownEvent]:
    """
    Extract distinct drawdown events from a wealth / drawdown series.

    Each event:
      - starts at a local peak (drawdown transitions from 0 to negative)
      - ends at trough (minimum in the episode)
      - recovers when drawdown returns to 0 (or end of series)
    """
    events = []
    in_dd = False
    peak_idx = None
    trough_idx = None
    trough_depth = 0.0
    rank = 1
    idx = drawdown.index

    for i, (dt, val) in enumerate(drawdown.items()):
        if not in_dd:
            if val < -1e-8:
                in_dd = True
                # Find the last peak before this drawdown started
                peak_idx = idx[max(0, i - 1)]
                trough_idx = dt
                trough_depth = val
        else:
            if val < trough_depth:
                trough_depth = val
                trough_idx = dt
            if val >= -1e-8:
                # Recovered
                recovery_idx = dt
                peak_pos = drawdown.index.get_loc(peak_idx)
                trough_pos = drawdown.index.get_loc(trough_idx)
                recov_pos = drawdown.index.get_loc(recovery_idx)
                events.append(DrawdownEvent(
                    rank=rank,
                    peak_date=str(peak_idx.date()),
                    trough_date=str(trough_idx.date()),
                    recovery_date=str(recovery_idx.date()),
                    depth=trough_depth,
                    duration_days=trough_pos - peak_pos,
                    recovery_days=recov_pos - trough_pos,
                    underwater_days=recov_pos - peak_pos,
                ))
                rank += 1
                in_dd = False
                peak_idx = None
                trough_depth = 0.0

    if in_dd and peak_idx is not None:
        # Still underwater at end of series
        peak_pos = drawdown.index.get_loc(peak_idx)
        trough_pos = drawdown.index.get_loc(trough_idx)
        last_pos = len(drawdown) - 1
        events.append(DrawdownEvent(
            rank=rank,
            peak_date=str(peak_idx.date()),
            trough_date=str(trough_idx.date()),
            recovery_date="",
            depth=trough_depth,
            duration_days=trough_pos - peak_pos,
            recovery_days=0,
            underwater_days=last_pos - peak_pos,
        ))

    return events


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_regime_report(
    vol_stats: Optional[Dict[str, RegimeStats]] = None,
    hurst_stats: Optional[Dict[str, RegimeStats]] = None,
    drawdowns: Optional[List[DrawdownEvent]] = None,
) -> str:
    """Format regime analysis results as a Markdown section for REPORT.md."""
    lines = ["## Regime Analysis", ""]

    if vol_stats:
        lines += [
            "### Volatility Regimes",
            "",
            "| Regime | Days | Ann. Return | Ann. Vol | Sharpe |"
            " Hit Rate | Max DD |",
            "|--------|------|-------------|----------|--------|"
            "----------|--------|",
        ]
        for name, s in sorted(vol_stats.items()):
            lines.append(
                f"| {name} | {s.n_days} | {s.annualized_return:.2%} |"
                f" {s.annualized_vol:.2%} | {s.sharpe:.2f} |"
                f" {s.hit_rate:.1%} | {s.max_drawdown:.2%} |"
            )
        lines.append("")

    if hurst_stats:
        regime_map = {
            "1.0": "Trending", "0.0": "Random Walk", "-1.0": "Mean-Reverting",
        }
        lines += [
            "### Trend Regimes (Hurst R/S)",
            "",
            "| Regime | Days | Ann. Return | Ann. Vol | Sharpe |"
            " Hit Rate | Max DD |",
            "|--------|------|-------------|----------|--------|"
            "----------|--------|",
        ]
        for name, s in hurst_stats.items():
            display = regime_map.get(name, name)
            lines.append(
                f"| {display} | {s.n_days} | {s.annualized_return:.2%} |"
                f" {s.annualized_vol:.2%} | {s.sharpe:.2f} |"
                f" {s.hit_rate:.1%} | {s.max_drawdown:.2%} |"
            )
        lines.append("")

    if drawdowns:
        lines += [
            "### Top Drawdowns",
            "",
            "| # | Peak | Trough | Recovery | Depth | Duration"
            " | Recovery Days |",
            "|---|------|--------|----------|-------|---------|"
            "---------------|",
        ]
        for d in drawdowns:
            rec = d.recovery_date if d.recovery_date else "—"
            lines.append(
                f"| {d.rank} | {d.peak_date} | {d.trough_date} |"
                f" {rec} | {d.depth:.2%} |"
                f" {d.duration_days}d | {d.recovery_days}d |"
            )
        lines.append("")

    return "\n".join(lines)


def regime_report_dict(
    vol_stats: Optional[Dict[str, RegimeStats]] = None,
    hurst_stats: Optional[Dict[str, RegimeStats]] = None,
    drawdowns: Optional[List[DrawdownEvent]] = None,
) -> dict:
    """Serializable dict for JSON output."""
    result: dict = {}

    def _stats_to_list(stats_dict):
        return [
            {
                "regime": s.regime,
                "n_days": s.n_days,
                "annualized_return": s.annualized_return,
                "annualized_vol": s.annualized_vol,
                "sharpe": s.sharpe,
                "hit_rate": s.hit_rate,
                "max_drawdown": s.max_drawdown,
            }
            for s in stats_dict.values()
        ]

    if vol_stats:
        result["volatility_regimes"] = _stats_to_list(vol_stats)
    if hurst_stats:
        result["hurst_regimes"] = _stats_to_list(hurst_stats)
    if drawdowns:
        result["top_drawdowns"] = [
            {
                "rank": d.rank,
                "peak_date": d.peak_date,
                "trough_date": d.trough_date,
                "recovery_date": d.recovery_date,
                "depth": d.depth,
                "duration_days": d.duration_days,
                "recovery_days": d.recovery_days,
                "underwater_days": d.underwater_days,
            }
            for d in drawdowns
        ]
    return result
