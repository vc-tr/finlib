"""
Hurst Exponent Regime Switching — R/S Analysis.

The Hurst exponent (H) characterizes the long-range dependence in a time series:
  H > 0.5: Trending / persistent (trend-following works)
  H = 0.5: Random walk (no predictability)
  H < 0.5: Mean-reverting / anti-persistent (mean-reversion works)

By estimating H on a rolling basis, we can switch between momentum and
mean-reversion strategies depending on the current market regime.

Reference: Hurst (1951) — "Long-term storage capacity of reservoirs"
           Peters (1994) — "Fractal Market Analysis"
           Lo (1991) — "Long-term memory in stock market prices"

Key insight: The Hurst exponent is NOT reliably estimated from daily equity
returns. Most financial series appear to have H ≈ 0.5 at daily frequency.
However, the regime-switching concept is valuable and the estimation noise
itself provides useful uncertainty signal.

Expected: Modest improvement over pure momentum or mean-reversion by
switching regimes. Estimation noise limits OOS Sharpe gains.
"""

import numpy as np
import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


def _hurst_rs(prices: pd.Series, min_n: int = 10) -> float:
    """
    Estimate Hurst exponent via Rescaled Range (R/S) analysis.

    Splits the series into sub-periods, computes R/S for each,
    then fits log(R/S) ~ H * log(n) via OLS.

    Args:
        prices: Price series (log returns used internally)
        min_n: Minimum sub-period length

    Returns:
        Estimated Hurst exponent H ∈ [0, 1], or 0.5 if estimation fails
    """
    log_returns = np.diff(np.log(prices.values))
    n = len(log_returns)
    if n < min_n * 2:
        return 0.5

    rs_list = []
    n_list = []

    for sub_n in range(min_n, n // 2 + 1, max(1, n // 20)):
        n_subseries = n // sub_n
        if n_subseries < 2:
            break
        rs_vals = []
        for i in range(n_subseries):
            sub = log_returns[i * sub_n: (i + 1) * sub_n]
            mean_adj = sub - sub.mean()
            cumdev = np.cumsum(mean_adj)
            r = cumdev.max() - cumdev.min()
            s = sub.std(ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            n_list.append(sub_n)

    if len(n_list) < 3:
        return 0.5

    log_n = np.log(n_list)
    log_rs = np.log(rs_list)
    # OLS: log(RS) = H * log(n) + c
    H = np.polyfit(log_n, log_rs, 1)[0]
    return float(np.clip(H, 0.0, 1.0))


@StrategyRegistry.register
class HurstExponentStrategy(Strategy):
    """
    Regime-adaptive strategy using rolling Hurst exponent estimate.

    - H > trending_threshold → trend-following (long if up, short if down)
    - H < reverting_threshold → mean-reversion (short if up, long if down)
    - Otherwise → flat (uncertain regime)
    """

    def __init__(
        self,
        hurst_window: int = 126,
        signal_lookback: int = 21,
        trending_threshold: float = 0.55,
        reverting_threshold: float = 0.45,
    ):
        """
        Args:
            hurst_window: Rolling window to estimate Hurst (days)
            signal_lookback: Return lookback for direction signal
            trending_threshold: H above this → trending regime
            reverting_threshold: H below this → mean-reverting regime
        """
        self.hurst_window = hurst_window
        self.signal_lookback = signal_lookback
        self.trending_threshold = trending_threshold
        self.reverting_threshold = reverting_threshold

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="hurst_exponent",
            category="econophysics",
            source="Hurst (1951) / Peters (1994) — Fractal Market Analysis",
            description=(
                f"Trend-follow when rolling H > {self.trending_threshold}, "
                f"mean-revert when H < {self.reverting_threshold}"
            ),
            hypothesis=(
                "Financial markets switch between trending (H > 0.5) and "
                "mean-reverting (H < 0.5) regimes. R/S analysis detects these "
                "regime shifts and allows adaptive strategy selection."
            ),
            expected_result=(
                "Modest Sharpe improvement vs fixed strategies. "
                "R/S estimation is noisy at daily frequency — "
                "true H for equity returns is close to 0.5, "
                "limiting regime discrimination power."
            ),
            source_url=None,
            tags=["hurst", "fractal", "regime", "econophysics", "adaptive"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        signals = pd.Series(0.0, index=prices.index)

        for i in range(self.hurst_window, len(prices)):
            window_prices = prices.iloc[i - self.hurst_window: i]
            H = _hurst_rs(window_prices)

            # Direction from recent return
            if i < self.signal_lookback:
                continue
            ret = (prices.iloc[i - 1] / prices.iloc[i - self.signal_lookback] - 1)
            direction = 1.0 if ret > 0 else -1.0

            if H > self.trending_threshold:
                signals.iloc[i] = direction        # trend-follow
            elif H < self.reverting_threshold:
                signals.iloc[i] = -direction       # mean-revert
            # else: flat (uncertain regime)

        return signals

    def parameter_grid(self):
        return {
            "hurst_window": [63, 126],
            "trending_threshold": [0.55, 0.60],
            "reverting_threshold": [0.40, 0.45],
        }
