"""
D.E. Shaw-style pairs trading with Kalman filter.

Uses Kalman filter for dynamic hedge ratio (vs static OLS).
Adapts to changing market conditions. Inspired by stat arb at Morgan Stanley APT / DE Shaw.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    from pykalman import KalmanFilter
    HAS_PYKALMAN = True
except ImportError:
    HAS_PYKALMAN = False


class KalmanPairsStrategy:
    """
    Pairs trading with Kalman filter for dynamic hedge ratio.
    
    Spread = y - beta * x, where beta is estimated by Kalman filter.
    Trade when z-score of spread exceeds thresholds.
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        transition_cov: float = 1e-5,
        observation_cov: float = 1e-4,
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.transition_cov = transition_cov
        self.observation_cov = observation_cov

    def _kalman_hedge_ratio(
        self, y: np.ndarray, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate dynamic hedge ratio (beta) and spread via Kalman filter."""
        if not HAS_PYKALMAN:
            # Fallback: rolling OLS
            n = len(y)
            beta = np.full(n, np.nan)
            for i in range(30, n):
                X = x[i - 30 : i].reshape(-1, 1)
                Y = y[i - 30 : i]
                X = np.column_stack([np.ones(len(X)), X])
                coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
                beta[i] = coeffs[1]
            spread = y - beta * x
            return beta, spread
        # State: [alpha, beta], observation: y = alpha + beta * x
        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=np.column_stack([np.ones_like(x), x]),
            initial_state_mean=[0, 1],
            initial_state_covariance=np.eye(2) * 1,
            transition_covariance=np.eye(2) * self.transition_cov,
            observation_covariance=self.observation_cov,
        )
        state_means, _ = kf.filter(y)
        alpha, beta = state_means[:, 0], state_means[:, 1]
        spread = y - (alpha + beta * x)
        return beta, spread

    def generate_signals(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Returns:
            (signals, spread, hedge_ratio_series)
        """
        a, b = price_a.align(price_b, join="inner")
        y = a.values.astype(float)
        x = b.values.astype(float)
        beta, spread_arr = self._kalman_hedge_ratio(y, x)
        spread = pd.Series(spread_arr, index=a.index)
        mean = spread.rolling(60).mean()
        std = spread.rolling(60).std()
        z = (spread - mean) / std.replace(0, np.nan)
        signals = pd.Series(0.0, index=spread.index)
        position = 0
        for i in range(60, len(spread)):
            if np.isnan(z.iloc[i]):
                continue
            if position == 0:
                if z.iloc[i] < -self.entry_z:
                    position = 1
                elif z.iloc[i] > self.entry_z:
                    position = -1
            elif position == 1:
                if z.iloc[i] >= -self.exit_z:
                    position = 0
            elif position == -1:
                if z.iloc[i] <= self.exit_z:
                    position = 0
            signals.iloc[i] = position
        return signals, spread, pd.Series(beta, index=a.index)

    def backtest_returns(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Returns (signals, strategy_returns). Uses spread diff normalized by price scale."""
        signals, spread, _ = self.generate_signals(price_a, price_b)
        scale = price_a.rolling(20).mean().abs()
        scale = scale.replace(0, np.nan).ffill().bfill()
        spread_ret = spread.diff() / scale
        spread_ret = spread_ret.fillna(0).clip(-0.1, 0.1)
        strategy_returns = signals.shift(1) * spread_ret
        return signals, strategy_returns
