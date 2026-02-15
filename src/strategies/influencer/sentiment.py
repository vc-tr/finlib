"""
Sentiment-based trading strategy.

Designed for integration with external sentiment sources:
- Reddit (r/wallstreetbets, r/stocks) via sentiment APIs
- Twitter/X via FinBERT or VADER
- News headlines via FinBERT

When sentiment_scores are provided: trades in direction of sentiment when it exceeds threshold.
Scores typically in [-1, 1] (negative=bearish, positive=bullish).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class SentimentStrategy:
    """
    Trade based on sentiment scores (e.g. from Reddit, Twitter, news).
    
    Signal = sign(sentiment) when |sentiment| > threshold.
    Use with: FinBERT, VADER, or custom sentiment APIs.
    """

    def __init__(
        self,
        bullish_threshold: float = 0.3,
        bearish_threshold: float = -0.3,
        exit_threshold: float = 0.1,
    ):
        """
        Args:
            bullish_threshold: Min sentiment to go long
            bearish_threshold: Max sentiment to go short
            exit_threshold: Revert to flat when |sentiment| < this
        """
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(
        self,
        sentiment_scores: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Generate signals from sentiment.
        
        Args:
            sentiment_scores: Sentiment in [-1, 1], index-aligned with prices
            prices: Optional, for index alignment only
        """
        s = sentiment_scores.dropna()
        if prices is not None:
            s = s.reindex(prices.index).ffill().bfill()
        signals = pd.Series(0.0, index=s.index)
        position = 0
        for i in range(len(s)):
            val = s.iloc[i]
            if np.isnan(val):
                continue
            if position == 0:
                if val >= self.bullish_threshold:
                    position = 1
                elif val <= self.bearish_threshold:
                    position = -1
            elif position == 1:
                if val <= self.exit_threshold:
                    position = 0
            elif position == -1:
                if val >= -self.exit_threshold:
                    position = 0
            signals.iloc[i] = position
        return signals.shift(1)

    def backtest_returns(
        self,
        sentiment_scores: pd.Series,
        prices: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Returns (signals, strategy_returns)."""
        signals = self.generate_signals(sentiment_scores, prices)
        returns = prices.pct_change()
        strategy_returns = signals * returns
        return signals, strategy_returns
