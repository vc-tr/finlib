# Archived Modules

Strategies and components that are not production-ready or are experimental.
Kept for reference; not part of the flagship demo.

## Archived Strategies

- **Influencer / Sentiment**: SentimentStrategy, VolumeSentimentStrategy
  - Require external sentiment APIs (Reddit, Twitter); not tested end-to-end
  - Volume-sentiment is a proxy without real sentiment data

- **Deep Learning models**: LSTM, Transformer, TCN, etc.
  - Require full training pipeline + evaluation; not in core demo
  - See `train.py` for DL training

## Usage

Archived strategies remain importable via `src.strategies` for backward compatibility.
The demo and quickstart use only: Momentum, Mean Reversion, Renaissance Ensemble.
