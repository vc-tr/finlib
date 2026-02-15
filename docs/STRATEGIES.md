# Strategy Reference

Detailed reference for all strategies, their sources, and recommended use cases.

---

## Academic Paper Strategies

See **[docs/PAPERS.md](PAPERS.md)** for full citations and usage.

| Paper | Strategy | Key idea |
|-------|----------|----------|
| Moskowitz et al. (2012) | Time Series Momentum | 12m return predicts next month |
| Jegadeesh & Titman (1993) | Cross-sectional momentum | Long winners, short losers |
| Gatev et al. (2006) | Pairs (min distance) | Normalized price SSD matching |
| De Bondt & Thaler (1985) | Long-term reversal | Buy losers, sell winners |

---

## Statistical Arbitrage

### Mean Reversion
- **Logic**: Z-score of price vs rolling mean; trade when |z| > threshold
- **Source**: Classic quant finance
- **Best for**: Range-bound markets, mean-reverting assets

### Pairs Trading (OLS)
- **Logic**: Cointegration + spread z-score; OLS hedge ratio
- **Source**: Statistical arbitrage, Morgan Stanley APT
- **Best for**: Cointegrated pairs (e.g. SPY/IVV, sector ETFs)

### Kalman Pairs (D.E. Shaw-style)
- **Logic**: Dynamic hedge ratio via Kalman filter; adapts to regime changes
- **Source**: D.E. Shaw, Morgan Stanley APT (publicly documented)
- **Best for**: Pairs with time-varying relationship
- **Optional**: `pip install pykalman` for full Kalman; falls back to rolling OLS otherwise

---

## Momentum & Trend

### Momentum
- **Logic**: Sign of lookback return
- **Source**: Academic (Jegadeesh, Titman)
- **Best for**: Trending markets

### Renaissance Ensemble
- **Logic**: Combines momentum, mean reversion, volatility regime
- **Source**: Inspired by Renaissance Technologies (pattern-based, multi-signal)
- **Best for**: Adaptive, regime-agnostic

---

## Influencer / Sentiment

### Sentiment Strategy
- **Logic**: Trade in direction of sentiment when |score| > threshold
- **Source**: Reddit/Twitter sentiment research (VADER, FinBERT)
- **Input**: Sentiment scores in [-1, 1] from external API (e.g. Reddit scraper, FinBERT)
- **Best for**: Meme stocks, event-driven

### Volume Sentiment
- **Logic**: Volume spike (> N × avg) + price momentum as crowd proxy
- **Source**: WSB-style retail flow proxy
- **Best for**: When no explicit sentiment data; proxy for influencer-driven moves

---

## Day Trader

### Scalping (EMA Crossover)
- **Logic**: Fast EMA crosses slow EMA; hold until opposite
- **Source**: Classic technical analysis
- **Best for**: Intraday (1m, 5m)

### Opening Range Breakout (ORB)
- **Logic**: First N bars define range; trade break of high/low
- **Source**: Popular day trading strategy
- **Best for**: Intraday, first 15–30 min of session

### EMA + Stochastic
- **Logic**: EMA trend filter + Stochastic overbought/oversold
- **Source**: Technical analysis combo
- **Best for**: Intraday, swing

---

## Institutional

### VWAP Reversion
- **Logic**: Mean reversion to VWAP (z-score of price − VWAP)
- **Source**: Institutional benchmark; algo execution
- **Best for**: Intraday, large-cap

### ATR Breakout
- **Logic**: Breakout when price exceeds recent high/low ± k×ATR
- **Source**: Volatility targeting, institutional
- **Best for**: Volatile regimes

---

## Option Pricing

### Black-Scholes
- **Source**: Black, Scholes, Merton (1973)
- **Use**: European options, Greeks

### Monte Carlo
- **Source**: Numerical methods for path-dependent options
- **Use**: Asian, barrier, exotic options

---

## Strategy Selection Guide

| Market regime   | Suggested strategies                    |
|-----------------|----------------------------------------|
| Range-bound     | Mean reversion, VWAP reversion, Pairs  |
| Trending        | Momentum, Scalping, ATR breakout       |
| High volatility | ATR breakout, ORB                      |
| Meme / social   | Sentiment, Volume sentiment            |
| Intraday        | Scalping, ORB, EMA+Stochastic, VWAP   |
| Pairs           | Pairs (OLS), Kalman pairs              |
