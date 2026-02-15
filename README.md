# Quant Forecast

A comprehensive quantitative finance research and trading framework with multiple models and strategies—from classical option pricing to deep learning and statistical arbitrage.

## Overview

This project provides:

- **Option pricing**: Black-Scholes, Monte Carlo simulation
- **Stochastic models**: Brownian motion, GBM, Ornstein-Uhlenbeck
- **Statistical strategies**: Mean reversion, pairs trading, momentum
- **Renaissance-style**: Multi-signal ensemble (pattern-based)
- **Influencer / sentiment**: Sentiment-based (Reddit/Twitter proxy), volume-sentiment
- **Day trader**: Scalping (EMA crossover), Opening Range Breakout, EMA+Stochastic
- **Institutional**: D.E. Shaw-style Kalman pairs, VWAP reversion, ATR breakout
- **Deep learning**: LSTM, BiLSTM, GRU, Transformer, TCN
- **Backtesting**: Unified engine for strategy evaluation

---

## Project Structure

```
quant-forecast/
├── src/
│   ├── models/           # Deep learning models
│   │   ├── lstm.py       # UniLSTM
│   │   ├── bilstm.py     # BiLSTM
│   │   ├── gru.py        # GRU
│   │   ├── transformer.py
│   │   ├── tcn.py        # Temporal Convolutional Network
│   │   └── factory.py    # Model creation
│   ├── strategies/      # Quantitative strategies
│   │   ├── options/      # Black-Scholes, Monte Carlo
│   │   ├── stochastic/   # Brownian, GBM, Ornstein-Uhlenbeck
│   │   ├── stats/        # Mean reversion, pairs, momentum
│   │   ├── renaissance/  # Signal ensemble
│   │   ├── influencer/   # Sentiment, volume-sentiment
│   │   ├── daytrader/    # Scalping, ORB, EMA+Stochastic
│   │   ├── institutional/ # Kalman pairs, VWAP, ATR breakout
│   │   └── papers/        # Academic: Moskowitz, JT, GGR, De Bondt-Thaler
│   ├── pipeline/         # Data & preprocessing
│   │   ├── data_fetcher*.py
│   │   ├── pipeline.py
│   │   ├── features.py   # RSI, MACD, Bollinger, VWAP
│   │   └── baselines.py  # Persistence, MA, ARIMA
│   └── backtest/         # Backtesting engine
├── scripts/
│   ├── backtest_strategies.py   # Run strategy backtests
│   ├── backtest_baselines.py   # Baseline forecast evaluation
│   └── run_options_demo.py     # Black-Scholes & Monte Carlo demo
├── tests/
└── train.py             # Main training entry point
```

---

## Pipeline

### 1. Data Flow

```
Data Source (Yahoo/Alpha) → Fetch OHLCV → Reindex & Backfill → Features → Model/Strategy
```

- **Data fetchers**: `YahooDataFetcher`, `AlphaDataFetcher` (abstract interface)
- **Pipeline**: `reindex_and_backfill()` for minute data; forward-fill OHLC, zero-fill volume
- **Features**: RSI, Bollinger Bands, MACD, VWAP via `TechnicalIndicators`

### 2. Model Pipeline (Deep Learning)

```
Raw OHLCV → % returns → Sliding windows (seq_len) → Train/Val split → Model → Predict
```

- **Input**: Percentage change of close price over `SEQ_LEN` steps
- **Output**: Next-step return prediction
- **Models**: LSTM, BiLSTM, GRU, Transformer, TCN
- **Training**: Adam, ReduceLROnPlateau, early stopping, gradient clipping
- **Logging**: MLflow, TensorBoard

### 3. Strategy Pipeline (Rule-Based)

```
Prices → Strategy logic → Signals (-1, 0, 1) → Backtester → Metrics
```

- **Mean reversion**: Z-score of price vs rolling mean
- **Momentum**: Sign of lookback return
- **Pairs trading**: Cointegration + spread z-score
- **Renaissance**: Combined momentum, mean reversion, volatility regime

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `pandas`, `numpy`, `scipy`, `statsmodels`, `yfinance`, `mlflow`, `scikit-learn`.

### Train a Deep Learning Model

```bash
# Default: LSTM with MSE loss
python train.py

# Choose model and loss
python train.py --model_type transformer --loss huber --epochs 30

# Available models: lstm, bilstm, gru, transformer, tcn
# Available losses: mse, mae, huber
```

### Backtest Strategies

```bash
# Core strategies (mean reversion, momentum, Renaissance)
python scripts/backtest_strategies.py --symbol SPY --period 365d

# All strategies (incl. scalping, ORB, VWAP, ATR, volume-sentiment)
python scripts/backtest_strategies.py --symbol SPY --period 365d --all

# Paper strategies (Moskowitz TSMOM, De Bondt-Thaler)
python scripts/backtest_papers.py --symbol SPY --period 5y

# GGR pairs (Gatev et al. 2006)
python scripts/backtest_papers.py --symbol1 SPY --symbol2 IVV --period 2y

# Baseline forecasts (persistence, MA, ARIMA)
python scripts/backtest_baselines.py --symbol SPY --period 365d
```

### Option Pricing Demo

```bash
python scripts/run_options_demo.py
```

### Use Strategies Programmatically

```python
from src.strategies import MeanReversionStrategy, BlackScholes, MonteCarloPricer
from src.backtest import Backtester

# Mean reversion
mr = MeanReversionStrategy(lookback=20, entry_z=2.0)
signals, returns = mr.backtest_returns(prices)
result = Backtester().run_from_signals(prices, signals)
print(f"Sharpe: {result.sharpe_ratio:.2f}, Return: {result.total_return:.2%}")

# Black-Scholes
bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
print(bs.call_price(), bs.delta("call"))

# Monte Carlo
mc = MonteCarloPricer(S=100, K=100, T=1, r=0.05, sigma=0.2, n_paths=100000)
print(mc.price("european", "call"))
```

---

## Models & Strategies Reference

### Deep Learning Models

| Model       | Description                          | Best for                    |
|------------|--------------------------------------|-----------------------------|
| LSTM       | Unidirectional LSTM                 | General sequence modeling   |
| BiLSTM     | Bidirectional LSTM                   | Context from both directions|
| GRU        | Gated Recurrent Unit                | Faster, fewer parameters    |
| Transformer| Self-attention over sequence        | Long-range dependencies     |
| TCN        | Temporal Convolutional Network      | Causal, dilated convolutions |

### Option Pricing

| Model        | Type        | Use case                          |
|-------------|-------------|-----------------------------------|
| Black-Scholes| Analytical  | European options, Greeks          |
| Monte Carlo | Simulation  | Path-dependent, exotic options   |

### Stochastic Processes

| Process            | Use case                    |
|--------------------|-----------------------------|
| Brownian Motion    | Random walk, diffusion      |
| GBM                | Stock prices, Black-Scholes |
| Ornstein-Uhlenbeck | Mean reversion, rates, spreads |

### Trading Strategies

| Strategy   | Logic                          | Source / Use case        |
|-----------|---------------------------------|--------------------------|
| Mean reversion | Z-score vs rolling mean     | Range-bound              |
| Momentum       | Past return sign             | Trending                 |
| Pairs trading  | Cointegrated spread z-score  | Stat arb                 |
| Renaissance    | Multi-signal ensemble        | Adaptive                 |
| **Sentiment**  | External sentiment scores    | Reddit/Twitter/News      |
| **Volume sentiment** | Volume spike + momentum  | Crowd/influencer proxy   |
| **Scalping**   | EMA crossover                | Day trading              |
| **ORB**        | Opening range breakout       | Day trading              |
| **EMA+Stochastic** | EMA + Stoch overbought/oversold | Day trading       |
| **Kalman pairs** | Dynamic hedge ratio (Kalman) | D.E. Shaw-style stat arb |
| **VWAP reversion** | Mean revert to VWAP       | Institutional            |
| **ATR breakout**  | Volatility breakout         | Institutional            |

---

## Recommendations

### Data

- **Daily**: Use `interval="1d"` for most strategies; `period="2y"` or more for robustness
- **Minute**: Use `reindex_and_backfill` for 1m data; consider regular hours only for equities
- **Pairs**: Ensure sufficient history for cointegration tests; avoid overfitting on in-sample pairs

### Training (Deep Learning)

- **Seq length**: 30–60 for daily; 20–30 for intraday
- **Validation**: Use time-based split (no shuffle) to avoid lookahead
- **Loss**: `huber` often more robust than MSE for noisy returns
- **Early stopping**: `patience=5` is reasonable; increase for transformer/TCN

### Strategies

- **Mean reversion**: Tune `entry_z` (1.5–2.5) and `lookback` (10–30) per asset
- **Momentum**: Longer lookback (20–60) for daily; shorter for intraday
- **Pairs**: Validate cointegration on out-of-sample; use rolling hedge ratio

### Backtesting

- Use transaction costs and slippage in production
- Avoid survivorship bias when selecting symbols
- Walk-forward or expanding-window validation for robustness

---

## Trading Platform Integration

To connect strategies to paper or live trading, see **[docs/INTEGRATION.md](docs/INTEGRATION.md)** for:

- **Alpaca** (recommended): Free paper + live, commission-free, Python SDK
- **Tradier**: Free sandbox, stocks & options
- **Interactive Brokers**: Paper account, global markets

---

## Configuration

Key hyperparameters in `train.py`:

```python
SEQ_LEN = 30      # Input sequence length
BATCH_SIZE = 64
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 20
PATIENCE = 5
CLIP_VALUE = 1.0  # Gradient clipping
```

---

## Strategy Details

- **[docs/STRATEGIES.md](docs/STRATEGIES.md)** — Sources, logic, selection guide
- **[docs/PAPERS.md](docs/PAPERS.md)** — Academic paper strategies with citations

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT
