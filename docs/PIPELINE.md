# Pipeline Documentation

## Data Pipeline

### 1. Data Acquisition

- **YahooDataFetcher**: Fetches OHLCV via `yfinance`
- **AlphaDataFetcher**: Alternative for Alpha Vantage (requires API key)
- Supports intervals: `1m`, `1h`, `1d`
- Supports period: `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`

### 2. Preprocessing

- **reindex_and_backfill**: For minute data, creates complete 1-min index, forward-fills OHLC, zero-fills volume
- Optional: filter to regular trading hours (09:30–16:00) and weekdays

### 3. Feature Engineering

- **TechnicalIndicators**: RSI, Bollinger Bands, MACD, VWAP
- **create_features()**: One-call to add all indicators to OHLCV DataFrame

### 4. Model Input Preparation

For deep learning (train.py):

1. Compute `pct_change()` on close
2. Create sliding windows of size `SEQ_LEN`
3. Target = next value
4. Random split 80/20 (consider time-based split for production)

---

## Strategy Pipeline

### Rule-Based Strategies

1. **Input**: Price series (or pair of series for pairs trading)
2. **Signal generation**: Strategy-specific logic (z-score, momentum, etc.)
3. **Output**: Discrete signals -1, 0, 1
4. **Backtest**: `Backtester.run_from_signals(prices, signals)`

### Option Pricing

1. **Black-Scholes**: Direct formula; requires S, K, T, r, sigma
2. **Monte Carlo**: Simulate paths under GBM; average discounted payoffs

---

## Training Pipeline

1. `prepare_data()` → train_ds, val_ds
2. `create_model(model_type)` → PyTorch model
3. Train loop: forward → loss → backward → step
4. Early stop on validation loss
5. Log to MLflow + TensorBoard
6. Save best model to MLflow artifacts

---

## Backtest Pipeline

1. Load prices
2. Instantiate strategy
3. `strategy.backtest_returns(prices)` → signals, strategy_returns
4. `Backtester().run_from_signals(prices, signals)` → BacktestResult
5. Metrics: Sharpe, total return, max drawdown, win rate
