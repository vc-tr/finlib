# System Design

## 1. Overview
High-level flow:  
1. DataFetcher →  
2. Pipeline (reindex, backfill, feature‐engineering) →  
3. Model training (LSTM/variants) →  
4. Backtester/strategy →  
5. API & Dashboard deployment.

---

## 2. Data Layer
- **DataFetcher**: abstract base + concrete implementations (`yfinance`, `AlphaVantage`)  
- **Reindex/Backfill**: full 1 min index, ffill, drop head NaNs.

---

## 3. Pipeline
- **FeatureEngine** (`src/pipeline/features.py`): RSI, MACD, Bollinger, momentum, vol  
- **Preprocessor** (`src/pipeline/preprocess.py`): scalers (Standard, MinMax, Robust), train/val/test split

---

## 4. Model Architecture
- **PriceLSTM** (`src/models/lstm.py`):  
  - Input dim = number of features  
  - Hidden layers, dropout, optional bidirectional, attention  
- **Variants** (`src/models/gru.py`, `src/models/bilstm.py`)

---

## 5. Training Loop
- **train.py**:  
  - Optimizer (Adam), scheduler, gradient clipping, early stopping  
  - Logging to MLflow/W&B

---

## 6. Backtesting Engine
- **Backtester** (`src/backtest/engine.py`):  
  - Ingest model forecasts + price series  
  - Apply strategy rules (thresholds, sizing, slippage)  
  - Compute P&L, Sharpe, drawdown

---

## 7. API & Deployment
- **FastAPI** serve model at `/predict`  
- **Streamlit** dashboard for live charting & parameter tweaking  
- **Docker** + **GitHub Actions** for CI/CD