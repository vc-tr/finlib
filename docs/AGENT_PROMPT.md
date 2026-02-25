# Quant Forecast — Agent Context Prompt

Use this prompt when seeking help from an external AI agent. Copy everything below the line.

---

## Project Summary

**Quant Forecast** is a Python quantitative finance research and trading framework. It combines:

- **Option pricing**: Black-Scholes, Monte Carlo
- **Stochastic models**: Brownian motion, GBM, Ornstein-Uhlenbeck
- **Statistical strategies**: Mean reversion, pairs trading, momentum
- **Renaissance-style**: Multi-signal ensemble
- **Influencer/sentiment**: Sentiment-based, volume-sentiment
- **Day trader**: Scalping (EMA crossover), Opening Range Breakout, EMA+Stochastic
- **Institutional**: Kalman pairs, VWAP reversion, ATR breakout
- **Deep learning**: LSTM, BiLSTM, GRU, Transformer, TCN
- **Portfolio**: Multi-strategy allocation (equal, risk parity, inverse volatility)
- **Backtesting**: Unified engine for strategy evaluation

---

## Tech Stack

- **Python 3.x**
- **PyTorch** — deep learning models
- **pandas, numpy, scipy, statsmodels** — data & stats
- **yfinance** — market data (Yahoo)
- **mlflow, tensorboard** — experiment tracking
- **pytest** — testing
- **click** — CLI scripts

---

## Project Structure

```
quant-forecast/
├── src/
│   ├── models/           # LSTM, BiLSTM, GRU, Transformer, TCN, factory.py
│   ├── strategies/       # All trading strategies by category
│   │   ├── options/      # Black-Scholes, Monte Carlo
│   │   ├── stochastic/   # Brownian, GBM, Ornstein-Uhlenbeck
│   │   ├── stats/        # Mean reversion, pairs, momentum
│   │   ├── renaissance/  # Signal ensemble
│   │   ├── influencer/   # Sentiment, volume-sentiment
│   │   ├── daytrader/    # Scalping, ORB, EMA+Stochastic
│   │   ├── institutional/ # Kalman pairs, VWAP, ATR breakout
│   │   └── papers/       # Moskowitz, JT, GGR, De Bondt-Thaler
│   ├── pipeline/         # Data fetchers, pipeline, features, baselines
│   ├── backtest/         # Backtesting engine
│   └── portfolio/        # Multi-strategy allocation
├── scripts/              # CLI entry points (backtest_*.py, run_options_demo.py)
├── tests/                # pytest tests
├── docs/                 # STRATEGIES.md, PAPERS.md, PORTFOLIO.md, INTEGRATION.md
├── train.py              # Main DL training entry point
└── requirements.txt
```

---

## Conventions

1. **Strategies** implement a common interface: they take prices (or OHLCV) and return signals (-1, 0, 1) and/or backtest returns.
2. **Data flow**: `DataFetcher → OHLCV → reindex_and_backfill (for minute) → Features → Model/Strategy`.
3. **DL pipeline**: `% returns → sliding windows (SEQ_LEN) → train/val split → model → predict`.
4. **Backtester**: Consumes signals or returns; outputs Sharpe, total return, drawdown, etc.
5. **Portfolio**: `MultiStrategyPortfolio` + `PortfolioAllocator` (EQUAL, RISK_PARITY, INVERSE_VOLATILITY, CUSTOM).

---

## Common Commands

```bash
# Install
pip install -r requirements.txt

# Train DL model (default LSTM)
python train.py
python train.py --model_type transformer --loss huber --epochs 30

# Backtest strategies
python scripts/backtest_strategies.py --symbol SPY --period 365d
python scripts/backtest_strategies.py --symbol SPY --period 365d --all
python scripts/backtest_portfolio.py --symbol SPY --period 2y --alloc risk_parity
python scripts/backtest_papers.py --symbol SPY --period 5y
python scripts/backtest_baselines.py --symbol SPY --period 365d

# Option pricing demo
python scripts/run_options_demo.py

# Tests
pytest tests/ -v
```

---

## Key Imports

```python
from src.strategies import MeanReversionStrategy, MomentumStrategy, BlackScholes, MonteCarloPricer
from src.backtest import Backtester
from src.portfolio import MultiStrategyPortfolio, PortfolioAllocator, AllocationMethod
from src.experimental.models import create_model
from src.pipeline.data_fetcher_yahoo import YahooDataFetcher
from src.pipeline.pipeline import reindex_and_backfill
```

---

## Documentation

- **README.md** — Overview, quick start, model/strategy reference
- **docs/STRATEGIES.md** — Strategy logic, sources, use cases
- **docs/PAPERS.md** — Academic strategies with citations
- **docs/PORTFOLIO.md** — Multi-strategy allocation
- **docs/INTEGRATION.md** — Alpaca, Tradier, IB paper/live trading
- **docs/PIPELINE.md** — Data and model pipeline details

---

## Instructions for the Agent

When helping with this project:

1. **Follow existing patterns** — Match the style of `src/` (type hints, docstrings, modular structure).
2. **Use the backtester** — New strategies should integrate with `Backtester` and return signals/returns in the expected format.
3. **Preserve interfaces** — Strategies, models, and allocators have established APIs; extend rather than break them.
4. **Add tests** — New code should have corresponding tests in `tests/`.
5. **Check docs** — Update `docs/` and `README.md` when adding strategies, models, or major features.
6. **Data** — Use `YahooDataFetcher` or `AlphaDataFetcher`; avoid hardcoding API keys; support `reindex_and_backfill` for minute data.
7. **Time-based splits** — For ML, use time-based train/val splits (no shuffle) to avoid lookahead bias.

---

## Current Focus Areas (Optional Context)

- Portfolio: multi-strategy allocation with equal, risk parity, inverse volatility
- Deep learning: LSTM, BiLSTM, GRU, Transformer, TCN with MSE/MAE/Huber
- Backtesting: unified engine across all strategy types
- Integration: Alpaca, Tradier, IB for paper/live trading

---

*End of agent prompt. Paste this entire section when requesting help from an external agent.*
