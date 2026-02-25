# Repo Reality Check

*Generated from Phase 0 scan. Use for recruiter evaluation and refactor planning.*

---

## What Works Today

### Commands that run successfully

| Command | Status | Notes |
|---------|--------|-------|
| `python scripts/backtest_strategies.py --symbol SPY --period 60d` | ✅ Works | Mean reversion, momentum, Renaissance ensemble |
| `python scripts/backtest_strategies.py --symbol SPY --period 365d --all` | ✅ Works | Adds scalping, ORB, VWAP, ATR, volume-sentiment |
| `python scripts/backtest_papers.py --symbol SPY --period 5y` | ✅ Works | Moskowitz TSMOM, De Bondt-Thaler (needs 800+ days) |
| `python scripts/backtest_papers.py --symbol1 SPY --symbol2 IVV --period 2y` | ✅ Works | GGR pairs |
| `python scripts/backtest_portfolio.py --symbol SPY --period 2y --alloc equal` | ✅ Works | Multi-strategy portfolio |
| `python scripts/run_options_demo.py` | ✅ Works | Black-Scholes, Monte Carlo |
| `pytest tests/test_strategies.py tests/test_data_fetcher.py tests/test_pipeline.py tests/test_features.py` | ✅ 29/30 pass | Alpha fetcher test fails (API/column mismatch) |

### Backtester interface

- **Location**: `src/backtest/engine.py`
- **API**: `Backtester.run(strategy_returns)` or `Backtester.run_from_signals(prices, signals)`
- **Signals**: `{-1, 0, 1}` (short, flat, long)
- **Lookahead**: Correct — `run_from_signals` uses `signals.shift(1) * returns` (signal at t-1 executes at t)
- **Output**: `BacktestResult` with `returns`, `cumulative_returns`, `sharpe_ratio`, `max_drawdown`, `total_return`, `n_trades`, `win_rate`

### Strategy interface

- Strategies implement `generate_signals(prices)` and/or `backtest_returns(prices)` → `(signals, strategy_returns)`
- Position representation: signals in {-1, 0, 1}; no explicit target weight/units
- Mean reversion, momentum, Renaissance ensemble are clean and tested

### Data fetch + OHLCV

- **YahooDataFetcher**: `fetch_ohlcv(symbol, interval, period)` → DataFrame with `open`, `high`, `low`, `close`, `volume`
- **Convention**: lowercase columns, datetime index
- **AlphaDataFetcher**: Exists but test fails (column naming mismatch with Alpha Vantage API)

### Current metrics outputs

- Sharpe ratio (annualized)
- Total return
- Max drawdown
- N trades (simplified: count of non-zero return days)
- Win rate

---

## What Is Broken / Limitations

| Item | Details |
|------|---------|
| AlphaDataFetcher test | KeyError on columns — Alpha Vantage returns different column names; requires API key |
| DL tests (test_lstm, test_model_variants, test_train) | ImportError if torch not installed — run with `--ignore` or install torch |
| scripts/walkforward.py | Removed — use `walkforward_demo.py` instead |

---

## What Is Missing for Realism

1. **Execution layer**: Fees (bps), slippage (bps, optional vol-based), execution timing (signal at close → fill at next open/close)
2. **Position sizing**: Target notional or units; turnover calculation
3. **Walk-forward**: Rolling windows, out-of-sample folds
4. **Reporting**: Equity curve, drawdown, rolling Sharpe, returns histogram, exposure, turnover, summary table
5. **Config-driven run**: YAML/JSON config for reproducible runs

---

## Quick Wins (Completed)

1. ~~Create `scripts/run_demo.py`~~ — Done
2. ~~Pick Momentum as flagship~~ — Done
3. ~~Add `src/backtest/execution.py`~~ — Done
4. ~~Add `src/reporting/tearsheet.py`~~ — Done
5. ~~Add `scripts/walkforward_demo.py`~~ — Done (replaces legacy walkforward.py)
6. ~~Mark influencer/sentiment as archived~~ — ARCHIVED header in volume_sentiment, sentiment
7. **Fix or skip Alpha fetcher test** — Mark as integration test requiring API key
8. **Slim requirements** — torch optional for core backtest

---

## Scripts Inventory

| Script | Purpose |
|--------|---------|
| `run_demo.py` | One-command demo: data → strategy → backtest → tear-sheet |
| `walkforward_demo.py` | Rolling OOS evaluation |
| `backtest_factors.py` | Cross-sectional factor backtest |
| `replay_trade.py` | Paper trading replay |
| `daily_run.py` | Production daily pipeline |
| `monitor_runs.py` | Monitor last N runs |
| `backtest_strategies.py` | Core + optional extended strategies |
| `backtest_papers.py` | Academic (Moskowitz, De Bondt-Thaler, GGR) |
| `backtest_portfolio.py` | Multi-strategy allocation |
| `backtest_baselines.py` | Persistence, MA, ARIMA forecasts |
| `run_options_demo.py` | Black-Scholes, Monte Carlo |

---

## Tests Status

| Test file | Pass | Fail | Notes |
|-----------|------|------|-------|
| test_strategies.py | 7 | 0 | Core strategies + backtester |
| test_data_fetcher.py | 2 | 1 | Alpha fetcher fails |
| test_pipeline.py | 3 | 0 | reindex, backfill |
| test_features.py | 18 | 0 | RSI, MACD, Bollinger, VWAP |
| test_lstm.py | - | collect error | torch import |
| test_model_variants.py | - | collect error | torch import |
| test_train.py | - | collect error | torch import |

---

## run_demo.py — Exact Files Used

| File | Role |
|------|------|
| `scripts/run_demo.py` | CLI entry point |
| `src/pipeline/data_fetcher_yahoo.py` | YahooDataFetcher.fetch_ohlcv |
| `src/pipeline/pipeline.py` | reindex_and_backfill (for 1m/5m) |
| `src/strategies/stats/momentum.py` | MomentumStrategy |
| `src/backtest/engine.py` | Backtester.run_from_signals |
| `src/backtest/execution.py` | ExecutionConfig, apply_execution_realism |
| `src/reporting/tearsheet.py` | generate_tearsheet → output/ |
