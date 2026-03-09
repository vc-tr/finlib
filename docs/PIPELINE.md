# Pipeline Documentation

## Data Pipeline

### 1. Data Acquisition

- **YahooDataFetcher**: Fetches OHLCV via `yfinance`
- Supports intervals: `1m`, `5m`, `1h`, `1d`
- Supports periods: `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`
- Minute-bar caps enforced: 1m → max 7d, 5m → max 60d (Yahoo Finance limits)

### 2. Preprocessing

- **reindex_and_backfill**: For minute data, creates complete 1-min index, forward-fills OHLC, zero-fills volume
- Optional: filter to regular trading hours (09:30–16:00) and weekdays

---

## Strategy Pipeline

### Rule-Based Strategies

1. **Input**: Price series (or pair of series for pairs trading)
2. **Signal generation**: Strategy-specific logic (z-score, momentum, crossover, etc.)
3. **Output**: Discrete signals {-1, 0, 1}; -1 = short, 0 = flat, 1 = long
4. **Backtest**: `Backtester.run_from_signals(prices, signals, execution_config)`

### Cross-Sectional Factor Strategies

1. **Universe**: Fetch OHLCV for all symbols (e.g., `liquid_etfs`, `sector_etfs`)
2. **Factor computation**: `compute_factor(df_by_symbol, factor_name)` → date × symbol DataFrame
3. **Ranking**: Cross-sectional rank; long top-k, short bottom-k
4. **Portfolio construction**: `weights_at_rebalance(factor_df, rebalance_freq)`
5. **Optional**: Beta-neutral hedging, ensemble combination (equal, IC-weighted, ridge, Sharpe-opt)
6. **Backtest**: `run_factor_backtest(weights, prices, execution_config)`

---

## Backtest Pipeline

1. Load prices via `YahooDataFetcher`
2. Instantiate strategy or compute factor weights
3. Generate signals: `strategy.generate_signals(prices)` → pd.Series of {-1, 0, 1}
4. Apply execution: `Backtester.run_from_signals(prices, signals, ExecutionConfig)` → `BacktestResult`
   - Signal at t → position held during bar t+1 (no lookahead)
   - Costs deducted on position changes: fee_bps + slippage_bps + spread_bps
5. Metrics: Sharpe, CAGR, Sortino, max drawdown, win rate, trades, turnover
6. Reporting: `generate_tearsheet(result, prices, signals, output_dir)` → HTML + PNG charts

---

## Walk-Forward Pipeline

1. `generate_folds(index, train_days, test_days, embargo_days)` → list of `WalkForwardFold`
2. For each fold:
   - Train/calibrate on `[train_start, train_end]`
   - Evaluate on `[test_start, test_end]` (never seen during calibration)
3. Aggregate: concatenate OOS returns across folds, compute mean/median/agg Sharpe
4. Output: `walkforward_summary.json`, `WALKFORWARD_REPORT.md`

---

## Cost Models

| Model | When to use |
|-------|-------------|
| `FixedBpsCostModel` | Simple research: fixed fee + slippage + spread per trade |
| `LiquidityAwareCostModel` | Capacity analysis: impact scales with participation rate vs ADV |

---

## Paper Trading Pipeline

Event-driven replay on historical bars:

1. `run_replay(df_by_symbol, args)` — chronological bar-by-bar simulation
2. On rebalance dates: compute target weights → `PaperBroker.submit_order()`
3. `PaperExchange.fill_order()` — fills at next open/close with configurable slippage
4. `RiskManager.check_limits()` — enforces position/drawdown limits
5. Output: `orders.csv`, `blotter.csv`, `equity_curve.csv`, `positions_snapshot.csv`
