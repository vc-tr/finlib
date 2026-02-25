# Project Facts — Quant Forecast

*Discovery report for run_demo and backtest pipeline.*

---

## 1. Default Interval

**run_demo uses `--interval 1d` by default.**

- `scripts/run_demo.py` line 51: `parser.add_argument("--interval", default="1d", ...)`
- Daily bars unless `--interval 1m` or `--interval 5m` is passed

---

## 2. MomentumStrategy Signal Rule

**Exact rule: sign of lookback return (time-series momentum).**

- Location: `src/strategies/stats/momentum.py`
- `compute_momentum(prices)` = `prices.pct_change(lookback)` = `(price[t] / price[t-lookback]) - 1`
- `generate_signals`: `1` if mom > threshold, `-1` if mom < -threshold, `0` otherwise (threshold default 0)
- **Not** MA crossover; **not** RSI/MACD. Pure sign-of-past-return.

---

## 3. Signal Execution Timing (Lookahead)

**Signals execute next-bar by default. No same-bar lookahead.**

- `src/backtest/execution.py`: `pos = signals.shift(delay).fillna(0)` with `execution_delay_bars=1`
- `src/backtest/engine.py` (no execution): `pos = signals.shift(1).fillna(0) * returns`
- Signal at close of bar t → position held during bar t+1
- Configurable via `ExecutionConfig.execution_delay_bars`

---

## 4. Trade Counting

**Trades count only on position changes (0→1, 1→0, 1→-1, etc.).**

- `src/backtest/engine.py` line 143: `n_trades = int((pos.diff().abs() > 1e-8).sum())`
- `pos` = position held each bar (from signals.shift(1))
- No longer counts every bar with nonzero return

---

## 5. Tear-Sheet Location and Outputs

**Location:** `src/reporting/tearsheet.py` → `generate_tearsheet()`

**Output directory:** `output/` (created by run_demo)

**Outputs:**

| File | Description |
|------|-------------|
| `summary.json` | Key metrics + config |
| `REPORT.md` | Markdown with embedded plots |
| `equity_curve.png` | Cumulative return |
| `drawdown.png` | Underwater curve |
| `rolling_sharpe.png` | Rolling Sharpe |
| `returns_hist.png` | Return distribution |
| `positions.png` | Position exposure |
| `turnover.png` | Turnover |
| `tearsheet.html` | Legacy HTML |

**Dependencies:** matplotlib, pandas only (no plotly, no heavy deps)

---

## 6. Minute Bars Support

**Supported today.**

- `--interval 1m` or `--interval 5m` in run_demo
- `reindex_and_backfill(df, freq="1min"|"5min")` in `src/pipeline/pipeline.py`
- **Guardrail:** `INTERVAL_PERIOD_CAP` caps period: 1m→7d (Yahoo limit), 5m→60d
- **Bypass:** `--period-override` skips cap
- Yahoo limits: 1m ~7d per request; 5m ~60d

---

## 7. Execution Realism (Current)

| Parameter | Default | Location |
|-----------|---------|----------|
| fee_bps | 1.0 | ExecutionConfig |
| slippage_bps | 2.0 | ExecutionConfig |
| spread_bps | 1.0 | ExecutionConfig |
| execution_delay_bars | 1 | ExecutionConfig |
| Costs only on trades | Yes | Applied via `pos.diff().abs()` |

---

## 8. Files Used by run_demo

| File | Role |
|------|------|
| `scripts/run_demo.py` | CLI entry |
| `src/pipeline/data_fetcher_yahoo.py` | YahooDataFetcher |
| `src/pipeline/pipeline.py` | reindex_and_backfill |
| `src/strategies/stats/momentum.py` | MomentumStrategy |
| `src/backtest/engine.py` | Backtester |
| `src/backtest/execution.py` | ExecutionConfig, apply_execution_realism |
| `src/reporting/tearsheet.py` | generate_tearsheet |
