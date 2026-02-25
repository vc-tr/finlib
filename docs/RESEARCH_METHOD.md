# Research Method

Documentation for recruiter evaluation: data assumptions, evaluation protocol, and production path.

---

## Data Assumptions

- **Source**: Yahoo Finance via `yfinance` (free, delayed)
- **OHLCV**: open, high, low, close, volume; lowercase columns; datetime index
- **Adjustments**: No automatic splits/dividend adjustments in demo (configurable in fetcher)
- **Survivorship**: Demo uses SPY; no survivorship bias for single-index ETF
- **Gaps**: Forward-fill for minute data; daily data assumed continuous

---

## Evaluation Protocol

### Anti-Leakage Rules

1. **Signal timing**: Signal computed from bar `t` data executes at bar `t+1`
   - Backtester uses `signals.shift(1) * returns`
   - No same-bar execution

2. **Walk-forward**: Rolling train/test windows; test period never used for calibration
   - Default: 252d train, 63d test, step 63d
   - Aggregate OOS metrics only

3. **Features**: Technical indicators use only past data (rolling windows)

### Realism Assumptions

- **Fees**: Configurable bps per trade (default 5 bps round-trip)
- **Slippage**: Configurable bps; optional volatility-scaled
- **Execution**: Signal at close → fill at next open (configurable)

---

## Metrics

| Metric | Definition |
|--------|-------------|
| CAGR | Compound annual growth rate |
| Sharpe | Annualized excess return / volatility |
| Sortino | Return / downside volatility |
| MaxDD | Maximum drawdown from peak |
| Win Rate | % of days with positive return |
| Turnover | \|Δposition\| per period |

---

## Production Path

For live deployment:

1. **Broker integration**: Alpaca, Tradier, or IB (see `docs/INTEGRATION.md`)
2. **Data**: Replace Yahoo with live feed (Polygon, IEX, broker API)
3. **Execution**: Market/limit orders; TWAP/VWAP for large size
4. **Risk**: Position limits, drawdown stops, exposure constraints
5. **Monitoring**: Logging, alerts, reconciliation
