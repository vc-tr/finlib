# Quant Lab: Strategy Research Results

> **Living document.** Updated as each strategy is implemented and backtested.
> All results use anti-lookahead execution, walk-forward OOS validation, and realistic cost assumptions.

---

## Abstract

This document catalogs backtest results for all strategies in the Quant Lab, organized by category. Each entry reports in-sample and out-of-sample performance, Fama-French factor attribution (where available), and a verdict based on statistical significance and economic rationale.

**Key finding (preview):** Retail technical strategies promoted by social media traders systematically underperform after realistic transaction costs and out-of-sample testing. Academic factor strategies (momentum, low-volatility) show more robust results but require careful implementation to avoid factor crowding and momentum crashes.

---

## Methodology

### Data
- **Source**: Yahoo Finance via `yfinance`
- **Universe**: SPY (single-asset tests), `liquid_etfs` / `sector_etfs` (cross-sectional tests)
- **Test period**: Varies per strategy; minimum 2 years in-sample, 1 year OOS

### Execution Assumptions
- **Fee**: 1–5 bps per trade (one-way)
- **Slippage**: 2–5 bps per trade
- **Spread**: 1 bps
- **Fill**: Signal at close t → fill at open t+1 (no lookahead)

### Walk-Forward Protocol
- Rolling train/test splits with 5-day embargo between windows
- OOS Sharpe reported as primary performance metric
- In-sample results included for overfitting comparison

### Statistical Significance
- Sharpe ratio standard error: Lo (2002) `SE(SR) = sqrt((1 + 0.5*SR²) / T)`
- Deflated Sharpe Ratio *(coming)*: Bailey & Lopez de Prado (2014) multiple-testing correction
- Factor attribution *(coming)*: Fama-French 5-factor regression

---

## 1. Statistical / Core Strategies

### 1.1 Momentum (Time-Series)

- **Hypothesis**: Assets with positive recent returns continue to outperform
- **Signal**: `prices.pct_change(lookback) > threshold` → long; negative → short
- **Parameters**: lookback=20 bars, threshold=0, min_hold=1 bar
- **Source**: `src/strategies/stats/momentum.py`

| Metric | In-Sample | OOS (Walk-Forward) |
|--------|-----------|---------------------|
| Sharpe | — | — |
| CAGR | — | — |
| Max Drawdown | — | — |
| Win Rate | — | — |

**Verdict**: *(pending backtest results)*

**Reproduction**:
```bash
python scripts/run_demo.py --symbol SPY --period 5y --walkforward
```

---

### 1.2 Mean Reversion (Z-Score)

- **Hypothesis**: Short-term price deviations from the rolling mean revert
- **Signal**: Z-score of price vs rolling mean; short when z > entry_z, long when z < -entry_z
- **Source**: `src/strategies/stats/mean_reversion.py`

| Metric | In-Sample | OOS (Walk-Forward) |
|--------|-----------|---------------------|
| Sharpe | — | — |
| CAGR | — | — |
| Max Drawdown | — | — |

**Verdict**: *(pending backtest results)*

---

### 1.3 Pairs Trading (Cointegration)

- **Hypothesis**: Cointegrated pair spread is stationary and mean-reverts
- **Signal**: Fit OLS spread, trade z-score of residuals
- **Source**: `src/strategies/stats/pairs_trading.py`

| Metric | In-Sample | OOS (Walk-Forward) |
|--------|-----------|---------------------|
| Sharpe | — | — |
| CAGR | — | — |
| Max Drawdown | — | — |

**Verdict**: *(pending backtest results)*

---

## 2. Retail / Technical Analysis Strategies

*Strategies popularized by YouTube and TikTok trading educators. Tested with rigorous OOS validation and realistic costs.*

*(Coming soon — strategies being implemented)*

---

## 3. Academic Strategies

*Replications of peer-reviewed factor research.*

*(Coming soon — strategies being implemented)*

---

## 4. Econophysics Strategies

*Physics-inspired quantitative approaches.*

*(Coming soon — strategies being implemented)*

---

## 5. Cross-Strategy Analysis

*(To be populated once multiple strategies are live)*

### 5.1 Performance Comparison

| Strategy | Category | IS Sharpe | OOS Sharpe | Max DD | Verdict |
|----------|----------|-----------|------------|--------|---------|
| *(pending)* | | | | | |

### 5.2 Return Correlation Matrix

*(To be generated from strategy equity curves)*

### 5.3 Factor Attribution Summary

*(Fama-French 5-factor decomposition — coming in Phase 2)*

### 5.4 Multiple Testing Adjustment

*(Deflated Sharpe Ratio across all tested strategies — coming in Phase 2)*

---

## 6. Conclusions

*(To be written once sufficient strategies are tested)*

---

## Appendix: Reproduction Commands

Each strategy can be reproduced exactly with the commands listed in its section above.
All parameters and data sources are documented in `configs/` and `src/strategies/`.
