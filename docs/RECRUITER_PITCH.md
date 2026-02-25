# Recruiter Pitch — Quant Forecast

Resume bullets and interview talking points for quant research roles.

---

## Resume Bullets (6)

1. **Built a quantitative backtesting platform** with anti-lookahead execution (signal at close t → fill at t+1), configurable fees/slippage, and walk-forward OOS evaluation.

2. **Designed execution realism layer** (5 bps fees, 5 bps slippage, fill-at-next-open) integrated into a unified backtester; supports both rule-based signals and model-based forecasts.

3. **Implemented walk-forward evaluation** with rolling train/test folds to prevent overfitting; aggregate OOS Sharpe, return, and max drawdown across folds.

4. **Developed tear-sheet reporting** (HTML + PNG): equity curve, drawdown, rolling Sharpe, returns histogram, exposure, turnover, and summary metrics (CAGR, Sortino, win rate).

5. **Architected modular strategy pipeline** (momentum, mean reversion, academic papers) with common interface; data fetchers (Yahoo, Alpha Vantage), feature engineering, and portfolio allocation.

6. **Maintained production-grade test suite** (34+ tests) for strategies, execution, data pipeline, and features; documented methodology and reality checks for recruiter evaluation.

---

## Interview Talking Points (4)

### 1. Anti-lookahead and execution timing

> "We enforce strict no-lookahead: the signal is computed from bar t's close, but the fill happens at bar t+1's open. The backtester uses `signals.shift(1)` so the position held during bar t+1 reflects what we knew at close t. We also apply fees and slippage on every position change."

### 2. Walk-forward and overfitting

> "We use rolling walk-forward: train on N days, test on M days, step forward, repeat. The test set is never used for calibration. We aggregate OOS results across folds to get a realistic performance estimate. This avoids the classic backtest-overfitting trap."

### 3. Execution realism

> "We model round-trip fees (5 bps default) and slippage (5 bps) on each trade. Slippage can optionally scale with volatility. The execution layer is pluggable—you can disable it for quick experiments or tune it for production."

### 4. Tear-sheet and reproducibility

> "Every run produces an HTML tear-sheet with equity curve, drawdown, rolling Sharpe, and a summary table. Config-driven runs (JSON) ensure reproducibility. We document what works and what's broken in a reality-check doc so recruiters can evaluate honestly."
