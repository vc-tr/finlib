# Quant Lab

A quantitative research platform and **library of 23 strategies** across five families — classical statistics, retail/technical lore, academic anomalies, econophysics, and **learned ML signals**. Every strategy runs through the same harness: anti-lookahead execution, walk-forward out-of-sample validation, realistic costs, and statistical-significance testing (deflated Sharpe). Strategies that fail are documented with analysis of why.

Built for serious strategy research — not toy demos.

---

## Strategy Catalog

| Family | Count | Examples | Approach |
|--------|-------|----------|----------|
| `stats` | 3 | momentum, mean_reversion, pairs_trading | Classical statistical signals |
| `retail` | 8 | golden_cross, macd_crossover, rsi_overbought, bollinger_breakout | Popular technical patterns, rigorously tested (most fail OOS) |
| `academic` | 5 | time_series_momentum, betting_against_beta, post_earnings_drift | Peer-reviewed anomaly replications |
| `econophysics` | 4 | hurst_exponent, ornstein_uhlenbeck, entropy_signal, power_law_tail | Physics-inspired models |
| `ml` | 3 | ml_logistic, ml_gradient_boost, ml_lstm | **Walk-forward learned signals — leak-free by construction** |

`python scripts/run_strategy.py --list` prints the live catalog. See [`docs/STRATEGY_RESULTS.md`](docs/STRATEGY_RESULTS.md) for backtest results and verdicts, and [`docs/ML_MODELS.md`](docs/ML_MODELS.md) for the ML methodology.

> The `ml_lstm` strategy needs PyTorch (`pip install -r requirements-ml.txt`); the other 22 strategies run on the core dependencies alone.

---

## Quickstart

```bash
pip install -r requirements.txt

# Single-symbol demo (SPY, 2 years, daily)
python scripts/run_demo.py --symbol SPY --period 2y

# Cross-sectional factor backtest
python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1 --rebalance M --top-k 10 --bottom-k 10

# Walk-forward out-of-sample validation
python scripts/walkforward_demo.py --symbol SPY --interval 1d --folds 6 --train-days 90 --test-days 30

# Walk-forward ML signal vs buy-and-hold (sklearn; add -r requirements-ml.txt for ml_lstm)
python scripts/run_ml_demo.py --strategy ml_gradient_boost --symbol SPY --period 8y
```

**Smoke test** (~30 sec): `python scripts/run_demo.py --symbol SPY --period 30d`

---

## Outputs

After `run_demo`, check `output/`:

| File | Description |
|------|-------------|
| `summary.json` | Key metrics + config |
| `REPORT.md` | Markdown report |
| `tearsheet.html` | HTML tearsheet + charts |
| `equity_curve.png` | Cumulative return |
| `drawdown.png` | Underwater curve |
| `rolling_sharpe.png` | Rolling Sharpe ratio |
| `returns_hist.png` | Return distribution |
| `positions.png` | Position exposure over time |
| `turnover.png` | Turnover |

---

## Methodology

- **No lookahead**: Signal at close `t` → fill at open `t+1`; ML models retrain on **past data only**
- **Walk-forward**: Rolling train/test splits; test window never used for calibration
- **Realistic costs**: Configurable fees, slippage, spread (1–5 bps each); liquidity-aware impact model
- **Statistical rigor**: Deflated Sharpe (Bailey & López de Prado 2014), Sharpe standard error (Lo 2002), IC/IR testing
- **Factor attribution**: Fama-French 5-factor regression (α, t-stat, p-value) via the Kenneth French data library

See [`docs/RESEARCH_METHOD.md`](docs/RESEARCH_METHOD.md) for full methodology and [`docs/ML_MODELS.md`](docs/ML_MODELS.md) for the ML approach.

---

## Architecture

```
Data (Yahoo) → Strategy / Factors → Execution (fees/slippage) → Backtester → Tearsheet
                                         │
                             signal at close t → fill at t+1 (no lookahead)
```

| Module | Purpose |
|--------|---------|
| `src/backtest/` | Engine, execution config, cost models, walk-forward |
| `src/factors/` | Cross-sectional factor research, universe registry, ensemble |
| `src/ml/` | Causal features, walk-forward driver, sklearn + PyTorch direction models |
| `src/research/` | Fama-French 5-factor attribution, deflated Sharpe, regime analysis |
| `src/paper/` | Event-driven paper trading replay |
| `src/ops/` | Daily portfolio generation, monitoring/alerts |
| `src/pipeline/` | Yahoo data fetcher, preprocessing |
| `src/reporting/` | HTML tearsheet + PNG charts |
| `src/strategies/` | Strategy library (stats, retail, academic, econophysics, ml) |
| `src/utils/` | CLI parsers, I/O, RunLock, JSON serialization |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_demo.py` | Single-symbol demo: data → strategy → backtest → tearsheet |
| `run_strategy.py` | Run any registered strategy (or a whole category) by name |
| `run_ml_demo.py` | Walk-forward ML strategy vs buy-and-hold (sklearn / PyTorch) |
| `walkforward_demo.py` | Rolling OOS validation |
| `backtest_factors.py` | Cross-sectional factor backtest |
| `replay_trade.py` | Event-driven paper trading |
| `daily_run.py` | Production daily portfolio generation |
| `monitor_runs.py` | Scan runs, generate alerts |

---

## Minute-Bar Guardrails

```bash
python scripts/run_demo.py --symbol SPY --period 7d --interval 1m
python scripts/run_demo.py --symbol SPY --period 30d --interval 5m
```

Yahoo limits: 1m ~7d, 5m ~60d. Period is auto-capped; use `--period-override` to bypass.

---

## Paper Trading Replay

```bash
python scripts/replay_trade.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M --start 2024-01-01 --end 2025-12-31

# Combo factors with auto_robust ensemble
python scripts/replay_trade.py --strategy factors --factor combo --combo "momentum_12_1,reversal_5d,lowvol_20d" --combo-method auto_robust --rebalance M
```

---

## Documentation

- [`docs/RESEARCH_METHOD.md`](docs/RESEARCH_METHOD.md) — Data assumptions, evaluation protocol, anti-lookahead guarantees
- [`docs/UNIVERSES.md`](docs/UNIVERSES.md) — Universe definitions (liquid_etfs, sector_etfs, bond_etfs, etc.)
- [`docs/STRATEGY_RESULTS.md`](docs/STRATEGY_RESULTS.md) — Research-paper-style results for all strategies

---

## License

MIT
