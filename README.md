# Quant Lab

A quantitative research platform and **growing library of 20+ strategies** spanning retail trading lore, academic finance, and econophysics. Every strategy is tested with anti-lookahead execution, walk-forward out-of-sample validation, realistic costs, and Fama-French factor attribution. Strategies that fail are documented with analysis of why.

Built for serious strategy research — not toy demos.

---

## Strategy Catalog

| Category | Strategy | Source | Status |
|----------|----------|--------|--------|
| `stats` | Momentum | Classic time-series momentum | ✅ |
| `stats` | Mean Reversion | Z-score mean reversion | ✅ |
| `stats` | Pairs Trading | Cointegration-based | ✅ |
| `retail` | *(coming soon)* | YouTube/TikTok guru debunking | 🔜 |
| `academic` | *(coming soon)* | Research paper replications | 🔜 |
| `econophysics` | *(coming soon)* | Physics-inspired approaches | 🔜 |

See [`docs/STRATEGY_RESULTS.md`](docs/STRATEGY_RESULTS.md) for full backtest results, factor attribution, and verdicts.

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

- **No lookahead**: Signal at close `t` → fill at open `t+1`
- **Walk-forward**: Rolling train/test splits; test window never used for calibration
- **Realistic costs**: Configurable fees, slippage, spread (1–5 bps each); liquidity-aware impact model
- **Statistical rigor**: Walk-forward Sharpe, IC/IR testing, out-of-sample validation
- **Factor attribution**: *(coming)* Fama-French 5-factor decomposition for each strategy

See [`docs/RESEARCH_METHOD.md`](docs/RESEARCH_METHOD.md) for full methodology.

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
| `src/paper/` | Event-driven paper trading replay |
| `src/ops/` | Daily portfolio generation, monitoring/alerts |
| `src/pipeline/` | Yahoo data fetcher, preprocessing |
| `src/reporting/` | HTML tearsheet + PNG charts |
| `src/strategies/` | Strategy library (stats, retail, academic, econophysics) |
| `src/utils/` | CLI parsers, I/O, RunLock, JSON serialization |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_demo.py` | Single-symbol demo: data → strategy → backtest → tearsheet |
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

## FAQ

**Q: Why can't I run minute-bar backtests for 6 months?**
A: Yahoo Finance restricts 1m data to ~7 days per request. Use daily data or a paid provider for longer history.

**Q: Config-driven run?**
A: `python scripts/run_demo.py --config configs/demo_spy_momentum.json`

---

## Documentation

- [`docs/RESEARCH_METHOD.md`](docs/RESEARCH_METHOD.md) — Data assumptions, evaluation protocol, anti-lookahead guarantees
- [`docs/UNIVERSES.md`](docs/UNIVERSES.md) — Universe definitions (liquid_etfs, sector_etfs, bond_etfs, etc.)
- [`docs/STRATEGY_RESULTS.md`](docs/STRATEGY_RESULTS.md) — Research-paper-style results for all strategies

---

## License

MIT
