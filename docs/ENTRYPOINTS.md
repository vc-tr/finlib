# Core Entrypoints

The ONLY supported entrypoints for the quant-forecast platform.

## Commands

| Entrypoint | Command | Purpose |
|------------|---------|---------|
| **run_demo** | `python scripts/run_demo.py --symbol SPY --period 30d --interval 1d` | Single-symbol demo: data → strategy → backtest → tear-sheet |
| **backtest_factors** | `python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1 --rebalance M --top-k 10 --bottom-k 10 --period 2y` | Cross-sectional factor backtest |
| **walkforward_demo** | `python scripts/walkforward_demo.py --symbol SPY --interval 1d --folds 6 --train-days 90 --test-days 30` | Rolling OOS evaluation |
| **replay_trade** | `python scripts/replay_trade.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M --start 2024-01-01 --end 2024-06-30` | Event-driven paper trading replay |
| **daily_run** | `python scripts/daily_run.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M --asof 2024-07-02` | Production daily pipeline (target portfolio + orders) |
| **monitor_runs** | `python scripts/monitor_runs.py -n 5 --since-hours 6` | Monitor last N runs, alerts |

## Smoke Test (all entrypoints)

```bash
python scripts/run_demo.py --symbol SPY --period 30d --interval 1d
python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1 --rebalance M --top-k 10 --bottom-k 10 --period 2y
python scripts/replay_trade.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M --start 2024-01-01 --end 2024-06-30
python scripts/daily_run.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M --asof 2024-07-02
python scripts/monitor_runs.py -n 5 --since-hours 6
```

## Non-Entrypoints (deprecated or auxiliary)

- `scripts/make_research_bundle.py` — Auxiliary (orchestrates multiple demos)
- `scripts/backtest_strategies.py`, `backtest_papers.py`, `backtest_portfolio.py`, `backtest_baselines.py`, `backtest_pairs.py` — Deprecated
- `scripts/run_options_demo.py` — Deprecated
- `scripts/sweep_momentum.py` — Auxiliary (parameter sweep)
