# Package Structure and Demos

## Package Layout (`src/`)

```
src/
├── backtest/          # Backtesting engine
│   ├── engine.py      # Backtester, BacktestResult
│   ├── execution.py  # ExecutionConfig, fees/slippage
│   ├── cost_models.py # FixedBpsCostModel, LiquidityAwareCostModel
│   ├── walkforward_demo.py # Rolling OOS evaluation
│   └── factor_backtest.py  # Cross-sectional factor backtest logic
├── factors/           # Factor construction and portfolio
│   ├── factors.py    # compute_factor, get_prices_wide
│   ├── universe.py   # UniverseRegistry, get_universe
│   ├── portfolio.py  # build_portfolio, weights_at_rebalance
│   ├── ensemble.py   # combine_factors (combo weighting)
│   ├── research.py   # forward_returns, IC/IR
│   ├── risk.py       # estimate_beta, rolling_portfolio_beta
│   └── runner.py     # run_factor_backtest, main (CLI)
├── paper/             # Paper trading
│   ├── orders.py     # Order, OrderSide, OrderType
│   ├── exchange.py   # PaperExchange, Fill
│   ├── broker.py     # PaperBroker
│   ├── risk.py       # RiskManager
│   ├── strategy_adapter.py  # get_factor_target_weights
│   └── runner.py     # run_replay
├── pipeline/          # Data and features
│   ├── data_fetcher.py
│   ├── data_fetcher_yahoo.py
│   ├── pipeline.py   # reindex_and_backfill
│   ├── features.py
│   └── baselines.py
├── reporting/         # Tear-sheet generation
│   └── tearsheet.py
├── ops/               # Operations
│   ├── daily.py      # run_daily, load/save portfolio, run_meta
│   └── monitor.py    # run_monitor, scan runs, alerts
├── strategies/        # Trading strategies (momentum, mean reversion, etc.)
├── models/            # LSTM, GRU, Transformer, TCN
├── portfolio/         # Multi-strategy allocation
└── utils/
    ├── io.py         # parse_period_days, fetch_universe_ohlcv, cap_period_for_interval
    ├── jsonable.py   # to_jsonable
    ├── cli.py        # shared argparse builders (factors, replay, daily, monitor)
    └── runlock.py    # RunLock (global run lock)
```

## Scripts (`scripts/`)

Scripts are thin wrappers: argparse + calls to `src/` modules.

| Script | Purpose |
|--------|---------|
| `run_demo.py` | One-command demo: data → strategy → backtest → tear-sheet |
| `walkforward_demo.py` | Walk-forward evaluation |
| `backtest_factors.py` | Cross-sectional factor backtest |
| `backtest_strategies.py` | Single-strategy backtest |
| `backtest_portfolio.py` | Multi-strategy portfolio backtest |
| `backtest_papers.py` | Academic strategy backtests |
| `backtest_baselines.py` | Baseline model backtests |
| `backtest_pairs.py` | Pairs trading backtest |
| `replay_trade.py` | Event-driven paper trading replay |
| `daily_run.py` | Production daily pipeline (target portfolio + orders) |
| `monitor_runs.py` | Monitor last N runs, alerts |
| `sweep_momentum.py` | Momentum parameter sweep |
| `make_research_bundle.py` | One-command recruiter packet |
| `run_options_demo.py` | Option pricing demo |

## How to Run Demos

```bash
# Install
pip install -r requirements.txt

# Quick smoke test (~30 sec)
python scripts/run_demo.py --symbol SPY --period 30d

# Full demo (2y daily)
python scripts/run_demo.py --symbol SPY --period 2y

# Walk-forward
python scripts/walkforward_demo.py --symbol SPY --interval 1d --folds 6 --train-days 90 --test-days 30

# Factor backtest
python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1 --rebalance M

# Paper trading replay
python scripts/replay_trade.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M

# Daily pipeline (dry-run)
python scripts/daily_run.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M

# Research bundle (all demos in one run)
python scripts/make_research_bundle.py --symbol SPY
```

## Output Locations

- `output/runs/<timestamp>_<type>_<suffix>/` — per-run artifacts (tear-sheets, reports)
- `output/latest/` — mirror of latest run (for research bundle)
- `output/monitor/` — monitor summary and report
- `data/state/` — current portfolio state (daily_run --apply)
- `data/cache/` — cached data (not committed)

## Tests

```bash
pytest tests/ -q
```
