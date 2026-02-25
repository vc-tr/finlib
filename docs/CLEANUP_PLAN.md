# Cleanup Plan — Every Tracked .py File

Generated from `tools/repo_audit.py`. Status: KEEP | MERGE | DEPRECATE | DELETE.

## Entrypoints (KEEP)

| File | Status | Reason |
|------|--------|--------|
| scripts/run_demo.py | KEEP | Core entrypoint |
| scripts/backtest_factors.py | KEEP | Core entrypoint |
| scripts/walkforward_demo.py | KEEP | Core entrypoint |
| scripts/replay_trade.py | KEEP | Core entrypoint |
| scripts/daily_run.py | KEEP | Core entrypoint |
| scripts/monitor_runs.py | KEEP | Core entrypoint |

## Reachable src/ (KEEP)

| File | Status | Reason |
|------|--------|--------|
| src/backtest/__init__.py | KEEP | Package |
| src/backtest/cost_models.py | KEEP | Canonical cost model |
| src/backtest/engine.py | KEEP | Backtester |
| src/backtest/execution.py | KEEP | Execution realism |
| src/backtest/factor_backtest.py | KEEP | Factor backtest |
| src/backtest/walkforward.py | KEEP | Walk-forward |
| src/factors/__init__.py | KEEP | Package |
| src/factors/ensemble.py | KEEP | Combo factors |
| src/factors/factors.py | KEEP | Canonical factor compute |
| src/factors/portfolio.py | KEEP | Portfolio build |
| src/factors/ranking.py | KEEP | Ranking |
| src/factors/research.py | KEEP | IC, forward returns |
| src/factors/risk.py | KEEP | Beta |
| src/factors/runner.py | KEEP | Factor backtest runner |
| src/factors/universe.py | KEEP | Universe registry |
| src/factors/weight_learning.py | KEEP | Weight learning |
| src/models/lstm.py | KEEP | Used by test_lstm |
| src/ops/__init__.py | KEEP | Package |
| src/ops/daily.py | KEEP | Daily pipeline |
| src/ops/monitor.py | KEEP | Monitor runs |
| src/paper/__init__.py | KEEP | Package |
| src/paper/broker.py | KEEP | Paper broker |
| src/paper/exchange.py | KEEP | Paper exchange |
| src/paper/orders.py | KEEP | Orders |
| src/paper/risk.py | KEEP | Risk manager |
| src/paper/runner.py | KEEP | Replay runner |
| src/paper/strategy_adapter.py | KEEP | Factor target weights |
| src/pipeline/*.py (reachable) | KEEP | Data, features |
| src/portfolio/*.py | KEEP | Portfolio allocator |
| src/reporting/tearsheet.py | KEEP | Canonical reporting |
| src/strategies/*.py (reachable) | KEEP | Strategies used by run_demo/tests |
| src/utils/*.py | KEEP | CLI, io, jsonable, runlock |

## Tests (KEEP)

| File | Status | Reason |
|------|--------|--------|
| tests/*.py | KEEP | All tests |

## DEPRECATE (stub + warning)

| File | Status | Reason | Replacement |
|------|--------|--------|-------------|
| scripts/backtest_baselines.py | DEPRECATE | Not entrypoint | backtest_factors for factors |
| scripts/backtest_pairs.py | DEPRECATE | Not entrypoint | — |
| scripts/backtest_papers.py | DEPRECATE | Not entrypoint | — |
| scripts/backtest_portfolio.py | DEPRECATE | Not entrypoint | — |
| scripts/backtest_strategies.py | DEPRECATE | Not entrypoint | run_demo for single strategy |
| scripts/make_research_bundle.py | DEPRECATE | Auxiliary | Run entrypoints manually |
| scripts/run_options_demo.py | DEPRECATE | Not entrypoint | — |
| scripts/sweep_momentum.py | DEPRECATE | Auxiliary | — |

## DELETE (unreachable, not experimental)

| File | Status | Reason |
|------|--------|--------|
| run_tensorboard.py | DELETE | DL-only, not in entrypoints |
| train.py | DELETE | DL-only, not in entrypoints |

## Move to src/experimental/

| File | Status | Reason |
|------|--------|--------|
| src/models/bilstm.py | → experimental | DL, unreachable from entrypoints |
| src/models/factory.py | → experimental | DL |
| src/models/gru.py | → experimental | DL |
| src/models/tcn.py | → experimental | DL |
| src/models/transformer.py | → experimental | DL |
| src/pipeline/baselines.py | → experimental | ARIMA, not used by entrypoints |
| src/pipeline/data_fetcher.py | KEEP | Abstract base for data_fetcher_yahoo |

## Package __init__.py (KEEP for structure)

| File | Status |
|------|--------|
| src/__init__.py | KEEP |
| src/pipeline/__init__.py | KEEP |
| src/reporting/__init__.py | KEEP |
| src/utils/__init__.py | KEEP |
| src/strategies/*/__init__.py | KEEP |

## Duplicate Symbols (no merge — different domains)

- generate_signals, backtest_returns: Strategy interface (each strategy implements)
- fetch_ohlcv: Abstract + implementations (data_fetcher base, yahoo, alpha)
- estimate_costs: CostModel ABC + implementations
- main: Script entry points

## Execution Order

1. Create src/experimental/, move models (except lstm), baselines
2. Update train.py to import from src.experimental or DELETE train.py
3. Replace deprecated scripts with stubs
4. DELETE run_tensorboard.py, train.py (or move to tools/)
5. Update tests that import from moved modules
