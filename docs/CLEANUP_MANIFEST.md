# Cleanup Manifest — Redundancy Audit

## 1. Duplicated Functions/Modules

| Item | Locations | Decision | Notes |
|------|-----------|----------|-------|
| **Cost calculations** | `cost_models.py`, `execution.py`, `portfolio.apply_rebalance_costs`, `paper/exchange._cost_for_trade` | KEEP (different domains) | Each serves distinct context: factor backtest trades, single-asset signals, portfolio turnover, paper fills. Consolidation would risk behavior change. |
| **Output-dir creation** | 7+ scripts: `ts = datetime.now().strftime(...)`; `Path("output")/ "runs" / f"{ts}_X"` | MERGE | Add `make_output_dir(base, suffix)` in `src/utils/io.py`; scripts call it. |
| **Timestamp for run prefix** | Same 7+ scripts | MERGE | Add `timestamp_for_run()` in `src/utils/io.py`. |
| **Factor computation** | `src/factors/factors.py` only | N/A | Single source; no duplication. |

## 2. Dead / Unused Files

| File | Decision | Notes |
|------|----------|-------|
| `scripts/walkforward.py` | DEPRECATE | Superseded by `walkforward_demo.py`. Uses legacy `run_walkforward_legacy`; `walkforward_demo` uses newer API and is used by `make_research_bundle`. Keep stub that prints warning and calls `walkforward_demo` or exits with migration hint. |

## 3. Outdated Scripts Superseded by Newer Flows

| Script | Superseded By | Decision |
|--------|---------------|----------|
| `scripts/walkforward.py` | `scripts/walkforward_demo.py` | DEPRECATE — stub + warning |

## 4. Duplicated Utilities

| Utility | Locations | Decision |
|---------|------------|----------|
| **JSON serialization** | `to_jsonable` in `jsonable.py`; some code uses `json.dumps` without it | MERGE | Ensure `json.dumps(to_jsonable(obj), indent=2)` where obj may contain numpy/pandas. |
| **Date/period parsing** | `parse_period_days` in `io.py` | N/A | Single source. |
| **Lock** | `RunLock` in `runlock.py` | N/A | Single implementation; scripts repeat lock *usage* pattern but that's expected. |

## 5. High-Priority Removals (per task)

| Priority | Item | Action |
|----------|------|--------|
| 1 | Multiple cost implementations | Document only — different domains, no safe merge |
| 2 | Multiple output-dir generators | MERGE — add `make_output_dir` + `timestamp_for_run` |
| 3 | Multiple lock implementations | N/A — single `RunLock` |
| 4 | Duplicated factor computations | N/A — single `compute_factor` in `factors.py` |

## 6. Execution Plan

1. **Add to `src/utils/io.py`**: `timestamp_for_run()`, `make_output_dir(base, suffix)`.
2. **Update scripts** to use new utils: `daily_run`, `replay_trade`, `factors/runner`, `monitor_runs` (monitor uses fixed `output/monitor`).
3. **DEPRECATE `scripts/walkforward.py`**: Replace with stub that prints deprecation and suggests `walkforward_demo.py`.
4. **Audit json.dumps**: Ensure `to_jsonable` used where needed in `runner.py`, `ops/daily.py`, `ops/monitor.py`.
