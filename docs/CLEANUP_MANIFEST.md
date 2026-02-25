# Cleanup Manifest — Redundancy Audit

## 1. Duplicated Functions/Modules

| Item | Locations | Decision | Notes |
|------|-----------|----------|-------|
| **Cost calculations** | `cost_models.py`, `execution.py`, `portfolio.apply_rebalance_costs`, `paper/exchange._cost_for_trade` | KEEP (different domains) | Each serves distinct context. |
| **Output-dir creation** | 7+ scripts | ✅ DONE | `make_output_dir()`, `timestamp_for_run()` in `src/utils/io.py`. |
| **Factor computation** | `src/factors/factors.py` only | N/A | Single source. |

## 2. Dead / Removed Files

| File | Decision | Notes |
|------|----------|-------|
| `scripts/walkforward.py` | ✅ DELETED | Superseded by `walkforward_demo.py`. |
| `docs/FACTS.md` | ✅ DELETED | Overlapped with STRUCTURE.md, RESEARCH_METHOD.md. |

## 3. Redundant Docs (Audit)

| Doc | Status |
|-----|--------|
| `docs/REALITY_CHECK.md` | Updated — fixed outdated "What Is Broken". |
| `docs/AGENT_PROMPT.md` | Keep — useful for external agents. |
| `docs/CLEANUP_MANIFEST.md` | This file — internal planning. |

## 4. Test Import Fixes

| File | Fix |
|------|-----|
| `tests/test_train.py` | ✅ `from src.models.lstm` (was `from models.lstm`) |
| `tests/test_dataset.py` | ✅ `from src.pipeline.scheduler` (was `from pipeline.scheduler`) |

## 5. Jupyter Notebooks

- **No `.ipynb` files** in the repo.
- **Added to `.gitignore`**: `*.ipynb`, `.ipynb_checkpoints/` — project uses scripts, not notebooks.

## 6. Scripts Reference

| Script | Status |
|--------|--------|
| `walkforward_demo.py` | Primary walk-forward; used by make_research_bundle |
| `run_demo.py` | Primary single-symbol demo |
| `backtest_factors.py` | Factor backtest |
| `replay_trade.py` | Paper replay |
| `daily_run.py` | Production daily pipeline |
| `monitor_runs.py` | Run monitoring |

## 7. Archived / Experimental (Not Removed)

| Item | Notes |
|------|-------|
| `src/strategies/influencer/` | ARCHIVED header; requires external APIs |
| `src/archive/README.md` | Describes archived strategies |
| `train.py`, `run_tensorboard.py` | DL path; torch optional |
