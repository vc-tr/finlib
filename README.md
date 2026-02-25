# Quant Forecast

A credible quantitative research platform: realistic backtesting, walk-forward evaluation, and tear-sheet reporting.

---

## Three Pillars

| Pillar | Description |
|--------|--------------|
| **A) Realistic Backtesting** | Fees, slippage, execution timing. No lookahead. |
| **B) Walk-Forward Evaluation** | Rolling OOS folds. No leakage. |
| **C) Tear-Sheet Reporting** | Equity curve, drawdown, rolling Sharpe, summary table. |

---

## One-Command Quickstart

```bash
pip install -r requirements.txt
python scripts/run_demo.py --symbol SPY --period 2y
```

This runs:
1. Data download (Yahoo)
2. Momentum strategy + execution realism (fees/slippage)
3. Backtest
4. Tear-sheet (HTML + PNG in `output/`)

**Smoke test** (30 seconds):
```bash
python scripts/run_demo.py --symbol SPY --period 30d
```

---

## Architecture

```
Data (Yahoo) → Strategy (signals) → Execution (fees/slippage) → Backtester → Tear-Sheet
                    ↑
              signal at close t → fill at t+1 (no lookahead)
```

---

## Anti-Leakage Rules

- Signal at bar `t` executes at bar `t+1`
- Walk-forward: train/test split; test never used for calibration
- Features use only past data (rolling windows)

---

## Realism Assumptions

- **Fees**: 5 bps per trade (configurable)
- **Slippage**: 5 bps (configurable; optional vol-scaled)
- **Execution**: Signal at close → fill at next open

---

## Config-Driven Run

```bash
python scripts/run_demo.py --config configs/demo_spy_momentum.json
```

Example config: `configs/demo_spy_momentum.json`

---

## Walk-Forward

```bash
python scripts/walkforward.py --symbol SPY --period 2y --train-days 252 --test-days 63
```

---

## Tests

```bash
pytest tests/test_strategies.py tests/test_execution.py tests/test_data_fetcher.py tests/test_pipeline.py tests/test_features.py -v
```

Core tests (no torch): 34+ pass.

---

## Example Tear-Sheet

After `run_demo`, open `output/tearsheet.html` for:
- Summary table (CAGR, Sharpe, Sortino, MaxDD, Win rate, etc.)
- Equity curve, drawdown, rolling Sharpe
- Returns histogram, exposure, turnover

---

## Documentation

- **[docs/REALITY_CHECK.md](docs/REALITY_CHECK.md)** — What works, what's broken, quick wins
- **[docs/RESEARCH_METHOD.md](docs/RESEARCH_METHOD.md)** — Data assumptions, evaluation protocol, production path
- **[docs/INTEGRATION.md](docs/INTEGRATION.md)** — Live broker integration (Alpaca, Tradier, IB)

---

## License

MIT
