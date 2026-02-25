# Quant Forecast

A quantitative research platform for realistic backtesting: no-lookahead execution, execution realism, walk-forward evaluation, and tear-sheet reporting. Built for serious strategy research, not toy demos.

---

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_demo.py --symbol SPY --period 2y
```

**Smoke test** (~30 sec): `python scripts/run_demo.py --symbol SPY --period 30d`

---

## Outputs

After `run_demo`, check `output/`:

| File | Description |
|------|-------------|
| `summary.json` | Key metrics + config |
| `REPORT.md` | Markdown report with embedded plots |
| `tearsheet.html` | HTML summary + charts |
| `equity_curve.png` | Cumulative return |
| `drawdown.png` | Underwater curve |
| `rolling_sharpe.png` | Rolling Sharpe ratio |
| `returns_hist.png` | Return distribution |
| `positions.png` | Position exposure over time |
| `turnover.png` | Turnover |

---

## Methodology

- **No lookahead**: Signal at close `t` → fill at open `t+1`
- **Time split**: Walk-forward train/test; test never used for calibration
- **Costs**: Configurable fees, slippage, spread (1–2 bps each)
- **No leakage**: Features use only past data (rolling windows)

---

## Architecture

```
Data (Yahoo) → Strategy → Execution (fees/slippage) → Backtester → Tear-sheet
                    │
                    └── signal at close t → fill at t+1 (no lookahead)
```

---

## Minute-Bar Guardrails

```bash
python scripts/run_demo.py --symbol SPY --period 7d --interval 1m
python scripts/run_demo.py --symbol SPY --period 30d --interval 5m
```

Yahoo limits: 1m ~7d, 5m ~60d. We auto-cap period; use `--period-override` to bypass. For minute bars, `min_hold_bars` defaults to 5 to reduce trade spam.

---

## Walk-Forward

```bash
python scripts/walkforward.py --symbol SPY --period 2y --train-days 252 --test-days 63
```

---

## Repo Layout

```
quant-forecast/
├── src/backtest/     # Engine, execution, walk-forward
├── src/strategies/   # Momentum, mean reversion, papers
├── src/pipeline/     # Data fetchers, features
├── src/reporting/    # Tear-sheet generation
├── scripts/          # run_demo.py, walkforward.py
├── configs/          # JSON configs
├── output/           # Tear-sheet outputs
└── tests/
```

---

## FAQ

**Q: Why can't I run minute-bar backtests for 6 months?**  
A: Yahoo Finance restricts 1m data to ~7 days per request. Use daily data or a paid provider for longer backtests.

**Q: Config-driven run?**  
A: `python scripts/run_demo.py --config configs/demo_spy_momentum.json`

---

## Documentation

- [docs/REALITY_CHECK.md](docs/REALITY_CHECK.md) — What works, what's broken
- [docs/RESEARCH_METHOD.md](docs/RESEARCH_METHOD.md) — Data assumptions, evaluation protocol
- [docs/INTEGRATION.md](docs/INTEGRATION.md) — Live broker integration

---

## License

MIT
