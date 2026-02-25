# Quant Forecast

A quantitative research platform for realistic backtesting: anti-lookahead execution, walk-forward evaluation, and tear-sheet reporting. Built for serious strategy research, not toy demos.

---

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_demo.py --symbol SPY --period 2y
```

**Smoke test** (~30 sec): `python scripts/run_demo.py --symbol SPY --period 30d`

---

## Demo Outputs

After `run_demo`, check `output/`:

| File | Description |
|------|-------------|
| `summary.json` | Key metrics + config used |
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

| Rule | Implementation |
|------|----------------|
| **Time split** | Walk-forward: train/test split; test never used for calibration |
| **Execution delay** | Signal at close `t` → fill at open `t+1` (no lookahead) |
| **Costs** | Configurable fees (1 bps), slippage (2 bps), spread (1 bps) per trade |
| **No leakage** | Features use only past data (rolling windows); no future peeking |

---

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐    ┌────────────┐
│ Data (Yahoo)│───▶│ Strategy     │───▶│ Execution layer │───▶│ Backtester │───▶│ Tear-sheet │
│ yfinance   │    │ (signals)    │    │ fees/slippage   │    │            │    │ HTML + PNG │
└─────────────┘    └──────────────┘    └─────────────────┘    └────────────┘    └────────────┘
                            │
                            ▼
                   signal at close t → fill at t+1 (no lookahead)
```

---

## Repo Layout

```
quant-forecast/
├── src/
│   ├── backtest/       # Engine, execution, walk-forward
│   ├── strategies/     # Momentum, mean reversion, papers
│   ├── pipeline/       # Data fetchers, features
│   └── reporting/      # Tear-sheet generation
├── scripts/
│   ├── run_demo.py     # One-command demo
│   └── walkforward.py  # Rolling OOS evaluation
├── configs/            # JSON configs (e.g. demo_spy_momentum.json)
├── output/             # Tear-sheet outputs
└── tests/
```

---

## Walk-Forward

```bash
python scripts/walkforward.py --symbol SPY --period 2y --train-days 252 --test-days 63
```

---

## Minute-Bar Demo

```bash
python scripts/run_demo.py --symbol SPY --period 7d --interval 1m
python scripts/run_demo.py --symbol SPY --period 30d --interval 5m
```

**Guardrails**: Yahoo limits 1m to ~7d, 5m to ~60d. We auto-cap period; use `--period-override` to bypass. For minute bars, `min_hold_bars` defaults to 5 to reduce trade spam.

---

## FAQ

**Q: Why can't I run minute-bar backtests for 6 months?**  
A: Yahoo Finance (yfinance) restricts 1-minute data to a **7-day window** per request. Longer intraday history requires chunked requests or a paid data provider. We cap `period` to `7d` when using `interval=1m` to avoid empty results.

**Q: How do I run with a config file?**  
A: `python scripts/run_demo.py --config configs/demo_spy_momentum.json`

---

## Next Steps

- [ ] Add chunked yfinance fetcher for multi-week minute data
- [ ] Integrate paid data (Polygon, Alpha Vantage) for production
- [ ] Extend walk-forward to strategy hyperparameter tuning
- [ ] Add live broker integration (Alpaca, IB) — see [docs/INTEGRATION.md](docs/INTEGRATION.md)

---

## Documentation

- [docs/REALITY_CHECK.md](docs/REALITY_CHECK.md) — What works, what's broken
- [docs/RESEARCH_METHOD.md](docs/RESEARCH_METHOD.md) — Data assumptions, evaluation protocol
- [docs/INTEGRATION.md](docs/INTEGRATION.md) — Live broker integration

---

## License

MIT
