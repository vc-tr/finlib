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
python scripts/walkforward_demo.py --symbol SPY --interval 1d --folds 6 --train-days 90 --test-days 30
```

---

## Paper Trading Replay

Event-driven paper trading replay on historical data (no broker accounts):

```bash
python scripts/replay_trade.py --strategy factors --universe liquid_etfs --factor momentum_12_1 --rebalance M --start 2024-01-01 --end 2025-12-31
```

Combo factors with auto_robust:

```bash
python scripts/replay_trade.py --strategy factors --factor combo --combo "momentum_12_1,reversal_5d,lowvol_20d" --combo-method auto_robust --rebalance M
```

Outputs: `orders.csv`, `blotter.csv`, `equity_curve.csv`, `positions_snapshot.csv`, `replay_report.md`

---

## One Command Research Bundle

Generate a recruiter-friendly packet (daily demo, walk-forward, intraday demo, sweep) in one run:

```bash
python scripts/make_research_bundle.py --symbol SPY
```

Output: `output/runs/<timestamp>_bundle_SPY/` with subfolders `daily_demo/`, `walkforward/`, `intraday_demo/`, `sweep/`, plus `INDEX.md` and `output/latest/` mirror.

---

## Repo Layout

```
quant-forecast/
├── src/backtest/     # Engine, execution, walk-forward
├── src/paper/        # Paper trading replay (orders, broker, exchange, risk)
├── src/strategies/   # Momentum, mean reversion, papers
├── src/pipeline/     # Data fetchers, features
├── src/reporting/    # Tear-sheet generation
├── scripts/          # run_demo.py, replay_trade.py, backtest_factors.py
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
- [docs/UNIVERSES.md](docs/UNIVERSES.md) — Universe definitions (liquid_etfs, sector_etfs, bond_etfs, etc.)
- [docs/INTEGRATION.md](docs/INTEGRATION.md) — Live broker integration

---

## License

MIT
