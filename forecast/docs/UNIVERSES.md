# Universe Reference

Cross-sectional factor backtests use predefined universes of ETFs. The `UniverseRegistry` in `src/factors/universe.py` provides symbols and metadata.

---

## Available Universes

| Universe | Category | Description |
|----------|----------|-------------|
| `liquid_etfs` | mixed | Broad mix of liquid US equity, bond, commodity ETFs (SPY, QQQ, GLD, TLT, etc.) |
| `sector_etfs` | sector | S&P 500 sector SPDRs (XLK, XLF, XLV, XLE, XLI, XLP, XLY, XLB, XLU, XLRE) |
| `bond_etfs` | bond | Treasury, corporate, and aggregate bond ETFs (TLT, IEF, LQD, HYG, BND, AGG, etc.) |
| `commodity_etfs` | commodity | Precious metals, energy, agriculture (GLD, SLV, USO, UNG, DBA, DBC, etc.) |
| `international_etfs` | international | Developed and emerging market equity ETFs (EFA, EEM, VEA, VWO, etc.) |

---

## Usage

### backtest_factors.py

```bash
# List available universes
python scripts/backtest_factors.py --list-universes

# Run backtest on sector ETFs
python scripts/backtest_factors.py --universe sector_etfs --factor momentum_12_1 --period 5y

# Run on bond universe
python scripts/backtest_factors.py --universe bond_etfs --factor lowvol_20d --rebalance M
```

### replay_trade.py

```bash
# List universes with symbol preview
python scripts/replay_trade.py --list-universes

# Inspect symbols for a universe
python scripts/replay_trade.py --universe commodity_etfs
python scripts/replay_trade.py --universe international_etfs --n 10
```

### Programmatic

```python
from src.factors import UniverseRegistry, get_universe

# Get symbols (backward compatible)
symbols = get_universe("liquid_etfs", n=20)

# Get symbols + metadata
symbols, meta = UniverseRegistry.get("sector_etfs", n=10)
print(meta.description)  # "S&P 500 sector SPDRs..."

# List all universe names
names = UniverseRegistry.list_names()
```

---

## Notes

- Universes are **hardcoded** to avoid survivorship bias in backtests.
- Symbols are snapshots of commonly traded ETFs; update periodically for production.
- For beta-neutral backtests, ensure the market symbol (e.g. SPY) is in the universe or use `--market-symbol`.
