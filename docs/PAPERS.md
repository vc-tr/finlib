# Academic Paper Strategies

Strategies implemented from peer-reviewed finance papers. Each includes citation and methodology.

---

## Moskowitz, Ooi, Pedersen (2012) — Time Series Momentum

**Paper:** "Time Series Momentum", *Journal of Financial Economics*, 104(2), 228-250.

**Strategy:** `MoskowitzTimeSeriesMomentum`
- Signal = sign(12-month past return)
- Hold for 1 month, then rebalance
- Tested on 58 futures/forwards (equity, FX, commodities, bonds)

**Usage:**
```python
from src.strategies import MoskowitzTimeSeriesMomentum
strat = MoskowitzTimeSeriesMomentum(formation_period=252, holding_period=21)
signals, returns = strat.backtest_returns(prices)
```

---

## Jegadeesh & Titman (1993) — Cross-Sectional Momentum

**Paper:** "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency", *Journal of Finance*, 48(1), 65-91.

**Strategy:** `JegadeeshTitmanMomentum`
- Rank stocks by past J-month return (J=3,6,9,12)
- Long top decile (winners), short bottom decile (losers)
- Hold for K months, rebalance monthly
- **Requires:** DataFrame of asset returns (multiple assets)

**Usage:**
```python
from src.strategies import JegadeeshTitmanMomentum
# returns: DataFrame (dates x assets)
strat = JegadeeshTitmanMomentum(formation_period=126, holding_period=126)
portfolio_returns, _ = strat.backtest_returns(returns)
```

---

## Gatev, Goetzmann, Rouwenhorst (2006) — Pairs Trading

**Paper:** "Pairs Trading: Performance of a Relative-Value Arbitrage Rule", *Review of Financial Studies*, 19(3), 797-827.

**Strategy:** `GatevGoetzmannRouwenhorstPairs`
- Match pairs by minimum sum of squared deviations of *normalized* prices (not cointegration)
- Normalize: price / first_price
- Enter when |spread| > 2σ; exit when spread crosses 0
- Paper: 1962-2002, annualized excess returns up to 11%

**Usage:**
```python
from src.strategies import GatevGoetzmannRouwenhorstPairs
strat = GatevGoetzmannRouwenhorstPairs(formation_period=252, entry_std=2.0)
signals, returns = strat.backtest_returns(price_a, price_b)
```

---

## De Bondt & Thaler (1985, 1987) — Long-Term Reversal

**Paper:** "Does the Stock Market Overreact?", *Journal of Finance*, 40(3), 793-805.  
"Further Evidence on Investor Overreaction and Stock Market Seasonality", *JF* 1987.

**Strategy:** `DeBondtThalerReversal`
- Contrarian: buy past *losers* (3-5 yr), sell past *winners*
- Formation and holding: 3-5 years
- Signal = -sign(formation-period return)

**Usage:**
```python
from src.strategies import DeBondtThalerReversal
strat = DeBondtThalerReversal(formation_period=756, holding_period=756)
signals, returns = strat.backtest_returns(prices)
```

---

## Data Requirements

| Strategy | Input | Notes |
|----------|-------|-------|
| Moskowitz TSMOM | Single price series | Daily; 12m formation needs ~252 bars |
| Jegadeesh-Titman | DataFrame of returns | Multiple assets; cross-sectional |
| GGR Pairs | Two price series | Or DataFrame to find best pair |
| De Bondt-Thaler | Single price series | Long formation (3-5 yr) |
