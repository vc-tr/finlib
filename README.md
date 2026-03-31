## Project Structure

```
.
├── engine/
│   ├── CMakeLists.txt
│   ├── include/qe/
│   │   ├── core/          types, constants
│   │   ├── math/          RNG, interpolation, root finding, linalg, statistics
│   │   ├── models/        BS, Heston, SABR, Merton JD, CIR
│   │   ├── instruments/   european, american (LSM), exotic (barrier, asian, lookback, basket)
│   │   ├── processes/     GBM, SDE, discretization, correlation
│   │   ├── montecarlo/    engine, convergence
│   │   ├── greeks/        finite difference, pathwise, likelihood ratio
│   │   ├── pde/           FDM solver (explicit, implicit, Crank-Nicolson)
│   │   ├── curves/        yield curve
│   │   ├── volatility/    implied vol, vol surface
│   │   └── risk/          VaR, CVaR, stress testing, portfolio
│   ├── src/               23 implementation files
│   ├── tests/             18 Catch2 test suites
│   ├── benchmarks/        performance benchmarks
│   └── python/            pybind11 bindings
│
└── forecast/
    ├── configs/            backtest configurations
    ├── scripts/            entry points (run_demo, walkforward, factor backtest, paper trading)
    ├── src/
    │   ├── strategies/     signal generators (momentum, mean reversion, pairs trading)
    │   ├── factors/        cross-sectional factors, ensemble, portfolio construction
    │   ├── backtest/       engine, execution model, walk-forward validation
    │   ├── pipeline/       data fetching (Yahoo Finance), preprocessing
    │   ├── reporting/      tearsheet generation (HTML, PNG, markdown)
    │   ├── paper/          event-driven paper trading (broker, exchange, risk)
    │   ├── ops/            daily portfolio generation, monitoring
    │   └── utils/          CLI, I/O, serialization
    ├── tests/
    └── requirements.txt

---

# quant-engine

A quantitative finance monorepo combining a **zero-dependency C++ pricing engine** with a **Python research and backtesting platform**. The engine handles derivatives pricing, Greeks, risk analytics, and PDE solvers from scratch. The forecast platform provides strategy research, cross-sectional factor models, walk-forward validation, and paper trading — all with strict anti-lookahead guarantees.

```
quant-engine/
├── engine/     C++ pricing library with Python bindings
└── forecast/   Strategy research, backtesting & paper trading
```

---

## Engine — C++ Quantitative Finance Library

A header-heavy C++20 library implementing all numerical methods from scratch — no QuantLib, no Boost, no external math libraries.

### Pricing Models

| Model | Method | Key Feature |
|-------|--------|-------------|
| **Black-Scholes** | Closed-form | Analytical Greeks (delta, gamma, vega, theta, rho) |
| **Heston** | Characteristic function + Gauss-Laguerre quadrature | QE scheme (Andersen 2008) for MC paths |
| **SABR** | Hagan et al. (2002) implied vol approximation | Smile/skew calibration |
| **Merton Jump-Diffusion** | Poisson-weighted BS series (50 terms) + MC | Compound Poisson jumps on GBM |
| **CIR** | Exact non-central chi-squared simulation | Mean-reverting short rates, closed-form bond pricing |

### Instruments

- **European** options (call/put)
- **American** options via Longstaff-Schwartz LSM (Laguerre polynomial basis)
- **Exotic** path-dependent options:
  - Barrier (down/up, in/out)
  - Asian (arithmetic & geometric averaging)
  - Lookback (floating strike)
  - Basket (multi-asset with Cholesky-correlated paths)

### Monte Carlo Engine

Universal MC pricer with three variance reduction techniques:

- **Antithetic variates** — paired ±dW paths
- **Control variates** — customizable control function
- **Importance sampling** — drift-shifted measure

Builder pattern API. Returns price, standard error, 95% CI, and timing.

### Greeks

Three independent computation methods:

| Method | Approach | Payoff Requirement |
|--------|----------|--------------------|
| **Finite Difference** | Bump-and-reprice | Any payoff |
| **Pathwise (IPA)** | Differentiate through paths | Smooth payoffs only |
| **Likelihood Ratio** | Score function | All payoffs (incl. digital) |

### PDE Solvers

Black-Scholes PDE via finite difference on log-space grid:

- **Explicit** (conditionally stable)
- **Implicit** (unconditionally stable)
- **Crank-Nicolson** (2nd-order accurate)

Configurable grid: up to 200 spot points × 1000 time steps. Extracts price, delta, gamma, theta.

### Yield Curves & Volatility

- **Yield curve**: Bootstrap from deposit/swap rates, cubic spline interpolation, discount factors, forward rates
- **Implied vol solver**: Newton-Raphson with Brenner-Subrahmanyam seed + bisection fallback
- **Vol surface**: Bilinear interpolation over (strike, maturity) grid with smile extraction

### Risk Analytics

- **VaR**: Historical percentile and parametric (Gaussian)
- **CVaR / Expected Shortfall**: Coherent tail risk measure
- **Stress testing**: Spot shocks, vol shocks, rate shocks across predefined scenarios
- **Portfolio aggregation**: Total value and Greek aggregation across positions

### Numerical Foundation

All implemented from scratch:

- Mersenne Twister RNG + Box-Muller normal generation
- Sobol quasi-random sequences
- Cubic spline and linear interpolation
- Newton-Raphson, bisection, Brent's method root finders
- Cholesky decomposition, Thomas tridiagonal solver
- Welford online running statistics

### Performance

Benchmarked on Apple Silicon (Release build, `-O3`):

| Operation | Time |
|-----------|------|
| BS call price | 0.45 μs |
| BS all Greeks | 0.13 μs |
| Implied vol solve | 0.40 μs |
| Heston analytical | 71 μs |
| MC European (1M paths) | 59 ms |
| Crank-Nicolson PDE (200×1000) | 3.4 ms |
| Portfolio risk (20 positions) | 4 μs |
| Stress test (20 pos, 7 scenarios) | 17 μs |

### Build

```bash
# Requirements: C++20 compiler (GCC 11+, Clang 14+, Apple Clang 14+)
cd engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests (Catch2 auto-fetched)
cd build && ctest --output-on-failure

# With Python bindings (optional)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build -j$(nproc)
```

### Python Bindings

```python
import quant_engine as qe

# Black-Scholes
price = qe.bs.call_price(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
greeks = qe.bs.greeks(S=100, K=100, r=0.05, sigma=0.2, T=1.0)

# Monte Carlo with variance reduction
result = qe.mc.price_european(S=100, K=100, r=0.05, sigma=0.2, T=1.0,
                               n_paths=1_000_000, antithetic=True)

# Heston stochastic vol
price = qe.heston.call_price(S=100, K=100, r=0.05, v0=0.04, kappa=2.0,
                              theta=0.04, xi=0.3, rho=-0.7, T=1.0)

# American option (LSM)
price = qe.american.price(S=100, K=100, r=0.05, sigma=0.2, T=1.0,
                           n_paths=100_000, n_steps=252)

# Risk
var_95 = qe.risk.historical_var(pnl_series, confidence=0.95)
cvar_95 = qe.risk.cvar(pnl_series, confidence=0.95)
```

Full submodule list: `qe.bs`, `qe.mc`, `qe.heston`, `qe.sabr`, `qe.merton`, `qe.cir`, `qe.american`, `qe.exotic`, `qe.fdm`, `qe.risk`, `qe.curves`, `qe.implied_vol`

---

## Forecast — Quantitative Research & Backtesting Platform

A Python platform for strategy research, factor investing, and paper trading with production-grade execution modeling.

### Anti-Lookahead Guarantees

Every backtest enforces strict temporal separation:

- Signal computed at close of bar *t* → position filled at open of bar *t+1*
- Walk-forward validation: test window is never used for calibration
- Optional embargo period between train and test folds

### Strategies

**Base interface:**

```python
class Strategy:
    def meta() -> StrategyMeta          # name, category, hypothesis, expected result
    def generate_signals(prices) -> Series  # returns {-1, 0, +1}
    def parameter_grid() -> Dict        # for grid search
```

**Implemented strategies:**

| Strategy | Type | Sharpe (SPY, 5Y) | Notes |
|----------|------|-------------------|-------|
| Mean Reversion | Z-score | +0.77 | Best performer, DSR 0.964 |
| Three-Bar Reversal | Pattern | +0.57 | Borderline significance |
| Time-Series Momentum | Trend | +0.49 | Sign of lookback return |
| Carry Trade | Yield | +0.44 | Weak after costs |
| Pairs Trading | Stat arb | N/A | Requires multi-asset universe |

Statistical significance tested via **Deflated Sharpe Ratio** (corrects for multiple testing across N strategies).

### Cross-Sectional Factors

Three factor definitions with ensemble methods:

| Factor | Definition |
|--------|------------|
| **Momentum 12-1** | 12-month return skipping last 21 days |
| **Reversal 5d** | Negative 5-day return |
| **Low Volatility 20d** | Negative 20-day rolling vol |

**Ensemble methods:** Equal-weighted, IC-weighted, ridge regression, Sharpe-optimized, and auto-select with robustness checks.

**Universes:** 5 predefined ETF universes (liquid ETFs, sectors, bonds, commodities, international) totaling 100+ symbols.

### Portfolio Construction

- **Rebalance scheduling**: Daily, weekly (Friday), month-end
- **Constraints**: Max gross exposure, max net exposure, target leverage
- **Beta-neutral hedging**: Automatic market-beta hedge via SPY
- **Turnover tracking**: Computed only on actual position changes

### Execution Model

Realistic cost simulation:

| Component | Default |
|-----------|---------|
| Commission | 1 bps |
| Slippage | 2 bps |
| Spread | 1 bps |
| Vol-scaled slippage | Optional multiplier |
| Liquidity impact | Scales with participation rate vs ADV |

### Walk-Forward Validation

Rolling out-of-sample testing:

- Default: 252-day train / 63-day test / 63-day step
- Optional embargo gap between train and test
- Aggregates OOS metrics across folds
- Outputs: `walkforward_summary.json`, `WALKFORWARD_REPORT.md`

### Paper Trading

Event-driven simulation engine:

- Bar-by-bar chronological replay
- Order lifecycle: submit → fill at next bar → update P&L
- Risk manager: position limits, drawdown stops, exposure constraints
- Outputs: `orders.csv`, `blotter.csv`, `equity_curve.csv`, `positions_snapshot.csv`

### Reporting

Generates a full tearsheet with 9 outputs:

- `summary.json` — key metrics (programmatic access)
- `REPORT.md` — markdown summary
- `tearsheet.html` — interactive HTML with embedded charts
- `equity_curve.png`, `drawdown.png`, `rolling_sharpe.png`, `returns_hist.png`, `positions.png`, `turnover.png`

Metrics: CAGR, Sharpe, Sortino, Calmar, max drawdown, recovery time, win rate, turnover.

### Usage

```bash
cd forecast
pip install -r requirements.txt

# Single-symbol backtest
python scripts/run_demo.py --config configs/demo_spy_momentum.json

# Walk-forward validation
python scripts/walkforward_demo.py --symbol SPY --strategy momentum --train-days 252 --test-days 63

# Factor backtest on universe
python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1

# Paper trading replay
python scripts/replay_trade.py --config configs/demo_spy_momentum.json

# Strategy research (parameter sweep + significance testing)
python scripts/run_strategy.py --strategy mean_reversion --symbol SPY --period 5y
```

---

## Dependencies

### Engine
- **Zero external dependencies** — all numerical methods from scratch
- Catch2 v3 (auto-fetched for tests)
- pybind11 (optional, for Python bindings)

### Forecast
```
numpy>=2.0        pandas>=2.0       scipy>=1.10
statsmodels>=0.14  scikit-learn>=1.3  yfinance>=0.2
matplotlib>=3.7    seaborn>=0.12     pytest>=7.0
```

---

## Testing

```bash
# Engine — 18 Catch2 test suites, 100+ test cases
cd engine/build && ctest --output-on-failure

# Forecast
cd forecast && pytest tests/ -v
```

Tests cover: analytical accuracy, put-call parity, MC convergence, model limiting cases (Heston → BS as ξ→0), Greek consistency across three methods, risk metric properties, yield curve roundtripping, strategy signal correctness, and backtest anti-lookahead verification.

```
