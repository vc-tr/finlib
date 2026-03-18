# Quant Engine

A high-performance quantitative finance library written in C++20 with Python bindings. Implements pricing models, Monte Carlo simulation, PDE solvers, Greeks computation, yield curve construction, and portfolio risk analytics from scratch.

## Architecture

```
quant-engine/
├── include/qe/          # Public headers
│   ├── core/            # Types, constants
│   ├── math/            # RNG, statistics, interpolation, linear algebra, root-finding
│   ├── models/          # Black-Scholes, Heston, SABR, Merton JD, CIR
│   ├── processes/       # SDE framework, GBM, correlated Brownians
│   ├── montecarlo/      # MC engine with variance reduction
│   ├── instruments/     # Payoffs (vanilla, digital, barrier, Asian, lookback, basket)
│   ├── pde/             # Finite difference solvers (explicit, implicit, Crank-Nicolson)
│   ├── greeks/          # Finite difference, pathwise (IPA), likelihood ratio Greeks
│   ├── curves/          # Yield curve bootstrapping
│   ├── volatility/      # Implied vol solver, vol surface
│   └── risk/            # VaR, CVaR, stress testing, portfolio risk
├── src/                 # Implementation files
├── tests/               # 18 Catch2 test suites
├── benchmarks/          # Performance benchmarks
└── python/              # pybind11 bindings
```

## Features

### Pricing Models
- **Black-Scholes** — Closed-form pricing and full Greeks (delta, gamma, vega, theta, rho)
- **Heston** — Stochastic volatility via characteristic function (Gauss-Laguerre quadrature) and Monte Carlo with full-truncation Euler scheme
- **SABR** — Hagan et al. (2002) implied volatility approximation for smile/skew modeling
- **Merton Jump-Diffusion** — Semi-analytical (Poisson-weighted BS sum) and Monte Carlo with compound Poisson jumps
- **CIR** — Cox-Ingersoll-Ross short rate model with closed-form bond pricing

### Numerical Methods
- **Monte Carlo Engine** — European option pricing with variance reduction (antithetic variates, control variates, importance sampling)
- **Exotic Options** — Barrier (up/down, in/out), Asian (arithmetic/geometric), lookback, basket options
- **American Options** — Longstaff-Schwartz least-squares Monte Carlo with Laguerre polynomial basis
- **PDE Solvers** — Black-Scholes PDE via explicit, implicit, and Crank-Nicolson finite difference schemes
- **SDE Framework** — Euler-Maruyama, Milstein, and exact (log-normal) discretization schemes

### Greeks (Three Independent Methods)
- **Finite Difference** — Bump-and-reprice (works for any model/payoff)
- **Pathwise (IPA)** — Differentiates through the simulation path (efficient, smooth payoffs only)
- **Likelihood Ratio** — Score function method (works for all payoffs including digitals)

### Market Data
- **Yield Curves** — Cubic spline interpolation, discount factors, forward rates, bootstrapping from deposits and swaps
- **Implied Volatility** — Newton-Raphson solver with Brenner-Subrahmanyam initial guess and bisection fallback
- **Volatility Surface** — Bilinear interpolation over (strike, maturity) grid, smile extraction

### Risk Analytics
- **Value-at-Risk** — Historical and parametric (Gaussian) VaR
- **CVaR / Expected Shortfall** — Coherent tail risk measure (always ≥ VaR)
- **Stress Testing** — Reprice portfolios under predefined or custom shock scenarios
- **Portfolio Aggregation** — Total value, delta, gamma, vega, theta, rho across positions

### Math Foundation
- Mersenne Twister RNG with Box-Muller normal generation
- Online (Welford) running statistics
- Cubic spline and linear interpolation
- Root-finding (Newton-Raphson, bisection, Brent's method)
- LU decomposition, tridiagonal solver (Thomas algorithm)

## Build

### Requirements
- C++20 compiler (GCC 11+, Clang 14+, Apple Clang 14+)
- CMake 3.20+
- Python 3.8+ and pybind11 (optional, for Python bindings)

### Build & Test
```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run all 18 test suites
cd build && ctest

# Run benchmarks
./build/benchmarks
```

### Python Bindings
```bash
pip install pybind11
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build -j$(nproc)

# Use from Python
PYTHONPATH=build python3 -c "import quant_engine; print(quant_engine.__version__)"
```

## Python Usage

```python
import quant_engine as qe

# Black-Scholes pricing
call = qe.bs.call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
greeks = qe.bs.greeks(S=100, K=100, r=0.05, sigma=0.2, T=1.0, type=qe.OptionType.Call)

# Monte Carlo
cfg = qe.mc.Config()
cfg.spot = 100; cfg.num_paths = 500000
payoff = qe.mc.VanillaPayoff(100.0, qe.OptionType.Call)
result = qe.mc.Engine(cfg).price(payoff)

# Heston stochastic vol
hp = qe.heston.Params()
hp.spot=100; hp.v0=0.04; hp.kappa=2.0; hp.theta=0.04; hp.xi=0.3; hp.rho=-0.7; hp.maturity=1.0
price = qe.heston.call(hp, strike=100.0)

# American options (LSM)
ac = qe.american.Config()
ac.spot=100; ac.strike=100; ac.sigma=0.2; ac.type=qe.OptionType.Put; ac.num_paths=100000
result = qe.american.Pricer(ac).price()

# Risk analytics
pnl = [...]  # your P&L data
var = qe.risk.historical_var(pnl, confidence=0.95)
cvar = qe.risk.historical_cvar(pnl, confidence=0.95)
```

## Benchmarks

Measured on Apple M-series, single-threaded, Release build (`-O3`):

| Operation | Time | Notes |
|---|---|---|
| BS Call Price | 0.45 μs | Closed-form |
| BS All Greeks | 0.13 μs | Delta, gamma, vega, theta, rho |
| Implied Vol (Newton) | 0.40 μs | Converges in ~3 iterations |
| MC European 100K | 8.5 ms | Single-step GBM |
| MC European 1M | 59 ms | Single-step GBM |
| Crank-Nicolson PDE | 3.4 ms | 200×1000 grid |
| Heston Analytical | 71 μs | Characteristic function + quadrature |
| Portfolio Risk (20 pos) | 4 μs | Aggregate Greeks |
| Stress Test (20 pos, 7 scen) | 17 μs | Full reprice under shocks |

## Testing

18 test suites with 100+ test cases covering:
- Analytical pricing accuracy against known results
- Put-call parity verification
- Monte Carlo convergence and variance reduction effectiveness
- Model limiting cases (e.g., Heston → BS when ξ=0)
- Greek consistency across three independent methods
- Risk metric properties (CVaR ≥ VaR, monotonicity)
- Yield curve roundtripping and bootstrapping
- Implied vol solver convergence

```bash
cd build && ctest --output-on-failure
```

## Dependencies

- [Catch2](https://github.com/catchorg/Catch2) v3 — Testing (auto-fetched via CMake)
- [pybind11](https://github.com/pybind/pybind11) — Python bindings (optional)

No other external dependencies. All numerical methods implemented from scratch.

## License

MIT
