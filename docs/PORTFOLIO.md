# Portfolio Management

Multi-strategy portfolio construction and allocation.

---

## Overview

The portfolio module combines multiple strategies into a single portfolio with configurable allocation:

- **Equal weight**: 1/N per strategy
- **Risk parity**: Each strategy contributes equally to portfolio risk
- **Inverse volatility**: Weight inversely proportional to volatility
- **Custom**: User-defined weights

---

## Quick Start

```python
from src.portfolio import MultiStrategyPortfolio, PortfolioAllocator, AllocationMethod
from src.strategies import MeanReversionStrategy, MomentumStrategy
from src.backtest import Backtester

# Build portfolio
portfolio = MultiStrategyPortfolio()
portfolio.add_strategy("Mean Rev", MeanReversionStrategy(lookback=20), lambda s: s.backtest_returns(prices))
portfolio.add_strategy("Momentum", MomentumStrategy(lookback=20), lambda s: s.backtest_returns(prices))

# Run with equal weight
allocator = PortfolioAllocator(method=AllocationMethod.EQUAL, rebalance_freq=21)
returns_df, weights_df, port_returns = portfolio.run(allocator=allocator)

# Backtest
result = Backtester().run(port_returns)
print(f"Portfolio Sharpe: {result.sharpe_ratio:.2f}, Return: {result.total_return:.2%}")
```

---

## Allocation Methods

| Method | Description | Best for |
|--------|-------------|----------|
| **EQUAL** | 1/N per strategy | Simple, robust |
| **INVERSE_VOLATILITY** | Weight ∝ 1/vol | Reduce vol contribution from noisy strategies |
| **RISK_PARITY** | Equal risk contribution | Balanced risk across strategies |
| **CUSTOM** | User-defined weights | Override with prior views |

---

## Parameters

- **lookback**: Bars for volatility/covariance estimation (default 63)
- **rebalance_freq**: Rebalance every N bars (0 = initial weights only)
- **custom_weights**: Dict of strategy_name -> weight for CUSTOM method

---

## CLI

```bash
# Equal weight, rebalance monthly
python scripts/backtest_portfolio.py --symbol SPY --period 2y

# Risk parity
python scripts/backtest_portfolio.py --symbol SPY --period 2y --alloc risk_parity

# Inverse volatility, rebalance weekly
python scripts/backtest_portfolio.py --symbol SPY --period 2y --alloc inverse_vol --rebalance 5
```

---

## Multi-Asset / Multi-Universe

For portfolios across multiple assets (e.g. SPY, QQQ, sector ETFs), add strategies per asset and use the same allocator. The `universe` field in `add_strategy` can tag strategies for grouping.
