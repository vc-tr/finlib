"""
Multi-strategy portfolio manager.

Runs multiple strategies on one or more universes, collects returns,
and combines them into a single portfolio using configurable allocation.
"""

import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field

from .allocator import PortfolioAllocator, AllocationMethod


@dataclass
class StrategySpec:
    """Specification for a strategy in the portfolio."""
    name: str
    strategy: Any
    run_fn: Callable[[Any], Tuple[pd.Series, pd.Series]]
    universe: Optional[str] = None


class MultiStrategyPortfolio:
    """
    Manage a portfolio of multiple strategies.
    
    Usage:
        portfolio = MultiStrategyPortfolio()
        portfolio.add_strategy("Momentum", MomentumStrategy(), lambda s: s.backtest_returns(prices))
        portfolio.add_strategy("Mean Rev", MeanReversionStrategy(), lambda s: s.backtest_returns(prices))
        returns_df, weights_df, port_returns = portfolio.run(allocator=PortfolioAllocator(method=EQUAL))
    """

    def __init__(self):
        self.strategies: List[StrategySpec] = []

    def add_strategy(
        self,
        name: str,
        strategy: Any,
        run_fn: Callable[[Any], Tuple[pd.Series, pd.Series]],
        universe: Optional[str] = None,
    ) -> "MultiStrategyPortfolio":
        """
        Add a strategy to the portfolio.
        
        Args:
            name: Strategy identifier
            strategy: Strategy instance
            run_fn: Function(strategy) -> (signals, returns)
            universe: Optional universe label (for multi-universe)
        """
        self.strategies.append(StrategySpec(
            name=name,
            strategy=strategy,
            run_fn=run_fn,
            universe=universe,
        ))
        return self

    def run(
        self,
        allocator: Optional[PortfolioAllocator] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Run all strategies and combine into portfolio.
        
        Returns:
            (strategy_returns_df, weights_df, portfolio_returns)
        """
        if not self.strategies:
            raise ValueError("No strategies added")
        allocator = allocator or PortfolioAllocator(method=AllocationMethod.EQUAL)
        returns_dict: Dict[str, pd.Series] = {}
        for spec in self.strategies:
            try:
                _, strat_returns = spec.run_fn(spec.strategy)
                returns_dict[spec.name] = strat_returns
            except Exception as e:
                returns_dict[spec.name] = pd.Series(dtype=float)
        returns_df = pd.DataFrame(returns_dict).fillna(0)
        weights_df = allocator.allocate(returns_df)
        port_returns = allocator.portfolio_returns(returns_df, weights_df)
        return returns_df, weights_df, port_returns

    def run_with_prices(
        self,
        prices: pd.Series,
        allocator: Optional[PortfolioAllocator] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Convenience: run strategies that use run_fn(strategy) -> (signals, returns).
        Assumes all strategies use the same price series.
        """
        return self.run(allocator)


def build_default_portfolio(
    prices: pd.Series,
    df_ohlcv: Optional[pd.DataFrame] = None,
) -> MultiStrategyPortfolio:
    """
    Build a portfolio with common strategies for single-asset backtest.
    
    prices: Close price series
    df_ohlcv: Optional full OHLCV for strategies that need it
    """
    from src.strategies import (
        MeanReversionStrategy,
        MomentumStrategy,
        RenaissanceSignalEnsemble,
        ScalpingStrategy,
        VWAPReversionStrategy,
        MoskowitzTimeSeriesMomentum,
    )
    portfolio = MultiStrategyPortfolio()
    portfolio.add_strategy(
        "Mean Reversion",
        MeanReversionStrategy(lookback=20, entry_z=2.0),
        lambda s: s.backtest_returns(prices),
    )
    portfolio.add_strategy(
        "Momentum",
        MomentumStrategy(lookback=20),
        lambda s: s.backtest_returns(prices),
    )
    portfolio.add_strategy(
        "Renaissance",
        RenaissanceSignalEnsemble(min_signal_agreement=0.3),
        lambda s: s.backtest_returns(prices),
    )
    portfolio.add_strategy(
        "Scalping",
        ScalpingStrategy(fast_ema=9, slow_ema=21),
        lambda s: s.backtest_returns(prices),
    )
    portfolio.add_strategy(
        "Moskowitz TSMOM",
        MoskowitzTimeSeriesMomentum(formation_period=252, holding_period=21),
        lambda s: s.backtest_returns(prices),
    )
    if df_ohlcv is not None:
        portfolio.add_strategy(
            "VWAP Reversion",
            VWAPReversionStrategy(entry_z=2.0),
            lambda s: s.backtest_returns(df_ohlcv),
        )
    return portfolio
