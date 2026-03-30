"""
Strategy registry for the Quant Lab.

Use @StrategyRegistry.register to register a strategy class.
The registry enables:
  - Discovery of all strategies by name or category
  - Catalog table generation for README / reports
  - Instantiation by name from CLI

Usage:
    from src.strategies.registry import StrategyRegistry
    from src.strategies.base import Strategy, StrategyMeta

    @StrategyRegistry.register
    class MyStrategy(Strategy):
        def meta(self):
            return StrategyMeta(name="my_strategy", category="retail", source="...")
        def generate_signals(self, prices):
            ...
"""

from typing import Dict, List, Type

from .base import Strategy, StrategyMeta


class StrategyRegistry:
    """Central registry for all Quant Lab strategies."""

    _strategies: Dict[str, Type[Strategy]] = {}

    @classmethod
    def register(cls, strategy_cls: Type[Strategy]) -> Type[Strategy]:
        """
        Class decorator to register a strategy.

            @StrategyRegistry.register
            class GoldenCross(Strategy): ...
        """
        instance = strategy_cls()
        name = instance.meta().name
        if name in cls._strategies:
            raise ValueError(
                f"Strategy name '{name}' already registered by "
                f"{cls._strategies[name].__name__}. Choose a unique name."
            )
        cls._strategies[name] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> Strategy:
        """Instantiate a strategy by name."""
        if name not in cls._strategies:
            available = sorted(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: '{name}'. "
                f"Available: {available}"
            )
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_all(cls) -> List[StrategyMeta]:
        """Return metadata for all registered strategies, sorted by category then name."""
        metas = [cls._strategies[n]().meta() for n in sorted(cls._strategies)]
        return sorted(metas, key=lambda m: (m.category, m.name))

    @classmethod
    def list_by_category(cls, category: str) -> List[StrategyMeta]:
        """Return metadata for all strategies in a category."""
        return [m for m in cls.list_all() if m.category == category]

    @classmethod
    def categories(cls) -> List[str]:
        """Return sorted list of unique categories."""
        return sorted({cls._strategies[n]().meta().category for n in cls._strategies})

    @classmethod
    def names(cls) -> List[str]:
        """Return sorted list of all registered strategy names."""
        return sorted(cls._strategies.keys())

    @classmethod
    def catalog_markdown(cls) -> str:
        """
        Generate a markdown table of all registered strategies.
        Useful for README / STRATEGY_RESULTS.md generation.
        """
        lines = [
            "| Category | Strategy | Source | Description |",
            "|----------|----------|--------|-------------|",
        ]
        for m in cls.list_all():
            src = f"[source]({m.source_url})" if m.source_url else m.source
            lines.append(f"| `{m.category}` | {m.name} | {src} | {m.description} |")
        return "\n".join(lines)

    @classmethod
    def _load_all(cls) -> None:
        """
        Import all strategy subpackages to trigger @register decorators.
        Called once at startup when the full registry is needed.
        """
        import importlib
        for pkg in ("stats", "retail", "academic", "econophysics"):
            try:
                importlib.import_module(f"src.strategies.{pkg}")
            except ModuleNotFoundError:
                pass
