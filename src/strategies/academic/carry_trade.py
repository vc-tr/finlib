"""
Carry Trade — Koijen, Moskowitz, Pedersen & Vrugt (2018).

"Carry" = expected return assuming price stays flat. Across assets:
- Equity: dividend yield or earnings yield
- Rates: yield on bond
- FX: interest rate differential
- Commodities: futures roll yield

Paper: "Carry" — Koijen, Moskowitz, Pedersen & Vrugt (2018)
Journal of Financial Economics, 127(2), 197-225.

Single-asset implementation: approximate carry using dividend yield proxy
(earnings yield = E/P). Use rolling 12-month return as a carry proxy —
assets earning > risk-free rate have positive carry.

True carry requires multi-asset context (long high-carry, short low-carry).
See docs for multi-asset carry extension.

Expected: Positive carry premium well-documented. However, severe crash risk
in stress periods (carry crashes — 2008, 2020). High skewness and kurtosis.
"Picking up nickels in front of a steamroller."
"""

import pandas as pd

from src.strategies.base import Strategy, StrategyMeta
from src.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class CarryTradeStrategy(Strategy):
    """
    Equity carry proxy: long when trailing yield > risk-free threshold.

    Uses rolling total return as a proxy for carry — when an asset's
    recent return (adjusted for momentum) exceeds a risk-free hurdle,
    it has positive carry. This is a simplification; production carry
    uses dividend yield or futures basis.
    """

    def __init__(
        self,
        carry_lookback: int = 252,
        risk_free_annual: float = 0.04,
        smooth_window: int = 21,
    ):
        """
        Args:
            carry_lookback: Lookback for trailing return estimation (days)
            risk_free_annual: Annual risk-free rate hurdle (default 4%)
            smooth_window: Smoothing window for carry signal
        """
        self.carry_lookback = carry_lookback
        self.risk_free_annual = risk_free_annual
        self.smooth_window = smooth_window

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="carry_trade",
            category="academic",
            source="Koijen, Moskowitz, Pedersen & Vrugt (2018)",
            description=(
                f"Long when {self.carry_lookback}d trailing return > "
                f"{self.risk_free_annual:.0%} annual hurdle; short when below"
            ),
            hypothesis=(
                "Assets with high carry (expected return assuming no price change) "
                "earn a risk premium. Carry predicts returns across asset classes "
                "— equities, fixed income, FX, commodities."
            ),
            expected_result=(
                "Positive average carry premium. "
                "Severe crash risk: carry strategies have negative skew. "
                "Single-asset implementation mixes carry and momentum signals."
            ),
            source_url="https://doi.org/10.1016/j.jfineco.2017.11.002",
            tags=["carry", "yield", "factor", "academic", "crash-risk"],
        )

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # Trailing annual return as carry proxy
        trailing_return = prices.pct_change(self.carry_lookback)
        # Daily risk-free rate
        daily_rf = (1 + self.risk_free_annual) ** (1 / 252) - 1
        # Annualized carry signal: excess return over risk-free
        cumulative_rf = (1 + daily_rf) ** self.carry_lookback - 1
        excess_return = trailing_return - cumulative_rf

        # Smooth carry signal
        carry_signal = excess_return.rolling(self.smooth_window).mean()

        signals = pd.Series(0.0, index=prices.index)
        signals[carry_signal > 0] = 1.0
        signals[carry_signal < 0] = -1.0
        return signals

    def parameter_grid(self):
        return {
            "carry_lookback": [126, 252],
            "risk_free_annual": [0.02, 0.04, 0.05],
            "smooth_window": [5, 21, 42],
        }
