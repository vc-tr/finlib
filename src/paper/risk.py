"""
Risk manager: position limits (max gross, max net, max single-name weight).
"""

from typing import Dict, Optional

from .orders import Order, OrderSide


class RiskManager:
    """
    Checks orders against risk limits before submission.

    - max_gross: Cap on sum(|position|) across all symbols (notional)
    - max_net: Cap on |sum(position)| (long - short) (notional)
    - max_single_weight: Max |weight| per symbol (as fraction of portfolio)
    - max_position_weight: Alias for max_single_weight
    """

    def __init__(
        self,
        max_gross: Optional[float] = None,
        max_net: Optional[float] = None,
        max_single_weight: Optional[float] = None,
        max_position_weight: Optional[float] = None,
    ) -> None:
        self.max_gross = max_gross
        self.max_net = max_net
        self.max_single_weight = max_single_weight or max_position_weight
        self.max_position_weight = self.max_single_weight

    def check_order(
        self,
        order: Order,
        positions: Dict[str, float],
        portfolio_value: float,
        prices: Optional[Dict[str, float]] = None,
    ) -> tuple[bool, str]:
        """
        Check if order would violate risk limits.

        positions: shares per symbol
        portfolio_value: total portfolio value (cash + positions)
        prices: optional {symbol: price} for weight calc (default: use shares as proxy)

        Returns:
            (ok, reason) - ok=True if order passes, reason explains rejection
        """
        if portfolio_value <= 0:
            return False, "portfolio_value must be positive"

        # Simulate new positions after fill
        new_pos = dict(positions)
        qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        new_pos[order.symbol] = new_pos.get(order.symbol, 0) + qty

        # Gross/net in notional if prices available, else in shares
        prices = prices or {}
        if prices:
            gross = sum(abs(new_pos.get(s, 0)) * prices.get(s, 0) for s in new_pos)
            net = sum(new_pos.get(s, 0) * prices.get(s, 0) for s in new_pos)
        else:
            gross = sum(abs(p) for p in new_pos.values())
            net = sum(new_pos.values())

        if self.max_gross is not None and gross > self.max_gross:
            return False, f"gross {gross:.2f} exceeds max_gross {self.max_gross}"

        if self.max_net is not None and abs(net) > self.max_net:
            return False, f"|net| {abs(net):.2f} exceeds max_net {self.max_net}"

        if self.max_single_weight is not None and portfolio_value > 0 and prices:
            for sym, pos in new_pos.items():
                pos_val = abs(pos) * prices.get(sym, 0)
                weight = pos_val / portfolio_value
                if weight > self.max_single_weight:
                    return False, f"symbol {sym} weight {weight:.2%} exceeds max_single_weight {self.max_single_weight:.2%}"

        return True, ""
