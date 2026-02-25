"""
Paper broker: holds cash/positions, submits orders, tracks PnL.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .orders import Order, OrderSide, OrderStatus, OrderType
from .exchange import PaperExchange, Fill
from .risk import RiskManager


@dataclass
class BlotterRow:
    """Single blotter entry."""

    timestamp: datetime
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    cost_bps: float
    cash_impact: float
    position_after: float


class PaperBroker:
    """
    Holds cash and positions, submits orders to exchange, tracks PnL.
    """

    def __init__(
        self,
        exchange: PaperExchange,
        initial_cash: float = 1_000_000.0,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        self.exchange = exchange
        self._cash = initial_cash
        self._positions: Dict[str, float] = {}
        self._risk = risk_manager or RiskManager()
        self._order_counter = 0
        self._blotter: List[BlotterRow] = []
        self._equity_curve: List[tuple] = []  # (timestamp, equity)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Dict[str, float]:
        return dict(self._positions)

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ord_{self._order_counter}"

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        """Mark-to-market portfolio value."""
        pos_val = sum(self._positions.get(s, 0) * prices.get(s, 0) for s in self._positions)
        return self._cash + pos_val

    def place_order(
        self,
        order: Order,
        submit_ts: Optional[datetime] = None,
        prices: Optional[Dict[str, float]] = None,
    ) -> Optional[Order]:
        """
        Place order. Assigns order_id if missing, checks risk, sends to exchange.
        Returns Order if accepted, None if rejected.
        """
        if not order.order_id:
            order.order_id = self._next_order_id()
        order.created_at = order.created_at or datetime.now()

        prices = prices or {}
        pv = self.portfolio_value(prices) if prices else self._cash

        ok, reason = self._risk.check_order(order, self._positions, pv, prices)
        if not ok:
            order.status = OrderStatus.REJECTED
            return order

        self.exchange.submit_order(order, submit_ts=submit_ts)
        return order

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        prices: Optional[Dict[str, float]] = None,
        submit_ts: Optional[datetime] = None,
    ) -> Optional[Order]:
        """
        Submit order. Checks risk, then sends to exchange.
        Returns Order if accepted, None if rejected.
        prices: Current prices for risk check (required for accurate portfolio value).
        """
        order_id = self._next_order_id()
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_id=order_id,
            order_type=order_type,
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
        )
        return self.place_order(order, submit_ts=submit_ts, prices=prices)

    def process_fills(self, fills: List[Fill]) -> None:
        """Process fills from exchange: update cash, positions, blotter."""
        for f in fills:
            cash_impact = -f.quantity * f.price if f.side == OrderSide.BUY else f.quantity * f.price
            self._cash += cash_impact
            prev_pos = self._positions.get(f.symbol, 0)
            delta = f.quantity if f.side == OrderSide.BUY else -f.quantity
            self._positions[f.symbol] = prev_pos + delta
            if self._positions[f.symbol] == 0:
                del self._positions[f.symbol]

            self._blotter.append(
                BlotterRow(
                    timestamp=f.timestamp,
                    order_id=f.order_id,
                    symbol=f.symbol,
                    side=f.side.value,
                    quantity=f.quantity,
                    price=f.price,
                    cost_bps=f.cost_bps,
                    cash_impact=cash_impact,
                    position_after=self._positions.get(f.symbol, 0),
                )
            )

    def record_equity(self, timestamp: datetime, prices: Dict[str, float]) -> None:
        """Record equity at timestamp for curve."""
        pv = self.portfolio_value(prices)
        self._equity_curve.append((timestamp, pv))

    def blotter_df(self) -> pd.DataFrame:
        """Blotter as DataFrame."""
        if not self._blotter:
            return pd.DataFrame(
                columns=["timestamp", "order_id", "symbol", "side", "quantity", "price", "cost_bps", "cash_impact", "position_after"]
            )
        return pd.DataFrame(
            [
                (
                    r.timestamp,
                    r.order_id,
                    r.symbol,
                    r.side,
                    r.quantity,
                    r.price,
                    r.cost_bps,
                    r.cash_impact,
                    r.position_after,
                )
                for r in self._blotter
            ],
            columns=["timestamp", "order_id", "symbol", "side", "quantity", "price", "cost_bps", "cash_impact", "position_after"],
        )

    def equity_curve_df(self) -> pd.DataFrame:
        """Equity curve as DataFrame."""
        if not self._equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])
        return pd.DataFrame(self._equity_curve, columns=["timestamp", "equity"])
