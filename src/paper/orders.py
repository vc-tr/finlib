"""
Order types and status for paper trading.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Order for paper trading.

    Attributes:
        order_id: Unique identifier (assigned by broker if None)
        symbol: Ticker symbol
        side: BUY or SELL
        quantity: Number of shares (positive)
        order_type: MARKET or LIMIT
        limit_price: Required for LIMIT orders
        status: Current status
        created_at: When order was created
        submitted_at: When submitted to exchange
        filled_at: When (fully) filled
        filled_price: Average fill price
        filled_quantity: Quantity filled
        fees: Fees paid (for logging)
        slippage: Slippage cost (for logging)
        impact: Market impact cost (for logging)
    """

    symbol: str
    side: OrderSide
    quantity: float
    order_id: str = ""
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    impact: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price required for LIMIT orders")
