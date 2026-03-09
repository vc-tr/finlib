"""
Paper exchange: replays bars and fills orders with realistic execution.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .orders import Order, OrderSide, OrderStatus, OrderType


@dataclass
class Bar:
    """OHLCV bar at a timestamp."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Fill:
    """Order fill event."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    cost_bps: float = 0.0


def _bar_from_row(symbol: str, idx: pd.Timestamp, row: pd.Series) -> Bar:
    """Build Bar from DataFrame row."""
    return Bar(
        timestamp=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
        symbol=symbol,
        open=float(row.get("open", row.get("Open", 0))),
        high=float(row.get("high", row.get("High", 0))),
        low=float(row.get("low", row.get("Low", 0))),
        close=float(row.get("close", row.get("Close", 0))),
        volume=float(row.get("volume", row.get("Volume", 0))),
    )


class PaperExchange:
    """
    Replays historical bars and fills orders deterministically.

    - Market orders: filled at next bar open or close (configurable via fill_mode)
    - Limit orders: filled when price touches limit (low<=limit<=high)
    - Cost model: fixed (bps) or liquidity-adjusted
    - No randomness: deterministic fills
    """

    def __init__(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        cost_model: str = "fixed",
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        spread_bps: Optional[float] = None,
        fill_mode: str = "next_close",
    ) -> None:
        """
        Args:
            bars_by_symbol: {symbol: DataFrame with open, high, low, close, volume}
            cost_model: "fixed" | "liquidity" (liquidity = higher cost for low volume)
            fee_bps: Fixed fee in bps (cost_model=fixed)
            slippage_bps: Slippage in bps
            spread_bps: Spread in bps (defaults to 1.0 if None)
            fill_mode: "next_open" | "next_close" - price for market order fills
        """
        self._bars = bars_by_symbol
        self._cost_model = cost_model
        self._fee_bps = fee_bps
        self._slippage_bps = slippage_bps
        self._spread_bps = spread_bps if spread_bps is not None else 1.0
        self._fill_mode = fill_mode
        self._pending_orders: List[Order] = []
        self._fills: List[Fill] = []
        self._current_ts: Optional[datetime] = None
        self._order_submit_ts: Dict[str, datetime] = {}  # order_id -> submit timestamp

    def _all_timestamps(self) -> pd.DatetimeIndex:
        """Union of all bar timestamps, sorted."""
        all_idx = set()
        for df in self._bars.values():
            all_idx.update(df.index)
        return pd.DatetimeIndex(sorted(all_idx))

    def get_bar(self, symbol: str, timestamp: datetime) -> Optional[Bar]:
        """Get bar for symbol at timestamp (or nearest prior)."""
        if symbol not in self._bars:
            return None
        df = self._bars[symbol]
        if timestamp in df.index:
            row = df.loc[timestamp]
            return _bar_from_row(symbol, timestamp, row)
        prior = df.index[df.index <= pd.Timestamp(timestamp)]
        if len(prior) == 0:
            return None
        ts = prior[-1]
        row = df.loc[ts]
        return _bar_from_row(symbol, ts, row)

    def get_close(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get close price for symbol at timestamp."""
        bar = self.get_bar(symbol, timestamp)
        return bar.close if bar else None

    def submit_order(self, order: Order, submit_ts: Optional[datetime] = None) -> None:
        """Submit order to exchange (queued for fill at next bar)."""
        order.status = OrderStatus.SUBMITTED
        ts = submit_ts or self._current_ts
        if ts:
            self._order_submit_ts[order.order_id] = ts
        self._pending_orders.append(order)

    def replay_bar(self, timestamp: datetime) -> List[Fill]:
        """
        Process bar at timestamp: attempt to fill pending orders.
        Only fills orders submitted before this timestamp (next-bar fill semantics).
        Returns list of fills.
        """
        self._current_ts = timestamp
        new_fills: List[Fill] = []
        still_pending: List[Order] = []

        for order in self._pending_orders:
            # Only fill orders submitted before current bar
            if order.order_id in self._order_submit_ts:
                if self._order_submit_ts[order.order_id] >= timestamp:
                    still_pending.append(order)
                    continue
            bar = self.get_bar(order.symbol, timestamp)
            if bar is None:
                still_pending.append(order)
                continue

            fill = self._try_fill(order, bar)
            if fill:
                new_fills.append(fill)
                prev_qty = order.filled_quantity
                order.filled_quantity += fill.quantity
                order.filled_price = (
                    fill.price
                    if prev_qty == 0
                    else (order.filled_price * prev_qty + fill.price * fill.quantity) / order.filled_quantity
                )
                if order.filled_quantity >= order.quantity:
                    order.status = OrderStatus.FILLED
                    order.filled_at = timestamp
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
                    still_pending.append(order)
            else:
                still_pending.append(order)

        self._pending_orders = still_pending
        self._fills.extend(new_fills)
        return new_fills

    def _try_fill(self, order: Order, bar: Bar) -> Optional[Fill]:
        """Attempt to fill order against bar. Returns Fill or None."""
        if order.order_type == OrderType.MARKET:
            price = bar.open if self._fill_mode == "next_open" else bar.close
            cost_bps = self._cost_for_trade(order.quantity, price, bar.volume)
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=price * (1 + cost_bps / 10000) if order.side == OrderSide.BUY else price * (1 - cost_bps / 10000),
                timestamp=bar.timestamp,
                cost_bps=cost_bps,
            )
        # LIMIT
        limit = order.limit_price or 0.0
        remaining = order.quantity - order.filled_quantity
        if order.side == OrderSide.BUY and bar.low <= limit:
            price = min(limit, bar.close)
            cost_bps = self._cost_for_trade(remaining, price, bar.volume)
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=remaining,
                price=price * (1 + cost_bps / 10000),
                timestamp=bar.timestamp,
                cost_bps=cost_bps,
            )
        if order.side == OrderSide.SELL and bar.high >= limit:
            price = max(limit, bar.close)
            cost_bps = self._cost_for_trade(remaining, price, bar.volume)
            return Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=remaining,
                price=price * (1 - cost_bps / 10000),
                timestamp=bar.timestamp,
                cost_bps=cost_bps,
            )
        return None

    def _cost_for_trade(self, quantity: float, price: float, volume: float) -> float:
        """Cost in bps for trade."""
        if self._cost_model == "fixed":
            return self._fee_bps + self._slippage_bps
        # liquidity: higher cost when trade size is large vs volume
        notional = quantity * price
        if volume <= 0:
            return self._fee_bps + self._slippage_bps * 2
        participation = notional / (volume * price) if price > 0 else 0
        # Scale up cost for participation > 1%
        extra = 0.0
        if participation > 0.01:
            extra = min(50, participation * 500)
        return self._fee_bps + self._slippage_bps + extra

    @property
    def fills(self) -> List[Fill]:
        return self._fills

    @property
    def pending_orders(self) -> List[Order]:
        return self._pending_orders
