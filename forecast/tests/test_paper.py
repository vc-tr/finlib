"""Tests for paper trading engine."""

import numpy as np
import pandas as pd

from src.paper import PaperExchange, PaperBroker, RiskManager
from src.paper.orders import Order, OrderSide, OrderType


def _synthetic_bars(symbol: str, n: int = 10, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV bars."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    ret = rng.randn(n) * 0.01
    close = start_price * (1 + pd.Series(ret, index=idx)).cumprod()
    return pd.DataFrame({
        "open": close.shift(1).fillna(start_price),
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": 1_000_000,
    }, index=idx)


def test_replay_buy_then_sell_pnl_matches() -> None:
    """Simple replay: 1 symbol, buy then sell, PnL matches expectation."""
    bars = {"S0": _synthetic_bars("S0", n=5, start_price=100.0)}
    exchange = PaperExchange(bars, cost_model="fixed", fee_bps=0, slippage_bps=0)
    broker = PaperBroker(exchange, initial_cash=100_000.0)

    ts_list = list(bars["S0"].index)
    # Bar 0: buy 100 shares at close
    t0 = ts_list[0].to_pydatetime()
    prices = {"S0": bars["S0"].loc[ts_list[0], "close"]}
    broker.submit_order("S0", OrderSide.BUY, 100.0, prices=prices)
    fills = exchange.replay_bar(t0)
    broker.process_fills(fills)
    broker.record_equity(t0, prices)

    buy_price = bars["S0"].loc[ts_list[0], "close"]
    assert broker.positions.get("S0", 0) == 100.0
    assert broker.cash == 100_000.0 - 100.0 * buy_price

    # Bar 1: sell 100 shares at close
    t1 = ts_list[1].to_pydatetime()
    sell_price = bars["S0"].loc[ts_list[1], "close"]
    prices = {"S0": sell_price}
    broker.submit_order("S0", OrderSide.SELL, 100.0, prices=prices)
    fills = exchange.replay_bar(t1)
    broker.process_fills(fills)
    broker.record_equity(t1, prices)

    assert broker.positions.get("S0", 0) == 0.0
    pnl = 100.0 * (sell_price - buy_price)
    expected_cash = 100_000.0 - 100.0 * buy_price + 100.0 * sell_price
    assert abs(broker.cash - expected_cash) < 1e-6
    assert abs(broker.cash - (100_000.0 + pnl)) < 1e-6


def test_limit_order_fill_when_touched() -> None:
    """Limit order fills when price touches limit (low<=limit for buy, high>=limit for sell)."""
    # Bar: open=100, high=105, low=95, close=102
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    bars = {
        "S0": pd.DataFrame(
            {"open": [100, 100, 100], "high": [105, 105, 105], "low": [95, 95, 95], "close": [102, 102, 102], "volume": [1e6, 1e6, 1e6]},
            index=idx,
        )
    }
    exchange = PaperExchange(bars, cost_model="fixed", fee_bps=0, slippage_bps=0)
    broker = PaperBroker(exchange, initial_cash=100_000.0)

    # Buy limit at 100: bar has low=95 <= 100, so fill at bar 1
    order = Order(symbol="S0", side=OrderSide.BUY, quantity=10, order_type=OrderType.LIMIT, limit_price=100.0)
    broker.place_order(order, submit_ts=idx[0].to_pydatetime())
    fills = exchange.replay_bar(idx[1].to_pydatetime())
    assert len(fills) == 1
    assert fills[0].quantity == 10
    assert fills[0].price <= 100.0  # Fill at or better than limit
    broker.process_fills(fills)

    # Sell limit at 105: bar has high=105 >= 105, so fill at bar 2
    order2 = Order(symbol="S0", side=OrderSide.SELL, quantity=5, order_type=OrderType.LIMIT, limit_price=105.0)
    broker.place_order(order2, submit_ts=idx[1].to_pydatetime())
    fills2 = exchange.replay_bar(idx[2].to_pydatetime())
    assert len(fills2) == 1
    assert fills2[0].quantity == 5
    assert fills2[0].price >= 105.0


def test_risk_manager_rejects_oversized_order() -> None:
    """Risk manager rejects order that would exceed max_position_weight."""
    bars = {"S0": _synthetic_bars("S0", n=5, start_price=100.0)}
    exchange = PaperExchange(bars, cost_model="fixed", fee_bps=0, slippage_bps=0)
    risk = RiskManager(max_position_weight=0.1)  # Max 10% per symbol
    broker = PaperBroker(exchange, initial_cash=100_000.0, risk_manager=risk)

    # 100 shares at 100 = 10k = 10% of 100k. At limit. Try 110 shares = 11% -> reject
    prices = {"S0": 100.0}
    order = Order(symbol="S0", side=OrderSide.BUY, quantity=110, order_type=OrderType.MARKET)
    result = broker.place_order(order, prices=prices)
    assert result is not None
    assert result.status.value == "rejected"
