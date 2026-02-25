"""Tests for paper trading engine."""

import numpy as np
import pandas as pd

from src.paper import PaperExchange, PaperBroker, RiskManager
from src.paper.orders import OrderSide


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
