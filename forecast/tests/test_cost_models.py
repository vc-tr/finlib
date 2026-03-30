"""Tests for transaction cost models."""

import numpy as np
import pandas as pd

from src.backtest.cost_models import (
    FixedBpsCostModel,
    LiquidityAwareCostModel,
    build_trades_from_weights,
    apply_costs_from_trades,
    compute_capacity_report,
    _compute_adv,
)


def _synthetic_ohlcv(n_days: int = 50, adv_target: float = 1e7, seed: int = 42) -> dict:
    """Create synthetic OHLCV with known ADV. ADV = mean(volume * close)."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    close = 100 * (1 + rng.randn(n_days) * 0.01).cumprod()
    vol_per_day = adv_target / close
    vol = (vol_per_day * (1 + rng.randn(n_days) * 0.1)).clip(1)
    df = pd.DataFrame({
        "open": close * (1 - 0.005),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": vol,
    }, index=idx)
    return {"A": df}


def test_synthetic_adv_known() -> None:
    """Synthetic OHLCV produces ADV close to target."""
    adv_target = 5e6
    ohlcv = _synthetic_ohlcv(n_days=100, adv_target=adv_target)
    adv = _compute_adv(ohlcv, window=20)
    assert "A" in adv.columns
    adv_mean = adv["A"].dropna().mean()
    assert 0.5 * adv_target < adv_mean < 2 * adv_target


def test_fixed_model_matches_old_behavior() -> None:
    """FixedBpsCostModel matches old apply_rebalance_costs behavior."""
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    weights = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    weights.loc[idx[:25], "A"] = 0.5
    weights.loc[idx[25:], "B"] = 0.5
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    trades = build_trades_from_weights(weights, prices, portfolio_value=1.0)
    assert len(trades) >= 1

    config = {"fee_bps": 1.0, "slippage_bps": 2.0, "spread_bps": 1.0}
    model = FixedBpsCostModel()
    out = model.estimate_costs(trades, {}, config)
    total_bps = 4.0
    for _, row in out.iterrows():
        expected_cost = (total_bps / 10_000) * row["trade_weight"]
        assert abs(row["total_cost"] - expected_cost) < 1e-10
        assert row["impact_cost"] == 0.0


def test_impact_increases_with_trade_notional() -> None:
    """LiquidityAwareCostModel: impact increases when trade_notional increases."""
    adv_target = 1e7
    ohlcv = _synthetic_ohlcv(n_days=100, adv_target=adv_target)
    config = {"impact_k": 10.0, "impact_alpha": 0.5, "max_impact_bps": 100.0, "adv_window": 20}

    trades_small = pd.DataFrame([
        {"timestamp": pd.Timestamp("2020-02-15"), "symbol": "A", "trade_weight": 0.01,
         "side": "buy", "fill_price": 100.0, "trade_notional": 1e4},
    ])
    trades_large = pd.DataFrame([
        {"timestamp": pd.Timestamp("2020-02-15"), "symbol": "A", "trade_weight": 0.10,
         "side": "buy", "fill_price": 100.0, "trade_notional": 1e6},
    ])
    model = LiquidityAwareCostModel()
    out_small = model.estimate_costs(trades_small, ohlcv, config)
    out_large = model.estimate_costs(trades_large, ohlcv, config)
    assert out_large["impact_cost"].iloc[0] > out_small["impact_cost"].iloc[0]


def test_apply_costs_from_trades() -> None:
    """apply_costs_from_trades subtracts costs from returns."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    port_ret = pd.Series(0.001, index=idx)
    trades = pd.DataFrame([
        {"timestamp": idx[0], "total_cost": 0.0005},
        {"timestamp": idx[1], "total_cost": 0.0003},
    ])
    out = apply_costs_from_trades(port_ret, trades)
    assert out.iloc[0] < port_ret.iloc[0]
    assert out.iloc[1] < port_ret.iloc[1]


def test_compute_capacity_report() -> None:
    """Capacity report has expected structure."""
    ohlcv = _synthetic_ohlcv(n_days=100, adv_target=1e7)
    trades = pd.DataFrame([
        {"timestamp": pd.Timestamp("2020-02-15"), "symbol": "A", "trade_weight": 0.05,
         "side": "buy", "fill_price": 100.0, "trade_notional": 5e4},
    ])
    config = {"impact_k": 10.0, "impact_alpha": 0.5}
    report = compute_capacity_report(trades, ohlcv, config, adv_window=20, target_impact_bps=10.0)
    assert "per_symbol_adv" in report
    assert "avg_trade_notional" in report
    assert "impact_bps" in report
    assert "capacity_notional_at_target_bps" in report
    assert "A" in report["per_symbol_adv"]
    assert report["impact_bps"]["median"] >= 0
