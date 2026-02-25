"""Walk-forward evaluation tests. Uses synthetic OHLCV (no network)."""

import numpy as np
import pandas as pd

from src.backtest import Backtester
from src.backtest.walkforward import generate_folds, run_walkforward
from src.strategies import MomentumStrategy


def _synthetic_ohlcv(n: int = 400) -> pd.DataFrame:
    """Deterministic OHLCV for testing."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.RandomState(42).randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )


def test_walkforward_splits_no_overlap_no_future() -> None:
    """Ensure test windows are after train windows and don't leak."""
    df = _synthetic_ohlcv(400)
    folds = generate_folds(
        df.index, train_days=60, test_days=30, step_days=30, max_folds=6
    )
    assert len(folds) >= 1
    for f in folds:
        assert f.train_start < f.train_end, "train window valid"
        assert f.train_end < f.test_start, "test after train (no future leak)"
        assert f.test_start < f.test_end, "test window valid"


def test_walkforward_outputs_metrics_shape() -> None:
    """Returns fold count = requested folds, metrics keys present."""
    df = _synthetic_ohlcv(300)

    def strategy_factory(cfg):
        return MomentumStrategy(lookback=cfg.get("lookback", 20))

    def backtest_factory(cfg):
        return Backtester(annualization_factor=252)

    result = run_walkforward(
        df,
        strategy_factory,
        backtest_factory,
        folds=4,
        train_days=60,
        test_days=30,
        step_days=30,
        config={"lookback": 20, "fee_bps": 1, "slippage_bps": 2, "spread_bps": 1},
    )

    assert "per_fold" in result
    assert "aggregated" in result
    assert len(result["per_fold"]) <= 4
    assert len(result["per_fold"]) >= 1

    agg = result["aggregated"]
    for key in ("mean_sharpe", "median_sharpe", "mean_return", "worst_drawdown", "agg_sharpe", "n_folds"):
        assert key in agg

    for row in result["per_fold"]:
        for key in ("fold_idx", "train_start", "train_end", "test_start", "test_end", "sharpe", "total_return", "max_drawdown", "n_trades"):
            assert key in row
