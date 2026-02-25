"""
Cross-sectional factor backtest logic.

Moved from scripts/backtest_factors.py for testability and reuse.
"""

from typing import Any

import pandas as pd

from src.backtest import Backtester
from src.factors import (
    build_portfolio,
    compute_factor,
    compute_factors,
    estimate_beta,
    forward_returns,
)
from src.factors.ensemble import combine_factors
from src.factors.factors import get_prices_wide
from src.factors.portfolio import (
    _resample_weights_to_rebalance,
    apply_beta_neutral,
    apply_constraints,
    rebalance_dates,
    weights_at_rebalance,
)
from src.backtest.cost_models import (
    FixedBpsCostModel,
    LiquidityAwareCostModel,
    build_trades_from_weights,
    apply_costs_from_trades,
    compute_capacity_report,
)
from src.backtest.walkforward import generate_folds


def run_factor_backtest(
    df_by_symbol: dict[str, pd.DataFrame],
    factor_df: pd.DataFrame,
    top_k: int,
    bottom_k: int,
    rebalance: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    annualization: float,
    beta_neutral: bool = False,
    market_symbol: str = "SPY",
    beta_window: int = 252,
    max_gross: float | None = None,
    max_net: float | None = None,
    cost_model: str = "fixed",
    impact_k: float = 10.0,
    impact_alpha: float = 0.5,
    max_impact_bps: float = 50.0,
    adv_window: int = 20,
    portfolio_value: float = 1e6,
) -> tuple:
    """
    Run factor backtest.

    Returns (order matters for robust unpacking):
        0: result (Backtester result)
        1: positions (Series)
        2: prices (DataFrame)
        3: turnover (Series)
        4: w_held (DataFrame)
        5: beta_before (Series or None)
        6: beta_after (Series or None)
        7: hedge_weight (Series or None)
        8: n_rebalance_events (int)
        9: capacity_report (dict or None)
    """
    weights = weights_at_rebalance(
        factor_df,
        rebalance=rebalance,
        top_k=top_k,
        bottom_k=bottom_k,
        method="zscore",
        long_short=True,
        gross_leverage=1.0,
        max_weight=0.1,
    )
    prices = get_prices_wide(df_by_symbol)
    beta_before = None
    beta_after = None
    hedge_weight = None

    if beta_neutral:
        if market_symbol not in df_by_symbol:
            raise ValueError(
                f"Beta-neutral requires {market_symbol} in universe; add it to --universe or fetch it."
            )
        returns = prices.pct_change()
        mr = (
            returns[market_symbol]
            if market_symbol in returns.columns
            else prices[market_symbol].pct_change()
        )
        betas = estimate_beta(returns, mr, window=beta_window)
        out = build_portfolio(
            weights,
            prices,
            rebalance=rebalance,
            execution_delay=1,
            max_gross=max_gross,
            max_net=max_net,
            beta_neutral=True,
            betas=betas,
            market_symbol=market_symbol,
        )
        port_ret, beta_before, beta_after, hedge_weight = out
        w_raw = weights.copy()
        if max_gross is not None or max_net is not None:
            w_raw = apply_constraints(
                w_raw, max_gross=max_gross, max_net=max_net, gross_leverage=1.0
            )
        w_raw, _, _ = apply_beta_neutral(w_raw, betas, market_symbol=market_symbol)
        w_held = _resample_weights_to_rebalance(w_raw, rebalance).shift(1).fillna(0)
    else:
        port_ret = build_portfolio(
            weights,
            prices,
            rebalance=rebalance,
            execution_delay=1,
            max_gross=max_gross,
            max_net=max_net,
        )
        w_held = _resample_weights_to_rebalance(weights, rebalance).shift(1).fillna(0)

    cost_config = {
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "spread_bps": spread_bps,
        "impact_k": impact_k,
        "impact_alpha": impact_alpha,
        "max_impact_bps": max_impact_bps,
        "adv_window": adv_window,
    }
    trades_df = build_trades_from_weights(w_held, prices, portfolio_value=portfolio_value)
    if cost_model == "liquidity":
        model = LiquidityAwareCostModel()
    else:
        model = FixedBpsCostModel()
    trades_df = model.estimate_costs(trades_df, df_by_symbol, cost_config)
    port_ret = apply_costs_from_trades(port_ret, trades_df)

    bt = Backtester(annualization_factor=annualization)
    result = bt.run(port_ret)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0)
    positions = w_held.sum(axis=1)
    rb_dates = rebalance_dates(factor_df.index, rebalance)
    n_rebalance = len(rb_dates)
    capacity_report = compute_capacity_report(
        trades_df,
        df_by_symbol,
        cost_config,
        adv_window=adv_window,
        target_impact_bps=10.0,
    )
    return (
        result,
        positions,
        prices,
        turnover,
        w_held,
        beta_before,
        beta_after,
        hedge_weight,
        n_rebalance,
        capacity_report,
    )


def get_factor_df(
    df_by_symbol: dict[str, pd.DataFrame],
    factor: str,
    combo_list: list[str] | None,
    combo_method: str,
    train_slice: slice | None = None,
    top_k: int = 10,
    bottom_k: int = 10,
    rebalance: str = "M",
    cost_bps: float = 4.0,
    auto_metric: str = "val_ic_ir",
    val_split: float = 0.3,
    shrinkage: float = 0.5,
) -> tuple[pd.DataFrame, dict | None, dict[str, pd.DataFrame] | None, dict]:
    """
    Get factor DataFrame for backtest. For single factor or combo.
    Returns (factor_df, combo_weights or None, zscored_dict or None, meta).
    """
    if combo_list is None:
        factor_df = compute_factor(df_by_symbol, factor)
        return factor_df, None, None, {}

    factors_dict = compute_factors(df_by_symbol, combo_list)
    prices = get_prices_wide(df_by_symbol)
    fwd_returns = prices.pct_change().shift(-1)
    fwd_returns_dict = forward_returns(prices, horizons=[1, 5, 21])

    combo_kw = dict(
        fwd_returns=fwd_returns,
        fwd_returns_dict=fwd_returns_dict,
        prices=prices,
        top_k=top_k,
        bottom_k=bottom_k,
        rebalance=rebalance,
        cost_bps=cost_bps,
    )
    if combo_method == "auto_robust":
        combo_kw["auto_metric"] = auto_metric
        combo_kw["val_split"] = val_split
        combo_kw["shrinkage"] = shrinkage

    if combo_method == "equal":
        combined, weights, zscored, meta = combine_factors(factors_dict, method="equal")
    else:
        if train_slice is None:
            idx = prices.index
            n = int(len(idx) * 0.7)
            if n < 30:
                combined, weights, zscored, meta = combine_factors(
                    factors_dict, method="equal"
                )
            else:
                ts = slice(idx[0], idx[n - 1])
                combined, weights, zscored, meta = combine_factors(
                    factors_dict,
                    method=combo_method,
                    train_slice=ts,
                    **combo_kw,
                )
        else:
            combined, weights, zscored, meta = combine_factors(
                factors_dict,
                method=combo_method,
                train_slice=train_slice,
                **combo_kw,
            )
    return combined, weights, zscored, meta


def run_factor_walkforward(
    df_by_symbol: dict[str, pd.DataFrame],
    factor: str,
    combo_list: list[str] | None,
    combo_method: str,
    top_k: int,
    bottom_k: int,
    rebalance: str,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    annualization: float,
    folds: int,
    train_days: int,
    test_days: int,
    embargo_days: int = 1,
    auto_metric: str = "val_ic_ir",
    val_split: float = 0.3,
    shrinkage: float = 0.5,
    beta_neutral: bool = False,
    market_symbol: str = "SPY",
    beta_window: int = 252,
    max_gross: float | None = None,
    max_net: float | None = None,
    cost_model: str = "fixed",
    impact_k: float = 10.0,
    impact_alpha: float = 0.5,
    max_impact_bps: float = 50.0,
    adv_window: int = 20,
    portfolio_value: float = 1e6,
) -> dict[str, Any]:
    """Run walk-forward: use history up to test_end, evaluate on test window only."""
    prices_wide = get_prices_wide(df_by_symbol)
    index = prices_wide.index
    fold_list = generate_folds(
        index,
        train_days,
        test_days,
        test_days,
        max_folds=folds,
        embargo_days=embargo_days,
    )

    per_fold = []
    all_returns = []
    combo_weights_per_fold = []
    factor_df_per_fold = []

    for fold in fold_list:
        test_start, test_end = fold.test_start, fold.test_end
        df_by_hist = {
            sym: df.loc[:test_end]
            for sym, df in df_by_symbol.items()
            if len(df.loc[:test_end]) >= 30
        }
        if len(df_by_hist) < top_k + bottom_k:
            continue

        if factor == "combo" and combo_list:
            train_slice = slice(fold.train_start, fold.train_end)
            cost_bps = fee_bps + slippage_bps + spread_bps
            factor_df, combo_weights, _, meta = get_factor_df(
                df_by_hist,
                factor,
                combo_list,
                combo_method,
                train_slice,
                top_k=top_k,
                bottom_k=bottom_k,
                rebalance=rebalance,
                cost_bps=cost_bps,
                auto_metric=auto_metric,
                val_split=val_split,
                shrinkage=shrinkage,
            )
            if combo_weights is not None:
                entry = {
                    "fold_idx": fold.fold_idx,
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "weights": combo_weights,
                }
                if meta.get("selected_method"):
                    entry["selected_method"] = meta["selected_method"]
                if meta.get("val_score") is not None:
                    entry["val_score"] = meta["val_score"]
                if meta.get("weights_before_shrink"):
                    entry["weights_before_shrink"] = meta["weights_before_shrink"]
                if meta.get("weights_final"):
                    entry["weights_final"] = meta["weights_final"]
                combo_weights_per_fold.append(entry)
        else:
            factor_df, _, _, _ = get_factor_df(
                df_by_hist, factor, None, "equal", None
            )
        out = run_factor_backtest(
            df_by_hist,
            factor_df,
            top_k,
            bottom_k,
            rebalance,
            fee_bps,
            slippage_bps,
            spread_bps,
            annualization,
            beta_neutral=beta_neutral,
            market_symbol=market_symbol,
            beta_window=beta_window,
            max_gross=max_gross,
            max_net=max_net,
            cost_model=cost_model,
            impact_k=impact_k,
            impact_alpha=impact_alpha,
            max_impact_bps=max_impact_bps,
            adv_window=adv_window,
            portfolio_value=portfolio_value,
        )
        result = out[0]
        test_ret = result.returns.loc[test_start:test_end].dropna()
        if len(test_ret) < 5:
            continue
        factor_df_per_fold.append((fold.fold_idx, factor_df))
        fold_result = Backtester(annualization_factor=annualization).run(test_ret)
        per_fold.append(
            {
                "fold_idx": fold.fold_idx,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "sharpe": fold_result.sharpe_ratio,
                "total_return": fold_result.total_return,
                "max_drawdown": fold_result.max_drawdown,
            }
        )
        all_returns.append(test_ret)

    if not all_returns:
        agg = {"mean_sharpe": 0, "median_sharpe": 0, "agg_sharpe": 0, "n_folds": 0}
        agg_ret = None
    else:
        agg_ret = pd.concat(all_returns).sort_index()
        agg_ret = agg_ret[~agg_ret.index.duplicated(keep="first")]
        bt = Backtester(annualization_factor=annualization)
        agg_result = bt.run(agg_ret)
        sharpes = [r["sharpe"] for r in per_fold]
        agg = {
            "mean_sharpe": float(sum(sharpes) / len(sharpes)),
            "median_sharpe": float(sorted(sharpes)[len(sharpes) // 2]),
            "agg_sharpe": agg_result.sharpe_ratio,
            "agg_total_return": agg_result.total_return,
            "n_folds": len(per_fold),
        }
    out = {"per_fold": per_fold, "aggregated": agg}
    if combo_weights_per_fold:
        out["combo_weights_per_fold"] = combo_weights_per_fold
    if agg_ret is not None:
        out["all_returns"] = agg_ret
    if factor_df_per_fold:
        out["factor_df_per_fold"] = factor_df_per_fold
    return out
