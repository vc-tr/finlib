"""
Shared argparse builders for CLI scripts.
"""

import argparse
from pathlib import Path

from src.factors import UniverseRegistry


def build_factors_parser(description: str = "Cross-sectional factor backtest") -> argparse.ArgumentParser:
    """Build parser for factor backtest / replay / daily (shared args)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--universe", default="liquid_etfs", help=f"Universe (choices: {', '.join(UniverseRegistry.list_names())})")
    p.add_argument("--factor", default="momentum_12_1", choices=["momentum_12_1", "reversal_5d", "lowvol_20d", "combo"])
    p.add_argument("--combo", default=None, help='Comma-separated factors (e.g. "momentum_12_1,reversal_5d,lowvol_20d")')
    p.add_argument("--combo-method", default="equal", choices=["equal", "ic_weighted", "ridge", "sharpe_opt", "auto", "auto_robust"])
    p.add_argument("--rebalance", default="M", choices=["D", "W", "M"])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bottom-k", type=int, default=10)
    p.add_argument("--period", default="5y")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--spread-bps", type=float, default=1.0)
    p.add_argument("--cost-model", default="fixed", choices=["fixed", "liquidity"])
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--lock-timeout", type=float, default=0)
    return p


def build_replay_parser() -> argparse.ArgumentParser:
    """Build parser for replay_trade.py."""
    p = build_factors_parser(description="Paper trading replay engine")
    p.add_argument("--strategy", default="factors", choices=["factors"])
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--fill-mode", default="next_close", choices=["next_open", "next_close"])
    return p


def build_daily_parser() -> argparse.ArgumentParser:
    """Build parser for daily_run.py."""
    p = build_factors_parser(description="Daily pipeline: target portfolio + staged orders")
    p.add_argument("--strategy", default="factors", choices=["factors"])
    p.add_argument("--asof", default=None, help="YYYY-MM-DD (default: latest)")
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--max-gross", type=float, default=None)
    p.add_argument("--max-net", type=float, default=None)
    p.add_argument("--max-position-weight", type=float, default=0.2)
    p.add_argument("--beta-threshold", type=float, default=0.5)
    p.add_argument("--state-path", default=None)
    p.add_argument("--apply", action="store_true", help="Update current_portfolio.json")
    p.add_argument("--force-rebalance", action="store_true")
    return p


def build_backtest_factors_parser() -> argparse.ArgumentParser:
    """Build full parser for backtest_factors.py."""
    p = argparse.ArgumentParser(description="Cross-sectional factor backtest")
    p.add_argument("--universe", default="liquid_etfs", help=f"Universe (choices: {', '.join(UniverseRegistry.list_names())})")
    p.add_argument("--list-universes", action="store_true", help="List universes and exit")
    p.add_argument("--factor", default="momentum_12_1", choices=["momentum_12_1", "reversal_5d", "lowvol_20d", "combo"])
    p.add_argument("--combo", default=None)
    p.add_argument("--combo-method", default="equal", choices=["equal", "ic_weighted", "ridge", "sharpe_opt", "auto", "auto_robust"])
    p.add_argument("--auto-metric", default="val_ic_ir", choices=["val_sharpe", "val_ic_ir"])
    p.add_argument("--val-split", type=float, default=0.3)
    p.add_argument("--shrinkage", type=float, default=0.5)
    p.add_argument("--rebalance", default="M", choices=["D", "W", "M"])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bottom-k", type=int, default=10)
    p.add_argument("--period", default="5y")
    p.add_argument("--interval", default="1d", choices=["1d", "1h"])
    p.add_argument("--fee-bps", type=float, default=1.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--spread-bps", type=float, default=1.0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--walkforward", action="store_true")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--train-days", type=int, default=252)
    p.add_argument("--test-days", type=int, default=63)
    p.add_argument("--embargo-days", type=int, default=1)
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--lock-timeout", type=float, default=0)
    p.add_argument("--beta-neutral", action="store_true")
    p.add_argument("--market-symbol", default="SPY")
    p.add_argument("--beta-window", type=int, default=252)
    p.add_argument("--max-gross", type=float, default=None)
    p.add_argument("--max-net", type=float, default=None)
    p.add_argument("--cost-model", default="fixed", choices=["fixed", "liquidity"])
    p.add_argument("--impact-k", type=float, default=10.0)
    p.add_argument("--impact-alpha", type=float, default=0.5)
    p.add_argument("--max-impact-bps", type=float, default=50.0)
    p.add_argument("--adv-window", type=int, default=20)
    p.add_argument("--portfolio-value", type=float, default=1e6)
    p.add_argument("--report-ic", action="store_true")
    p.add_argument("--ic-horizons", default="1,5,21")
    p.add_argument("--ic-method", default="spearman", choices=["spearman", "pearson"])
    return p


def build_monitor_parser() -> argparse.ArgumentParser:
    """Build parser for monitor_runs.py."""
    p = argparse.ArgumentParser(description="Monitor runs (daily, replay, factors)")
    p.add_argument("--runs-dir", default=None, help="Default: output/runs")
    p.add_argument("--output-dir", default=None, help="Default: output/monitor")
    p.add_argument("-n", type=int, default=10, help="Last N runs to scan")
    p.add_argument("--only-type", choices=["daily", "replay", "factors", "all"], default="all")
    time_group = p.add_mutually_exclusive_group()
    time_group.add_argument("--since-hours", type=float, default=6)
    time_group.add_argument("--since-days", type=float, default=None)
    time_group.add_argument("--no-since", action="store_true")
    p.add_argument("--turnover-threshold", type=float, default=0.5)
    p.add_argument("--turnover-applies-to", choices=["daily", "all"], default="daily")
    p.add_argument("--ignore-initial-deploy", action="store_true", default=True)
    p.add_argument("--no-ignore-initial-deploy", action="store_false", dest="ignore_initial_deploy")
    p.add_argument("--beta-threshold", type=float, default=0.5)
    p.add_argument("--cost-spike-factor", type=float, default=2.0)
    return p
