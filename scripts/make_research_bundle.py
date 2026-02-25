#!/usr/bin/env python3
"""
Research bundle: one command to generate a recruiter-friendly packet.

Runs: daily demo, walk-forward, intraday demo, sweep. Writes INDEX.md and updates output/latest.

Usage:
    python scripts/make_research_bundle.py --symbol SPY
    python scripts/make_research_bundle.py --symbol SPY --no-lock
"""

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_demo(output_dir: Path, symbol: str, period: str, interval: str, **overrides) -> None:
    """Invoke run_demo logic with output_dir. Uses --no-mirror and --no-lock."""
    argv = [
        "run_demo",
        "--symbol", symbol,
        "--period", period,
        "--interval", interval,
        "--output-dir", str(output_dir),
        "--no-mirror",
        "--no-lock",
    ]
    for k, v in overrides.items():
        if v is not None:
            argv.extend([f"--{k.replace('_', '-')}", str(v)])
    orig_argv = sys.argv
    sys.argv = argv
    try:
        spec = importlib.util.spec_from_file_location(
            "run_demo", Path(__file__).parent / "run_demo.py"
        )
        run_demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_demo)
        run_demo.main()
    finally:
        sys.argv = orig_argv


def _run_walkforward(output_dir: Path, symbol: str, period: str, interval: str, folds: int, train_days: int, test_days: int) -> None:
    """Invoke walkforward_demo logic with output_dir."""
    argv = [
        "walkforward_demo",
        "--symbol", symbol,
        "--period", period,
        "--interval", interval,
        "--folds", str(folds),
        "--train-days", str(train_days),
        "--test-days", str(test_days),
        "--output-dir", str(output_dir),
        "--no-lock",
    ]
    orig_argv = sys.argv
    sys.argv = argv
    try:
        spec = importlib.util.spec_from_file_location(
            "walkforward_demo", Path(__file__).parent / "walkforward_demo.py"
        )
        wf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wf)
        wf.main()
    finally:
        sys.argv = orig_argv


def _run_sweep(output_dir: Path, symbol: str, period: str, interval: str, lookbacks: str, min_holds: str, decision_intervals: str) -> None:
    """Invoke sweep_momentum logic with output_dir."""
    argv = [
        "sweep_momentum",
        "--symbol", symbol,
        "--period", period,
        "--interval", interval,
        "--lookbacks", lookbacks,
        "--min_holds", min_holds,
        "--decision_intervals", decision_intervals,
        "--output-dir", str(output_dir),
        "--no-lock",
    ]
    orig_argv = sys.argv
    sys.argv = argv
    try:
        spec = importlib.util.spec_from_file_location(
            "sweep_momentum", Path(__file__).parent / "sweep_momentum.py"
        )
        sweep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweep)
        sweep.main()
    finally:
        sys.argv = orig_argv


def _read_summary(path: Path) -> dict:
    """Read JSON summary or return empty dict."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_index(run_dir: Path, configs: dict, summaries: dict) -> None:
    """Write INDEX.md with configs, headline metrics, and links."""
    lines = [
        "# Research Bundle",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## What Was Run",
        "",
        "| Step | Config |",
        "|------|--------|",
    ]
    for step, cfg in configs.items():
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
        lines.append(f"| {step} | {cfg_str} |")

    lines.extend([
        "",
        "## Headline Metrics",
        "",
    ])

    # Daily demo
    daily = summaries.get("daily_demo", {})
    if daily:
        sharpe = daily.get("sharpe", "N/A")
        ret = daily.get("total_return", "N/A")
        if isinstance(sharpe, (int, float)):
            sharpe = f"{sharpe:.2f}"
        if isinstance(ret, (int, float)):
            ret = f"{ret:.2%}"
        lines.append(f"- **Daily demo:** Sharpe={sharpe}, Return={ret}")
        lines.append("")

    # Walkforward
    wf = summaries.get("walkforward", {})
    agg = wf.get("aggregated", {}) if isinstance(wf.get("aggregated"), dict) else {}
    if agg:
        lines.append(f"- **Walk-forward:** Mean Sharpe={agg.get('mean_sharpe', 0):.2f}, Agg Return={agg.get('agg_total_return', 0):.2%}, Folds={agg.get('n_folds', 0)}")
        lines.append("")

    # Intraday
    intraday = summaries.get("intraday_demo", {})
    if intraday:
        sharpe = intraday.get("sharpe", "N/A")
        ret = intraday.get("total_return", "N/A")
        if isinstance(sharpe, (int, float)):
            sharpe = f"{sharpe:.2f}"
        if isinstance(ret, (int, float)):
            ret = f"{ret:.2%}"
        lines.append(f"- **Intraday demo:** Sharpe={sharpe}, Return={ret}")
        lines.append("")

    # Sweep
    sweep_csv = list((run_dir / "sweep").glob("*.csv")) if (run_dir / "sweep").exists() else []
    if sweep_csv:
        lines.append(f"- **Sweep:** {len(sweep_csv)} CSV(s)")
        lines.append("")

    lines.extend([
        "## Links",
        "",
        "- [daily_demo/REPORT.md](daily_demo/REPORT.md)",
        "- [daily_demo/tearsheet.html](daily_demo/tearsheet.html)",
        "- [walkforward/WALKFORWARD_REPORT.md](walkforward/WALKFORWARD_REPORT.md)",
        "- [intraday_demo/REPORT.md](intraday_demo/REPORT.md)",
        "- [intraday_demo/tearsheet.html](intraday_demo/tearsheet.html)",
        "- [sweep/momentum_results.csv](sweep/momentum_results.csv)",
        "",
    ])

    (run_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate recruiter-friendly research bundle")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--output-dir", help="Run directory (default: output/runs/<timestamp>_bundle_<symbol>/)")
    parser.add_argument("--no-lock", action="store_true", help="Disable global run lock")
    parser.add_argument("--lock-timeout", type=float, default=0)
    parser.add_argument("--quick", action="store_true", help="Quick run: 30d 1d for daily/walkforward (no network for 1m if cached)")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = Path("output") / "runs" / f"{ts}_bundle_{args.symbol}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    configs = {}
    summaries = {}

    # A) Daily demo
    daily_dir = run_dir / "daily_demo"
    daily_dir.mkdir(exist_ok=True)
    period_daily = "1y" if args.quick else "2y"
    print("[A] Daily demo...")
    _run_demo(daily_dir, args.symbol, period_daily, "1d")
    configs["daily_demo"] = {"symbol": args.symbol, "period": period_daily, "interval": "1d"}
    summaries["daily_demo"] = _read_summary(daily_dir / "summary.json")

    # B) Walk-forward
    wf_dir = run_dir / "walkforward"
    wf_dir.mkdir(exist_ok=True)
    folds, train_days, test_days = (3, 60, 20) if args.quick else (4, 90, 30)
    print("[B] Walk-forward...")
    _run_walkforward(wf_dir, args.symbol, period_daily, "1d", folds=folds, train_days=train_days, test_days=test_days)
    configs["walkforward"] = {"folds": folds, "train_days": train_days, "test_days": test_days, "interval": "1d"}
    summaries["walkforward"] = _read_summary(wf_dir / "walkforward_summary.json")

    # C) Intraday demo (1m, 7d, safe defaults)
    intraday_dir = run_dir / "intraday_demo"
    intraday_dir.mkdir(exist_ok=True)
    print("[C] Intraday demo...")
    _run_demo(
        intraday_dir,
        args.symbol,
        "7d",
        "1m",
        lookback=50,
        min_hold_bars=30,
        decision_interval_bars=30,
    )
    configs["intraday_demo"] = {"symbol": args.symbol, "period": "7d", "interval": "1m", "lookback": 50, "min_hold": 30, "decision_interval": 30}
    summaries["intraday_demo"] = _read_summary(intraday_dir / "summary.json")

    # D) Sweep (tiny grid)
    sweep_dir = run_dir / "sweep"
    sweep_dir.mkdir(exist_ok=True)
    print("[D] Sweep...")
    _run_sweep(
        sweep_dir,
        args.symbol,
        "7d",
        "1m",
        lookbacks="20,50",
        min_holds="15,30",
        decision_intervals="15,30",
    )

    # E) INDEX.md
    print("[E] Writing INDEX.md...")
    _write_index(run_dir, configs, summaries)

    # F) Update output/latest
    latest_dir = Path("output") / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(str(run_dir), str(latest_dir))

    print(f"\nOutput directory: {run_dir}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--no-lock", action="store_true")
    _parser.add_argument("--lock-timeout", type=float, default=0)
    _pre, _ = _parser.parse_known_args()
    if _pre.no_lock:
        main()
    else:
        from src.utils.runlock import RunLock
        _lock_path = Path(__file__).resolve().parent.parent / ".runlock"
        with RunLock(lock_path=str(_lock_path), timeout_s=_pre.lock_timeout):
            main()
