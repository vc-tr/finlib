#!/usr/bin/env python3
"""
Cross-sectional factor backtest.

Usage:
    python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1 --rebalance M --top-k 10 --bottom-k 10
    python scripts/backtest_factors.py --factor lowvol_20d --period 5y --walkforward
    python scripts/backtest_factors.py --factor combo --combo "momentum_12_1,reversal_5d,lowvol_20d" --combo-method equal --walkforward
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.factors.runner import main as run_factor_backtest_main
from src.utils.cli import build_backtest_factors_parser
from src.utils.runlock import RunLock


def main() -> None:
    parser = build_backtest_factors_parser()
    args = parser.parse_args()

    cmd = "python scripts/backtest_factors.py " + " ".join(sys.argv[1:])

    def _run() -> None:
        output_dir = run_factor_backtest_main(args, cmd=cmd)
        print(f"Output: {output_dir}")

    if args.no_lock:
        _run()
    else:
        with RunLock(lock_path=str(Path(__file__).resolve().parent.parent / ".runlock"), timeout_s=args.lock_timeout):
            _run()


if __name__ == "__main__":
    main()
