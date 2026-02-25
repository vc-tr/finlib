#!/usr/bin/env python3
"""
Monitor last N runs: summary + alerts.

Scans output/runs for run directories (_daily_, _replay_, _factors_) and produces:
- output/monitor/summary.csv
- output/monitor/monitor_report.md

Alerts: missing files (type-specific), turnover > threshold (daily only by default),
beta abs > threshold, cost spike.

Filters: --since-hours / --since-days (timestamp prefix), --only-type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ops import run_monitor
from src.utils.cli import build_monitor_parser


def main() -> None:
    parser = build_monitor_parser()
    args = parser.parse_args()

    if args.no_since:
        since_hours = None
        since_days = None
    elif args.since_days is not None:
        since_hours = None
        since_days = args.since_days
    else:
        since_hours = args.since_hours
        since_days = None

    root = Path(__file__).resolve().parent.parent
    runs_dir = Path(args.runs_dir) if args.runs_dir else root / "output" / "runs"
    output_dir = Path(args.output_dir) if args.output_dir else root / "output" / "monitor"

    result = run_monitor(
        runs_dir,
        output_dir,
        n=args.n,
        turnover_threshold=args.turnover_threshold,
        beta_threshold=args.beta_threshold,
        cost_spike_factor=args.cost_spike_factor,
        only_type=args.only_type,
        since_hours=since_hours,
        since_days=since_days,
        turnover_applies_to=args.turnover_applies_to,
        ignore_initial_deploy=args.ignore_initial_deploy,
    )

    print(f"Scanned {result['n_runs']} runs. Alerts: {len(result['alerts'])}")
    for a in result["alerts"]:
        print(f"  - {a['type']}: {a['detail']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
