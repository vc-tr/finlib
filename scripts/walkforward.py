#!/usr/bin/env python3
"""
DEPRECATED: Use walkforward_demo.py instead.

This script used the legacy run_walkforward_legacy API.
The recommended replacement provides more features and uses the newer API:

    python scripts/walkforward_demo.py --symbol SPY --interval 1d --folds 6 --train-days 90 --test-days 30

For factor-based walk-forward, use:

    python scripts/backtest_factors.py --universe liquid_etfs --factor momentum_12_1 --rebalance M --walkforward
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    print(
        "[DEPRECATED] scripts/walkforward.py is deprecated. Use walkforward_demo.py instead:\n"
        "  python scripts/walkforward_demo.py --symbol SPY --interval 1d --folds 6 --train-days 90 --test-days 30\n"
        "For factor walk-forward: python scripts/backtest_factors.py --factor momentum_12_1 --walkforward"
    )
    sys.exit(1)
