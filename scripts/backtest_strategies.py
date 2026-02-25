#!/usr/bin/env python3
"""DEPRECATED. REMOVE AFTER 2025-06-01. Use run_demo for single-symbol strategy backtest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
print("[DEPRECATED] backtest_strategies.py removed. Use: python scripts/run_demo.py --symbol SPY --period 2y")
sys.exit(1)
