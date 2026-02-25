#!/usr/bin/env python3
"""DEPRECATED. REMOVE AFTER 2025-06-01. Use backtest_factors or run_demo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
print("[DEPRECATED] backtest_baselines.py removed. Use: python scripts/backtest_factors.py or scripts/run_demo.py")
sys.exit(1)
