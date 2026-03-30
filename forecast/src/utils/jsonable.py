"""
Convert numpy/pandas types to JSON-serializable Python types.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(x: Any) -> Any:
    """
    Recursively convert numpy/pandas scalars and containers to
    JSON-serializable types.

    - str, int, float, bool, None: return as-is
    - numpy scalar (np.bool_, np.float64, etc.): return x.item()
    - pandas Timestamp: return x.isoformat()
    - Path: return str(x)
    - dict: recurse on values
    - list/tuple: recurse on elements
    - fallback: return str(x)
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return str(x)
