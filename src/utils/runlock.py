"""
Simple process lock to prevent concurrent Python runs.

Usage:
    with RunLock():
        main()
"""

import sys
import time
from pathlib import Path
from typing import Optional


class RunLock:
    """Context manager: acquire file lock on enter, release on exit."""

    def __init__(self, path: str = ".runlock", timeout: float = 0) -> None:
        self.path = Path(path).resolve()
        self.timeout = timeout
        self._acquired = False

    def __enter__(self) -> "RunLock":
        deadline = time.monotonic() + self.timeout if self.timeout > 0 else 0
        while self.path.exists():
            if self.timeout <= 0:
                print("[RunLock] Another run is active; exiting.", file=sys.stderr)
                sys.exit(1)
            if time.monotonic() >= deadline:
                print("[RunLock] Timeout waiting for lock; exiting.", file=sys.stderr)
                sys.exit(1)
            time.sleep(0.5)
        self.path.write_text("")
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._acquired and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
        return None
