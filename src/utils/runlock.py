"""
Cross-platform process lock to prevent concurrent Python runs.

Usage:
    with RunLock():
        main()

    with RunLock(lock_path=".runlock", timeout_s=10):
        main()
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional


class RunLock:
    """Context manager: acquire file lock on enter, release on exit."""

    def __init__(
        self,
        lock_path: str = ".runlock",
        timeout_s: float = 0,
        poll_s: float = 0.2,
    ) -> None:
        self.lock_path = Path(lock_path).resolve()
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._acquired = False

    def __enter__(self) -> "RunLock":
        deadline = time.monotonic() + self.timeout_s if self.timeout_s > 0 else 0
        while self.lock_path.exists():
            if self.timeout_s == 0:
                print(
                    "[RunLock] Another run is active; exiting.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if time.monotonic() >= deadline:
                print(
                    "[RunLock] Timeout waiting for lock; exiting.",
                    file=sys.stderr,
                )
                sys.exit(2)
            time.sleep(self.poll_s)
        content = f"{os.getpid()}\n{time.time():.0f}\n"
        self.lock_path.write_text(content, encoding="utf-8")
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except OSError:
                pass
        return None
