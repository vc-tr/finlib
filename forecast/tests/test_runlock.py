"""Tests for RunLock: prevents concurrent acquisition."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.runlock import RunLock


def test_runlock_prevents_concurrent_acquisition(tmp_path: Path) -> None:
    """When lock exists and timeout_s=0, second acquisition exits with code 2."""
    lock_path = tmp_path / "test.lock"
    lock_path.write_text("12345\n1000000\n")  # simulate another process holding lock

    proj_root = Path(__file__).resolve().parent.parent
    code = f"""
import sys
sys.path.insert(0, {repr(str(proj_root))})
from src.utils.runlock import RunLock
with RunLock(lock_path={repr(str(lock_path))}, timeout_s=0):
    pass
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=proj_root,
        capture_output=True,
    )
    assert result.returncode == 2


def test_runlock_acquires_and_releases(tmp_path: Path) -> None:
    """RunLock acquires when free and releases on exit."""
    lock_path = tmp_path / "test.lock"
    with RunLock(lock_path=str(lock_path), timeout_s=0):
        assert lock_path.exists()
        content = lock_path.read_text()
        assert "\n" in content  # pid\ntimestamp
    assert not lock_path.exists()
