#!/usr/bin/env python3
"""
Repo audit: compute reachable modules, unreachable modules, duplicate symbols.
Uses AST to parse imports. Fast, no network.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ENTRYPOINTS = [
    "scripts/run_demo.py",
    "scripts/backtest_factors.py",
    "scripts/walkforward_demo.py",
    "scripts/replay_trade.py",
    "scripts/daily_run.py",
    "scripts/monitor_runs.py",
]


def load_tracked_py() -> set[str]:
    """Load git-tracked .py files."""
    tracked = REPO_ROOT / "docs" / "GIT_TRACKED_FILES.txt"
    if not tracked.exists():
        import subprocess
        out = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return set(out.stdout.strip().splitlines()) if out.returncode == 0 else set()
    return {p for p in tracked.read_text().strip().splitlines() if p.endswith(".py")}


def module_to_path(module: str) -> str | None:
    """Convert module name to file path. Returns path if file exists in repo."""
    p = module.replace(".", "/") + ".py"
    if (REPO_ROOT / p).exists():
        return p
    pkg = module.replace(".", "/") + "/__init__.py"
    if (REPO_ROOT / pkg).exists():
        return pkg
    return None


def path_to_module(path: str) -> str:
    """Convert file path to module name."""
    return path[:-3].replace("/", ".")


def extract_local_imports(filepath: Path) -> set[str]:
    """Extract imports of local src/ modules from a Python file."""
    try:
        tree = ast.parse(filepath.read_text())
    except Exception:
        return set()
    imports: set[str] = set()
    from_path = str(filepath.relative_to(REPO_ROOT)).replace("\\", "/")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and node.level == 0:
                continue
            level = node.level or 0
            if level > 0 and from_path.startswith("src/"):
                # Relative: from .foo import X or from ..bar import X
                # Path like src/backtest/__init__.py -> package src.backtest
                path_no_ext = from_path.replace("/__init__.py", "").replace(".py", "")
                parts = path_no_ext.split("/")
                if path_no_ext.endswith("__init__"):
                    parts = parts[:-1]
                base = ".".join(parts[: len(parts) - level + 1])
                imports.add(f"{base}.{node.module}" if node.module else base)
            elif node.module.startswith("src."):
                imports.add(node.module)
            elif not any(
                node.module.startswith(x)
                for x in (
                    "argparse", "json", "sys", "os", "pathlib", "datetime", "re", "time",
                    "dataclasses", "typing", "abc", "enum", "itertools", "math", "warnings",
                    "subprocess", "importlib", "shutil", "traceback", "numpy", "pandas",
                    "torch", "scipy", "statsmodels", "yfinance", "matplotlib", "mlflow",
                )
            ):
                if from_path.startswith("scripts/") or from_path.startswith("tests/"):
                    imports.add(f"src.{node.module}")

    return imports


def collect_reachable(py_files: set[str], seeds: list[str]) -> set[str]:
    """BFS from seed paths to collect all reachable module paths."""
    path_to_mod = {p: path_to_module(p) for p in py_files}
    mod_to_path = {path_to_module(p): p for p in py_files}

    reachable_paths: set[str] = set(seeds)
    queue = list(seeds)

    while queue:
        path = queue.pop(0)
        full = REPO_ROOT / path
        if not full.exists():
            continue
        for imp in extract_local_imports(full):
            target_path = module_to_path(imp)
            if target_path and target_path in py_files and target_path not in reachable_paths:
                reachable_paths.add(target_path)
                queue.append(target_path)
            elif imp in mod_to_path:
                tp = mod_to_path[imp]
                if tp not in reachable_paths:
                    reachable_paths.add(tp)
                    queue.append(tp)

    return reachable_paths


def find_duplicate_symbols(py_files: set[str]) -> list[tuple[str, list[str]]]:
    """Find same function/class name in multiple files."""
    symbols: dict[str, list[str]] = {}
    for pf in sorted(py_files):
        path = REPO_ROOT / pf
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_"):
                    continue
                symbols.setdefault(node.name, []).append(pf)
    return [(n, p) for n, p in symbols.items() if len(p) > 1]


def main() -> None:
    py_files = load_tracked_py()
    seeds = ENTRYPOINTS + [p for p in py_files if p.startswith("tests/")]
    seeds = [s for s in seeds if s in py_files]

    reachable = collect_reachable(py_files, seeds)
    unreachable = py_files - reachable

    docs = REPO_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "REACHABLE_MODULES.txt").write_text("\n".join(sorted(reachable)) + "\n")
    (docs / "UNREACHABLE_MODULES.txt").write_text("\n".join(sorted(unreachable)) + "\n")

    dups = find_duplicate_symbols(py_files)
    lines = []
    for name, paths in sorted(dups, key=lambda x: -len(x[1])):
        lines.append(f"{name}:")
        for p in paths:
            lines.append(f"  {p}")
        lines.append("")
    (docs / "DUPLICATE_SYMBOLS.txt").write_text("\n".join(lines))

    print(f"Reachable: {len(reachable)}")
    print(f"Unreachable: {len(unreachable)}")
    print(f"Duplicate symbols: {len(dups)}")


if __name__ == "__main__":
    main()
    sys.exit(0)
