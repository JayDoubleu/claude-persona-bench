from __future__ import annotations

import json
import tempfile
from pathlib import Path

from persona_bench.models import RunKey, RunResult


def save_result(runs_dir: Path, result: RunResult) -> None:
    """Atomically save a single run result as JSON."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / result.key.filename

    # Atomic write: write to temp file then rename
    fd, tmp_path = tempfile.mkstemp(dir=runs_dir, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        Path(tmp_path).rename(target)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_result(path: Path) -> RunResult:
    """Load a single run result from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunResult.from_dict(data)


def load_all_results(runs_dir: Path) -> list[RunResult]:
    """Load all run results from a directory."""
    if not runs_dir.exists():
        return []
    results = []
    for path in sorted(runs_dir.glob("*.json")):
        results.append(load_result(path))
    return results


def get_completed_keys(runs_dir: Path) -> set[RunKey]:
    """Scan existing result files to determine which runs are already done."""
    if not runs_dir.exists():
        return set()
    keys = set()
    for path in runs_dir.glob("*.json"):
        try:
            keys.add(RunKey.from_filename(path.name))
        except (ValueError, IndexError):
            continue
    return keys


def update_result(runs_dir: Path, result: RunResult) -> None:
    """Update an existing result file (e.g., after evaluation)."""
    save_result(runs_dir, result)
