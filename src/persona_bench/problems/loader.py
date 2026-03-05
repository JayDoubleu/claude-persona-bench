from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

from persona_bench.models import Problem

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
CACHE_FILENAME = "HumanEval.jsonl.gz"


def download_humaneval(cache_dir: Path) -> Path:
    """Download HumanEval dataset if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_FILENAME
    if path.exists():
        return path
    urllib.request.urlretrieve(HUMANEVAL_URL, path)
    return path


def parse_humaneval(path: Path) -> list[Problem]:
    """Parse HumanEval JSONL.gz into Problem objects."""
    problems = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            problems.append(
                Problem(
                    task_id=data["task_id"],
                    prompt=data["prompt"],
                    entry_point=data["entry_point"],
                    canonical_solution=data["canonical_solution"],
                    test=data["test"],
                )
            )
    return problems


def load_problems(cache_dir: Path, max_problems: int | None = None) -> list[Problem]:
    """Download (if needed) and load HumanEval problems."""
    path = download_humaneval(cache_dir)
    problems = parse_humaneval(path)
    if max_problems is not None:
        problems = problems[:max_problems]
    return problems
