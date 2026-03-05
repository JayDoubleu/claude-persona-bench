from pathlib import Path

from persona_bench.models import Condition, RunKey, RunResult, ThinkingMode
from persona_bench.results.store import (
    get_completed_keys,
    load_all_results,
    load_result,
    save_result,
)


def test_save_and_load(tmp_path: Path):
    key = RunKey("HumanEval/0", Condition.BASELINE, ThinkingMode.DISABLED, 1)
    result = RunResult(
        key=key,
        completion="    return 1",
        raw_response="    return 1",
        input_tokens=50,
        output_tokens=20,
    )
    save_result(tmp_path, result)

    path = tmp_path / key.filename
    assert path.exists()

    loaded = load_result(path)
    assert loaded.key == key
    assert loaded.completion == "    return 1"


def test_get_completed_keys(tmp_path: Path):
    for i in range(3):
        key = RunKey(f"HumanEval/{i}", Condition.BASELINE, ThinkingMode.DISABLED, 1)
        result = RunResult(key=key, completion="x", raw_response="x")
        save_result(tmp_path, result)

    completed = get_completed_keys(tmp_path)
    assert len(completed) == 3
    assert RunKey("HumanEval/0", Condition.BASELINE, ThinkingMode.DISABLED, 1) in completed


def test_load_all_results(tmp_path: Path):
    for i in range(2):
        key = RunKey(f"HumanEval/{i}", Condition.PROFESSIONAL, ThinkingMode.ENABLED, 1)
        result = RunResult(key=key, completion="x", raw_response="x")
        save_result(tmp_path, result)

    results = load_all_results(tmp_path)
    assert len(results) == 2


def test_load_all_empty(tmp_path: Path):
    results = load_all_results(tmp_path / "nonexistent")
    assert results == []
