import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from click.testing import CliRunner

from persona_bench.cli import main
from persona_bench.models import Condition, Problem, RunKey, RunResult, ThinkingMode
from persona_bench.results.store import load_result, save_result


def test_run_rejects_unsupported_thinking_model(monkeypatch):
    def fail_load(*args, **kwargs):
        raise AssertionError("load_problems should not be called")

    monkeypatch.setattr("persona_bench.cli.load_problems", fail_load)

    result = CliRunner().invoke(
        main,
        ["run", "--model", "claude-3-5-haiku-latest", "--thinking", "enabled"],
    )

    assert result.exit_code != 0
    assert "does not support thinking mode" in result.output


def test_evaluate_includes_generation_errors(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "exp1" / "runs"
    result = RunResult(
        key=RunKey("HumanEval/0", Condition.BASELINE, ThinkingMode.DISABLED, 1),
        completion="",
        raw_response="",
        generation_error="Response truncated (hit max_tokens limit)",
    )
    save_result(runs_dir, result)

    monkeypatch.setattr(
        "persona_bench.cli.load_problems",
        lambda *_args, **_kwargs: [
            Problem(
                task_id="HumanEval/0",
                prompt='def add(a, b):\n    """Add two numbers."""\n',
                entry_point="add",
                canonical_solution="",
                test="def check(f):\n    assert f(1, 2) == 3\n",
            )
        ],
    )
    monkeypatch.setattr("persona_bench.cli.ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(
        "persona_bench.cli.run_sandboxed",
        lambda *_args, **_kwargs: (False, "IndentationError: expected an indented block"),
    )

    outcome = CliRunner().invoke(main, ["evaluate", "exp1", "--results-dir", str(tmp_path)])
    assert outcome.exit_code == 0

    stored = load_result(runs_dir / result.key.filename)
    assert stored.passed is False
    assert stored.generation_error == "Response truncated (hit max_tokens limit)"
    assert stored.evaluation_error == "IndentationError: expected an indented block"


def test_report_ignores_non_experiment_files(tmp_path: Path, monkeypatch):
    runs_dir = tmp_path / "exp1" / "runs"
    save_result(
        runs_dir,
        RunResult(
            key=RunKey("HumanEval/0", Condition.BASELINE, ThinkingMode.DISABLED, 1),
            completion="    return 1",
            raw_response="    return 1",
            passed=True,
        ),
    )

    captured: dict[str, int] = {}

    def fake_render(results, k_values):
        captured["results"] = len(results)
        captured["k"] = len(k_values)

    monkeypatch.setattr("persona_bench.cli.render_console_table", fake_render)

    time.sleep(1)
    (tmp_path / "notes.txt").write_text("newer than the experiment", encoding="utf-8")

    outcome = CliRunner().invoke(main, ["report", "--results-dir", str(tmp_path)])
    assert outcome.exit_code == 0
    assert captured == {"results": 1, "k": 1}
