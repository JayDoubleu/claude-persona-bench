from persona_bench.models import Condition, RunKey, RunResult, ThinkingMode


def test_run_key_filename():
    key = RunKey(
        task_id="HumanEval/0",
        condition=Condition.BASELINE,
        thinking=ThinkingMode.DISABLED,
        run_n=1,
    )
    assert key.filename == "task=HumanEval%2F0__baseline__disabled__1.json"


def test_run_key_roundtrip():
    key = RunKey(
        task_id="HumanEval/42",
        condition=Condition.PROFESSIONAL,
        thinking=ThinkingMode.ENABLED,
        run_n=3,
    )
    restored = RunKey.from_filename(key.filename)
    assert restored == key


def test_run_key_roundtrip_with_underscores():
    key = RunKey(
        task_id="Foo_Bar/7",
        condition=Condition.BASELINE,
        thinking=ThinkingMode.DISABLED,
        run_n=1,
    )
    restored = RunKey.from_filename(key.filename)
    assert restored == key


def test_run_result_dict_roundtrip():
    key = RunKey(
        task_id="HumanEval/5",
        condition=Condition.ABSURD,
        thinking=ThinkingMode.DISABLED,
        run_n=2,
    )
    result = RunResult(
        key=key,
        completion="    return 42",
        raw_response="    return 42",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        passed=True,
        generation_error="Response truncated (hit max_tokens limit)",
    )
    data = result.to_dict()
    restored = RunResult.from_dict(data)
    assert restored.key == key
    assert restored.completion == "    return 42"
    assert restored.passed is True
    assert restored.cost_usd == 0.001
    assert restored.generation_error == "Response truncated (hit max_tokens limit)"
