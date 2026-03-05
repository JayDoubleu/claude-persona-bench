from persona_bench.evaluator.scorer import pass_at_k, compute_pass_rates
from persona_bench.models import Condition, RunKey, RunResult, ThinkingMode


def test_pass_at_k_all_correct():
    assert pass_at_k(3, 3, 1) == 1.0


def test_pass_at_k_none_correct():
    assert pass_at_k(3, 0, 1) == 0.0


def test_pass_at_k_partial():
    # 3 samples, 1 correct, k=1
    result = pass_at_k(3, 1, 1)
    assert 0.0 < result < 1.0


def test_compute_pass_rates():
    results = []
    for i in range(3):
        key = RunKey("HumanEval/0", Condition.BASELINE, ThinkingMode.DISABLED, i + 1)
        results.append(RunResult(
            key=key,
            completion="x",
            raw_response="x",
            passed=(i < 2),  # 2 pass, 1 fail
        ))

    rates = compute_pass_rates(results, k_values=[1])
    group = (Condition.BASELINE, ThinkingMode.DISABLED)
    assert group in rates
    assert 0.0 < rates[group]["pass@1"] < 1.0
