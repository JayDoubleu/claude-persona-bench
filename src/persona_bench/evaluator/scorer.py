from __future__ import annotations

import math
from collections import defaultdict

from persona_bench.models import Condition, RunResult, ThinkingMode


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric.

    n: total number of samples
    c: number of correct samples
    k: k in pass@k

    Uses the unbiased estimator: 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c - k + 1, n - c + 1)) / math.prod(range(n - k + 1, n + 1))


GroupKey = tuple[Condition, ThinkingMode]


def compute_pass_rates(
    results: list[RunResult],
    k_values: list[int] | None = None,
) -> dict[GroupKey, dict[str, float]]:
    """Compute pass@k rates grouped by (condition, thinking_mode).

    Returns a dict mapping (condition, thinking) to {"pass@1": rate, "pass@3": rate, ...}.
    """
    if k_values is None:
        k_values = [1]

    # Group results by (condition, thinking, task_id)
    by_task: dict[tuple[GroupKey, str], list[bool]] = defaultdict(list)
    for r in results:
        if r.passed is None:
            continue
        group = (r.key.condition, r.key.thinking)
        by_task[(group, r.key.task_id)].append(r.passed)

    # Collect all group keys
    groups: set[GroupKey] = {g for (g, _) in by_task}

    rates: dict[GroupKey, dict[str, float]] = {}
    for group in sorted(groups):
        task_results = {tid: outcomes for (g, tid), outcomes in by_task.items() if g == group}
        if not task_results:
            continue

        pass_rates_for_k: dict[str, float] = {}
        for k in k_values:
            total = 0.0
            for outcomes in task_results.values():
                n = len(outcomes)
                c = sum(outcomes)
                total += pass_at_k(n, c, k)
            pass_rates_for_k[f"pass@{k}"] = total / len(task_results)

        rates[group] = pass_rates_for_k

    return rates
