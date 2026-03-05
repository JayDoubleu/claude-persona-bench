from __future__ import annotations

from dataclasses import dataclass

from persona_bench.evaluator.scorer import GroupKey
from persona_bench.models import Condition, RunResult, ThinkingMode


@dataclass
class GroupStats:
    condition: Condition
    thinking: ThinkingMode
    total_runs: int = 0
    evaluated: int = 0
    passed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_thinking_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.evaluated if self.evaluated else 0.0

    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / self.total_runs if self.total_runs else 0.0

    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / self.total_runs if self.total_runs else 0.0

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.total_runs if self.total_runs else 0.0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.total_runs if self.total_runs else 0.0


def compute_group_stats(results: list[RunResult]) -> dict[GroupKey, GroupStats]:
    """Aggregate statistics by (condition, thinking_mode)."""
    stats: dict[GroupKey, GroupStats] = {}

    for r in results:
        key = (r.key.condition, r.key.thinking)
        if key not in stats:
            stats[key] = GroupStats(condition=r.key.condition, thinking=r.key.thinking)

        s = stats[key]
        s.total_runs += 1
        s.total_input_tokens += r.input_tokens
        s.total_output_tokens += r.output_tokens
        s.total_thinking_tokens += r.thinking_tokens
        s.total_cost_usd += r.cost_usd
        s.total_duration_ms += r.duration_ms

        if r.passed is not None:
            s.evaluated += 1
            if r.passed:
                s.passed += 1
            else:
                s.failed += 1

    return stats
