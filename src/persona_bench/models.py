from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Condition(str, Enum):
    BASELINE = "baseline"
    PROFESSIONAL = "professional"
    ABSURD = "absurd"
    MICKEY = "mickey"


class ThinkingMode(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass(frozen=True)
class Problem:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test: str


@dataclass(frozen=True)
class RunKey:
    task_id: str
    condition: Condition
    thinking: ThinkingMode
    run_n: int

    @property
    def filename(self) -> str:
        safe_task = self.task_id.replace("/", "_")
        return f"{safe_task}__{self.condition.value}__{self.thinking.value}__{self.run_n}.json"

    @classmethod
    def from_filename(cls, name: str) -> RunKey:
        stem = name.removesuffix(".json")
        parts = stem.rsplit("__", maxsplit=3)
        task_id = parts[0].replace("_", "/", 1)
        return cls(
            task_id=task_id,
            condition=Condition(parts[1]),
            thinking=ThinkingMode(parts[2]),
            run_n=int(parts[3]),
        )


@dataclass
class RunResult:
    key: RunKey
    completion: str
    raw_response: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    passed: bool | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.key.task_id,
            "condition": self.key.condition.value,
            "thinking": self.key.thinking.value,
            "run_n": self.key.run_n,
            "completion": self.completion,
            "raw_response": self.raw_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "passed": self.passed,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        key = RunKey(
            task_id=data["task_id"],
            condition=Condition(data["condition"]),
            thinking=ThinkingMode(data["thinking"]),
            run_n=data["run_n"],
        )
        return cls(
            key=key,
            completion=data["completion"],
            raw_response=data["raw_response"],
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            thinking_tokens=data.get("thinking_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            duration_ms=data.get("duration_ms", 0),
            passed=data.get("passed"),
            error=data.get("error"),
        )
