from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import quote, unquote


class Condition(StrEnum):
    BASELINE = "baseline"
    PROFESSIONAL = "professional"
    ABSURD = "absurd"
    MICKEY = "mickey"


class ThinkingMode(StrEnum):
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
        safe_task = f"task={quote(self.task_id, safe='')}"
        return f"{safe_task}__{self.condition.value}__{self.thinking.value}__{self.run_n}.json"

    @classmethod
    def from_filename(cls, name: str) -> RunKey:
        stem = name.removesuffix(".json")
        parts = stem.rsplit("__", maxsplit=3)
        if len(parts) != 4:
            raise ValueError(f"Invalid run filename: {name}")

        task_part = parts[0]
        if not task_part.startswith("task="):
            raise ValueError(f"Invalid run filename (expected task= prefix): {name}")
        task_id = unquote(task_part.removeprefix("task="))
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
    generation_error: str | None = None
    evaluation_error: str | None = None

    @property
    def error(self) -> str | None:
        if self.generation_error and self.evaluation_error:
            return f"{self.generation_error} | {self.evaluation_error}"
        return self.generation_error or self.evaluation_error

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
            "generation_error": self.generation_error,
            "evaluation_error": self.evaluation_error,
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
            generation_error=data.get("generation_error"),
            evaluation_error=data.get("evaluation_error"),
        )
