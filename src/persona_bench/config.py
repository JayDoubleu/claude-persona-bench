from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from persona_bench.models import Condition, ThinkingMode


@dataclass
class ExperimentConfig:
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.2
    conditions: list[Condition] = field(
        default_factory=lambda: list(Condition),
    )
    thinking_modes: list[ThinkingMode] = field(
        default_factory=lambda: list(ThinkingMode),
    )
    runs_per_combination: int = 3
    max_problems: int | None = None
    concurrency: int = 5
    thinking_budget_tokens: int = 10_000
    results_dir: Path = field(default_factory=lambda: Path("results/experiments"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))

    @property
    def experiment_id(self) -> str:
        cond_names = "+".join(sorted(c.value for c in self.conditions))
        think_names = "+".join(sorted(t.value for t in self.thinking_modes))
        parts = [
            self.model.replace("-", ""),
            cond_names,
            think_names,
            f"{self.runs_per_combination}runs",
        ]
        if self.max_problems:
            parts.append(f"{self.max_problems}prob")
        return "_".join(parts)

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / self.experiment_id / "runs"
