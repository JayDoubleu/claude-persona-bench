from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from persona_bench.config import ExperimentConfig
from persona_bench.models import Problem, RunKey, RunResult
from persona_bench.results.store import get_completed_keys, save_result
from persona_bench.runner.client import call_model

logger = logging.getLogger(__name__)


def generate_all_keys(
    problems: list[Problem],
    config: ExperimentConfig,
) -> list[tuple[Problem, RunKey]]:
    """Generate all (problem, key) pairs for the experiment."""
    pairs = []
    for problem in problems:
        for condition in config.conditions:
            for thinking in config.thinking_modes:
                for run_n in range(1, config.runs_per_combination + 1):
                    key = RunKey(
                        task_id=problem.task_id,
                        condition=condition,
                        thinking=thinking,
                        run_n=run_n,
                    )
                    pairs.append((problem, key))
    return pairs


async def run_experiment(
    problems: list[Problem],
    config: ExperimentConfig,
    on_complete: Callable[[RunResult], None] | None = None,
) -> list[RunResult]:
    """Run the full experiment with resumability."""
    runs_dir = config.runs_dir
    completed = get_completed_keys(runs_dir)
    all_pairs = generate_all_keys(problems, config)
    pending = [(p, k) for p, k in all_pairs if k not in completed]

    total = len(all_pairs)
    done = len(completed)

    logger.info(
        "Experiment: %d total, %d completed, %d pending",
        total,
        done,
        len(pending),
    )

    if not pending:
        logger.info("All runs already completed.")
        return []

    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[RunResult] = []

    async def process_one(problem: Problem, key: RunKey) -> RunResult:
        result = await call_model(problem, key, config, semaphore)
        save_result(runs_dir, result)
        if on_complete:
            on_complete(result)
        return result

    tasks = [process_one(p, k) for p, k in pending]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)

    return results
