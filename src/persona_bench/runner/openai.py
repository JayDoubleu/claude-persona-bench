from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import openai

from persona_bench.config import ExperimentConfig
from persona_bench.models import (
    Problem,
    RunKey,
    RunResult,
    ThinkingMode,
)
from persona_bench.personas.registry import get_system_prompt
from persona_bench.runner.client import compute_cost, parse_model
from persona_bench.runner.extract import extract_code

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_RETRY_DELAY = 10.0
RATE_LIMIT_DELAY = 60.0

_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(max_retries=0)
    return _client


async def call_openai(
    problem: Problem,
    key: RunKey,
    config: ExperimentConfig,
    semaphore: asyncio.Semaphore,
) -> RunResult:
    """Call OpenAI API with concurrency control and retry."""
    async with semaphore:
        return await _call_with_retry(problem, key, config)


async def _call_with_retry(
    problem: Problem,
    key: RunKey,
    config: ExperimentConfig,
) -> RunResult:
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            return await _single_call(problem, key, config)

        except openai.RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Attempt %d/%d for %s: rate limited, retrying in %ds",
                    attempt + 1,
                    MAX_RETRIES,
                    key.task_id,
                    int(RATE_LIMIT_DELAY),
                )
                await asyncio.sleep(RATE_LIMIT_DELAY)

        except openai.APIStatusError as e:
            last_error = e
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Attempt %d/%d for %s: server error %d, retrying",
                    attempt + 1,
                    MAX_RETRIES,
                    key.task_id,
                    e.status_code,
                )
                await asyncio.sleep(BASE_RETRY_DELAY * (attempt + 1))
            elif e.status_code < 500:
                logger.error("Client error for %s: %s", key.task_id, e)
                return RunResult(
                    key=key,
                    completion="",
                    raw_response="",
                    generation_error=f"APIStatusError {e.status_code}: {e.message}",
                )

        except openai.APIConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Attempt %d/%d for %s: connection error, retrying",
                    attempt + 1,
                    MAX_RETRIES,
                    key.task_id,
                )
                await asyncio.sleep(BASE_RETRY_DELAY * (attempt + 1))

        except ValueError as e:
            return RunResult(
                key=key,
                completion="",
                raw_response="",
                generation_error=str(e),
            )

        except Exception as e:
            last_error = e
            logger.warning(
                "Attempt %d/%d for %s: %s: %s",
                attempt + 1,
                MAX_RETRIES,
                key.task_id,
                type(e).__name__,
                e,
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BASE_RETRY_DELAY * (attempt + 1))

    return RunResult(
        key=key,
        completion="",
        raw_response="",
        generation_error=f"All {MAX_RETRIES} attempts failed: {last_error}",
    )


async def _single_call(
    problem: Problem,
    key: RunKey,
    config: ExperimentConfig,
) -> RunResult:
    _, bare_model = parse_model(config.model)
    system_prompt = get_system_prompt(key.condition)

    kwargs: dict[str, Any] = {
        "model": bare_model,
        "instructions": system_prompt,
        "input": problem.prompt,
        "max_output_tokens": 16_384,
    }

    if key.thinking == ThinkingMode.ENABLED:
        kwargs["reasoning"] = {"effort": "medium"}
        kwargs["max_output_tokens"] = 16_000
    else:
        kwargs["temperature"] = config.temperature

    start = time.monotonic()
    response = await _get_client().responses.create(**kwargs)
    duration_ms = int((time.monotonic() - start) * 1000)

    raw_response = response.output_text or ""
    completion = extract_code(raw_response) if raw_response else ""

    input_tokens = response.usage.input_tokens if response.usage else 0
    output_tokens = response.usage.output_tokens if response.usage else 0
    thinking_tokens = 0
    if response.usage and response.usage.output_tokens_details:
        thinking_tokens = response.usage.output_tokens_details.reasoning_tokens

    cost = compute_cost(bare_model, "openai", input_tokens, output_tokens, thinking_tokens)

    generation_error = None
    if response.status == "incomplete":
        reason = ""
        if response.incomplete_details:
            reason = f" ({response.incomplete_details.reason})"
        generation_error = f"Response incomplete{reason}"
    elif not raw_response and not completion:
        generation_error = f"Empty response (status={response.status})"

    return RunResult(
        key=key,
        completion=completion,
        raw_response=raw_response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        cost_usd=cost,
        duration_ms=duration_ms,
        generation_error=generation_error,
    )
