from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import anthropic

from persona_bench.config import ExperimentConfig
from persona_bench.models import (
    Problem,
    RunKey,
    RunResult,
    ThinkingMode,
)
from persona_bench.personas.registry import get_system_prompt
from persona_bench.runner.extract import extract_code

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_RETRY_DELAY = 10.0
RATE_LIMIT_DELAY = 60.0

# Pricing (USD per million tokens): (input, output, thinking)
_MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "claude-opus-4-6": (5.0, 25.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0, 5.0),
    "claude-opus-4-5": (5.0, 25.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0, 15.0),
    "claude-opus-4-1": (5.0, 25.0, 25.0),
    "claude-opus-4-0": (5.0, 25.0, 25.0),
    "claude-sonnet-4-0": (3.0, 15.0, 15.0),
}

# Models that support adaptive thinking (budget_tokens is deprecated)
_ADAPTIVE_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}

# Models that support budget_tokens thinking
_BUDGET_THINKING_MODELS = {
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-haiku-4-5",
}


def _get_pricing(model: str) -> tuple[float, float, float]:
    for prefix, pricing in _MODEL_PRICING.items():
        if model.startswith(prefix):
            return pricing
    return (5.0, 25.0, 25.0)  # default to most expensive


def _supports_adaptive(model: str) -> bool:
    return any(model.startswith(p) for p in _ADAPTIVE_MODELS)


def _supports_budget_thinking(model: str) -> bool:
    return any(model.startswith(p) for p in _BUDGET_THINKING_MODELS)


_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(max_retries=0)
    return _client


async def call_claude(
    problem: Problem,
    key: RunKey,
    config: ExperimentConfig,
    semaphore: asyncio.Semaphore,
) -> RunResult:
    """Call Anthropic API for a single problem, with concurrency control and retry."""
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

        except anthropic.RateLimitError as e:
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

        except anthropic.APIStatusError as e:
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
                # Client error (4xx), not retryable
                logger.error("Client error for %s: %s", key.task_id, e)
                return RunResult(
                    key=key,
                    completion="",
                    raw_response="",
                    error=f"APIStatusError {e.status_code}: {e.message}",
                )

        except anthropic.APIConnectionError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Attempt %d/%d for %s: connection error, retrying",
                    attempt + 1,
                    MAX_RETRIES,
                    key.task_id,
                )
                await asyncio.sleep(BASE_RETRY_DELAY * (attempt + 1))

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
        error=f"All {MAX_RETRIES} attempts failed: {last_error}",
    )


async def _single_call(
    problem: Problem,
    key: RunKey,
    config: ExperimentConfig,
) -> RunResult:
    system_prompt = get_system_prompt(key.condition)

    kwargs: dict[str, Any] = {
        "model": config.model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": problem.prompt}],
    }

    if key.thinking == ThinkingMode.ENABLED:
        # API requires temperature=1 when thinking is enabled
        kwargs["temperature"] = 1
        kwargs["max_tokens"] = 16_000
        if _supports_adaptive(config.model):
            kwargs["thinking"] = {"type": "adaptive"}
        elif _supports_budget_thinking(config.model):
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.thinking_budget_tokens,
            }
        else:
            logger.warning(
                "Model %s does not support thinking; falling back to standard mode",
                config.model,
            )
            kwargs["temperature"] = config.temperature
            kwargs["max_tokens"] = 1024
    else:
        kwargs["temperature"] = config.temperature

    start = time.monotonic()
    response = await _get_client().messages.create(**kwargs)
    duration_ms = int((time.monotonic() - start) * 1000)

    # Extract text from response blocks
    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)

    raw_response = "\n".join(text_parts)
    completion = extract_code(raw_response) if raw_response else ""

    # Token counts
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    full_input = input_tokens + cache_creation + cache_read

    thinking_tokens = getattr(response.usage, "thinking_tokens", 0) or 0

    # Cost calculation (model-aware)
    input_price, output_price, thinking_price = _get_pricing(config.model)
    cost = (
        full_input * input_price + output_tokens * output_price + thinking_tokens * thinking_price
    ) / 1_000_000

    # Build error string if needed
    error = None
    if response.stop_reason == "max_tokens":
        error = "Response truncated (hit max_tokens limit)"
    elif response.stop_reason == "refusal":
        error = "Model refused to respond"
    elif not raw_response and not completion:
        error = f"Empty response (stop_reason={response.stop_reason})"

    return RunResult(
        key=key,
        completion=completion,
        raw_response=raw_response,
        input_tokens=full_input,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        cost_usd=cost,
        duration_ms=duration_ms,
        error=error,
    )
