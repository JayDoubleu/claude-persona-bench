from __future__ import annotations

import asyncio
import logging

from genai_prices import Usage, calc_price

from persona_bench.config import ExperimentConfig
from persona_bench.models import (
    Problem,
    RunKey,
    RunResult,
)

logger = logging.getLogger(__name__)

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


def parse_model(model: str) -> tuple[str, str]:
    """Split a model string into (provider_id, bare_model_name).

    "groq/qwen3-32b" -> ("groq", "qwen3-32b")
    "claude-haiku-4-5" -> ("anthropic", "claude-haiku-4-5")
    """
    if model.startswith("groq/"):
        return ("groq", model.removeprefix("groq/"))
    return ("anthropic", model)


def supports_thinking(model: str) -> bool:
    """Check if a model supports thinking/reasoning mode."""
    provider_id, bare_model = parse_model(model)
    if provider_id == "anthropic":
        return _supports_adaptive(bare_model) or _supports_budget_thinking(bare_model)
    # For non-Anthropic providers, let the API decide
    return True


def _supports_adaptive(model: str) -> bool:
    return any(model.startswith(p) for p in _ADAPTIVE_MODELS)


def _supports_budget_thinking(model: str) -> bool:
    return any(model.startswith(p) for p in _BUDGET_THINKING_MODELS)


def compute_cost(
    model: str,
    provider_id: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
) -> float:
    """Calculate cost using genai-prices. Returns 0.0 if model is not recognized."""
    try:
        price = calc_price(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens + thinking_tokens),
            model_ref=model,
            provider_id=provider_id,
        )
        return float(price.total_price) if price else 0.0
    except Exception:
        return 0.0


async def call_model(
    problem: Problem,
    key: RunKey,
    config: ExperimentConfig,
    semaphore: asyncio.Semaphore,
) -> RunResult:
    """Dispatch to the appropriate provider based on the model string."""
    provider_id, _ = parse_model(config.model)

    if provider_id == "groq":
        from persona_bench.runner.groq import call_groq

        return await call_groq(problem, key, config, semaphore)

    from persona_bench.runner.anthropic import call_anthropic

    return await call_anthropic(problem, key, config, semaphore)
