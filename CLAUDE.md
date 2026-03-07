# persona-bench

Benchmarking tool that tests whether system prompt personas affect LLM code generation quality on HumanEval problems. Supports multiple providers (Anthropic, Groq).

## Setup

```bash
uv sync
```

## Usage

```bash
uv run persona-bench run          # Run experiment (resumable)
uv run persona-bench evaluate     # Execute generated code against tests
uv run persona-bench report       # Generate comparison tables
uv run persona-bench failures     # Inspect failed runs with syntax check
uv run persona-bench status       # Show experiment progress

# Provider routing via model name prefix:
uv run persona-bench run --model claude-haiku-4-5       # Anthropic (default, no prefix)
uv run persona-bench run --model groq/qwen/qwen3-32b    # Groq
```

## Project Structure

- `src/persona_bench/` - Main package
  - `cli.py` - Click CLI entry point
  - `config.py` - ExperimentConfig dataclass
  - `models.py` - Core data models (Problem, Condition, RunKey, RunResult)
  - `problems/loader.py` - HumanEval dataset download and parsing
  - `personas/registry.py` - Persona condition definitions
  - `runner/` - Async orchestrator, provider dispatch, code extraction
    - `client.py` - Dispatcher (`call_model`), `parse_model`, `supports_thinking`, `compute_cost` (via `genai-prices`)
    - `anthropic.py` - Anthropic provider (`call_anthropic`)
    - `groq.py` - Groq provider (`call_groq`)
    - `engine.py` - Experiment orchestrator
    - `extract.py` - Code extraction from LLM responses
  - `evaluator/` - Sandboxed code execution and pass@k scoring
  - `results/store.py` - File-per-run JSON storage
  - `reporter/` - Rich tables, stats aggregation, CSV/markdown export

## Development

```bash
uv run pytest               # Run tests
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run mypy src/            # Type check
```

## Conventions

- Python 3.11+
- uv for dependency management
- ruff for linting and formatting (line-length 100)
- mypy strict mode for type checking
- Async throughout the runner layer
- One JSON file per run result for resumability
- Results directory is gitignored
