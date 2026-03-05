# persona-bench

Benchmarking tool that tests whether system prompt personas affect Claude's code generation quality on HumanEval problems.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
persona-bench run          # Run experiment (resumable)
persona-bench evaluate     # Execute generated code against tests
persona-bench report       # Generate comparison tables
persona-bench failures     # Inspect failed runs with syntax check
persona-bench status       # Show experiment progress
```

## Project Structure

- `src/persona_bench/` - Main package
  - `cli.py` - Click CLI entry point
  - `config.py` - ExperimentConfig dataclass
  - `models.py` - Core data models (Problem, Condition, RunKey, RunResult)
  - `problems/loader.py` - HumanEval dataset download and parsing
  - `personas/registry.py` - Persona condition definitions
  - `runner/` - Async orchestrator, SDK client, code extraction
  - `evaluator/` - Sandboxed code execution and pass@k scoring
  - `results/store.py` - File-per-run JSON storage
  - `reporter/` - Rich tables, stats aggregation, CSV/markdown export

## Development

```bash
pytest
```

## Conventions

- Python 3.11+
- Async throughout the runner layer
- One JSON file per run result for resumability
- Results directory is gitignored
