# Design

## Goal

Test whether system prompt personas (professional, absurd, none) affect Claude's code generation quality on HumanEval problems. Uses the Anthropic Python SDK for direct API access with reasoning enabled/disabled as a second experimental axis.

## Experimental Design

- **4 persona conditions**: baseline (no persona), professional, absurd (GigaChad), mickey (Mickey Mouse)
- **2 thinking modes**: enabled (budget=10k tokens), disabled
- **164 HumanEval problems** per condition
- **10 runs per combination** for pass@k estimation
- Total: 164 x 4 x 2 x 10 = 13,120 API calls

## Architecture

Data flows through three decoupled phases:

1. **Run** - Generate completions via Anthropic API, save as individual JSON files
2. **Evaluate** - Execute completions in sandboxed processes against HumanEval tests
3. **Report** - Aggregate results into comparison tables

Each phase is a separate CLI command, allowing partial runs and re-evaluation.

## Key Decisions

- **Direct API access** via `anthropic.AsyncAnthropic` for proper token counting, temperature control, and no subprocess overhead
- **Temperature control**: configurable (default 0.2), automatically set to 1.0 when thinking is enabled (API requirement)
- **File-per-run storage** enables resumability without complex state management
- **Sandboxed evaluation** uses multiprocessing with timeouts and disabled destructive builtins
- **Thinking mode** controlled via API's `thinking` parameter (`enabled`/`disabled`)
- **Concurrency** configurable via asyncio.Semaphore (default 5, recommended 20 for Haiku)
- **Complete function prompting**: model returns the full function in a markdown fence, then body is extracted structurally (no indentation guessing)
- **Parallel evaluation** via ProcessPoolExecutor for faster test execution
- **Cost calculation** from actual token counts using model-specific pricing

## Limitations

- API requires `temperature=1` when thinking is enabled, so temperature config is ignored for thinking runs
- HumanEval is a well-known benchmark, so models may have seen solutions during training
