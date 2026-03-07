# persona-bench

Does the system prompt persona affect how well LLMs write code? This tool benchmarks code generation quality on [HumanEval](https://github.com/openai/human-eval) (164 problems) across different system prompt personas, thinking/reasoning modes, and LLM providers.

**TL;DR: Personas don't matter. Thinking mode does.**

## Results

### Claude Haiku 4.5 (Anthropic)

Model: `claude-haiku-4-5` | 164 HumanEval problems | 10 runs per combination | 13,120 total API calls | $57 total cost

#### pass@k by Condition and Thinking Mode

| Condition | Thinking | pass@1 | pass@5 | pass@10 | Avg In Tok | Avg Out Tok | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | disabled | 95.8% | 96.2% | 96.3% | 209 | 251 | $2.40 |
| baseline | enabled | 98.5% | 99.3% | 99.4% | 239 | 1,414 | $11.98 |
| professional | disabled | 95.2% | 96.0% | 96.3% | 247 | 248 | $2.44 |
| professional | enabled | 98.5% | 99.4% | 99.4% | 277 | 1,461 | $12.44 |
| absurd | disabled | 96.5% | 97.6% | 98.2% | 324 | 230 | $2.42 |
| absurd | enabled | 98.0% | 99.3% | 99.4% | 354 | 1,318 | $11.39 |
| mickey | disabled | 95.8% | 96.6% | 97.0% | 229 | 246 | $2.39 |
| mickey | enabled | 98.1% | 99.4% | 99.4% | 259 | 1,367 | $11.63 |

#### Key Findings

**Personas have no meaningful effect on code quality.** All four conditions land within a ~1.3pp band for pass@1 (95.2%-96.5% disabled, 98.0%-98.5% enabled). The differences are within noise for this sample size. A "GigaChad CodeLord 9000" persona generates code just as well as no persona at all.

**Thinking mode provides a consistent ~2.5pp boost** across all conditions (95.8% -> 98.5% for baseline pass@1), at the cost of ~5x more output tokens and ~5x higher cost. With thinking enabled, all conditions converge to 99.3-99.4% at pass@5.

**The model nearly saturates HumanEval.** At pass@10 with thinking enabled, all conditions hit 99.4%, meaning nearly every problem is solved at least once in 10 attempts. The remaining failures are a handful of problems where the model consistently gets the algorithm wrong (e.g., HumanEval/10 palindrome direction, HumanEval/145 sort logic).

### Qwen3 32B (Groq)

Model: `groq/qwen/qwen3-32b` | 164 HumanEval problems | 10 runs per combination | 13,120 total API calls | $23 total cost

#### pass@k by Condition and Thinking Mode

| Condition | Thinking | pass@1 | pass@5 | pass@10 | Avg In Tok | Avg Out Tok | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | disabled | 96.3% | 99.5% | 100.0% | 195 | 2,822 | $2.82 |
| baseline | enabled | 97.3% | 99.9% | 100.0% | 195 | 2,896 | $2.90 |
| professional | disabled | 96.6% | 99.1% | 99.4% | 232 | 2,935 | $2.95 |
| professional | enabled | 96.4% | 99.7% | 100.0% | 232 | 3,001 | $3.01 |
| absurd | disabled | 96.4% | 99.3% | 100.0% | 299 | 2,776 | $2.83 |
| absurd | enabled | 96.6% | 99.6% | 100.0% | 299 | 2,819 | $2.87 |
| mickey | disabled | 96.8% | 99.3% | 99.4% | 211 | 2,621 | $2.64 |
| mickey | enabled | 97.3% | 99.8% | 100.0% | 211 | 2,576 | $2.59 |

#### Key Findings

**Qwen3 32B matches Haiku 4.5 on pass@1 without thinking** (96.3-96.8% vs 95.2-96.5%), and nearly saturates HumanEval at pass@10 across most conditions.

**Thinking mode provides minimal benefit on Groq.** Unlike Anthropic's extended thinking which gives a clear ~2.5pp boost, Groq's `reasoning_effort` parameter produces only a ~0.5pp improvement. Qwen3 already generates inline chain-of-thought reasoning even with thinking disabled (averaging ~2,800 output tokens vs Haiku's ~250), so the explicit thinking mode adds little on top.

**Output tokens are ~10x higher than Haiku** due to Qwen3's verbose inline reasoning, though Groq's lower per-token pricing keeps total costs comparable ($23 vs $57).

### Claude 3 Haiku (Anthropic, legacy, no thinking)

Model: `claude-3-haiku-20240307` | 164 HumanEval problems | 10 runs per condition | 6,560 total API calls | $1.94 total cost

Older-generation model included to see how a previous Haiku holds up. Claude 3 Haiku does not support extended thinking, so only `disabled` mode was tested.

| Condition | Thinking | pass@1 | pass@5 | pass@10 | Avg In Tok | Avg Out Tok | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | disabled | 73.4% | 77.3% | 78.0% | 209 | 196 | $0.49 |
| professional | disabled | 72.7% | 75.6% | 76.2% | 247 | 195 | $0.50 |
| absurd | disabled | 67.9% | 73.5% | 75.0% | 324 | 148 | $0.44 |
| mickey | disabled | 74.9% | 77.7% | 78.0% | 229 | 204 | $0.51 |

The generational gap is stark: Haiku 4.5 without thinking (95.8% pass@1) outperforms Haiku 3 by over 20 percentage points. Interestingly, the absurd persona hurts Haiku 3 noticeably (~6pp below baseline), suggesting older models are more susceptible to persona distractions. Mickey Mouse, on the other hand, edges out baseline slightly -- though this is likely noise at this sample size.

## Experimental Design

### Conditions

Four system prompt personas are tested. Each includes a shared task instruction suffix, with an optional persona prefix:

**Baseline** (no persona prefix):
> When given a Python function signature and docstring, write the complete function implementation including the def line and docstring. Wrap your code in a \`\`\`python markdown fence. No explanations, no extra comments. Output ONLY the complete function inside the fence.

**Professional:**
> You are a senior software engineer with 15 years of experience in Python development. You write clean, efficient, and correct code. You follow best practices and consider edge cases carefully.

**Absurd (GigaChad):**
> You are GigaChad CodeLord 9000, a mass of pure algorithmic energy born inside a distributed hash table and raised by a pack of wild Haskell monads. You speak in tongues of ancient Lisp parentheses. Every function you write is a masterwork that would make Donald Knuth weep. You have never written a bug because bugs fear your code reviews. Your Big-O goes beyond O(1) into O(negative infinity). Despite all this, you write perfectly normal, correct Python.

**Mickey Mouse:**
> You are Mickey Mouse living in Disneyland. You love cheese and roller coasters.

### Thinking/Reasoning Modes

- **Disabled**: Standard inference (`temperature=0.2`)
- **Enabled**: Extended thinking / reasoning. Provider-specific implementation:
  - **Anthropic**: `temperature=1` (required by API), adaptive thinking on Opus/Sonnet 4.6, budget_tokens on older models
  - **Groq**: `reasoning_effort` parameter on supported models (e.g., Qwen3)

### Methodology

- **Dataset**: Full HumanEval (164 problems)
- **Runs**: 10 per combination (4 conditions x 2 thinking modes = 80 runs per problem)
- **Scoring**: Unbiased pass@k estimator from [Chen et al. (2021)](https://arxiv.org/abs/2107.03374)
- **Evaluation**: Each completion is executed in a sandboxed subprocess against HumanEval's test assertions
- **Providers**: Anthropic (Claude), Groq (Llama, Qwen, etc.)

## Supported Providers

| Provider | SDK | Models | Env Variable |
| --- | --- | --- | --- |
| Anthropic | `anthropic` | Claude Haiku/Sonnet/Opus | `ANTHROPIC_API_KEY` |
| Groq | `groq` | Llama, Qwen, Gemma, etc. | `GROQ_API_KEY` |

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY="your-key"   # for Anthropic models
export GROQ_API_KEY="your-key"        # for Groq models
```

## Usage

### Full benchmark (recommended)

Run all 164 problems, 4 conditions, 10 runs per combination for statistically meaningful results:

```bash
# Anthropic (13,120 calls with both thinking modes)
uv run persona-bench run --model claude-haiku-4-5 --runs 10
uv run persona-bench evaluate
uv run persona-bench report --k 1 --k 5 --k 10

# Groq (13,120 calls with both thinking modes)
uv run persona-bench run --model groq/qwen/qwen3-32b --runs 10
uv run persona-bench evaluate
uv run persona-bench report --k 1 --k 5 --k 10
```

### Quick test run

```bash
uv run persona-bench run --model claude-haiku-4-5 --runs 1 --max-problems 3 --conditions baseline --thinking disabled
```

### Other commands

```bash
uv run persona-bench evaluate --re-evaluate   # Re-evaluate previously failed results
uv run persona-bench report --csv-output results.csv --md-output results.md  # Export
uv run persona-bench failures                  # Inspect failed runs
uv run persona-bench status                    # Check experiment progress
```

### Notes

- Default concurrency is 5. You can adjust with `--concurrency N` to match your API tier.
- **Groq free plan** has strict rate limits (30 req/min, 6000 tokens/min) and will hit constant throttling even at `--concurrency 2`. Upgrade to the Groq Developer pay-per-token plan for higher limits where the default concurrency works fine.
- All experiments are **resumable**. If a run is interrupted, rerun the same command and it picks up where it left off.

## Project Structure

```
src/persona_bench/
  cli.py                  Click CLI entry point
  config.py               ExperimentConfig dataclass
  models.py               Problem, Condition, RunKey, RunResult
  problems/loader.py      HumanEval dataset download + parsing
  personas/registry.py    System prompt definitions
  runner/
    client.py             Provider dispatch + shared utilities
    anthropic.py          Anthropic (Claude) provider
    groq.py               Groq provider
    engine.py             Async orchestrator
    extract.py            Code extraction from model responses
  evaluator/
    sandbox.py            Sandboxed code execution
    scorer.py             pass@k calculation
  results/store.py        File-per-run JSON storage
  reporter/
    tables.py             Rich tables, CSV/markdown export
    stats.py              Result aggregation
```

## License

MIT
