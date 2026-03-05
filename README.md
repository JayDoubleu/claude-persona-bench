# persona-bench

Does the system prompt persona affect how well Claude writes code? This tool benchmarks Claude's code generation quality on [HumanEval](https://github.com/openai/human-eval) (164 problems) across different system prompt personas and thinking modes.

**TL;DR: Personas don't matter. Thinking mode does.**

## Results

### Claude Haiku 4.5

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

### Claude 3 Haiku (legacy, no thinking)

Model: `claude-3-haiku-20240307` | 164 HumanEval problems | 10 runs per condition | 6,560 total API calls | $1.94 total cost

Older-generation model included to see how a previous Haiku holds up. Claude 3 Haiku does not support extended thinking, so only `disabled` mode was tested.

| Condition | Thinking | pass@1 | pass@5 | pass@10 | Avg In Tok | Avg Out Tok | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | disabled | 73.4% | 77.3% | 78.0% | 209 | 196 | $0.49 |
| professional | disabled | 72.7% | 75.6% | 76.2% | 247 | 195 | $0.50 |
| absurd | disabled | 67.9% | 73.5% | 75.0% | 324 | 148 | $0.44 |
| mickey | disabled | 74.9% | 77.7% | 78.0% | 229 | 204 | $0.51 |

The generational gap is stark: Haiku 4.5 without thinking (95.8% pass@1) outperforms Haiku 3 by over 20 percentage points. Interestingly, the absurd persona hurts Haiku 3 noticeably (~6pp below baseline), suggesting older models are more susceptible to persona distractions. Mickey Mouse, on the other hand, edges out baseline slightly -- though this is likely noise at this sample size.

### Claude 3 Haiku + APM OSS Architect (experimental)

Model: `claude-3-haiku-20240307` | 164 HumanEval problems | 10 runs per condition | 4,920 total API calls | $1.61 total cost

Testing the [APM OSS Architect](https://github.com/microsoft/apm/blob/main/.github/chatmodes/oss-architect.chatmode.md) chatmode from Microsoft's APM project. The system prompt includes the full chatmode persona, the [APM Manifesto](https://github.com/microsoft/apm/blob/main/MANIFESTO.md), and project context -- a much longer prompt (~770 input tokens vs ~210 for baseline).

| Condition | Thinking | pass@1 | pass@5 | pass@10 | Avg In Tok | Avg Out Tok | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | disabled | 72.7% | 77.0% | 78.7% | 209 | 197 | $0.49 |
| apm_oss_architect | disabled | 73.4% | 77.4% | 79.3% | 773 | 142 | $0.61 |
| mickey | disabled | 75.8% | 78.5% | 79.3% | 229 | 204 | $0.51 |

The APM OSS Architect persona performs on par with baseline despite being ~3.7x more input tokens. The extra context (manifesto, project description) neither helps nor hurts code generation quality. Mickey Mouse continues to slightly edge out both, though the differences remain within noise. Notably, apm_oss_architect produces shorter outputs (142 avg tokens vs 197 for baseline), suggesting the verbose system prompt may cause the model to be more concise in its responses.

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

### Thinking Modes

- **Disabled**: Standard inference (`temperature=0.2`)
- **Enabled**: Extended thinking (`temperature=1`, required by API). Uses adaptive thinking on Opus 4.6/Sonnet 4.6, budget_tokens on legacy models.

### Methodology

- **Dataset**: Full HumanEval (164 problems)
- **Runs**: 10 per combination (4 conditions x 2 thinking modes = 80 runs per problem)
- **Scoring**: Unbiased pass@k estimator from [Chen et al. (2021)](https://arxiv.org/abs/2107.03374)
- **Evaluation**: Each completion is executed in a sandboxed subprocess against HumanEval's test assertions
- **API**: Direct Anthropic API via `anthropic` Python SDK (no CLI wrapper overhead)

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY="your-key"
```

## Usage

```bash
# Run experiment (resumable)
persona-bench run --runs 10 --concurrency 20

# Run with specific conditions/modes
persona-bench run --runs 5 --thinking disabled --conditions baseline --conditions absurd

# Evaluate completions against tests
persona-bench evaluate

# Re-evaluate previously failed results (after pipeline fixes)
persona-bench evaluate --re-evaluate

# Generate report
persona-bench report --k 1 --k 5 --k 10

# Export results
persona-bench report --csv-output results.csv --md-output results.md

# Inspect failures
persona-bench failures

# Check experiment progress
persona-bench status
```

## Project Structure

```
src/persona_bench/
  cli.py                  Click CLI entry point
  config.py               ExperimentConfig dataclass
  models.py               Problem, Condition, RunKey, RunResult
  problems/loader.py      HumanEval dataset download + parsing
  personas/registry.py    System prompt definitions
  runner/
    client.py             Anthropic API client with retry logic
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
