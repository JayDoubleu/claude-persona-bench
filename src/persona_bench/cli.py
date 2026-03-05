from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

from persona_bench.config import ExperimentConfig
from persona_bench.evaluator.sandbox import run_sandboxed
from persona_bench.models import Condition, RunResult, ThinkingMode
from persona_bench.runner.extract import ensure_indented, extract_body
from persona_bench.problems.loader import load_problems
from persona_bench.reporter.tables import (
    export_csv,
    export_markdown,
    render_console_table,
)
from persona_bench.results.store import (
    get_completed_keys,
    load_all_results,
    update_result,
)
from persona_bench.runner.engine import generate_all_keys, run_experiment

console = Console()


@click.group()
def main() -> None:
    """Persona vs Task Prompt Benchmarking Tool."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@main.command()
@click.option("--model", default="claude-haiku-4-5", help="Model to use")
@click.option("--temperature", default=0.2, type=float, help="Sampling temperature (ignored when thinking is enabled)")
@click.option(
    "--conditions",
    multiple=True,
    type=click.Choice([c.value for c in Condition]),
    help="Conditions to test (repeat for multiple). Default: all",
)
@click.option(
    "--thinking",
    multiple=True,
    type=click.Choice([t.value for t in ThinkingMode]),
    help="Thinking modes (repeat for multiple). Default: all",
)
@click.option("--runs", default=3, type=int, help="Runs per combination")
@click.option("--max-problems", default=None, type=int, help="Limit number of problems")
@click.option("--concurrency", default=5, type=int, help="Max concurrent API calls")
@click.option("--results-dir", default="results/experiments", type=click.Path())
def run(
    model: str,
    temperature: float,
    conditions: tuple[str, ...],
    thinking: tuple[str, ...],
    runs: int,
    max_problems: int | None,
    concurrency: int,
    results_dir: str,
) -> None:
    """Run the experiment (resumable)."""
    config = ExperimentConfig(
        model=model,
        temperature=temperature,
        conditions=[Condition(c) for c in conditions] if conditions else list(Condition),
        thinking_modes=[ThinkingMode(t) for t in thinking] if thinking else list(ThinkingMode),
        runs_per_combination=runs,
        max_problems=max_problems,
        concurrency=concurrency,
        results_dir=Path(results_dir),
    )

    console.print(f"[bold]Experiment ID:[/bold] {config.experiment_id}")
    console.print(f"[bold]Model:[/bold] {config.model}")
    console.print(f"[bold]Conditions:[/bold] {[c.value for c in config.conditions]}")
    console.print(f"[bold]Thinking:[/bold] {[t.value for t in config.thinking_modes]}")
    console.print(f"[bold]Runs per combo:[/bold] {config.runs_per_combination}")

    problems = load_problems(config.cache_dir, config.max_problems)
    console.print(f"[bold]Problems loaded:[/bold] {len(problems)}")

    all_keys = generate_all_keys(problems, config)
    completed = get_completed_keys(config.runs_dir)
    pending = len(all_keys) - len(completed)
    console.print(f"[bold]Total runs:[/bold] {len(all_keys)}, [bold]Pending:[/bold] {pending}")

    if pending == 0:
        console.print("[green]All runs already completed!")
        return

    with Progress() as progress:
        task = progress.add_task("Running...", total=pending)

        def on_complete(result: RunResult) -> None:
            status = "ok" if not result.error else f"err: {result.error[:40]}"
            progress.advance(task)
            progress.console.print(
                f"  {result.key.task_id} [{result.key.condition.value}/"
                f"{result.key.thinking.value}/r{result.key.run_n}] {status}"
            )

        results = asyncio.run(run_experiment(problems, config, on_complete))

    console.print(f"\n[green bold]Completed {len(results)} runs.[/green bold]")
    console.print(f"Results saved to: {config.runs_dir}")


@main.command("evaluate")
@click.argument("experiment_id", required=False)
@click.option("--results-dir", default="results/experiments", type=click.Path())
@click.option("--re-evaluate", is_flag=True, help="Re-evaluate previously failed results")
def evaluate_cmd(experiment_id: str | None, results_dir: str, re_evaluate: bool) -> None:
    """Execute generated code against HumanEval tests."""
    base = Path(results_dir)

    if experiment_id:
        runs_dir = base / experiment_id / "runs"
    else:
        # Find the most recent experiment
        dirs = sorted(base.iterdir(), key=lambda d: d.stat().st_mtime) if base.exists() else []
        if not dirs:
            console.print("[red]No experiments found.")
            return
        runs_dir = dirs[-1] / "runs"

    console.print(f"[bold]Evaluating:[/bold] {runs_dir}")

    results = load_all_results(runs_dir)
    if re_evaluate:
        # Reset failed results so they get re-evaluated
        for r in results:
            if r.passed is False:
                r.passed = None
                r.error = None
                update_result(runs_dir, r)
    unevaluated = [r for r in results if r.passed is None and not r.error]
    console.print(f"[bold]Total results:[/bold] {len(results)}, [bold]To evaluate:[/bold] {len(unevaluated)}")

    if not unevaluated:
        console.print("[green]Nothing to evaluate.")
        return

    # Load problems for test data
    problems = load_problems(Path(".cache"))
    problem_map = {p.task_id: p for p in problems}

    evaluated = 0
    passed = 0
    workers = min(os.cpu_count() or 4, len(unevaluated))

    # Prepare all eval jobs upfront
    jobs: list[tuple[RunResult, str, str, str]] = []
    for result in unevaluated:
        problem = problem_map.get(result.key.task_id)
        if not problem:
            result.error = f"Problem {result.key.task_id} not found"
            update_result(runs_dir, result)
            continue
        imports, body = extract_body(result.completion)
        full_code = imports + problem.prompt + ensure_indented(body)
        jobs.append((result, full_code, problem.test, problem.entry_point))

    with Progress() as progress:
        task = progress.add_task("Evaluating...", total=len(jobs))

        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_result = {
                pool.submit(run_sandboxed, code, test, ep): result
                for result, code, test, ep in jobs
            }
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                ok, error = future.result()
                result.passed = ok
                if error:
                    result.error = error

                update_result(runs_dir, result)
                evaluated += 1
                if ok:
                    passed += 1

                progress.advance(task)
                progress.console.print(
                    f"  {result.key.task_id} [{result.key.condition.value}]: "
                    f"{'PASS' if ok else 'FAIL'}"
                )

    console.print(f"\n[bold]Evaluated:[/bold] {evaluated}, [green]Passed:[/green] {passed}, [red]Failed:[/red] {evaluated - passed}")


@main.command()
@click.argument("experiment_id", required=False)
@click.option("--results-dir", default="results/experiments", type=click.Path())
@click.option("--csv-output", default=None, type=click.Path(), help="Export CSV to this path")
@click.option("--md-output", default=None, type=click.Path(), help="Export markdown to this path")
@click.option("--k", "k_values", multiple=True, type=int, default=[1], help="k values for pass@k")
def report(
    experiment_id: str | None,
    results_dir: str,
    csv_output: str | None,
    md_output: str | None,
    k_values: tuple[int, ...],
) -> None:
    """Generate comparison tables from results."""
    base = Path(results_dir)

    if experiment_id:
        runs_dir = base / experiment_id / "runs"
    else:
        dirs = sorted(base.iterdir(), key=lambda d: d.stat().st_mtime) if base.exists() else []
        if not dirs:
            console.print("[red]No experiments found.")
            return
        runs_dir = dirs[-1] / "runs"

    results = load_all_results(runs_dir)
    if not results:
        console.print("[red]No results found.")
        return

    k_list = list(k_values)
    render_console_table(results, k_list)

    if csv_output:
        export_csv(results, Path(csv_output), k_list)
        console.print(f"[green]CSV exported to {csv_output}")

    if md_output:
        export_markdown(results, Path(md_output), k_list)
        console.print(f"[green]Markdown exported to {md_output}")


@main.command()
@click.argument("experiment_id", required=False)
@click.option("--results-dir", default="results/experiments", type=click.Path())
def status(experiment_id: str | None, results_dir: str) -> None:
    """Show experiment progress."""
    base = Path(results_dir)

    if not base.exists():
        console.print("[yellow]No experiments directory found.")
        return

    if experiment_id:
        dirs = [base / experiment_id]
    else:
        dirs = sorted(base.iterdir(), key=lambda d: d.stat().st_mtime)

    for exp_dir in dirs:
        if not exp_dir.is_dir():
            continue
        runs_dir = exp_dir / "runs"
        if not runs_dir.exists():
            continue

        results = load_all_results(runs_dir)
        total = len(results)
        evaluated = sum(1 for r in results if r.passed is not None)
        passed = sum(1 for r in results if r.passed is True)
        errors = sum(1 for r in results if r.error)
        cost = sum(r.cost_usd for r in results)

        console.print(f"\n[bold cyan]{exp_dir.name}[/bold cyan]")
        console.print(f"  Runs: {total}")
        console.print(f"  Evaluated: {evaluated}/{total}")
        console.print(f"  Passed: {passed}, Failed: {evaluated - passed}, Errors: {errors}")
        console.print(f"  Total cost: ${cost:.2f}")


@main.command()
@click.argument("experiment_id", required=False)
@click.option("--results-dir", default="results/experiments", type=click.Path())
def failures(experiment_id: str | None, results_dir: str) -> None:
    """Show details of failed runs for quick triage."""
    base = Path(results_dir)

    if experiment_id:
        runs_dir = base / experiment_id / "runs"
    else:
        dirs = sorted(base.iterdir(), key=lambda d: d.stat().st_mtime) if base.exists() else []
        if not dirs:
            console.print("[red]No experiments found.")
            return
        runs_dir = dirs[-1] / "runs"

    results = load_all_results(runs_dir)
    failed = [r for r in results if r.passed is False]

    if not failed:
        console.print("[green]No failures found!")
        return

    console.print(f"[bold red]{len(failed)} failure(s)[/bold red] in {runs_dir.parent.name}\n")

    # Load problems to show the prompt for context
    problems = load_problems(Path(".cache"))
    problem_map = {p.task_id: p for p in problems}

    for r in sorted(failed, key=lambda r: (r.key.task_id, r.key.condition.value)):
        console.print(f"[bold]{r.key.task_id}[/bold] [{r.key.condition.value}/{r.key.thinking.value}/r{r.key.run_n}]")

        if r.error:
            console.print(f"  [red]Error:[/red] {r.error}")

        # Show what extract_body + ensure_indented would produce
        imports, body = extract_body(r.completion)
        indented = ensure_indented(body)

        # Check for structural issues
        problem = problem_map.get(r.key.task_id)
        if problem:
            full_code = imports + problem.prompt + indented
            # Quick check: does it even parse?
            try:
                compile(full_code, "<check>", "exec")
                console.print("  [green]Syntax:[/green] OK")
            except SyntaxError as e:
                console.print(f"  [red]Syntax:[/red] {e}")

        # Show first few lines of completion
        lines = r.completion.split("\n")
        preview = "\n".join(lines[:8])
        if len(lines) > 8:
            preview += f"\n    ... ({len(lines) - 8} more lines)"
        console.print(f"  [dim]Completion:[/dim]\n{preview}\n")


if __name__ == "__main__":
    main()
