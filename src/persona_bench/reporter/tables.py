from __future__ import annotations

import csv
from pathlib import Path

from rich.console import Console
from rich.table import Table

from persona_bench.evaluator.scorer import GroupKey, compute_pass_rates
from persona_bench.models import RunResult
from persona_bench.reporter.stats import GroupStats, compute_group_stats


def _sorted_keys(stats: dict[GroupKey, GroupStats]) -> list[GroupKey]:
    return sorted(stats.keys(), key=lambda k: (k[0].value, k[1].value))


def render_console_table(results: list[RunResult], k_values: list[int] | None = None) -> None:
    """Print a Rich table to the console with pass rates and stats."""
    if k_values is None:
        k_values = [1]

    console = Console()
    stats = compute_group_stats(results)
    pass_rates = compute_pass_rates(results, k_values)

    table = Table(title="Persona Benchmark Results")
    table.add_column("Condition", style="cyan")
    table.add_column("Thinking", style="magenta")
    table.add_column("Runs", justify="right")
    table.add_column("Evaluated", justify="right")
    for k in k_values:
        table.add_column(f"pass@{k}", justify="right", style="green")
    table.add_column("Avg In Tok", justify="right")
    table.add_column("Avg Out Tok", justify="right")
    table.add_column("Avg Cost $", justify="right")
    table.add_column("Total Cost $", justify="right", style="yellow")

    for key in _sorted_keys(stats):
        s = stats[key]
        pr = pass_rates.get(key, {})
        row = [
            s.condition.value,
            s.thinking.value,
            str(s.total_runs),
            str(s.evaluated),
        ]
        for k in k_values:
            rate = pr.get(f"pass@{k}", 0.0)
            row.append(f"{rate:.1%}")
        row.extend(
            [
                f"{s.avg_input_tokens:.0f}",
                f"{s.avg_output_tokens:.0f}",
                f"${s.avg_cost_usd:.4f}",
                f"${s.total_cost_usd:.2f}",
            ]
        )
        table.add_row(*row)

    console.print(table)


def export_csv(results: list[RunResult], path: Path, k_values: list[int] | None = None) -> None:
    """Export results summary to CSV."""
    if k_values is None:
        k_values = [1]

    stats = compute_group_stats(results)
    pass_rates = compute_pass_rates(results, k_values)

    headers = [
        "condition",
        "thinking",
        "runs",
        "evaluated",
        *[f"pass@{k}" for k in k_values],
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_cost_usd",
        "total_cost_usd",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for key in _sorted_keys(stats):
            s = stats[key]
            pr = pass_rates.get(key, {})
            row = [
                s.condition.value,
                s.thinking.value,
                s.total_runs,
                s.evaluated,
                *[f"{pr.get(f'pass@{k}', 0.0):.4f}" for k in k_values],
                f"{s.avg_input_tokens:.0f}",
                f"{s.avg_output_tokens:.0f}",
                f"{s.avg_cost_usd:.6f}",
                f"{s.total_cost_usd:.4f}",
            ]
            writer.writerow(row)


def export_markdown(
    results: list[RunResult],
    path: Path,
    k_values: list[int] | None = None,
) -> None:
    """Export results summary as a markdown table."""
    if k_values is None:
        k_values = [1]

    stats = compute_group_stats(results)
    pass_rates = compute_pass_rates(results, k_values)

    headers = [
        "Condition",
        "Thinking",
        "Runs",
        "Evaluated",
        *[f"pass@{k}" for k in k_values],
        "Avg In Tok",
        "Avg Out Tok",
        "Avg Cost $",
        "Total Cost $",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for key in _sorted_keys(stats):
        s = stats[key]
        pr = pass_rates.get(key, {})
        cells = [
            s.condition.value,
            s.thinking.value,
            str(s.total_runs),
            str(s.evaluated),
            *[f"{pr.get(f'pass@{k}', 0.0):.1%}" for k in k_values],
            f"{s.avg_input_tokens:.0f}",
            f"{s.avg_output_tokens:.0f}",
            f"${s.avg_cost_usd:.4f}",
            f"${s.total_cost_usd:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
