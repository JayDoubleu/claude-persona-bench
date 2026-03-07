"""Generate pass@1 comparison chart for README."""

import matplotlib.pyplot as plt
import numpy as np

# Data: (group_label, {condition: pass@1})
groups = [
    ("Claude 3\nHaiku\n(disabled)", {
        "baseline": 73.4, "professional": 72.7, "absurd": 67.9, "mickey": 74.9,
    }),
    ("Haiku 4.5\n(disabled)", {
        "baseline": 95.8, "professional": 95.2, "absurd": 96.5, "mickey": 95.8,
    }),
    ("Haiku 4.5\n(thinking)", {
        "baseline": 98.5, "professional": 98.5, "absurd": 98.0, "mickey": 98.1,
    }),
    ("Qwen3 32B\n(disabled)", {
        "baseline": 96.3, "professional": 96.6, "absurd": 96.4, "mickey": 96.8,
    }),
    ("Qwen3 32B\n(thinking)", {
        "baseline": 97.3, "professional": 96.4, "absurd": 96.6, "mickey": 97.3,
    }),
    ("GPT-4.1\n(disabled)", {
        "baseline": 96.8, "professional": 95.7, "absurd": 96.2, "mickey": 96.2,
    }),
]

conditions = ["baseline", "professional", "absurd", "mickey"]
colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

fig, ax = plt.subplots(figsize=(12, 5.5))

n_groups = len(groups)
n_bars = len(conditions)
bar_width = 0.18
group_width = n_bars * bar_width + 0.15

x = np.arange(n_groups) * group_width

for i, cond in enumerate(conditions):
    values = [g[1][cond] for g in groups]
    offset = (i - (n_bars - 1) / 2) * bar_width
    bars = ax.bar(x + offset, values, bar_width, label=cond, color=colors[i],
                  edgecolor="white", linewidth=0.5)

ax.set_ylabel("pass@1 (%)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylim(60, 100)
ax.set_yticks(range(60, 101, 5))
ax.yaxis.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

# Add a subtle divider between the old model and the frontier cluster
ax.axvline(x=(x[0] + x[1]) / 2, color="#cccccc", linestyle=":", linewidth=1)

ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("HumanEval pass@1 by Model and Persona", fontsize=14, fontweight="bold", y=0.97)
ax.set_title(
    "Personas cluster tightly within each model. Thinking mode is the real differentiator.",
    fontsize=10, color="#666666", pad=10,
)

fig.tight_layout()
fig.savefig("docs/pass_at_1_chart.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved docs/pass_at_1_chart.png")
