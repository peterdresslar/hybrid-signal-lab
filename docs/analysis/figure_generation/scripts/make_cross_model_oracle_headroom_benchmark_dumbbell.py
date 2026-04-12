from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.figurelib.common import configure_matplotlib


OUT_PNG = REPO_ROOT / "docs" / "figures" / "manuscript" / "figure_cross_model_oracle_headroom_by_benchmark.png"
OUT_CSV = REPO_ROOT / "docs" / "analysis" / "reports" / "cross_model_oracle_headroom_by_benchmark.csv"

ROWS = [
    {"benchmark": "ARC-Challenge", "model": "Qwen 9B", "best_fixed_delta": 0.4, "oracle_delta": 8.1, "n": 1172},
    {"benchmark": "ARC-Challenge", "model": "Olmo Hybrid", "best_fixed_delta": -0.2, "oracle_delta": 10.2, "n": 1172},
    {"benchmark": "abstract_algebra", "model": "Qwen 9B", "best_fixed_delta": 2.0, "oracle_delta": 19.0, "n": 100},
    {"benchmark": "abstract_algebra", "model": "Olmo Hybrid", "best_fixed_delta": -3.0, "oracle_delta": 11.0, "n": 100},
    {"benchmark": "college_math", "model": "Qwen 9B", "best_fixed_delta": 1.0, "oracle_delta": 15.0, "n": 100},
    {"benchmark": "college_math", "model": "Olmo Hybrid", "best_fixed_delta": -1.0, "oracle_delta": 19.0, "n": 100},
    {"benchmark": "college_cs", "model": "Qwen 9B", "best_fixed_delta": -1.0, "oracle_delta": 10.0, "n": 100},
    {"benchmark": "college_cs", "model": "Olmo Hybrid", "best_fixed_delta": 2.0, "oracle_delta": 21.0, "n": 100},
]

MODEL_COLORS = {
    "Qwen 9B": "#4C78A8",
    "Olmo Hybrid": "#F58518",
}

BENCHMARK_ORDER = [
    "ARC-Challenge",
    "abstract_algebra",
    "college_math",
    "college_cs",
]

MODEL_ORDER = ["Qwen 9B", "Olmo Hybrid"]


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["benchmark", "model", "best_fixed_delta", "oracle_delta", "n"],
        )
        writer.writeheader()
        writer.writerows(ROWS)


def plot() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_CSV)
    configure_matplotlib(font_family="sans-serif", font_size=11)

    fig, ax = plt.subplots(figsize=(9.7, 5.3))

    y_positions = {
        ("ARC-Challenge", "Qwen 9B"): 7.8,
        ("ARC-Challenge", "Olmo Hybrid"): 7.0,
        ("abstract_algebra", "Qwen 9B"): 5.6,
        ("abstract_algebra", "Olmo Hybrid"): 4.8,
        ("college_math", "Qwen 9B"): 3.4,
        ("college_math", "Olmo Hybrid"): 2.6,
        ("college_cs", "Qwen 9B"): 1.2,
        ("college_cs", "Olmo Hybrid"): 0.4,
    }
    benchmark_centers = {
        "ARC-Challenge": 7.4,
        "abstract_algebra": 5.2,
        "college_math": 3.0,
        "college_cs": 0.8,
    }
    benchmark_n = {
        "ARC-Challenge": 1172,
        "abstract_algebra": 100,
        "college_math": 100,
        "college_cs": 100,
    }

    fixed_min = min(row["best_fixed_delta"] for row in ROWS)
    fixed_max = max(row["best_fixed_delta"] for row in ROWS)
    ax.axvspan(fixed_min, fixed_max, color="#E6E6E6", alpha=0.65, zorder=0)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)

    for row in ROWS:
        y = y_positions[(row["benchmark"], row["model"])]
        color = MODEL_COLORS[row["model"]]
        ax.plot(
            [row["best_fixed_delta"], row["oracle_delta"]],
            [y, y],
            color=color,
            linewidth=2.2,
            alpha=0.9,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            row["best_fixed_delta"],
            y,
            s=72,
            color="white",
            edgecolors=color,
            linewidths=2.2,
            marker="o",
            zorder=3,
        )
        ax.scatter(
            row["oracle_delta"],
            y,
            s=82,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            marker="D",
            zorder=4,
        )

    for benchmark in BENCHMARK_ORDER:
        center = benchmark_centers[benchmark]
        ax.text(-5.8, center, benchmark, ha="right", va="center", fontsize=11, fontweight="bold")
        ax.text(24.85, center, f"n={benchmark_n[benchmark]}", ha="right", va="center", fontsize=9, color="#444444")

    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS["Qwen 9B"], lw=2.2, label="Qwen 9B"),
        Line2D([0], [0], color=MODEL_COLORS["Olmo Hybrid"], lw=2.2, label="Olmo Hybrid"),
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.5, label="Best panel-fixed profile"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=8, markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.8, label="Oracle (per-prompt panel-best)"),
        Patch(facecolor="#E6E6E6", edgecolor="none", alpha=0.65, label="Fixed-result range"),
    ]
    legend1 = ax.legend(
        handles=model_handles,
        loc="upper right",
        bbox_to_anchor=(0.915, 0.99),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.92,
    )
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, loc="upper center", bbox_to_anchor=(0.54, -0.12), ncol=3, frameon=False)

    ax.set_xlim(-6.6, 25.15)
    ax.set_ylim(-0.2, 8.5)
    ax.set_yticks([])
    ax.set_xlabel("Accuracy delta over baseline (percentage points)")
    ax.set_facecolor("white")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot()
