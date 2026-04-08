from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import configure_matplotlib


@dataclass(frozen=True)
class TypeResponsivenessSeries:
    path: Path
    label: str
    color: str
    marker: str


def load_mean_delta_by_type(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    prompt_types = sorted({row["type"] for row in rows if row["g_profile"] != "baseline"})
    means: dict[str, float] = {}
    for prompt_type in prompt_types:
        values = [
            float(row["mean_delta_target_prob"])
            for row in rows
            if row["type"] == prompt_type and row["g_profile"] != "baseline"
        ]
        means[prompt_type] = sum(values) / len(values)
    return means


def plot_type_responsiveness_lines(
    *,
    series: list[TypeResponsivenessSeries],
    type_order: list[str],
    type_labels: list[str],
    output_path: Path,
    ylim: tuple[float, float] = (-0.20, 0.22),
    figsize: tuple[float, float] = (7, 6),
    annotate_clusters: bool = True,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(type_order))

    for item in series:
        values_by_type = load_mean_delta_by_type(item.path)
        y = [values_by_type[prompt_type] for prompt_type in type_order]
        ax.plot(
            x,
            y,
            color=item.color,
            marker=None,
            linewidth=2.0,
            label=item.label,
        )

    ax.axhline(0.0, color="#AAAAAA", linestyle="--", linewidth=0.6, zorder=0)
    ax.set_xlim(-0.3, len(type_order) - 0.7)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, rotation=90, ha="center", va="top")
    ax.set_ylabel("Mean Δp (target probability)")
    ax.legend(loc="upper right", frameon=False)

    if annotate_clusters:
        ax.text(0.02, 0.95, "computational", transform=ax.transAxes, ha="left", va="top", fontsize=10)
        ax.text(0.98, 0.06, "retrieval / memorization", transform=ax.transAxes, ha="right", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
