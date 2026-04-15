from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.analysis.pca.scripts.pca_figure6_diagnostics import TYPE_LEGEND_ORDER
from docs.figurelib.common import configure_matplotlib, prettify_type, qualitative_11_color_map


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_pad = 0.05 * (float(x.max()) - float(x.min()))
    y_pad = 0.08 * (float(y.max()) - float(y.min()))
    return (float(x.min()) - x_pad, float(x.max()) + x_pad), (float(y.min()) - y_pad, float(y.max()) + y_pad)


def eta_squared(values: np.ndarray, groups: list[str]) -> float:
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    buckets: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values.tolist(), groups):
        buckets[group].append(value)
    ss_between = sum(len(bucket) * (float(np.mean(bucket)) - grand_mean) ** 2 for bucket in buckets.values())
    return float(ss_between / ss_total)


def plot_compare(rows: list[dict[str, str]], out_path: Path) -> dict[str, float]:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    types = [row["type"] for row in rows]
    tokens = np.array([float(row["tokens_approx"]) for row in rows], dtype=float)
    raw_x = np.array([float(row["pc1"]) for row in rows], dtype=float)
    raw_y = np.array([float(row["pc2"]) for row in rows], dtype=float)
    resid_x = np.array([float(row["length_resid_pc1"]) for row in rows], dtype=float)
    resid_y = np.array([float(row["length_resid_pc2"]) for row in rows], dtype=float)

    type_order = [t for t in TYPE_LEGEND_ORDER if t in set(types)]
    colors = qualitative_11_color_map(type_order)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), sharex=False, sharey=False)
    panels = [
        (axes[0], raw_x, raw_y, "Raw PC1 vs PC2", "PC1", "PC2"),
        (axes[1], resid_x, resid_y, "Length-Residualized PC1 vs PC2", "Length-resid PC1", "Length-resid PC2"),
    ]

    for ax, x, y, title, xlabel, ylabel in panels:
        xlim, ylim = axis_limits(x, y)
        for prompt_type in type_order:
            idx = [i for i, t in enumerate(types) if t == prompt_type]
            ax.scatter(
                x[idx],
                y[idx],
                s=18,
                alpha=0.78,
                color=colors[prompt_type],
                linewidths=0,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.set_facecolor("white")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=colors[t], markeredgewidth=0)
        for t in type_order
    ]
    labels = [prettify_type(t) for t in type_order]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.18, wspace=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = {
        "corr_raw_pc1_tokens": float(np.corrcoef(raw_x, tokens)[0, 1]),
        "corr_raw_pc2_tokens": float(np.corrcoef(raw_y, tokens)[0, 1]),
        "corr_resid_pc1_tokens": float(np.corrcoef(resid_x, tokens)[0, 1]),
        "corr_resid_pc2_tokens": float(np.corrcoef(resid_y, tokens)[0, 1]),
        "raw_pc1_type_eta": eta_squared(raw_x, types),
        "raw_pc2_type_eta": eta_squared(raw_y, types),
        "resid_pc1_type_eta": eta_squared(resid_x, types),
        "resid_pc2_type_eta": eta_squared(resid_y, types),
    }
    return stats


def write_summary(stats: dict[str, float], out_path: Path) -> None:
    text = f"""PC1/PC2 raw vs length-residualized comparison

Token-count correlations:
- raw PC1 vs tokens_approx: {stats['corr_raw_pc1_tokens']:.3f}
- raw PC2 vs tokens_approx: {stats['corr_raw_pc2_tokens']:.3f}
- length-resid PC1 vs tokens_approx: {stats['corr_resid_pc1_tokens']:.3f}
- length-resid PC2 vs tokens_approx: {stats['corr_resid_pc2_tokens']:.3f}

Task structure (eta^2 by task class):
- raw PC1: {stats['raw_pc1_type_eta']:.3f}
- raw PC2: {stats['raw_pc2_type_eta']:.3f}
- length-resid PC1: {stats['resid_pc1_type_eta']:.3f}
- length-resid PC2: {stats['resid_pc2_type_eta']:.3f}

Read:
- Raw PC1 is strongly length-loaded.
- After residualizing on token count, PC1 is no longer length-dominated but still carries substantial task structure.
- Residualized PC2 remains task-structured rather than dissolving into noise.
"""
    out_path.write_text(text)


def main() -> None:
    csv_path = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "csv" / "figure6_main_task.csv"
    out_dir = REPO_ROOT / "docs" / "figures" / "diagnostics" / "pca_pc1_pc2_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "pc1_pc2_raw_vs_length_resid.png"
    out_summary = out_dir / "summary.txt"

    rows = load_rows(csv_path)
    stats = plot_compare(rows, out_fig)
    write_summary(stats, out_summary)
    print(out_fig)
    print(out_summary)


if __name__ == "__main__":
    main()
