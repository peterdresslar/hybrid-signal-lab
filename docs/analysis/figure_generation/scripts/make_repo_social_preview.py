from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.analysis.pca.scripts.pca_figure6_diagnostics import TYPE_LEGEND_ORDER
from docs.figurelib.common import configure_matplotlib, prettify_type, qualitative_11_color_map


def load_points(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_span = float(x.max()) - float(x.min())
    y_span = float(y.max()) - float(y.min())
    return (
        float(x.min()) - 0.08 * x_span,
        float(x.max()) + 0.08 * x_span,
    ), (
        float(y.min()) - 0.12 * y_span,
        float(y.max()) + 0.12 * y_span,
    )


def main() -> None:
    in_csv = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "csv" / "figure6_main_task.csv"
    out_path = REPO_ROOT / "docs" / "figures" / "social" / "repo_social_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_points(in_csv)
    type_order = [t for t in TYPE_LEGEND_ORDER if any(r["type"] == t for r in rows)]
    colors = qualitative_11_color_map(type_order)

    x = np.array([float(r["pc1"]) for r in rows], dtype=float)
    y = np.array([float(r["pc2"]) for r in rows], dtype=float)
    xlim, ylim = axis_limits(x, y)

    configure_matplotlib(font_family="sans-serif", font_size=12)
    fig = plt.figure(figsize=(12, 6.3), facecolor="#07111f")
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.76], facecolor="#07111f")

    # Faint white underlayer improves readability at thumbnail scale.
    ax.scatter(x, y, s=34, color="white", alpha=0.08, linewidths=0, zorder=1)

    for prompt_type in type_order:
        idx = [i for i, row in enumerate(rows) if row["type"] == prompt_type]
        ax.scatter(
            x[idx],
            y[idx],
            s=22,
            color=colors[prompt_type],
            alpha=0.82,
            linewidths=0,
            zorder=2,
        )

    # Label only the larger classes to keep the card legible.
    for prompt_type in type_order:
        idx = [i for i, row in enumerate(rows) if row["type"] == prompt_type]
        if len(idx) < 85:
            continue
        cx = float(np.mean(x[idx]))
        cy = float(np.mean(y[idx]))
        ax.text(
            cx,
            cy,
            prettify_type(prompt_type),
            color="white",
            fontsize=10,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#07111f",
                "edgecolor": "none",
                "alpha": 0.70,
            },
            zorder=3,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.065,
        0.93,
        "Hybrid Signal Lab",
        color="white",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.065,
        0.885,
        "Battery 4 prompt geometry in Qwen 3.5 9B",
        color="#b7c9e2",
        fontsize=13,
        ha="left",
        va="top",
    )
    fig.text(
        0.065,
        0.065,
        "PCA of baseline final-token attention-head entropy",
        color="#8ca0ba",
        fontsize=11,
        ha="left",
        va="bottom",
    )

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
