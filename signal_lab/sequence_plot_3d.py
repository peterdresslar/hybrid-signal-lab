"""Render 3D PCA scatters from sequence-analysis family CSV exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from docs.figurelib.common import configure_matplotlib, prettify_type, qualitative_11_color_map
from signal_lab.paths import DATA_DIR_ENV_VAR, configure_data_dir, resolve_input_path

TYPE_LEGEND_ORDER = [
    "code_comprehension",
    "reasoning_numerical",
    "algorithmic",
    "reasoning_tracking",
    "structural_copying",
    "syntactic_pattern",
    "factual_recall",
    "factual_retrieval",
    "domain_knowledge",
    "cultural_memorized",
    "long_range_retrieval",
]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def axis_limits(values: list[float], pad_frac: float = 0.08) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = hi - lo
    pad = span * pad_frac if span > 1e-12 else 1.0
    return lo - pad, hi + pad


def plot_pca_3d(
    *,
    rows: list[dict[str, str]],
    family_name: str,
    model_label: str,
    output_path: Path,
    mode: str,
    elev: float,
    azim: float,
) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=11)
    if mode == "raw":
        x_key, y_key, z_key = "pc1", "pc2", "pc3"
        axis_labels = ("PC1", "PC2", "PC3")
    elif mode == "length_resid":
        x_key, y_key, z_key = "length_resid_pc1", "length_resid_pc2", "length_resid_pc3"
        axis_labels = ("Length-resid PC1", "Length-resid PC2", "Length-resid PC3")
    elif mode == "attn_resid":
        x_key, y_key, z_key = "attn_resid_pc1", "attn_resid_pc2", "attn_resid_pc3"
        axis_labels = ("Length+Attn-resid PC1", "Length+Attn-resid PC2", "Length+Attn-resid PC3")
    else:
        raise ValueError(f"Unsupported plot mode: {mode}")

    types = [row["type"] for row in rows]
    type_order = [t for t in TYPE_LEGEND_ORDER if t in set(types)]
    colors = qualitative_11_color_map(type_order)

    x_vals = [float(row[x_key]) for row in rows]
    y_vals = [float(row[y_key]) for row in rows]
    z_vals = [float(row[z_key]) for row in rows]

    fig = plt.figure(figsize=(9.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    for prompt_type in type_order:
        idx = [i for i, t in enumerate(types) if t == prompt_type]
        ax.scatter(
            [x_vals[i] for i in idx],
            [y_vals[i] for i in idx],
            [z_vals[i] for i in idx],
            s=18,
            alpha=0.76,
            color=colors[prompt_type],
            linewidths=0,
            label=prettify_type(prompt_type),
            depthshade=False,
        )

    ax.set_xlim(*axis_limits(x_vals))
    ax.set_ylim(*axis_limits(y_vals))
    ax.set_zlim(*axis_limits(z_vals))
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_title(f"{model_label}: {family_name}\n{mode} PCA in 3D", fontsize=12)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        title="type",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 3D PCA scatters from sequence-analysis family CSVs.")
    parser.add_argument("--analysis-dir", required=True, help="Sequence analysis directory containing *_pca.csv files.")
    parser.add_argument("--family", required=True, action="append", help="Family name to plot. Repeat for multiple families.")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <analysis-dir>/3d_plots).")
    parser.add_argument("--model-label", default=None, help="Optional model label for titles.")
    parser.add_argument("--mode", choices=["raw", "length_resid", "attn_resid"], default="raw", help="Which PCA coordinates to plot.")
    parser.add_argument("--elev", type=float, default=24.0, help="3D elevation angle.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D azimuth angle.")
    parser.add_argument("--data-dir", default=None, help=f"Optional base directory override. Also supports {DATA_DIR_ENV_VAR}.")
    args = parser.parse_args()

    configure_data_dir(args.data_dir)
    analysis_dir = resolve_input_path(args.analysis_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (analysis_dir / "3d_plots").resolve()
    ensure_dir(output_dir)

    model_label = args.model_label or analysis_dir.name
    written_paths: list[str] = []
    for family_name in args.family:
        csv_path = analysis_dir / f"{family_name}_pca.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing family PCA CSV: {csv_path}")
        rows = load_csv_rows(csv_path)
        output_path = output_dir / f"{family_name}_3d_{args.mode}.png"
        plot_pca_3d(
            rows=rows,
            family_name=family_name,
            model_label=model_label,
            output_path=output_path,
            mode=args.mode,
            elev=args.elev,
            azim=args.azim,
        )
        written_paths.append(str(output_path))

    print("Wrote 3D PCA plots:")
    for path in written_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
