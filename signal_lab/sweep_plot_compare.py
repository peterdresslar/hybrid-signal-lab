#!/usr/bin/env python3
"""Generate comparison plots from pairwise sweep analysis outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
from signal_lab.paths import DATA_DIR_ENV_VAR, configure_data_dir, resolve_input_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_MIN_FAMILY_POINTS = 50
DEFAULT_PREFIX = "compare"
DEFAULT_DPI = 220
FAMILY_ORDER = [
    "baseline",
    "constant",
    "early_boost",
    "middle_bump",
    "late_boost",
    "early_suppress",
    "late_suppress",
    "early_high_late_low",
    "late_high_early_low",
    "edges_high",
    "edges_low",
    "ramp_up",
    "ramp_down",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pairwise sweep comparison outputs.")
    parser.add_argument(
        "--compare-dir",
        type=str,
        required=True,
        help="Directory containing pairwise comparison CSV outputs.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Optional base directory to use in place of data/. Also supports {DATA_DIR_ENV_VAR}.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Comparison file prefix. Auto-detected from *_files.csv when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated plot files (default: <compare-dir>/plots).",
    )
    parser.add_argument(
        "--min-family-points",
        type=int,
        default=DEFAULT_MIN_FAMILY_POINTS,
        help=f"Minimum prompt rows required to include a family facet (default: {DEFAULT_MIN_FAMILY_POINTS}).",
    )
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=3,
        help="How many outliers to annotate in each panel (default: 3).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output DPI for PNGs (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--intervention-folders",
        action="store_true",
        help="Write per-g_profile comparison folders under the plots directory.",
    )
    parser.add_argument(
        "--best-interventions-top-n",
        type=int,
        default=12,
        help="How many top interventions to index in plots/best_interventions (default: 12).",
    )
    parser.add_argument(
        "--disagreement-top-n",
        type=int,
        default=12,
        help="How many top interventions to index in plots/biggest_model_disagreements (default: 12).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_directory(path_str: str) -> Path:
    path = resolve_input_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")
    return path


def discover_prefix(compare_dir: Path) -> str:
    matches = sorted(compare_dir.glob("*_files.csv"))
    if not matches:
        return DEFAULT_PREFIX
    if len(matches) == 1:
        return matches[0].name.removesuffix("_files.csv")
    raise ValueError(
        "Could not auto-detect a unique prefix in "
        f"{compare_dir}; please pass --prefix explicitly."
    )


def infer_labels(files_rows: list[dict[str, str]]) -> tuple[str, str]:
    by_slot = {row.get("slot"): row for row in files_rows}
    row_a = by_slot.get("a")
    row_b = by_slot.get("b")
    if row_a is None or row_b is None:
        raise ValueError("Expected 'a' and 'b' rows in pairwise files CSV.")
    label_a = row_a.get("label")
    label_b = row_b.get("label")
    if not label_a or not label_b:
        raise ValueError("Expected non-empty labels in pairwise files CSV.")
    return label_a, label_b


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def clean_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned or "plot"


def family_sort_key(family: str) -> tuple[int, str]:
    if family in FAMILY_ORDER:
        return (FAMILY_ORDER.index(family), family)
    return (len(FAMILY_ORDER), family)


def type_order(rows: list[dict[str, Any]]) -> list[str]:
    scored: list[tuple[float, str]] = []
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row["type"]), []).append(row)
    for type_name, type_rows in by_type.items():
        score = sum(abs(row["delta_gap"]) for row in type_rows) / max(len(type_rows), 1)
        scored.append((score, type_name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [type_name for _, type_name in scored]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def family_order(rows: list[dict[str, Any]], min_points: int) -> list[str]:
    counts = Counter(str(row["g_family"]) for row in rows if str(row["g_family"]) != "baseline")
    families = [
        family
        for family, count in counts.items()
        if count >= min_points
    ]
    return sorted(families, key=family_sort_key)


def load_prompt_rows(compare_dir: Path, prefix: str, label_a: str, label_b: str) -> list[dict[str, Any]]:
    prompt_path = compare_dir / f"{prefix}_prompt_pairwise.csv"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing prompt pairwise CSV: {prompt_path}")
    raw_rows = read_csv(prompt_path)
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        parsed: dict[str, Any] = {
            "prompt_id": row.get("prompt_id", ""),
            "g_profile": row.get("g_profile", ""),
            "g_family": row.get("g_family", "other"),
            "rep": row.get("rep", ""),
            "type": row.get("type", "unknown"),
            "tier": row.get("tier", ""),
            "source": row.get("source", ""),
            "target": row.get("target", ""),
            "prompt": row.get("prompt", ""),
            "a_target_prob": to_float(row.get(f"{label_a}__target_prob")),
            "b_target_prob": to_float(row.get(f"{label_b}__target_prob")),
            "a_delta_prob": to_float(row.get(f"{label_a}__delta_target_prob")),
            "b_delta_prob": to_float(row.get(f"{label_b}__delta_target_prob")),
        }
        parsed["delta_gap"] = parsed["a_delta_prob"] - parsed["b_delta_prob"]
        rows.append(parsed)

    baseline_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        if row["g_profile"] == "baseline":
            baseline_lookup[(str(row["prompt_id"]), str(row["rep"]))] = (
                float(row["a_target_prob"]),
                float(row["b_target_prob"]),
            )

    for row in rows:
        baseline_a, baseline_b = baseline_lookup.get(
            (str(row["prompt_id"]), str(row["rep"])),
            (math.nan, math.nan),
        )
        row["a_baseline_prob"] = baseline_a
        row["b_baseline_prob"] = baseline_b
        row["baseline_gap"] = baseline_a - baseline_b
    return rows


def build_color_map(values: list[str], cmap_name: str) -> dict[str, Any]:
    unique_values = list(dict.fromkeys(values))
    cmap = plt.get_cmap(cmap_name, max(len(unique_values), 1))
    return {
        value: cmap(idx % cmap.N)
        for idx, value in enumerate(unique_values)
    }


def finite_rows(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if math.isfinite(float(row[x_key])) and math.isfinite(float(row[y_key]))
    ]


def axis_limits(rows: list[dict[str, Any]], x_key: str, y_key: str, equal: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    clean = finite_rows(rows, x_key, y_key)
    if not clean:
        return ((-1.0, 1.0), (-1.0, 1.0))

    x_values = [float(row[x_key]) for row in clean]
    y_values = [float(row[y_key]) for row in clean]
    if equal:
        max_abs = max(max(abs(value) for value in x_values), max(abs(value) for value in y_values), 0.05)
        pad = max_abs * 0.08
        bound = max_abs + pad
        return ((-bound, bound), (-bound, bound))

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    x_pad = max((x_max - x_min) * 0.08, 0.03)
    y_pad = max((y_max - y_min) * 0.08, 0.03)
    return ((x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad))


def annotate_outliers(
    ax: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    top_n: int,
) -> None:
    if top_n <= 0:
        return
    clean = finite_rows(rows, x_key, y_key)
    ranked = sorted(
        clean,
        key=lambda row: abs(float(row[y_key]) - float(row[x_key])),
        reverse=True,
    )[:top_n]
    for row in ranked:
        ax.annotate(
            str(row["prompt_id"]),
            (float(row[x_key]), float(row[y_key])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
        )


def add_reference_lines(ax: Any, xlim: tuple[float, float], ylim: tuple[float, float], diagonal: bool) -> None:
    ax.axhline(0.0, color="0.65", linewidth=0.9, linestyle="--", zorder=0)
    ax.axvline(0.0, color="0.65", linewidth=0.9, linestyle="--", zorder=0)
    if diagonal:
        low = min(xlim[0], ylim[0])
        high = max(xlim[1], ylim[1])
        ax.plot([low, high], [low, high], color="0.4", linewidth=1.0, linestyle=":", zorder=0)


def scatter_groups(
    ax: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    group_key: str,
    color_map: dict[str, Any],
    size: float = 18.0,
    alpha: float = 0.55,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in finite_rows(rows, x_key, y_key):
        grouped.setdefault(str(row[group_key]), []).append(row)
    for group_name, group_rows in grouped.items():
        ax.scatter(
            [float(row[x_key]) for row in group_rows],
            [float(row[y_key]) for row in group_rows],
            s=size,
            alpha=alpha,
            color=color_map[group_name],
            edgecolors="none",
            label=group_name,
        )


def layout_grid(n_panels: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, math.ceil(math.sqrt(n_panels))))
    rows = math.ceil(n_panels / cols)
    return rows, cols


def make_legend_handles(color_map: dict[str, Any]) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="none", label=label)
        for label, color in color_map.items()
    ]


def make_single_color_handles(label: str, color: Any) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="none", label=label)
    ]


def scatter_single_color(
    ax: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    color: Any,
    size: float = 18.0,
    alpha: float = 0.5,
) -> None:
    clean = finite_rows(rows, x_key, y_key)
    if not clean:
        return
    ax.scatter(
        [float(row[x_key]) for row in clean],
        [float(row[y_key]) for row in clean],
        s=size,
        alpha=alpha,
        color=color,
        edgecolors="none",
    )


def summarize_interventions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["g_profile"] == "baseline":
            continue
        grouped[str(row["g_profile"])].append(row)

    summaries: list[dict[str, Any]] = []
    for g_profile, group_rows in grouped.items():
        a_values = [float(row["a_delta_prob"]) for row in group_rows if math.isfinite(float(row["a_delta_prob"]))]
        b_values = [float(row["b_delta_prob"]) for row in group_rows if math.isfinite(float(row["b_delta_prob"]))]
        gap_values = [float(row["delta_gap"]) for row in group_rows if math.isfinite(float(row["delta_gap"]))]
        baseline_gap_values = [float(row["baseline_gap"]) for row in group_rows if math.isfinite(float(row["baseline_gap"]))]
        summaries.append(
            {
                "g_profile": g_profile,
                "g_family": str(group_rows[0]["g_family"]),
                "n": len(group_rows),
                "mean_a_delta_prob": mean(a_values),
                "mean_b_delta_prob": mean(b_values),
                "mean_joint_delta_prob": mean(a_values + b_values),
                "mean_delta_gap": mean(gap_values),
                "mean_abs_delta_gap": mean([abs(value) for value in gap_values]),
                "median_abs_delta_gap": statistics.median([abs(value) for value in gap_values]) if gap_values else math.nan,
                "mean_baseline_gap": mean(baseline_gap_values),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["value"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_intervention_scatter_all(
    rows: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    g_profile: str,
    output_path: Path,
    type_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] == g_profile]
    if not clean:
        return
    xlim, ylim = axis_limits(clean, "a_delta_prob", "b_delta_prob", equal=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    scatter_groups(ax, clean, "a_delta_prob", "b_delta_prob", "type", type_colors, size=18.0, alpha=0.5)
    add_reference_lines(ax, xlim, ylim, diagonal=True)
    annotate_outliers(ax, clean, "a_delta_prob", "b_delta_prob", top_n)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{label_a} delta target prob")
    ax.set_ylabel(f"{label_b} delta target prob")
    ax.set_title(f"{g_profile}: cross-model delta target prob")
    ax.grid(True, alpha=0.15)
    handles = make_legend_handles(type_colors)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="type")
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_intervention_scatter_by_type(
    rows: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    g_profile: str,
    output_path: Path,
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] == g_profile]
    if not clean:
        return
    ordered_types = type_order(clean)
    xlim, ylim = axis_limits(clean, "a_delta_prob", "b_delta_prob", equal=True)
    n_rows, n_cols = layout_grid(len(ordered_types))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.7 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    point_color = plt.get_cmap("tab20")(0)
    for idx, type_name in enumerate(ordered_types):
        ax = axes.flat[idx]
        ax.set_visible(True)
        type_rows = [row for row in clean if row["type"] == type_name]
        scatter_single_color(ax, type_rows, "a_delta_prob", "b_delta_prob", point_color, size=14.0, alpha=0.5)
        add_reference_lines(ax, xlim, ylim, diagonal=True)
        annotate_outliers(ax, type_rows, "a_delta_prob", "b_delta_prob", top_n)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{type_name} (n={len(type_rows)})")
        ax.grid(True, alpha=0.12)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel(label_a)
        if idx % n_cols == 0:
            ax.set_ylabel(label_b)

    handles = make_single_color_handles(g_profile, point_color)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="g_profile")
    fig.suptitle(f"{g_profile}: cross-model delta target prob by prompt type", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.97))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_intervention_gap_by_type(
    rows: list[dict[str, Any]],
    g_profile: str,
    output_path: Path,
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] == g_profile]
    if not clean:
        return
    ordered_types = type_order(clean)
    xlim, ylim = axis_limits(clean, "baseline_gap", "delta_gap", equal=False)
    n_rows, n_cols = layout_grid(len(ordered_types))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.5 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    point_color = plt.get_cmap("tab20")(2)
    for idx, type_name in enumerate(ordered_types):
        ax = axes.flat[idx]
        ax.set_visible(True)
        type_rows = [row for row in clean if row["type"] == type_name]
        scatter_single_color(ax, type_rows, "baseline_gap", "delta_gap", point_color, size=14.0, alpha=0.5)
        ax.axhline(0.0, color="0.65", linewidth=0.9, linestyle="--", zorder=0)
        ax.axvline(0.0, color="0.65", linewidth=0.9, linestyle="--", zorder=0)
        annotate_outliers(ax, type_rows, "baseline_gap", "delta_gap", top_n)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{type_name} (n={len(type_rows)})")
        ax.grid(True, alpha=0.12)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Baseline prob gap (A - B)")
        if idx % n_cols == 0:
            ax.set_ylabel("Delta prob gap (A - B)")

    handles = make_single_color_handles(g_profile, point_color)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="g_profile")
    fig.suptitle(f"{g_profile}: baseline gap vs delta gap by prompt type", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.97))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_intervention_folder(
    rows: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    g_profile: str,
    parent_dir: Path,
    type_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> dict[str, Any]:
    folder = parent_dir / clean_filename(g_profile)
    folder.mkdir(parents=True, exist_ok=True)

    scatter_all_path = folder / "scatter_delta_prob_all.png"
    plot_intervention_scatter_all(rows, label_a, label_b, g_profile, scatter_all_path, type_colors, top_n, dpi)

    scatter_by_type_path = folder / "scatter_delta_prob_by_type.png"
    plot_intervention_scatter_by_type(rows, label_a, label_b, g_profile, scatter_by_type_path, top_n, dpi)

    gap_by_type_path = folder / "scatter_baseline_gap_vs_delta_gap_by_type.png"
    plot_intervention_gap_by_type(rows, g_profile, gap_by_type_path, top_n, dpi)

    summary = next(item for item in summarize_interventions(rows) if item["g_profile"] == g_profile)
    written_plots = [
        str(scatter_all_path),
        str(scatter_by_type_path),
        str(gap_by_type_path),
    ]
    summary_path = folder / "summary.json"
    write_manifest(
        summary_path,
        {
            "label_a": label_a,
            "label_b": label_b,
            "g_profile": g_profile,
            "g_family": summary["g_family"],
            "n": summary["n"],
            "mean_a_delta_prob": summary["mean_a_delta_prob"],
            "mean_b_delta_prob": summary["mean_b_delta_prob"],
            "mean_joint_delta_prob": summary["mean_joint_delta_prob"],
            "mean_delta_gap": summary["mean_delta_gap"],
            "mean_abs_delta_gap": summary["mean_abs_delta_gap"],
            "plots": written_plots,
        },
    )
    written_plots.append(str(summary_path))
    return {
        **summary,
        "folder": str(folder),
        "written_plots": written_plots,
    }


def plot_scout_entropy_alignment(
    scout_data: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """Scatter plot of mean scout entropy: model A vs model B, colored by type."""
    points = scout_data.get("prompt_scout_summary", [])
    if not points:
        return
    label_a = scout_data.get("label_a", "A")
    label_b = scout_data.get("label_b", "B")
    r_val = scout_data["scout_entropy_alignment"]["r"]

    x = [pt["mean_scout_entropy_a"] for pt in points]
    y = [pt["mean_scout_entropy_b"] for pt in points]

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.scatter(x, y, s=16, alpha=0.5, color="steelblue", edgecolors="none")
    ax.set_xlabel(f"{label_a} mean scout entropy")
    ax.set_ylabel(f"{label_b} mean scout entropy")
    ax.set_title(
        f"Cross-model scout entropy alignment\n"
        f"r = {r_val:.4f} (top-{scout_data.get('top_k', '?')} scouts, {len(points)} prompts)"
    )
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_cross_model_prediction_summary(
    scout_data: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """Bar chart showing cross-model scout prediction strength per gain profile."""
    profiles = scout_data.get("profile_results", [])
    if not profiles:
        return
    label_a = scout_data.get("label_a", "A")
    label_b = scout_data.get("label_b", "B")

    # Sort by delta agreement.
    profiles = sorted(profiles, key=lambda p: p["r_delta_agreement"], reverse=True)
    names = [p["g_profile"] for p in profiles]
    agree = [p["r_delta_agreement"] for p in profiles]
    a_to_b = [p["r_a_scouts_predict_b_delta"] for p in profiles]
    b_to_a = [p["r_b_scouts_predict_a_delta"] for p in profiles]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14.0, 6.0))
    ax.bar(x - width, agree, width, label="Δp agreement (A vs B)", color="steelblue", alpha=0.8)
    ax.bar(x, a_to_b, width, label=f"{label_a} scouts → {label_b} Δp", color="coral", alpha=0.8)
    ax.bar(x + width, b_to_a, width, label=f"{label_b} scouts → {label_a} Δp", color="seagreen", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r")
    ax.set_title("Cross-model scout prediction by gain profile")
    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.12, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_scout_entropy_vs_delta(
    scout_data: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    """Scatter: mean scout entropy (A) vs mean delta_p (A and B), showing cross-model alignment."""
    points = scout_data.get("prompt_scout_summary", [])
    if not points:
        return
    label_a = scout_data.get("label_a", "A")
    label_b = scout_data.get("label_b", "B")

    se_a = [pt["mean_scout_entropy_a"] for pt in points]
    dp_a = [pt["mean_delta_p_a"] for pt in points]
    dp_b = [pt["mean_delta_p_b"] for pt in points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)

    ax1.scatter(se_a, dp_a, s=16, alpha=0.5, color="coral", edgecolors="none")
    ax1.set_xlabel(f"{label_a} mean scout entropy")
    ax1.set_ylabel("Mean Δp (across profiles)")
    ax1.set_title(f"{label_a} scout entropy → {label_a} Δp")
    ax1.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax1.grid(True, alpha=0.15)

    ax2.scatter(se_a, dp_b, s=16, alpha=0.5, color="seagreen", edgecolors="none")
    ax2.set_xlabel(f"{label_a} mean scout entropy")
    ax2.set_title(f"{label_a} scout entropy → {label_b} Δp")
    ax2.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax2.grid(True, alpha=0.15)

    fig.suptitle("Cross-model scout entropy predicting intervention response", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_cross_model_scout_plots(
    scout_data: dict[str, Any],
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> list[str]:
    """Generate all cross-model scout alignment plots."""
    written: list[str] = []

    alignment_path = output_dir / f"{clean_filename(prefix)}_scout_entropy_alignment.png"
    plot_scout_entropy_alignment(scout_data, alignment_path, dpi)
    if alignment_path.exists():
        written.append(str(alignment_path))

    prediction_path = output_dir / f"{clean_filename(prefix)}_cross_model_prediction_summary.png"
    plot_cross_model_prediction_summary(scout_data, prediction_path, dpi)
    if prediction_path.exists():
        written.append(str(prediction_path))

    entropy_delta_path = output_dir / f"{clean_filename(prefix)}_scout_entropy_vs_delta.png"
    plot_scout_entropy_vs_delta(scout_data, entropy_delta_path, dpi)
    if entropy_delta_path.exists():
        written.append(str(entropy_delta_path))

    return written


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    configure_data_dir(args.data_dir)
    compare_dir = resolve_directory(args.compare_dir)
    prefix = args.prefix or discover_prefix(compare_dir)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else compare_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    files_path = compare_dir / f"{prefix}_files.csv"
    if not files_path.is_file():
        raise FileNotFoundError(f"Missing pairwise files CSV: {files_path}")
    files_rows = read_csv(files_path)
    label_a, label_b = infer_labels(files_rows)

    prompt_rows = load_prompt_rows(compare_dir, prefix, label_a, label_b)
    nonbaseline_rows = [row for row in prompt_rows if row["g_profile"] != "baseline"]
    intervention_summaries = summarize_interventions(prompt_rows)
    family_colors = build_color_map(
        sorted({str(row["g_family"]) for row in nonbaseline_rows}, key=family_sort_key),
        "tab20",
    )
    type_colors = build_color_map(type_order(nonbaseline_rows), "tab10")

    written_paths: list[str] = []

    manifest_path = output_dir / f"{clean_filename(prefix)}_plot_manifest.json"

    if args.intervention_folders:
        interventions_dir = output_dir / "interventions"
        best_dir = output_dir / "best_interventions"
        disagreement_dir = output_dir / "biggest_model_disagreements"

        intervention_rows: list[dict[str, Any]] = []
        for summary in sorted(intervention_summaries, key=lambda row: str(row["g_profile"])):
            intervention_rows.append(
                write_intervention_folder(
                    rows=prompt_rows,
                    label_a=label_a,
                    label_b=label_b,
                    g_profile=str(summary["g_profile"]),
                    parent_dir=interventions_dir,
                    type_colors=type_colors,
                    top_n=args.label_top_n,
                    dpi=args.dpi,
                )
            )

        strongest = sorted(
            intervention_rows,
            key=lambda row: (
                -float(row["mean_joint_delta_prob"]) if math.isfinite(float(row["mean_joint_delta_prob"])) else math.inf,
                str(row["g_profile"]),
            ),
        )[: max(args.best_interventions_top_n, 0)]
        strongest_rows = [
            {
                "rank": idx + 1,
                "g_profile": row["g_profile"],
                "g_family": row["g_family"],
                "n": row["n"],
                "mean_a_delta_prob": row["mean_a_delta_prob"],
                "mean_b_delta_prob": row["mean_b_delta_prob"],
                "mean_joint_delta_prob": row["mean_joint_delta_prob"],
                "mean_abs_delta_gap": row["mean_abs_delta_gap"],
                "folder": row["folder"],
            }
            for idx, row in enumerate(strongest)
        ]
        best_csv = best_dir / "best_interventions.csv"
        write_csv(best_csv, strongest_rows)
        written_paths.append(str(best_csv))
        best_manifest = best_dir / "manifest.json"
        write_manifest(
            best_manifest,
            {
                "label_a": label_a,
                "label_b": label_b,
                "best_interventions_top_n": args.best_interventions_top_n,
                "interventions": strongest_rows,
            },
        )
        written_paths.append(str(best_manifest))

        disagreements = sorted(
            intervention_rows,
            key=lambda row: (
                -float(row["mean_abs_delta_gap"]) if math.isfinite(float(row["mean_abs_delta_gap"])) else math.inf,
                str(row["g_profile"]),
            ),
        )[: max(args.disagreement_top_n, 0)]
        disagreement_rows = [
            {
                "rank": idx + 1,
                "g_profile": row["g_profile"],
                "g_family": row["g_family"],
                "n": row["n"],
                "mean_delta_gap": row["mean_delta_gap"],
                "mean_abs_delta_gap": row["mean_abs_delta_gap"],
                "median_abs_delta_gap": row["median_abs_delta_gap"],
                "mean_a_delta_prob": row["mean_a_delta_prob"],
                "mean_b_delta_prob": row["mean_b_delta_prob"],
                "folder": row["folder"],
            }
            for idx, row in enumerate(disagreements)
        ]
        disagreement_csv = disagreement_dir / "biggest_model_disagreements.csv"
        write_csv(disagreement_csv, disagreement_rows)
        written_paths.append(str(disagreement_csv))
        disagreement_manifest = disagreement_dir / "manifest.json"
        write_manifest(
            disagreement_manifest,
            {
                "label_a": label_a,
                "label_b": label_b,
                "disagreement_top_n": args.disagreement_top_n,
                "interventions": disagreement_rows,
            },
        )
        written_paths.append(str(disagreement_manifest))

        interventions_manifest = interventions_dir / "manifest.json"
        write_manifest(
            interventions_manifest,
            {
                "label_a": label_a,
                "label_b": label_b,
                "intervention_count": len(intervention_rows),
                "interventions": [
                    {
                        "g_profile": row["g_profile"],
                        "g_family": row["g_family"],
                        "mean_joint_delta_prob": row["mean_joint_delta_prob"],
                        "mean_abs_delta_gap": row["mean_abs_delta_gap"],
                        "folder": row["folder"],
                    }
                    for row in intervention_rows
                ],
            },
        )
        written_paths.append(str(interventions_manifest))

    # Cross-model scout alignment plots.
    scout_json_path = compare_dir / f"{prefix}_cross_model_scouts.json"
    if scout_json_path.is_file():
        with open(scout_json_path, "r", encoding="utf-8") as f:
            scout_data = json.load(f)
        scout_plots = write_cross_model_scout_plots(
            scout_data=scout_data,
            output_dir=output_dir,
            prefix=prefix,
            dpi=args.dpi,
        )
        written_paths.extend(scout_plots)
    else:
        print(f"(No cross-model scout data found at {scout_json_path}; skipping scout plots.)")

    write_manifest(
        manifest_path,
        {
            "compare_dir": str(compare_dir),
            "prefix": prefix,
            "label_a": label_a,
            "label_b": label_b,
            "plots": written_paths,
        },
    )
    written_paths.append(str(manifest_path))

    print(f"Pairwise comparison directory: {compare_dir}")
    print(f"Prefix: {prefix}")
    print(f"Model A label: {label_a}")
    print(f"Model B label: {label_b}")
    print("Wrote plots:")
    for path_str in written_paths:
        print(f"- {path_str}")


if __name__ == "__main__":
    main()
