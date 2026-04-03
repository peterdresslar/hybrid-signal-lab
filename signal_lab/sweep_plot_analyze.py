#!/usr/bin/env python3
"""Generate single-run scatter plots from sweep analysis outputs."""

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


DEFAULT_PREFIX = "analysis"
DEFAULT_DPI = 220
DEFAULT_MIN_FAMILY_POINTS = 50
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
X_AXIS_CHOICES = [
    "tokens_approx",
    "baseline_target_prob",
    "baseline_target_geo_mean_prob",
    "baseline_final_entropy_bits",
    "baseline_mean_entropy_bits",
    "baseline_top1_top2_logit_margin",
    "baseline_attn_entropy_mean",
    "target_prob",
    "baseline_target_rank",
    "target_rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot single-run sweep analysis outputs.")
    parser.add_argument(
        "--analysis-dir",
        type=str,
        required=True,
        help="Directory containing analysis outputs from signal_lab.sweep_analyze.",
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
        help="Analysis file prefix. Auto-detected from *_files.csv when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated plot files (default: <analysis-dir>/plots).",
    )
    parser.add_argument(
        "--x-metric",
        type=str,
        default="tokens_approx",
        choices=X_AXIS_CHOICES,
        help="Primary x-axis metric for by-type scatter plots.",
    )
    parser.add_argument(
        "--x-metrics",
        nargs="+",
        choices=X_AXIS_CHOICES,
        default=None,
        help="Optional list of x-axis metrics to render as a batch.",
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
    intervention_group = parser.add_mutually_exclusive_group()
    intervention_group.add_argument(
        "--intervention-folders",
        dest="intervention_folders",
        action="store_true",
        help=(
            "Write intervention-folder plot bundles (interventions/, baseline/, "
            "best_interventions/). Enabled by default."
        ),
    )
    intervention_group.add_argument(
        "--no-intervention-folders",
        dest="intervention_folders",
        action="store_false",
        help="Disable intervention-folder plot bundles.",
    )
    parser.set_defaults(intervention_folders=True)
    parser.add_argument(
        "--best-interventions-top-n",
        type=int,
        default=12,
        help="How many top interventions to include in plots/best_interventions (default: 12).",
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


def discover_prefix(analysis_dir: Path) -> str:
    matches = sorted(analysis_dir.glob("*_files.csv"))
    if not matches:
        return DEFAULT_PREFIX
    if len(matches) == 1:
        return matches[0].name.removesuffix("_files.csv")
    raise ValueError(
        "Could not auto-detect a unique prefix in "
        f"{analysis_dir}; please pass --prefix explicitly."
    )


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


def build_color_map(values: list[str], cmap_name: str) -> dict[str, Any]:
    unique_values = list(dict.fromkeys(values))
    cmap = plt.get_cmap(cmap_name, max(len(unique_values), 1))
    return {value: cmap(idx % cmap.N) for idx, value in enumerate(unique_values)}


def infer_model_label(files_rows: list[dict[str, str]], fallback_rows: list[dict[str, Any]]) -> str:
    for row in files_rows:
        model_name = row.get("model")
        if model_name:
            return model_name
    for row in fallback_rows:
        model_name = row.get("model")
        if model_name:
            return str(model_name)
    return "model"


def load_rows(analysis_dir: Path, prefix: str) -> tuple[list[dict[str, Any]], str]:
    files_path = analysis_dir / f"{prefix}_files.csv"
    joined_path = analysis_dir / f"{prefix}_joined_long.csv"
    if not files_path.is_file():
        raise FileNotFoundError(f"Missing analysis files CSV: {files_path}")
    if not joined_path.is_file():
        raise FileNotFoundError(f"Missing joined analysis CSV: {joined_path}")

    files_rows = read_csv(files_path)
    raw_rows = read_csv(joined_path)
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
            "prompt": row.get("prompt", ""),
            "target": row.get("target", ""),
            "tokens_approx": to_float(row.get("tokens_approx")),
            "target_prob": to_float(row.get("target_prob")),
            "baseline_target_prob": to_float(row.get("baseline_target_prob")),
            "baseline_target_geo_mean_prob": to_float(row.get("baseline_target_geo_mean_prob")),
            "baseline_final_entropy_bits": to_float(row.get("baseline_final_entropy_bits")),
            "baseline_mean_entropy_bits": to_float(row.get("baseline_mean_entropy_bits")),
            "baseline_top1_top2_logit_margin": to_float(row.get("baseline_top1_top2_logit_margin")),
            "baseline_attn_entropy_mean": to_float(row.get("baseline_attn_entropy_mean")),
            "delta_target_prob": to_float(row.get("delta_target_prob")),
            "target_rank": to_float(row.get("target_rank")),
            "baseline_target_rank": to_float(row.get("baseline_target_rank")),
            "delta_target_rank": to_float(row.get("delta_target_rank")),
            "final_entropy_bits": to_float(row.get("final_entropy_bits")),
            "delta_final_entropy_bits": to_float(row.get("delta_final_entropy_bits")),
            "model": row.get("model", ""),
        }
        rows.append(parsed)
    model_label = infer_model_label(files_rows, rows)
    return rows, model_label


def load_overall_profile_summary(analysis_dir: Path, prefix: str) -> list[dict[str, Any]]:
    summary_path = analysis_dir / f"{prefix}_overall_profile_summary.csv"
    if not summary_path.is_file():
        return []

    raw_rows = read_csv(summary_path)
    parsed_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        parsed_rows.append(
            {
                "g_profile": row.get("g_profile", ""),
                "top_8_mean_delta_target_prob": to_float(row.get("top_8_mean_delta_target_prob")),
                "positive_prompt_count": to_float(row.get("positive_prompt_count")),
                "positive_mass_delta_target_prob": to_float(row.get("positive_mass_delta_target_prob")),
                "rank_top_8_mean_delta_target_prob": to_float(row.get("rank_top_8_mean_delta_target_prob")),
            }
        )
    return parsed_rows


def top_positive_cluster_profiles(
    overall_profile_summary: list[dict[str, Any]],
    top_n: int = 1,
) -> list[dict[str, Any]]:
    ranked = [
        row
        for row in overall_profile_summary
        if row.get("g_profile")
        and math.isfinite(float(row.get("rank_top_8_mean_delta_target_prob", math.nan)))
        and math.isfinite(float(row.get("top_8_mean_delta_target_prob", math.nan)))
        and float(row.get("top_8_mean_delta_target_prob", math.nan)) > 0.0
    ]
    ranked.sort(
        key=lambda row: (
            float(row.get("rank_top_8_mean_delta_target_prob", math.inf)),
            str(row.get("g_profile", "")),
        )
    )
    return ranked[: max(top_n, 0)]


def finite_rows(rows: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if math.isfinite(float(row[x_key])) and math.isfinite(float(row[y_key]))
    ]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def type_order(rows: list[dict[str, Any]]) -> list[str]:
    scored: list[tuple[float, str]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["type"]), []).append(row)
    for type_name, type_rows in grouped.items():
        score = sum(abs(float(row["delta_target_prob"])) for row in type_rows) / max(len(type_rows), 1)
        scored.append((score, type_name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [type_name for _, type_name in scored]


def family_order(rows: list[dict[str, Any]], min_points: int) -> list[str]:
    counts = Counter(str(row["g_family"]) for row in rows if str(row["g_profile"]) != "baseline")
    families = [family for family, count in counts.items() if count >= min_points]
    return sorted(families, key=family_sort_key)


def axis_limits(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[tuple[float, float], tuple[float, float]]:
    clean = finite_rows(rows, x_key, y_key)
    if not clean:
        return ((-1.0, 1.0), (-1.0, 1.0))
    x_values = [float(row[x_key]) for row in clean]
    y_values = [float(row[y_key]) for row in clean]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    x_pad = max((x_max - x_min) * 0.08, 0.03)
    y_pad = max((y_max - y_min) * 0.08, 0.03)
    return ((x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad))


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


def scatter_groups(
    ax: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    group_key: str,
    color_map: dict[str, Any],
    size: float = 16.0,
    alpha: float = 0.45,
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


def scatter_single_color(
    ax: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    color: Any,
    size: float = 16.0,
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
        key=lambda row: abs(float(row[y_key])),
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


def x_axis_label(metric: str) -> str:
    labels = {
        "tokens_approx": "Prompt tokens (approx)",
        "baseline_target_prob": "Baseline target prob",
        "baseline_target_geo_mean_prob": "Baseline target geo-mean prob",
        "baseline_final_entropy_bits": "Baseline final entropy (bits)",
        "baseline_mean_entropy_bits": "Baseline mean entropy (bits)",
        "baseline_top1_top2_logit_margin": "Baseline top-1 vs top-2 logit margin",
        "baseline_attn_entropy_mean": "Baseline attention entropy mean",
        "target_prob": "Observed target prob",
        "baseline_target_rank": "Baseline target rank",
        "target_rank": "Observed target rank",
    }
    return labels.get(metric, metric)


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def metric_has_finite_values(rows: list[dict[str, Any]], metric: str) -> bool:
    return any(math.isfinite(float(row.get(metric, math.nan))) for row in rows)


def recommended_x_metrics(rows: list[dict[str, Any]], primary_metric: str) -> list[str]:
    preferred = [
        primary_metric,
        "tokens_approx",
        "baseline_target_prob",
        "baseline_final_entropy_bits",
        "baseline_top1_top2_logit_margin",
        "baseline_target_geo_mean_prob",
        "baseline_attn_entropy_mean",
        "baseline_mean_entropy_bits",
    ]
    return [
        metric
        for metric in unique_in_order(preferred)
        if metric in X_AXIS_CHOICES and metric_has_finite_values(rows, metric)
    ]


def add_zero_line(ax: Any) -> None:
    ax.axhline(0.0, color="0.65", linewidth=0.9, linestyle="--", zorder=0)


def plot_delta_by_type(
    rows: list[dict[str, Any]],
    model_label: str,
    x_key: str,
    output_path: Path,
    family_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] != "baseline"]
    ordered_types = type_order(clean)
    xlim, ylim = axis_limits(clean, x_key, "delta_target_prob")
    n_rows, n_cols = layout_grid(len(ordered_types))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.6 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, type_name in enumerate(ordered_types):
        ax = axes.flat[idx]
        ax.set_visible(True)
        type_rows = [row for row in clean if row["type"] == type_name]
        scatter_groups(ax, type_rows, x_key, "delta_target_prob", "g_family", family_colors, size=14.0, alpha=0.45)
        add_zero_line(ax)
        annotate_outliers(ax, type_rows, x_key, "delta_target_prob", top_n)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{type_name} (n={len(type_rows)})")
        ax.grid(True, alpha=0.12)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel(x_axis_label(x_key))
        if idx % n_cols == 0:
            ax.set_ylabel("Delta target prob")

    handles = make_legend_handles(family_colors)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="g_family")
    fig.suptitle(f"{model_label}: delta target prob by prompt type", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.97))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_delta_all(
    rows: list[dict[str, Any]],
    model_label: str,
    x_key: str,
    output_path: Path,
    family_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] != "baseline"]
    xlim, ylim = axis_limits(clean, x_key, "delta_target_prob")
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    scatter_groups(ax, clean, x_key, "delta_target_prob", "g_family", family_colors, size=16.0, alpha=0.45)
    add_zero_line(ax)
    annotate_outliers(ax, clean, x_key, "delta_target_prob", top_n)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(x_axis_label(x_key))
    ax.set_ylabel("Delta target prob")
    ax.set_title(f"{model_label}: delta target prob across all prompts")
    ax.grid(True, alpha=0.15)
    handles = make_legend_handles(family_colors)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="g_family")
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_delta_by_family(
    rows: list[dict[str, Any]],
    model_label: str,
    x_key: str,
    output_path: Path,
    type_colors: dict[str, Any],
    min_points: int,
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] != "baseline"]
    ordered_families = family_order(clean, min_points)
    if not ordered_families:
        return
    xlim, ylim = axis_limits(clean, x_key, "delta_target_prob")
    n_rows, n_cols = layout_grid(len(ordered_families), max_cols=4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, family_name in enumerate(ordered_families):
        ax = axes.flat[idx]
        ax.set_visible(True)
        family_rows = [row for row in clean if row["g_family"] == family_name]
        scatter_groups(ax, family_rows, x_key, "delta_target_prob", "type", type_colors, size=14.0, alpha=0.45)
        add_zero_line(ax)
        annotate_outliers(ax, family_rows, x_key, "delta_target_prob", top_n)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{family_name} (n={len(family_rows)})")
        ax.grid(True, alpha=0.12)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel(x_axis_label(x_key))
        if idx % n_cols == 0:
            ax.set_ylabel("Delta target prob")

    handles = make_legend_handles(type_colors)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="type")
    fig.suptitle(f"{model_label}: delta target prob by gain family", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.97))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_baseline_vs_delta_by_type(
    rows: list[dict[str, Any]],
    model_label: str,
    output_path: Path,
    family_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] != "baseline"]
    ordered_types = type_order(clean)
    xlim, ylim = axis_limits(clean, "baseline_target_prob", "delta_target_prob")
    n_rows, n_cols = layout_grid(len(ordered_types))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.6 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for idx, type_name in enumerate(ordered_types):
        ax = axes.flat[idx]
        ax.set_visible(True)
        type_rows = [row for row in clean if row["type"] == type_name]
        scatter_groups(ax, type_rows, "baseline_target_prob", "delta_target_prob", "g_family", family_colors, size=14.0, alpha=0.45)
        add_zero_line(ax)
        annotate_outliers(ax, type_rows, "baseline_target_prob", "delta_target_prob", top_n)
        ax.set_xlim(max(-0.02, xlim[0]), min(1.02, xlim[1]))
        ax.set_ylim(*ylim)
        ax.set_title(f"{type_name} (n={len(type_rows)})")
        ax.grid(True, alpha=0.12)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Baseline target prob")
        if idx % n_cols == 0:
            ax.set_ylabel("Delta target prob")

    handles = make_legend_handles(family_colors)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="g_family")
    fig.suptitle(f"{model_label}: baseline target prob vs delta target prob", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.97))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_baseline_prob_all(
    rows: list[dict[str, Any]],
    model_label: str,
    x_key: str,
    output_path: Path,
    type_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> None:
    xlim, ylim = axis_limits(rows, x_key, "baseline_target_prob")
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    scatter_groups(ax, rows, x_key, "baseline_target_prob", "type", type_colors, size=16.0, alpha=0.45)
    annotate_outliers(ax, rows, x_key, "baseline_target_prob", top_n)
    ax.set_xlim(*xlim)
    ax.set_ylim(max(-0.02, ylim[0]), min(1.02, ylim[1]))
    ax.set_xlabel(x_axis_label(x_key))
    ax.set_ylabel("Baseline target prob")
    ax.set_title(f"{model_label}: baseline target prob across all prompts")
    ax.grid(True, alpha=0.15)
    handles = make_legend_handles(type_colors)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="type")
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_baseline_prob_by_type(
    rows: list[dict[str, Any]],
    model_label: str,
    x_key: str,
    output_path: Path,
    family_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> None:
    baseline_rows = [row for row in rows if row["g_profile"] == "baseline"]
    ordered_types = type_order(rows)
    xlim, ylim = axis_limits(baseline_rows, x_key, "baseline_target_prob")
    n_rows, n_cols = layout_grid(len(ordered_types))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.6 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    baseline_color = family_colors.get("baseline", plt.get_cmap("tab20")(0))
    for idx, type_name in enumerate(ordered_types):
        ax = axes.flat[idx]
        ax.set_visible(True)
        type_rows = [row for row in baseline_rows if row["type"] == type_name]
        scatter_single_color(ax, type_rows, x_key, "baseline_target_prob", baseline_color, size=16.0, alpha=0.5)
        annotate_outliers(ax, type_rows, x_key, "baseline_target_prob", top_n)
        ax.set_xlim(*xlim)
        ax.set_ylim(max(-0.02, ylim[0]), min(1.02, ylim[1]))
        ax.set_title(f"{type_name} (n={len(type_rows)})")
        ax.grid(True, alpha=0.12)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel(x_axis_label(x_key))
        if idx % n_cols == 0:
            ax.set_ylabel("Baseline target prob")

    handles = make_single_color_handles("baseline", baseline_color)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="g_profile")
    fig.suptitle(f"{model_label}: baseline target prob by prompt type", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.97))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_baseline_folder(
    rows: list[dict[str, Any]],
    model_label: str,
    x_metrics: list[str],
    parent_dir: Path,
    family_colors: dict[str, Any],
    type_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> list[str]:
    baseline_rows = [row for row in rows if row["g_profile"] == "baseline"]
    if not baseline_rows:
        return []

    folder = parent_dir / "baseline"
    folder.mkdir(parents=True, exist_ok=True)
    written_plots: list[str] = []
    for x_metric in x_metrics:
        all_path = folder / f"scatter_baseline_target_prob_all__x-{clean_filename(x_metric)}.png"
        plot_baseline_prob_all(baseline_rows, model_label, x_metric, all_path, type_colors, top_n, dpi)
        written_plots.append(str(all_path))

        by_type_path = folder / f"scatter_baseline_target_prob_by_type__x-{clean_filename(x_metric)}.png"
        plot_baseline_prob_by_type(rows, model_label, x_metric, by_type_path, family_colors, top_n, dpi)
        written_plots.append(str(by_type_path))

    write_manifest(
        folder / "summary.json",
        {
            "model": model_label,
            "g_profile": "baseline",
            "n": len(baseline_rows),
            "plots": written_plots,
        },
    )
    written_plots.append(str(folder / "summary.json"))
    return written_plots


def summarize_interventions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["g_profile"] == "baseline":
            continue
        grouped[str(row["g_profile"])].append(row)

    summaries: list[dict[str, Any]] = []
    for g_profile, group_rows in grouped.items():
        delta_values = [
            float(row["delta_target_prob"])
            for row in group_rows
            if math.isfinite(float(row["delta_target_prob"]))
        ]
        baseline_values = [
            float(row["baseline_target_prob"])
            for row in group_rows
            if math.isfinite(float(row["baseline_target_prob"]))
        ]
        summaries.append(
            {
                "g_profile": g_profile,
                "g_family": str(group_rows[0]["g_family"]),
                "n": len(group_rows),
                "mean_delta_target_prob": mean(delta_values),
                "median_delta_target_prob": statistics.median(delta_values) if delta_values else math.nan,
                "pct_delta_target_prob_positive": (
                    100.0 * sum(value > 0 for value in delta_values) / len(delta_values)
                    if delta_values
                    else math.nan
                ),
                "mean_baseline_target_prob": mean(baseline_values),
            }
        )
    summaries.sort(
        key=lambda row: (
            -float(row["mean_delta_target_prob"]) if math.isfinite(float(row["mean_delta_target_prob"])) else math.inf,
            str(row["g_profile"]),
        )
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


def plot_intervention_by_type(
    rows: list[dict[str, Any]],
    model_label: str,
    g_profile: str,
    x_key: str,
    output_path: Path,
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] == g_profile]
    if not clean:
        return
    ordered_types = type_order(clean)
    xlim, ylim = axis_limits(clean, x_key, "delta_target_prob")
    n_rows, n_cols = layout_grid(len(ordered_types))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.6 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    point_color = plt.get_cmap("tab20")(0)
    for idx, type_name in enumerate(ordered_types):
        ax = axes.flat[idx]
        ax.set_visible(True)
        type_rows = [row for row in clean if row["type"] == type_name]
        scatter_single_color(ax, type_rows, x_key, "delta_target_prob", point_color, size=16.0, alpha=0.5)
        add_zero_line(ax)
        annotate_outliers(ax, type_rows, x_key, "delta_target_prob", top_n)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{type_name} (n={len(type_rows)})")
        ax.grid(True, alpha=0.12)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel(x_axis_label(x_key))
        if idx % n_cols == 0:
            ax.set_ylabel("Delta target prob")

    handles = make_single_color_handles(g_profile, point_color)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="g_profile")
    fig.suptitle(f"{model_label}: {g_profile} by prompt type", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.97))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_intervention_all(
    rows: list[dict[str, Any]],
    model_label: str,
    g_profile: str,
    x_key: str,
    output_path: Path,
    type_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> None:
    clean = [row for row in rows if row["g_profile"] == g_profile]
    if not clean:
        return
    xlim, ylim = axis_limits(clean, x_key, "delta_target_prob")
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    scatter_groups(ax, clean, x_key, "delta_target_prob", "type", type_colors, size=18.0, alpha=0.5)
    add_zero_line(ax)
    annotate_outliers(ax, clean, x_key, "delta_target_prob", top_n)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(x_axis_label(x_key))
    ax.set_ylabel("Delta target prob")
    ax.set_title(f"{model_label}: {g_profile} across all prompts")
    ax.grid(True, alpha=0.15)
    handles = make_legend_handles(type_colors)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, title="type")
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_intervention_folder(
    rows: list[dict[str, Any]],
    model_label: str,
    g_profile: str,
    x_metrics: list[str],
    parent_dir: Path,
    type_colors: dict[str, Any],
    top_n: int,
    dpi: int,
) -> dict[str, Any]:
    clean_rows = [row for row in rows if row["g_profile"] == g_profile]
    if not clean_rows:
        return {"g_profile": g_profile, "written_plots": []}

    folder = parent_dir / clean_filename(g_profile)
    folder.mkdir(parents=True, exist_ok=True)
    written_plots: list[str] = []
    for x_metric in x_metrics:
        by_type_path = folder / f"scatter_delta_target_prob_by_type__x-{clean_filename(x_metric)}.png"
        plot_intervention_by_type(clean_rows, model_label, g_profile, x_metric, by_type_path, top_n, dpi)
        written_plots.append(str(by_type_path))

        all_path = folder / f"scatter_delta_target_prob_all__x-{clean_filename(x_metric)}.png"
        plot_intervention_all(clean_rows, model_label, g_profile, x_metric, all_path, type_colors, top_n, dpi)
        written_plots.append(str(all_path))

    summary = summarize_interventions(clean_rows)[0]
    metadata_path = folder / "summary.json"
    write_manifest(
        metadata_path,
        {
            "model": model_label,
            "g_profile": g_profile,
            "g_family": summary["g_family"],
            "n": summary["n"],
            "mean_delta_target_prob": summary["mean_delta_target_prob"],
            "median_delta_target_prob": summary["median_delta_target_prob"],
            "pct_delta_target_prob_positive": summary["pct_delta_target_prob_positive"],
            "plots": written_plots,
        },
    )
    written_plots.append(str(metadata_path))
    return {
        "g_profile": g_profile,
        "g_family": summary["g_family"],
        "n": summary["n"],
        "mean_delta_target_prob": summary["mean_delta_target_prob"],
        "median_delta_target_prob": summary["median_delta_target_prob"],
        "pct_delta_target_prob_positive": summary["pct_delta_target_prob_positive"],
        "folder": str(folder),
        "written_plots": written_plots,
    }


def plot_baseline_attn_pca(
    pca_data: dict[str, Any],
    model_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Render a 2D PCA scatter of baseline per-head attention entropy, colored by type."""
    points = pca_data.get("points", [])
    if not points:
        return

    explained = pca_data.get("explained_variance_ratio", [0.0, 0.0])
    n_layers = pca_data.get("n_layers", "?")
    n_heads = pca_data.get("n_heads", "?")
    n_features = pca_data.get("n_features", "?")

    types_in_order = list(dict.fromkeys(pt["type"] for pt in points))
    type_colors = build_color_map(types_in_order, "tab10")

    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for pt in points:
        grouped.setdefault(pt["type"], []).append(pt)

    for type_name in types_in_order:
        group = grouped[type_name]
        ax.scatter(
            [pt["pc1"] for pt in group],
            [pt["pc2"] for pt in group],
            s=22,
            alpha=0.6,
            color=type_colors[type_name],
            edgecolors="none",
            label=type_name,
        )

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title(
        f"{model_label}: baseline attention entropy PCA\n"
        f"({n_layers} layers × {n_heads} heads = {n_features} features)"
    )
    ax.grid(True, alpha=0.15)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="type",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_pca_intervention(
    pca_data: dict[str, Any],
    rows: list[dict[str, Any]],
    g_profile: str,
    model_label: str,
    output_path: Path,
    dpi: int,
    vmax: float | None = None,
) -> None:
    """Render a PCA scatter with points colored by delta_target_prob for one gain profile."""
    points = pca_data.get("points", [])
    if not points:
        return

    explained = pca_data.get("explained_variance_ratio", [0.0, 0.0])

    # Build lookup: (prompt_id, rep) -> delta_target_prob for this profile.
    delta_lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        if str(row.get("g_profile", "")) != g_profile:
            continue
        pid = str(row.get("prompt_id", ""))
        rep = str(row.get("rep", "1"))
        dp = to_float(row.get("delta_target_prob"))
        if math.isfinite(dp):
            delta_lookup[(pid, rep)] = dp

    # Join PCA coordinates with deltas.
    pc1_vals: list[float] = []
    pc2_vals: list[float] = []
    delta_vals: list[float] = []
    for pt in points:
        key = (pt["prompt_id"], str(pt["rep"]))
        if key not in delta_lookup:
            continue
        pc1_vals.append(pt["pc1"])
        pc2_vals.append(pt["pc2"])
        delta_vals.append(delta_lookup[key])

    if not delta_vals:
        return

    # Symmetric colour limits centred on zero.
    if vmax is None:
        vmax = max(abs(v) for v in delta_vals)
    if vmax < 1e-9:
        vmax = 0.01

    # Signed power transform: sign(d) * |d|^gamma.  gamma < 1 stretches
    # near-zero values away from white so subtle deltas are visible.
    gamma = 0.4
    transformed = [math.copysign(abs(d) ** gamma, d) for d in delta_vals]
    t_vmax = vmax ** gamma

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    sc = ax.scatter(
        pc1_vals,
        pc2_vals,
        c=transformed,
        cmap="RdBu_r",
        vmin=-t_vmax,
        vmax=t_vmax,
        s=24,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    # Relabel colorbar ticks with original (un-transformed) delta values.
    raw_ticks = np.linspace(-t_vmax, t_vmax, 9)
    cbar.set_ticks(raw_ticks)
    cbar.set_ticklabels(
        [f"{math.copysign(abs(t) ** (1.0 / gamma), t):.3f}" for t in raw_ticks]
    )
    cbar.set_label("Δ target prob")

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title(f"{model_label}: {g_profile}\nattention entropy PCA colored by Δ target prob")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_pca_intervention_folder(
    pca_data: dict[str, Any],
    rows: list[dict[str, Any]],
    model_label: str,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> list[str]:
    """Write one PCA-delta plot per gain profile into a subdirectory."""
    points = pca_data.get("points", [])
    if not points:
        return []

    profiles = sorted(
        {str(row["g_profile"]) for row in rows if str(row.get("g_profile", "")) != "baseline"}
    )
    if not profiles:
        return []

    pca_dir = output_dir / "pca_interventions"
    pca_dir.mkdir(parents=True, exist_ok=True)

    # Compute a shared vmax so all plots use the same colour scale.
    all_deltas: list[float] = []
    for row in rows:
        if str(row.get("g_profile", "")) == "baseline":
            continue
        dp = to_float(row.get("delta_target_prob"))
        if math.isfinite(dp):
            all_deltas.append(abs(dp))
    shared_vmax = sorted(all_deltas)[int(len(all_deltas) * 0.98)] if all_deltas else 0.1

    written: list[str] = []
    for g_profile in profiles:
        plot_path = pca_dir / f"{clean_filename(prefix)}_pca_delta__{clean_filename(g_profile)}.png"
        plot_pca_intervention(
            pca_data=pca_data,
            rows=rows,
            g_profile=g_profile,
            model_label=model_label,
            output_path=plot_path,
            dpi=dpi,
            vmax=shared_vmax,
        )
        if plot_path.exists():
            written.append(str(plot_path))

    manifest_path = pca_dir / "manifest.json"
    write_manifest(
        manifest_path,
        {
            "model": model_label,
            "description": "PCA of baseline attention entropy colored by delta_target_prob per gain profile",
            "shared_vmax": round(shared_vmax, 6),
            "profiles": profiles,
            "plots": written,
        },
    )
    written.append(str(manifest_path))
    return written


def plot_head_correlation_heatmap(
    corr_matrix: list[list[float]],
    layer_indices: list[int] | None,
    model_label: str,
    g_profile: str,
    output_path: Path,
    dpi: int,
    vmax: float = 0.4,
    top_heads: list[dict[str, Any]] | None = None,
    title_suffix: str = "",
) -> None:
    """Render a layer×head heatmap of Pearson r between baseline entropy and delta_p."""
    arr = np.array(corr_matrix, dtype=np.float64)
    n_layers, n_heads = arr.shape

    fig_width = max(6.0, n_heads * 0.28 + 2.0)
    fig_height = max(3.0, n_layers * 0.55 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        arr,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Pearson r")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels(range(n_heads), fontsize=6)
    ax.set_xlabel("Head")
    y_labels = [str(layer_indices[i]) if layer_indices else str(i) for i in range(n_layers)]
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylabel("Layer")
    if top_heads:
        for rank, head in enumerate(top_heads, start=1):
            layer_slot = head.get("layer_slot")
            head_idx = head.get("head")
            if not isinstance(layer_slot, int) or not isinstance(head_idx, int):
                continue
            if not (0 <= layer_slot < n_layers and 0 <= head_idx < n_heads):
                continue
            rect = plt.Rectangle(
                (head_idx - 0.5, layer_slot - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.6,
            )
            ax.add_patch(rect)
            ax.text(
                head_idx,
                layer_slot,
                str(rank),
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
    ax.set_title(
        f"{model_label}: {g_profile}\n"
        f"head entropy vs Δ target prob correlation{title_suffix}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_scout_heads_heatmap(
    scout_matrix: list[list[int]],
    layer_indices: list[int] | None,
    model_label: str,
    n_profiles: int,
    output_path: Path,
    dpi: int,
    title_suffix: str = "",
    colorbar_label: str | None = None,
) -> None:
    """Render a layer×head heatmap showing how often each head is in the top-10 across profiles."""
    arr = np.array(scout_matrix, dtype=np.float64)
    n_layers, n_heads = arr.shape

    fig_width = max(6.0, n_heads * 0.28 + 2.0)
    fig_height = max(3.0, n_layers * 0.55 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        arr,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(float(arr.max()), 1.0),
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(colorbar_label or f"Top-10 appearances (of {n_profiles} profiles)")

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels(range(n_heads), fontsize=6)
    ax.set_xlabel("Head")
    y_labels = [str(layer_indices[i]) if layer_indices else str(i) for i in range(n_layers)]
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylabel("Layer")
    ax.set_title(
        f"{model_label}: scout heads\n"
        f"frequency in top-10 correlated heads across {n_profiles} gain profiles{title_suffix}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_head_correlation_plots(
    head_corr_data: dict[str, Any],
    overall_profile_summary: list[dict[str, Any]],
    model_label: str,
    output_dir: Path,
    prefix: str,
    dpi: int,
) -> list[str]:
    """Write per-profile heatmaps and scout-heads aggregate from head correlation JSON."""
    profiles = head_corr_data.get("profiles", [])
    layer_indices = head_corr_data.get("layer_indices")
    n_profiles = head_corr_data.get("n_profiles", len(profiles))
    n_layers = int(head_corr_data.get("n_layers", 0))
    n_heads = int(head_corr_data.get("n_heads", 0))

    if not profiles:
        return []

    heatmap_dir = output_dir / "head_correlations"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Shared vmax: use the max |r| across all profiles, capped at 0.5 for readability.
    all_r = []
    for p in profiles:
        for row in p["correlation_matrix"]:
            all_r.extend(abs(v) for v in row)
    shared_vmax = min(max(sorted(all_r)[int(len(all_r) * 0.98)] * 1.2, 0.1) if all_r else 0.3, 0.5)

    written: list[str] = []

    for p in profiles:
        g_profile = p["g_profile"]
        plot_path = heatmap_dir / f"{clean_filename(prefix)}_head_corr__{clean_filename(g_profile)}.png"
        plot_head_correlation_heatmap(
            corr_matrix=p["correlation_matrix"],
            layer_indices=layer_indices,
            model_label=model_label,
            g_profile=g_profile,
            output_path=plot_path,
            dpi=dpi,
            vmax=shared_vmax,
        )
        if plot_path.exists():
            written.append(str(plot_path))

    # Scout heads aggregate.
    scout_matrix = head_corr_data.get("scout_heads_matrix")
    if scout_matrix:
        scout_path = output_dir / f"{clean_filename(prefix)}_scout_heads.png"
        plot_scout_heads_heatmap(
            scout_matrix=scout_matrix,
            layer_indices=layer_indices,
            model_label=model_label,
            n_profiles=n_profiles,
            output_path=scout_path,
            dpi=dpi,
        )
        if scout_path.exists():
            written.append(str(scout_path))

    top_cluster_rows = top_positive_cluster_profiles(overall_profile_summary, top_n=1)
    if top_cluster_rows and n_layers > 0 and n_heads > 0:
        selected_names = {str(row["g_profile"]) for row in top_cluster_rows}
        selected_profiles = [p for p in profiles if str(p.get("g_profile", "")) in selected_names]
        if selected_profiles:
            selected_summary = top_cluster_rows[0]
            selected_profile = selected_profiles[0]
            selected_name = str(selected_summary["g_profile"])
            selected_rank = int(float(selected_summary["rank_top_8_mean_delta_target_prob"]))
            selected_top8 = float(selected_summary["top_8_mean_delta_target_prob"])
            selected_path = (
                output_dir
                / f"{clean_filename(prefix)}_head_corr__top_positive_cluster__{clean_filename(selected_name)}.png"
            )
            plot_head_correlation_heatmap(
                corr_matrix=selected_profile["correlation_matrix"],
                layer_indices=layer_indices,
                model_label=model_label,
                g_profile=selected_name,
                output_path=selected_path,
                dpi=dpi,
                vmax=shared_vmax,
                top_heads=selected_profile.get("top_10_heads", []),
                title_suffix=f"\nTop positive-cluster profile #{selected_rank}; boxed cells are top-10 scout candidates (top8={selected_top8:.2f})",
            )
            if selected_path.exists():
                written.append(str(selected_path))

    manifest_path = heatmap_dir / "manifest.json"
    write_manifest(
        manifest_path,
        {
            "model": model_label,
            "description": "Per-head correlation heatmaps: baseline attention entropy vs delta_target_prob",
            "shared_vmax": round(shared_vmax, 6),
            "n_profiles": n_profiles,
            "top_positive_cluster_profiles": top_cluster_rows,
            "plots": written,
        },
    )
    written.append(str(manifest_path))
    return written


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    configure_data_dir(args.data_dir)
    analysis_dir = resolve_directory(args.analysis_dir)
    prefix = args.prefix or discover_prefix(analysis_dir)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else analysis_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, model_label = load_rows(analysis_dir, prefix)
    nonbaseline_rows = [row for row in rows if row["g_profile"] != "baseline"]
    ordered_types = type_order(nonbaseline_rows)
    ordered_families = sorted({str(row["g_family"]) for row in nonbaseline_rows}, key=family_sort_key)
    family_colors = build_color_map(ordered_families, "tab20")
    type_colors = build_color_map(ordered_types, "tab10")
    intervention_summaries = summarize_interventions(rows)
    selected_x_metrics = args.x_metrics or [args.x_metric]
    selected_x_metrics = [
        metric for metric in unique_in_order(selected_x_metrics) if metric_has_finite_values(rows, metric)
    ]
    if not selected_x_metrics:
        raise ValueError("No selected x metrics have finite values in the analysis rows.")

    written_paths: list[str] = []

    for x_metric in selected_x_metrics:
        all_path = output_dir / f"{clean_filename(prefix)}_scatter_delta_target_prob_all__x-{clean_filename(x_metric)}.png"
        plot_delta_all(rows, model_label, x_metric, all_path, family_colors, args.label_top_n, args.dpi)
        written_paths.append(str(all_path))

        by_type_path = output_dir / f"{clean_filename(prefix)}_scatter_delta_target_prob_by_type__x-{clean_filename(x_metric)}.png"
        plot_delta_by_type(rows, model_label, x_metric, by_type_path, family_colors, args.label_top_n, args.dpi)
        written_paths.append(str(by_type_path))

        by_family_path = output_dir / f"{clean_filename(prefix)}_scatter_delta_target_prob_by_family__x-{clean_filename(x_metric)}.png"
        plot_delta_by_family(
            rows,
            model_label,
            x_metric,
            by_family_path,
            type_colors,
            args.min_family_points,
            args.label_top_n,
            args.dpi,
        )
        if by_family_path.exists():
            written_paths.append(str(by_family_path))

    baseline_path = output_dir / f"{clean_filename(prefix)}_scatter_baseline_target_prob_vs_delta_target_prob_by_type.png"
    plot_baseline_vs_delta_by_type(rows, model_label, baseline_path, family_colors, args.label_top_n, args.dpi)
    written_paths.append(str(baseline_path))

    # Second-order plots: baseline attention entropy PCA.
    pca_json_path = analysis_dir / f"{prefix}_baseline_attn_pca.json"
    if pca_json_path.is_file():
        with open(pca_json_path, "r", encoding="utf-8") as f:
            pca_data = json.load(f)
        pca_plot_path = output_dir / f"{clean_filename(prefix)}_baseline_attn_pca.png"
        plot_baseline_attn_pca(pca_data, model_label, pca_plot_path, args.dpi)
        written_paths.append(str(pca_plot_path))

        # PCA intervention atlas: one plot per gain profile.
        pca_intervention_paths = write_pca_intervention_folder(
            pca_data=pca_data,
            rows=rows,
            model_label=model_label,
            output_dir=output_dir,
            prefix=prefix,
            dpi=args.dpi,
        )
        written_paths.extend(pca_intervention_paths)
    else:
        print(f"(No PCA data found at {pca_json_path}; skipping baseline attention PCA plots.)")

    # Head-level correlation heatmaps.
    head_corr_path = analysis_dir / f"{prefix}_head_correlations.json"
    if head_corr_path.is_file():
        with open(head_corr_path, "r", encoding="utf-8") as f:
            head_corr_data = json.load(f)
        overall_profile_summary = load_overall_profile_summary(analysis_dir, prefix)
        head_corr_plots = write_head_correlation_plots(
            head_corr_data=head_corr_data,
            overall_profile_summary=overall_profile_summary,
            model_label=model_label,
            output_dir=output_dir,
            prefix=prefix,
            dpi=args.dpi,
        )
        written_paths.extend(head_corr_plots)
    else:
        print(f"(No head correlation data found at {head_corr_path}; skipping head heatmaps.)")

    manifest_path = output_dir / f"{clean_filename(prefix)}_plot_manifest.json"

    if args.intervention_folders:
        interventions_dir = output_dir / "interventions"
        best_dir = output_dir / "best_interventions"
        raw_x_metrics = args.x_metrics or recommended_x_metrics(rows, args.x_metric)
        x_metrics = [metric for metric in raw_x_metrics if metric_has_finite_values(rows, metric)]
        baseline_x_metrics = [
            metric
            for metric in x_metrics
            if metric != "baseline_target_prob" and metric_has_finite_values(rows, metric)
        ]
        if not baseline_x_metrics:
            baseline_x_metrics = ["tokens_approx"]

        intervention_rows: list[dict[str, Any]] = []
        for summary in intervention_summaries:
            intervention_rows.append(
                write_intervention_folder(
                    rows=rows,
                    model_label=model_label,
                    g_profile=str(summary["g_profile"]),
                    x_metrics=x_metrics,
                    parent_dir=interventions_dir,
                    type_colors=type_colors,
                    top_n=args.label_top_n,
                    dpi=args.dpi,
                )
            )

        top_interventions = intervention_rows[: max(args.best_interventions_top_n, 0)]
        best_rows = [
            {
                "rank": idx + 1,
                "g_profile": row["g_profile"],
                "g_family": row["g_family"],
                "n": row["n"],
                "mean_delta_target_prob": row["mean_delta_target_prob"],
                "median_delta_target_prob": row["median_delta_target_prob"],
                "pct_delta_target_prob_positive": row["pct_delta_target_prob_positive"],
                "folder": row["folder"],
            }
            for idx, row in enumerate(top_interventions)
        ]
        best_summary_path = best_dir / "best_interventions.csv"
        write_csv(best_summary_path, best_rows)
        written_paths.append(str(best_summary_path))
        write_manifest(
            best_dir / "manifest.json",
            {
                "model": model_label,
                "best_interventions_top_n": args.best_interventions_top_n,
                "interventions": best_rows,
            },
        )
        written_paths.append(str(best_dir / "manifest.json"))

        write_manifest(
            interventions_dir / "manifest.json",
            {
                "model": model_label,
                "intervention_count": len(intervention_rows),
                "x_metrics": x_metrics,
                "interventions": [
                    {
                        "g_profile": row["g_profile"],
                        "g_family": row["g_family"],
                        "mean_delta_target_prob": row["mean_delta_target_prob"],
                        "folder": row["folder"],
                    }
                    for row in intervention_rows
                ],
            },
        )
        written_paths.append(str(interventions_dir / "manifest.json"))

        baseline_written = write_baseline_folder(
            rows=rows,
            model_label=model_label,
            x_metrics=baseline_x_metrics,
            parent_dir=output_dir,
            family_colors=family_colors,
            type_colors=type_colors,
            top_n=args.label_top_n,
            dpi=args.dpi,
        )
        written_paths.extend(baseline_written)

    write_manifest(
        manifest_path,
        {
            "analysis_dir": str(analysis_dir),
            "prefix": prefix,
            "model": model_label,
            "x_metric": args.x_metric,
            "x_metrics": selected_x_metrics,
            "plots": written_paths,
        },
    )
    written_paths.append(str(manifest_path))

    print(f"Analysis directory: {analysis_dir}")
    print(f"Prefix: {prefix}")
    print(f"Model: {model_label}")
    print(f"Primary x metric: {args.x_metric}")
    print(f"Rendered x metrics: {', '.join(selected_x_metrics)}")
    print("Wrote plots:")
    for path_str in written_paths:
        print(f"- {path_str}")


if __name__ == "__main__":
    main()
