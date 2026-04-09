from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .common import configure_matplotlib, prettify_type, qualitative_11_color_map


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

ROUTE_ORDER = [
    "edges_narrow",
    "late_boost_4.0",
    "shifted_ramp_down",
    "tent_steep",
    "off",
]

ROUTE_COLORS = {
    "edges_narrow": "#1b9e77",
    "late_boost_4.0": "#d95f02",
    "shifted_ramp_down": "#7570b3",
    "tent_steep": "#e7298a",
    "off": "#9E9E9E",
}


def load_pca_points(pca_path: Path) -> tuple[list[dict[str, float | str]], tuple[float, float]]:
    obj = json.loads(pca_path.read_text())
    points = obj["points"]
    explained = tuple(obj["explained_variance_ratio"][:2])
    return points, explained


def derive_oracle_route_labels(joined_long_path: Path, profiles: list[str]) -> dict[str, str]:
    allowed_profiles = set(profiles + ["baseline"])
    by_prompt: dict[str, dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader(joined_long_path.open()):
        profile = row["g_profile"]
        if profile not in allowed_profiles:
            continue
        by_prompt[row["prompt_id"]][profile] = float(row["delta_target_prob"])

    labels: dict[str, str] = {}
    for prompt_id, values in by_prompt.items():
        best_label = "off"
        best_value = 0.0
        for profile in profiles:
            candidate = values.get(profile, float("-inf"))
            if candidate > best_value:
                best_value = candidate
                best_label = profile
        labels[prompt_id] = best_label
    return labels


def _eta_squared(values: list[float], groups: list[str]) -> float:
    grand_mean = statistics.mean(values)
    ss_total = sum((value - grand_mean) ** 2 for value in values)
    buckets: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        buckets[group].append(value)
    ss_between = sum(
        len(bucket) * (statistics.mean(bucket) - grand_mean) ** 2
        for bucket in buckets.values()
    )
    return ss_between / ss_total


def summarize_routing_signal(points: list[dict[str, float | str]]) -> dict[str, float]:
    pc1 = [float(point["pc1"]) for point in points]
    pc2 = [float(point["pc2"]) for point in points]
    routes = [str(point["route"]) for point in points]
    types = [str(point["type"]) for point in points]

    pc2_on = [float(point["pc2"]) for point in points if point["route"] != "off"]
    pc2_off = [float(point["pc2"]) for point in points if point["route"] == "off"]
    mean_on = statistics.mean(pc2_on)
    mean_off = statistics.mean(pc2_off)
    std_on = statistics.stdev(pc2_on)
    std_off = statistics.stdev(pc2_off)
    pooled = math.sqrt(
        (
            (len(pc2_on) - 1) * std_on**2
            + (len(pc2_off) - 1) * std_off**2
        )
        / (len(pc2_on) + len(pc2_off) - 2)
    )

    return {
        "eta_pc1_by_type": _eta_squared(pc1, types),
        "eta_pc1_by_route": _eta_squared(pc1, routes),
        "eta_pc2_by_route": _eta_squared(pc2, routes),
        "mean_pc2_on": mean_on,
        "mean_pc2_off": mean_off,
        "cohen_d_pc2_on_vs_off": (mean_on - mean_off) / pooled,
    }


def plot_pca_baseline_entropy(
    *,
    pca_path: Path,
    joined_long_path: Path,
    output_path: Path,
    route_profiles: list[str],
    figsize: tuple[float, float] = (12, 5.2),
) -> dict[str, float]:
    configure_matplotlib(font_family="sans-serif", font_size=11)

    raw_points, explained = load_pca_points(pca_path)
    route_by_prompt = derive_oracle_route_labels(joined_long_path, route_profiles)
    points = [
        {
            **point,
            "route": route_by_prompt[str(point["prompt_id"])],
        }
        for point in raw_points
    ]
    stats = summarize_routing_signal(points)

    type_order = [prompt_type for prompt_type in TYPE_LEGEND_ORDER if any(point["type"] == prompt_type for point in points)]
    type_colors = qualitative_11_color_map(type_order)

    all_pc1 = [float(point["pc1"]) for point in points]
    all_pc2 = [float(point["pc2"]) for point in points]
    x_pad = 0.05 * (max(all_pc1) - min(all_pc1))
    y_pad = 0.08 * (max(all_pc2) - min(all_pc2))
    xlim = (min(all_pc1) - x_pad, max(all_pc1) + x_pad)
    ylim = (min(all_pc2) - y_pad, max(all_pc2) + y_pad)

    fig, (ax_type, ax_route) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for prompt_type in type_order:
        subset = [point for point in points if point["type"] == prompt_type]
        ax_type.scatter(
            [float(point["pc1"]) for point in subset],
            [float(point["pc2"]) for point in subset],
            s=18,
            alpha=0.75,
            color=type_colors[prompt_type],
            linewidths=0,
            label=prettify_type(prompt_type),
        )

    for route in ["off"] + [route for route in ROUTE_ORDER if route != "off"]:
        subset = [point for point in points if point["route"] == route]
        ax_route.scatter(
            [float(point["pc1"]) for point in subset],
            [float(point["pc2"]) for point in subset],
            s=18,
            alpha=0.82 if route != "off" else 0.6,
            color=ROUTE_COLORS[route],
            linewidths=0,
            label=route,
            zorder=2 if route != "off" else 1,
        )

    ax_route.axhline(0.0, color="#B5B5B5", linestyle="--", linewidth=0.7, zorder=0)

    for ax in (ax_type, ax_route):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)
        ax.set_facecolor("white")

    ax_type.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax_type.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax_route.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")

    ax_type.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    route_handles, route_labels = ax_route.get_legend_handles_labels()
    ax_route.legend(
        route_handles,
        route_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    fig.subplots_adjust(bottom=0.34, wspace=0.12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats["pc1_pct"] = explained[0] * 100
    stats["pc2_pct"] = explained[1] * 100
    return stats
