from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import add_mode_inset, configure_matplotlib, prettify_type


def load_oracle_by_type(prompt_winners_path: Path) -> tuple[list[str], dict[str, list[float]]]:
    rows = list(csv.DictReader(prompt_winners_path.open()))
    by_type: dict[str, list[float]] = {}
    for row in rows:
        by_type.setdefault(row["type"], []).append(float(row["best_delta_target_prob"]))

    type_order = [
        t
        for t, _ in sorted(
            by_type.items(),
            key=lambda kv: statistics.median(kv[1]),
            reverse=True,
        )
    ]
    return type_order, by_type


def load_best_constant_means(type_gain_summary_path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(type_gain_summary_path.open()))
    best: dict[str, float] = {}
    for row in rows:
        if not row["g_profile"].startswith("constant_"):
            continue
        prompt_type = row["type"]
        try:
            value = float(row["mean_delta_target_prob"])
        except (TypeError, ValueError):
            continue
        if prompt_type not in best or value > best[prompt_type]:
            best[prompt_type] = value
    return best


def plot_oracle_headroom_distribution(
    *,
    prompt_winners_path: Path,
    type_gain_summary_path: Path,
    output_path: Path,
    mode_label: str | None = None,
    xlim: tuple[float, float] = (-0.05, 0.85),
    figsize: tuple[float, float] = (10, 5),
) -> None:
    type_order, oracle = load_oracle_by_type(prompt_winners_path)
    best_constant = load_best_constant_means(type_gain_summary_path)

    configure_matplotlib(font_family="sans-serif", font_size=11)

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(type_order), 0, -1)
    data = [oracle[t] for t in type_order]

    ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#85C1E9", edgecolor="#5D8AA8", linewidth=1.0),
        medianprops=dict(color="#1F1F1F", linewidth=1.2),
        whiskerprops=dict(color="#5D8AA8", linewidth=1.0),
        capprops=dict(color="#5D8AA8", linewidth=1.0),
    )

    rng = np.random.default_rng(0)
    for pos, prompt_type in zip(positions, type_order):
        xs = np.array(oracle[prompt_type], dtype=float)
        ys = pos + rng.uniform(-0.22, 0.22, size=len(xs))
        ax.scatter(xs, ys, s=10, color="#7F8C8D", alpha=0.3, linewidths=0, zorder=2)

    marker_x = [best_constant[t] for t in type_order]
    ax.scatter(
        marker_x,
        positions,
        marker="D",
        s=34,
        color="#E67E22",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="Best constant-profile mean Δp",
    )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Best-profile Δp (oracle headroom)")
    ax.set_yticks(positions)
    ax.set_yticklabels([prettify_type(t) for t in type_order])
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.3)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    if mode_label is not None:
        add_mode_inset(ax, mode_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_oracle_headroom_distribution_two_panel(
    *,
    qwen_prompt_winners_path: Path,
    qwen_type_gain_summary_path: Path,
    olmo_prompt_winners_path: Path,
    olmo_type_gain_summary_path: Path,
    output_path: Path,
    mode_label: str | None = None,
    xlim: tuple[float, float] = (-0.05, 0.85),
    figsize: tuple[float, float] = (14, 5.6),
) -> None:
    qwen_order, qwen_oracle = load_oracle_by_type(qwen_prompt_winners_path)
    olmo_order, olmo_oracle = load_oracle_by_type(olmo_prompt_winners_path)
    qwen_constant = load_best_constant_means(qwen_type_gain_summary_path)
    olmo_constant = load_best_constant_means(olmo_type_gain_summary_path)

    all_types = sorted(set(qwen_order) | set(olmo_order))
    joint_order = [
        prompt_type
        for prompt_type, _ in sorted(
            (
                (
                    prompt_type,
                    statistics.mean(
                        [
                            statistics.median(qwen_oracle[prompt_type]),
                            statistics.median(olmo_oracle[prompt_type]),
                        ]
                    ),
                )
                for prompt_type in all_types
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
    ]

    configure_matplotlib(font_family="sans-serif", font_size=11)
    fig, (ax_qwen, ax_olmo) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    positions = np.arange(len(joint_order), 0, -1)

    def _draw_panel(ax: plt.Axes, oracle: dict[str, list[float]], best_constant: dict[str, float], title: str, seed: int) -> None:
        data = [oracle[prompt_type] for prompt_type in joint_order]
        ax.boxplot(
            data,
            vert=False,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#85C1E9", edgecolor="#5D8AA8", linewidth=1.0),
            medianprops=dict(color="#1F1F1F", linewidth=1.2),
            whiskerprops=dict(color="#5D8AA8", linewidth=1.0),
            capprops=dict(color="#5D8AA8", linewidth=1.0),
        )

        rng = np.random.default_rng(seed)
        for pos, prompt_type in zip(positions, joint_order):
            xs = np.array(oracle[prompt_type], dtype=float)
            ys = pos + rng.uniform(-0.22, 0.22, size=len(xs))
            ax.scatter(xs, ys, s=8, color="#7F8C8D", alpha=0.24, linewidths=0, zorder=2)

        marker_x = [best_constant[prompt_type] for prompt_type in joint_order]
        ax.scatter(
            marker_x,
            positions,
            marker="D",
            s=30,
            color="#E67E22",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            label="Best constant-profile mean Δp",
        )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.3)
        ax.grid(axis="y", visible=False)
        if mode_label is not None:
            add_mode_inset(ax, mode_label)

    _draw_panel(ax_qwen, qwen_oracle, qwen_constant, "Qwen 3.5 9B", 0)
    _draw_panel(ax_olmo, olmo_oracle, olmo_constant, "Olmo Hybrid 7B", 1)

    ax_qwen.set_xlabel("Best-profile Δp (oracle headroom)")
    ax_olmo.set_xlabel("Best-profile Δp (oracle headroom)")
    ax_qwen.set_yticks(positions)
    ax_qwen.set_yticklabels([prettify_type(prompt_type) for prompt_type in joint_order])
    ax_olmo.set_yticks(positions)
    ax_olmo.tick_params(axis="y", labelleft=False)
    ax_qwen.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
