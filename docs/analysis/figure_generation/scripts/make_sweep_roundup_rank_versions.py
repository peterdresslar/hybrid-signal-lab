from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.figurelib.common import configure_matplotlib


OUT_SPARSE = REPO_ROOT / "docs" / "figures" / "diagnostics" / "figure_sweep_roundup_rank_sparse.png"
OUT_DENSE = REPO_ROOT / "docs" / "figures" / "diagnostics" / "figure_sweep_roundup_rank_dense.png"
OUT_CSV = REPO_ROOT / "docs" / "analysis" / "reports" / "sweep_roundup_rank_data.csv"

SWEEPS = [
    {
        "label": "Qwen 2B attn-contr",
        "group": "Hybrid attention-contribution",
        "family": "qwen_hybrid",
        "analysis_dir": REPO_ROOT / "data" / "022-balanced-attention-hybrid" / "2B" / "analysis",
    },
    {
        "label": "Qwen 9B attn-contr",
        "group": "Hybrid attention-contribution",
        "family": "qwen_hybrid",
        "analysis_dir": REPO_ROOT / "data" / "022-balanced-attention-hybrid" / "9B" / "analysis",
    },
    {
        "label": "Qwen 35B attn-contr",
        "group": "Hybrid attention-contribution",
        "family": "qwen_hybrid",
        "analysis_dir": REPO_ROOT / "data" / "022-balanced-attention-hybrid" / "35B" / "analysis",
    },
    {
        "label": "Olmo attn-contr",
        "group": "Hybrid attention-contribution",
        "family": "olmo_hybrid",
        "analysis_dir": REPO_ROOT / "data" / "022-balanced-attention-hybrid" / "OLMO" / "analysis",
    },
    {
        "label": "Qwen 2B block-out",
        "group": "Hybrid block-output",
        "family": "qwen_hybrid",
        "analysis_dir": REPO_ROOT / "data" / "022-balanced-block-hybrid" / "2B" / "analysis",
    },
    {
        "label": "Qwen 9B block-out",
        "group": "Hybrid block-output",
        "family": "qwen_hybrid",
        "analysis_dir": REPO_ROOT / "data" / "022-balanced-block-hybrid" / "9B" / "analysis",
    },
    {
        "label": "Qwen 35B block-out",
        "group": "Hybrid block-output",
        "family": "qwen_hybrid",
        "analysis_dir": REPO_ROOT / "data" / "022-balanced-block-hybrid" / "35B" / "analysis",
    },
    {
        "label": "Olmo block-out",
        "group": "Hybrid block-output",
        "family": "olmo_hybrid",
        "analysis_dir": REPO_ROOT / "data" / "022-balanced-block-hybrid" / "OLMO" / "analysis",
    },
    {
        "label": "Qwen3-8B all-layer",
        "group": "Pure-transformer all-layer",
        "family": "pure",
        "analysis_dir": REPO_ROOT / "data" / "022-pure-all" / "Q3_8B" / "analysis",
    },
    {
        "label": "Qwen3-30B all-layer",
        "group": "Pure-transformer all-layer",
        "family": "pure",
        "analysis_dir": REPO_ROOT / "data" / "022-pure-all" / "Q3_30B" / "analysis",
    },
    {
        "label": "Olmo-3 all-layer",
        "group": "Pure-transformer all-layer",
        "family": "pure",
        "analysis_dir": REPO_ROOT / "data" / "022-pure-all" / "OLMO_3" / "analysis",
    },
    {
        "label": "Qwen3-8B mimic",
        "group": "Pure-transformer mimic",
        "family": "pure",
        "analysis_dir": REPO_ROOT / "data" / "022-pure-mimic" / "Q3_8B" / "analysis",
    },
    {
        "label": "Qwen3-30B mimic",
        "group": "Pure-transformer mimic",
        "family": "pure",
        "analysis_dir": REPO_ROOT / "data" / "022-pure-mimic" / "Q3_30B" / "analysis",
    },
    {
        "label": "Olmo-3 mimic",
        "group": "Pure-transformer mimic",
        "family": "pure",
        "analysis_dir": REPO_ROOT / "data" / "022-pure-mimic" / "OLMO_3" / "analysis",
    },
]

FAMILY_COLORS = {
    "qwen_hybrid": "#4C78A8",
    "olmo_hybrid": "#F58518",
    "pure": "#6E6E6E",
}

GROUPS = [
    "Hybrid attention-contribution",
    "Hybrid block-output",
    "Pure-transformer all-layer",
    "Pure-transformer mimic",
]

SHORT_LABELS = {
    "Qwen 2B attn-contr": "Qwen2B",
    "Qwen 9B attn-contr": "Qwen9B",
    "Qwen 35B attn-contr": "Qwen35B",
    "Olmo attn-contr": "OlmoH",
    "Qwen 2B block-out": "Qwen2B",
    "Qwen 9B block-out": "Qwen9B",
    "Qwen 35B block-out": "Qwen35B",
    "Olmo block-out": "OlmoH",
    "Qwen3-8B all-layer": "Qwen8B",
    "Qwen3-30B all-layer": "Qwen30B",
    "Olmo-3 all-layer": "Olmo3",
    "Qwen3-8B mimic": "Qwen8B",
    "Qwen3-30B mimic": "Qwen30B",
    "Olmo-3 mimic": "Olmo3",
}


def _safe_float(value: str | None) -> float | None:
    if value in (None, "", "NaN", "nan"):
        return None
    return float(value)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _profile_mean_rank(rows: list[dict]) -> float:
    vals = [float(r["target_rank"]) for r in rows if r["target_rank"] not in ("", None)]
    return _mean(vals)


def load_sweep_summary(spec: dict) -> dict:
    analysis_dir = spec["analysis_dir"]
    joined = list(csv.DictReader((analysis_dir / "analysis_joined_long.csv").open()))
    overall = {row["g_profile"]: row for row in csv.DictReader((analysis_dir / "analysis_overall_profile_summary.csv").open())}

    baseline_rows = [row for row in joined if row["g_profile"] == "baseline"]
    baseline_mean_rank = _mean([float(row["target_rank"]) for row in baseline_rows])
    baseline_mean_entropy = _mean([float(row["final_entropy_bits"]) for row in baseline_rows])

    collapse_excluded: set[str] = set()
    if spec["label"] == "Olmo block-out":
        for prof, row in overall.items():
            ent = _safe_float(row["mean_delta_final_entropy_bits"])
            if ent is None:
                if prof.startswith("constant_"):
                    collapse_excluded.add(prof)
            elif ent >= 3.0:
                collapse_excluded.add(prof)

    profile_rows: dict[str, list[dict]] = defaultdict(list)
    by_prompt: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in joined:
        if row["target_rank"] not in ("", None):
            if row["g_profile"] not in collapse_excluded:
                by_prompt[(row["prompt_id"], row["rep"])].append(float(row["target_rank"]))
        if row["g_profile"] == "baseline":
            continue
        if row["g_profile"] in collapse_excluded:
            continue
        profile_rows[row["g_profile"]].append(row)

    constant_profiles = [p for p in profile_rows if p.startswith("constant_")]
    shaped_profiles = [p for p in profile_rows if not p.startswith("constant_")]

    best_const = min(constant_profiles, key=lambda p: _profile_mean_rank(profile_rows[p]))
    worst_const = max(constant_profiles, key=lambda p: _profile_mean_rank(profile_rows[p]))
    best_shaped = min(shaped_profiles, key=lambda p: _profile_mean_rank(profile_rows[p]))
    worst_shaped = max(shaped_profiles, key=lambda p: _profile_mean_rank(profile_rows[p]))
    oracle_mean_rank = _mean([min(ranks) for ranks in by_prompt.values()])

    def rank_for(profile: str) -> float:
        return _profile_mean_rank(profile_rows[profile])

    def ent_delta_for(profile: str) -> float | None:
        return _safe_float(overall[profile]["mean_delta_final_entropy_bits"])

    return {
        "sweep": spec["label"],
        "group": spec["group"],
        "family": spec["family"],
        "baseline_mean_rank": baseline_mean_rank,
        "baseline_mean_entropy": baseline_mean_entropy,
        "best_const_name": best_const,
        "best_const_rank": rank_for(best_const),
        "best_const_ent_delta": ent_delta_for(best_const),
        "worst_const_name": worst_const,
        "worst_const_rank": rank_for(worst_const),
        "worst_const_ent_delta": ent_delta_for(worst_const),
        "best_shaped_name": best_shaped,
        "best_shaped_rank": rank_for(best_shaped),
        "best_shaped_ent_delta": ent_delta_for(best_shaped),
        "worst_shaped_name": worst_shaped,
        "worst_shaped_rank": rank_for(worst_shaped),
        "worst_shaped_ent_delta": ent_delta_for(worst_shaped),
        "oracle_mean_rank": oracle_mean_rank,
        "collapse_filtered": bool(collapse_excluded),
        "collapse_excluded_profiles": ",".join(sorted(collapse_excluded)),
    }


def write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sweep",
        "group",
        "family",
        "baseline_mean_rank",
        "baseline_mean_entropy",
        "best_const_name",
        "best_const_rank",
        "best_const_ent_delta",
        "worst_const_name",
        "worst_const_rank",
        "worst_const_ent_delta",
        "best_shaped_name",
        "best_shaped_rank",
        "best_shaped_ent_delta",
        "worst_shaped_name",
        "worst_shaped_rank",
        "worst_shaped_ent_delta",
        "oracle_mean_rank",
        "collapse_filtered",
        "collapse_excluded_profiles",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _entropy_stub(ax: plt.Axes, x: float, y: float, delta: float | None, *, direction_row: int) -> None:
    if delta is None or math.isnan(delta):
        return
    log_extent = min(0.08 * abs(delta), 0.25)
    factor = 10**log_extent
    x2 = x * factor if delta > 0 else x / factor
    y_offset = -0.11 if direction_row == 0 else 0.11
    ax.plot([x, x2], [y + y_offset, y + y_offset], color="black", linewidth=1.2, solid_capstyle="round", zorder=5)


def _entropy_arrow_vertical(ax: plt.Axes, x: float, y: float, delta: float | None) -> None:
    if delta is None or math.isnan(delta) or abs(delta) < 1e-9:
        return
    log_extent = min(0.085 * abs(delta), 0.24)
    y2 = y * (10**log_extent) if delta > 0 else y / (10**log_extent)
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "linewidth": 1.0,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 8,
            "alpha": 0.8,
        },
        zorder=4,
    )


def _group_spans(rows: list[dict]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    current = rows[0]["group"]
    for i, row in enumerate(rows):
        if row["group"] != current:
            spans.append((start, i - 1, current))
            start = i
            current = row["group"]
    spans.append((start, len(rows) - 1, current))
    return spans


def plot_version(rows: list[dict], *, dense: bool, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(font_family="sans-serif", font_size=10.5)

    if dense:
        fig, ax = plt.subplots(figsize=(12.2, 9.6))
        x = np.arange(len(rows), dtype=float)

        for idx, (start, end, _group) in enumerate(_group_spans(rows)):
            if idx % 2 == 0:
                ax.axvspan(start - 0.5, end + 0.5, color="#F7F7F7", zorder=0)

        for i, row in enumerate(rows):
            color = FAMILY_COLORS[row["family"]]
            xi = x[i]

            # Baseline tick, made more prominent as a short horizontal bar.
            ax.hlines(row["baseline_mean_rank"], xi - 0.24, xi + 0.24, color="black", linewidth=2.0, zorder=4)

            # Constant markers.
            ax.scatter(xi - 0.12, row["best_const_rank"], marker="o", s=60, color=color, edgecolors="none", zorder=5)
            ax.scatter(xi - 0.12, row["worst_const_rank"], marker="o", s=60, facecolors="white", edgecolors=color, linewidths=1.7, zorder=5)

            # Shaped markers.
            ax.scatter(xi + 0.12, row["best_shaped_rank"], marker="D", s=60, color=color, edgecolors="none", zorder=5)
            ax.scatter(xi + 0.12, row["worst_shaped_rank"], marker="D", s=60, facecolors="white", edgecolors=color, linewidths=1.7, zorder=5)

            # Oracle marker.
            ax.scatter(xi, row["oracle_mean_rank"], marker="^", s=74, color=color, edgecolors="white", linewidths=0.6, zorder=6)

        ax.set_yscale("log")
        ax.set_ylim(2e5, 10.0)
        ax.set_xlim(-0.7, len(rows) - 0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=1)
        ax.axhline(10.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
        ax.set_ylabel("Mean target rank (log scale; lower value (up) is better)")
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for start, end, group in _group_spans(rows):
            ax.text(
                (start + end) / 2,
                1.08,
                group,
                transform=ax.get_xaxis_transform(),
                fontsize=9,
                color="#555555",
                ha="center",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                zorder=7,
            )
        for xi, row in zip(x, rows, strict=True):
            ax.text(
                xi,
                0.985,
                SHORT_LABELS[row["sweep"]],
                transform=ax.get_xaxis_transform(),
                fontsize=8.5,
                color="#444444",
                ha="center",
                va="top",
                zorder=7,
            )
    else:
        fig, ax = plt.subplots(figsize=(11.2, 8.1))
        y = np.arange(len(rows), dtype=float)

        for idx, (start, end, _group) in enumerate(_group_spans(rows)):
            if idx % 2 == 0:
                ax.axhspan(start - 0.5, end + 0.5, color="#F7F7F7", zorder=0)

        for i, row in enumerate(rows):
            color = FAMILY_COLORS[row["family"]]
            yi = y[i]

            ax.scatter(row["baseline_mean_rank"], yi, marker="|", s=160, color="black", linewidths=1.4, zorder=4)
            ax.scatter(row["best_const_rank"], yi - 0.10, marker="o", s=58, color=color, edgecolors="none", zorder=4)
            ax.scatter(row["best_shaped_rank"], yi + 0.10, marker="D", s=58, facecolors="white", edgecolors=color, linewidths=1.7, zorder=4)
            ax.scatter(row["oracle_mean_rank"], yi, marker="^", s=72, color=color, edgecolors="white", linewidths=0.6, zorder=4)

            if row["collapse_filtered"]:
                ax.text(row["oracle_mean_rank"] * 1.12, yi + 0.18, "†", fontsize=10, ha="left", va="center", color="black")

        ax.set_xscale("log")
        ax.set_xlim(0.8, 2e5)
        ax.set_yticks(y)
        ax.set_yticklabels([row["sweep"] for row in rows])
        ax.invert_yaxis()
        ax.axvline(1.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=1)
        ax.set_xlabel("Mean target rank (log scale; lower is better)")
        ax.tick_params(axis="y", length=0)
        ax.set_facecolor("white")
        ax.spines["left"].set_visible(False)

        for start, end, group in _group_spans(rows):
            ax.text(
                0.82,
                (start + end) / 2,
                group,
                fontsize=9,
                color="#555555",
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                zorder=6,
            )

    from matplotlib.lines import Line2D

    if dense:
        legend_handles = [
            Line2D([0, 1], [0, 0], color="black", linewidth=2.0, label="Baseline"),
            Line2D([0], [0], marker="^", color="#444444", linestyle="None", markersize=7, label="Oracle"),
            Line2D([0], [0], marker="o", color="#444444", linestyle="None", markersize=7, label="Best constant"),
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#444444", linestyle="None", markersize=7, label="Worst constant"),
            Line2D([0], [0], marker="D", color="#444444", linestyle="None", markersize=7, label="Best shaped"),
            Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor="#444444", linestyle="None", markersize=7, label="Worst shaped"),
        ]
    else:
        legend_handles = [Line2D([0], [0], marker="|", color="black", linestyle="None", markersize=14, markeredgewidth=1.4, label="Baseline")]
        legend_handles += [
            Line2D([0], [0], marker="o", color="#444444", linestyle="None", markersize=7, label="Best constant"),
            Line2D([0], [0], marker="D", markerfacecolor="white", markeredgecolor="#444444", linestyle="None", markersize=7, label="Best shaped"),
        ]
    if not dense:
        legend_handles += [Line2D([0], [0], marker="^", color="#444444", linestyle="None", markersize=7, label="Oracle")]
    ncol = 6 if dense else 3
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = [load_sweep_summary(spec) for spec in SWEEPS]
    write_csv(rows)
    plot_version(rows, dense=False, output_path=OUT_SPARSE)
    plot_version(rows, dense=True, output_path=OUT_DENSE)


if __name__ == "__main__":
    main()
