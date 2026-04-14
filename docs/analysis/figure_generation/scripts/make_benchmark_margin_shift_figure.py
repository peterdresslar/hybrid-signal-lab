from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.figurelib.common import configure_matplotlib

DATA_ROOT = REPO_ROOT / "data" / "030-bench"
OUT_FIG = REPO_ROOT / "docs" / "figures" / "diagnostics" / "figure_benchmark_margin_shift.png"
OUT_CSV = REPO_ROOT / "docs" / "analysis" / "reports" / "benchmark_margin_shift_summary.csv"

RUNS = [
    {
        "run": "routed_9B",
        "model_label": "Qwen 9B",
        "color": "#2A6FBB",
    },
    {
        "run": "routed_OLMO",
        "model_label": "Olmo Hybrid",
        "color": "#D17A22",
    },
]

TASK_ORDER = [
    "arc_challenge",
    "mmlu_abstract_algebra",
    "mmlu_college_math",
    "mmlu_college_cs",
]

TASK_LABELS = {
    "arc_challenge": "ARC-Challenge",
    "mmlu_abstract_algebra": "abstract_algebra",
    "mmlu_college_math": "college_math",
    "mmlu_college_cs": "college_cs",
}


def margin(scores: list[float], correct_idx: int) -> float:
    correct = scores[correct_idx]
    distractor = max(score for idx, score in enumerate(scores) if idx != correct_idx)
    return correct - distractor


def load_task_records(run_dir: Path, task: str) -> dict[str, dict[str, dict]]:
    path = run_dir / f"{task}_records.jsonl"
    rows: dict[str, dict[str, dict]] = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            rows.setdefault(obj["example_id"], {})[obj["condition"]] = obj
    return rows


def load_best_fixed(results_path: Path) -> str:
    data = json.loads(results_path.read_text())
    return data["best_fixed"]["profile"]


def summarize_run(run_spec: dict) -> tuple[list[dict], list[dict]]:
    run_dir = DATA_ROOT / run_spec["run"]
    summary_rows: list[dict] = []
    plot_rows: list[dict] = []
    for task in TASK_ORDER:
        records = load_task_records(run_dir, task)
        best_fixed_profile = load_best_fixed(run_dir / f"{task}_results.json")
        best_fixed_condition = f"fixed_{best_fixed_profile}"
        fixed_shifts: list[float] = []
        oracle_shifts: list[float] = []
        baseline_margins: list[float] = []
        for example_id, by_condition in records.items():
            if "baseline" not in by_condition or "oracle" not in by_condition or best_fixed_condition not in by_condition:
                continue
            baseline = by_condition["baseline"]
            oracle = by_condition["oracle"]
            fixed = by_condition[best_fixed_condition]
            base_margin = margin(baseline["scores"], baseline["correct_idx"])
            fixed_margin = margin(fixed["scores"], fixed["correct_idx"])
            oracle_margin = margin(oracle["scores"], oracle["correct_idx"])
            fixed_shift = fixed_margin - base_margin
            oracle_shift = oracle_margin - base_margin
            baseline_margins.append(base_margin)
            fixed_shifts.append(fixed_shift)
            oracle_shifts.append(oracle_shift)
            plot_rows.append(
                {
                    "model": run_spec["model_label"],
                    "task": task,
                    "example_id": example_id,
                    "comparison": "best_fixed",
                    "margin_shift": fixed_shift,
                }
            )
            plot_rows.append(
                {
                    "model": run_spec["model_label"],
                    "task": task,
                    "example_id": example_id,
                    "comparison": "oracle",
                    "margin_shift": oracle_shift,
                }
            )
        summary_rows.extend(
            [
                {
                    "model": run_spec["model_label"],
                    "task": task,
                    "comparison": "best_fixed",
                    "n": len(fixed_shifts),
                    "best_fixed_profile": best_fixed_profile,
                    "mean_baseline_margin": statistics.mean(baseline_margins),
                    "mean_margin_shift": statistics.mean(fixed_shifts),
                    "median_margin_shift": statistics.median(fixed_shifts),
                    "pct_positive_margin_shift": 100 * sum(v > 0 for v in fixed_shifts) / len(fixed_shifts),
                },
                {
                    "model": run_spec["model_label"],
                    "task": task,
                    "comparison": "oracle",
                    "n": len(oracle_shifts),
                    "best_fixed_profile": best_fixed_profile,
                    "mean_baseline_margin": statistics.mean(baseline_margins),
                    "mean_margin_shift": statistics.mean(oracle_shifts),
                    "median_margin_shift": statistics.median(oracle_shifts),
                    "pct_positive_margin_shift": 100 * sum(v > 0 for v in oracle_shifts) / len(oracle_shifts),
                },
            ]
        )
    return summary_rows, plot_rows


def write_summary(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render(summary_rows: list[dict], plot_rows: list[dict]) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=10.5)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), sharey=True)
    for ax, run_spec in zip(axes, RUNS, strict=True):
        model = run_spec["model_label"]
        positions = []
        data = []
        colors = []
        centers = []
        for idx, task in enumerate(TASK_ORDER):
            center = idx * 2.8
            centers.append(center)
            fixed_vals = [row["margin_shift"] for row in plot_rows if row["model"] == model and row["task"] == task and row["comparison"] == "best_fixed"]
            oracle_vals = [row["margin_shift"] for row in plot_rows if row["model"] == model and row["task"] == task and row["comparison"] == "oracle"]
            positions.extend([center - 0.35, center + 0.35])
            data.extend([fixed_vals, oracle_vals])
            colors.extend(["#B8B8B8", run_spec["color"]])
        parts = ax.boxplot(
            data,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"color": "#666666", "linewidth": 0.9},
            capprops={"color": "#666666", "linewidth": 0.9},
            boxprops={"linewidth": 0.9, "edgecolor": "#555555"},
        )
        for patch, color in zip(parts["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.88)
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_xticks(centers)
        ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], rotation=25, ha="right")
        ax.set_title(model, fontsize=12)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Margin shift vs baseline\n(correct option minus best distractor)")
    handles = [
        plt.Line2D([0], [0], color="#B8B8B8", linewidth=8, label="Best fixed"),
        plt.Line2D([0], [0], color="#2A6FBB", linewidth=8, label="Oracle"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    summary_rows: list[dict] = []
    plot_rows: list[dict] = []
    for run_spec in RUNS:
        run_summary, run_plot = summarize_run(run_spec)
        summary_rows.extend(run_summary)
        plot_rows.extend(run_plot)
    write_summary(summary_rows)
    render(summary_rows, plot_rows)


if __name__ == "__main__":
    main()
