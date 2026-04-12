from __future__ import annotations

import csv
import json
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


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


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


def load_main_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def bootstrap_eta(values: np.ndarray, groups: list[str], *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    vals = np.empty(n_boot, dtype=float)
    groups_arr = np.array(groups, dtype=object)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = eta_squared(values[idx], groups_arr[idx].tolist())
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    out_fig = REPO_ROOT / "docs" / "figures" / "diagnostics" / "figure6d_pc2_winner_within_class.png"
    out_csv = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "reports" / "figure6d_pc2_winner_within_class.csv"
    out_summary = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "reports" / "figure6d_pc2_winner_within_class_summary.txt"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    main_csv = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "csv" / "figure6_main_task.csv"
    rows = load_main_csv(main_csv)

    pc1 = np.array([float(r["pc1"]) for r in rows], dtype=float)
    pc2 = np.array([float(r["pc2"]) for r in rows], dtype=float)
    token_count = np.array([float(r["tokens_approx"]) for r in rows], dtype=float)
    winners = [r["winner"] for r in rows]
    task_classes = [r["type"] for r in rows]
    pc2_resid = residualize(pc2, np.column_stack([pc1, token_count]))

    pooled_eta = eta_squared(pc2_resid, winners)

    order = sorted(set(task_classes), key=lambda t: (-sum(tt == t for tt in task_classes), TYPE_LEGEND_ORDER.index(t)))
    rows_out: list[dict] = []
    for task_class in order:
        idx = [i for i, t in enumerate(task_classes) if t == task_class]
        values = pc2_resid[idx]
        groups = [winners[i] for i in idx]
        eta = eta_squared(values, groups)
        ci_lo, ci_hi = bootstrap_eta(values, groups, n_boot=1000, seed=200 + TYPE_LEGEND_ORDER.index(task_class))
        constant_frac = float(np.mean(np.array(groups, dtype=object) == "constant_2.6"))
        rows_out.append(
            {
                "task_class": task_class,
                "n": len(idx),
                "winner_eta2": eta,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "pooled_eta2": pooled_eta,
                "constant_2.6_fraction": constant_frac,
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_class", "n", "winner_eta2", "ci_lo", "ci_hi", "pooled_eta2", "constant_2.6_fraction"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    configure_matplotlib(font_family="sans-serif", font_size=10.5)
    colors = qualitative_11_color_map(TYPE_LEGEND_ORDER)
    x = np.arange(len(rows_out))
    vals = np.array([r["winner_eta2"] for r in rows_out], dtype=float)
    lows = vals - np.array([r["ci_lo"] for r in rows_out], dtype=float)
    highs = np.array([r["ci_hi"] for r in rows_out], dtype=float) - vals
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x,
        vals,
        width=0.72,
        color=[colors[r["task_class"]] for r in rows_out],
        alpha=0.88,
        edgecolor="none",
        yerr=np.vstack([lows, highs]),
        ecolor="black",
        capsize=3,
        linewidth=0.8,
    )
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.axhline(pooled_eta, color="black", linestyle="--", linewidth=1.0)
    ax.text(len(rows_out) - 0.15, pooled_eta + 0.01, f"pooled η² = {pooled_eta:.3f}", ha="right", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([prettify_type(r["task_class"]) for r in rows_out], rotation=38, ha="right")
    ax.set_ylabel("Within-class η² of PC2 residual by winner")
    ax.set_ylim(0.0, 0.5)
    for xi, row in zip(x, rows_out, strict=True):
        ax.text(xi, 0.012, f"n={row['n']}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top_constant = sorted(rows_out, key=lambda r: r["constant_2.6_fraction"], reverse=True)[:5]
    out_summary.write_text(
        "\n".join(
            [
                "Figure 6d summary",
                f"Pooled η² of PC2 residual by winner: {pooled_eta:.4f}",
                "Top classes by constant_2.6 winner fraction:",
                *[
                    f"- {row['task_class']}: frac={row['constant_2.6_fraction']:.3f}, within-class η²={row['winner_eta2']:.3f}"
                    for row in top_constant
                ],
            ]
        )
        + "\n"
    )

    print(json.dumps({"figure": str(out_fig), "csv": str(out_csv), "pooled_eta2": pooled_eta}, indent=2))


if __name__ == "__main__":
    main()
