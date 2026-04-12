from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.analysis.pca.scripts.pca_figure6_diagnostics import TYPE_LEGEND_ORDER
from docs.figurelib.common import configure_matplotlib, prettify_type, qualitative_11_color_map


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def load_main_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_baseline_verbose(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if all(abs(float(x) - 1.0) < 1e-9 for x in row["g_attention_scales"]):
                out[row["prompt_id"]] = row
    return out


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    out_fig = REPO_ROOT / "docs" / "figures" / "manuscript" / "figure6c.png"
    out_csv = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "reports" / "figure6c_within_class_mean_entropy_bits.csv"
    out_summary = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "reports" / "figure6c_within_class_mean_entropy_bits_summary.txt"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    main_csv = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "csv" / "figure6_main_task.csv"
    verbose_path = REPO_ROOT / "data" / "022-balanced-attention-hybrid" / "9B" / "verbose.jsonl"
    pooled_csv = REPO_ROOT / "docs" / "figures" / "pc2_difficulty_probe" / "pc2_difficulty_correlations.csv"

    rows = load_main_csv(main_csv)
    baseline = load_baseline_verbose(verbose_path)
    pooled_rows = {row["proxy"]: row for row in csv.DictReader(pooled_csv.open())}
    pooled_rho = float(pooled_rows["mean_entropy_bits"]["partial_rho"])

    merged: list[dict] = []
    for row in rows:
        pid = row["prompt_id"]
        if pid not in baseline:
            continue
        b = baseline[pid]
        merged.append(
            {
                "prompt_id": pid,
                "task_class": row["type"],
                "tokens_approx": float(row["tokens_approx"]),
                "pc1": float(row["pc1"]),
                "pc2": float(row["pc2"]),
                "mean_entropy_bits": float(b["mean_entropy_bits"]),
            }
        )

    pc1 = np.array([r["pc1"] for r in merged], dtype=float)
    pc2 = np.array([r["pc2"] for r in merged], dtype=float)
    token_count = np.array([r["tokens_approx"] for r in merged], dtype=float)
    pc2_resid = residualize(pc2, np.column_stack([pc1, token_count]))

    rows_out: list[dict] = []
    type_order_idx = {t: i for i, t in enumerate(TYPE_LEGEND_ORDER)}
    for task_class in sorted(set(r["task_class"] for r in merged), key=lambda t: (-sum(r["task_class"] == t for r in merged), type_order_idx[t])):
        idx = [i for i, r in enumerate(merged) if r["task_class"] == task_class]
        x = np.array([merged[i]["mean_entropy_bits"] for i in idx], dtype=float)
        class_pc1 = np.array([merged[i]["pc1"] for i in idx], dtype=float)
        class_tokens = np.array([merged[i]["tokens_approx"] for i in idx], dtype=float)
        x_resid = residualize(x, np.column_stack([class_pc1, class_tokens]))
        y = pc2_resid[idx]
        rho, p = spearmanr(y, x_resid)
        rho = float(rho)
        p = float(p)
        ci_lo, ci_hi = bootstrap_spearman(y, x_resid, n_boot=1000, seed=100 + type_order_idx[task_class])
        flag = (abs(rho - pooled_rho) > 0.2) or ((rho == 0.0 and pooled_rho != 0.0) or (rho * pooled_rho < 0))
        rows_out.append(
            {
                "task_class": task_class,
                "n": len(idx),
                "within_class_rho": rho,
                "p_value": p,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "pooled_partial_rho": pooled_rho,
                "flag": flag,
            }
        )

    weighted_mean = float(sum(r["n"] * r["within_class_rho"] for r in rows_out) / sum(r["n"] for r in rows_out))
    for row in rows_out:
        row["weighted_mean_within_class_rho"] = weighted_mean

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_class",
                "n",
                "within_class_rho",
                "p_value",
                "ci_lo",
                "ci_hi",
                "weighted_mean_within_class_rho",
                "pooled_partial_rho",
                "flag",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    configure_matplotlib(font_family="sans-serif", font_size=10.5)
    colors = qualitative_11_color_map(TYPE_LEGEND_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows_out))
    vals = np.array([r["within_class_rho"] for r in rows_out], dtype=float)
    lows = vals - np.array([r["ci_lo"] for r in rows_out], dtype=float)
    highs = np.array([r["ci_hi"] for r in rows_out], dtype=float) - vals
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
    ax.axhline(pooled_rho, color="black", linestyle="--", linewidth=1.0)
    ax.text(len(rows_out) - 0.15, pooled_rho + 0.015, f"pooled partial ρ = {pooled_rho:.3f}", ha="right", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([prettify_type(r["task_class"]) for r in rows_out], rotation=38, ha="right")
    ax.set_ylabel("Within-class partial Spearman rho")
    ax.set_ylim(-0.5, 0.5)
    for xi, row in zip(x, rows_out, strict=True):
        y_text = -0.475 if row["within_class_rho"] >= 0 else 0.455
        va = "bottom" if row["within_class_rho"] >= 0 else "top"
        ax.text(xi, y_text, f"n={row['n']}", ha="center", va=va, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    flagged = [r["task_class"] for r in rows_out if r["flag"]]
    out_summary.write_text(
        "\n".join(
            [
                "Figure 6c summary",
                f"Weighted mean within-class rho: {weighted_mean:.4f}",
                f"Pooled partial rho: {pooled_rho:.4f}",
                f"Flagged classes: {', '.join(flagged) if flagged else 'none'}",
            ]
        )
        + "\n"
    )

    print(json.dumps({"figure": str(out_fig), "csv": str(out_csv), "weighted_mean": weighted_mean, "pooled_rho": pooled_rho}, indent=2))


if __name__ == "__main__":
    main()
