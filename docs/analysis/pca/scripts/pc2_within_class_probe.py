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

from docs.figurelib.common import configure_matplotlib, prettify_type


PROXIES = ["mean_entropy_bits", "target_prob"]


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


def plot_bar(rows: list[dict], pooled_rho: float, output_path: Path, title: str) -> None:
    configure_matplotlib(font_family="sans-serif", font_size=10.5)
    rows = sorted(rows, key=lambda r: r["n"], reverse=True)
    labels = [prettify_type(r["task_class"]) for r in rows]
    values = [r["within_class_rho"] for r in rows]
    colors = ["#E45756" if r["flag"] else "#4C78A8" for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(range(len(rows)), values, color=colors, alpha=0.85)
    ax.axhline(pooled_rho, color="black", linestyle="--", linewidth=1.0, label=f"Pooled partial rho = {pooled_rho:.3f}")
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Within-class Spearman rho")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    out_dir = REPO_ROOT / "docs" / "figures" / "pc2_within_class_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_csv = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "csv" / "figure6_main_task.csv"
    verbose_path = REPO_ROOT / "data" / "022-balanced-attention-hybrid" / "9B" / "verbose.jsonl"
    pooled_csv = REPO_ROOT / "docs" / "figures" / "pc2_difficulty_probe" / "pc2_difficulty_correlations.csv"

    rows = load_main_csv(main_csv)
    baseline = load_baseline_verbose(verbose_path)
    pooled_rows = {row["proxy"]: row for row in csv.DictReader(pooled_csv.open())}

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
                "target_prob": float(b["target_prob"]),
            }
        )

    pc1 = np.array([r["pc1"] for r in merged], dtype=float)
    pc2 = np.array([r["pc2"] for r in merged], dtype=float)
    token_count = np.array([r["tokens_approx"] for r in merged], dtype=float)
    pc2_resid = residualize(pc2, np.column_stack([pc1, token_count]))
    task_classes = sorted(set(r["task_class"] for r in merged))

    summary_lines = [
        "PC2 within-class conditioning probe",
        "",
        f"Matched baseline prompts: {len(merged)}",
        "- PC2 residuals are computed by regressing PC2 on PC1 and token count.",
        "- Within each task class, each proxy is residualized on token count only, then Spearman rho is computed against the class-matched PC2 residuals.",
        "",
    ]

    for proxy in PROXIES:
        pooled_rho = float(pooled_rows[proxy]["partial_rho"])
        proxy_values = np.array([r[proxy] for r in merged], dtype=float)
        per_class: list[dict] = []
        for task_class in task_classes:
            idx = [i for i, r in enumerate(merged) if r["task_class"] == task_class]
            x = proxy_values[idx]
            x_resid = residualize(x, token_count[idx, None])
            rho, p = spearmanr(pc2_resid[idx], x_resid)
            rho = float(rho)
            p = float(p)
            flag = (abs(rho - pooled_rho) > 0.2) or ((rho == 0.0 and pooled_rho != 0.0) or (rho * pooled_rho < 0))
            per_class.append(
                {
                    "task_class": task_class,
                    "n": len(idx),
                    "within_class_rho": rho,
                    "p_value": p,
                    "flag": flag,
                }
            )

        weighted_mean = float(sum(row["n"] * row["within_class_rho"] for row in per_class) / sum(row["n"] for row in per_class))
        for row in per_class:
            row["weighted_mean_within_class_rho"] = weighted_mean
            row["pooled_partial_rho"] = pooled_rho

        table_path = out_dir / f"{proxy}_within_class_table.csv"
        with table_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task_class",
                    "n",
                    "within_class_rho",
                    "p_value",
                    "weighted_mean_within_class_rho",
                    "pooled_partial_rho",
                    "flag",
                ],
            )
            writer.writeheader()
            writer.writerows(per_class)

        plot_bar(
            per_class,
            pooled_rho,
            out_dir / f"{proxy}_within_class_bar.png",
            title=f"{proxy}: within-class rho vs pooled partial rho",
        )

        flagged = [row["task_class"] for row in per_class if row["flag"]]
        summary_lines.append(
            f"{proxy}: weighted within-class rho = {weighted_mean:.4f}; pooled partial rho = {pooled_rho:.4f}; "
            f"flagged classes = {', '.join(flagged) if flagged else 'none'}."
        )

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(json.dumps({"out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
