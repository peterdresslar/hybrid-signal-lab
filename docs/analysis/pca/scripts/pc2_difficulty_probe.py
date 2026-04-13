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

from docs.figurelib.common import configure_matplotlib, qualitative_11_color_map
from docs.analysis.pca.scripts.pca_figure6_diagnostics import TYPE_LEGEND_ORDER


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


def main() -> None:
    out_dir = REPO_ROOT / "docs" / "figures" / "pc2_difficulty_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_csv = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "qwen9b" / "csv" / "figure6_main_task.csv"
    verbose_path = REPO_ROOT / "data" / "022-balanced-attention-hybrid" / "9B" / "verbose.jsonl"
    rows = load_main_csv(main_csv)
    baseline = load_baseline_verbose(verbose_path)

    merged: list[dict] = []
    for row in rows:
        pid = row["prompt_id"]
        if pid not in baseline:
            continue
        b = baseline[pid]
        merged.append(
            {
                "prompt_id": pid,
                "type": row["type"],
                "tokens_approx": float(row["tokens_approx"]),
                "pc1": float(row["pc1"]),
                "pc2": float(row["pc2"]),
                "target_prob": float(b["target_prob"]),
                "target_rank": float(b["target_rank"]),
                "final_entropy_bits": float(b["final_entropy_bits"]),
                "mean_entropy_bits": float(b["mean_entropy_bits"]),
            }
        )

    types = [r["type"] for r in merged]
    type_order = [t for t in TYPE_LEGEND_ORDER if t in set(types)]
    type_colors = qualitative_11_color_map(type_order)

    pc1 = np.array([r["pc1"] for r in merged], dtype=float)
    pc2 = np.array([r["pc2"] for r in merged], dtype=float)
    token_count = np.array([r["tokens_approx"] for r in merged], dtype=float)

    # Probe y-axis used for the plot: residualize PC2 on both PC1 and token count.
    pc2_resid = residualize(pc2, np.column_stack([pc1, token_count]))

    proxies = ["target_prob", "target_rank", "final_entropy_bits", "mean_entropy_bits"]
    results: list[dict] = []
    for proxy in proxies:
        x = np.array([r[proxy] for r in merged], dtype=float)
        raw = spearmanr(pc2, x)
        x_resid = residualize(x, token_count[:, None])
        partial = spearmanr(pc2_resid, x_resid)
        pc1_only = spearmanr(residualize(pc2, pc1[:, None]), x)
        results.append(
            {
                "proxy": proxy,
                "raw_rho": float(raw.statistic),
                "raw_p": float(raw.pvalue),
                "pc1_resid_rho": float(pc1_only.statistic),
                "pc1_resid_p": float(pc1_only.pvalue),
                "partial_rho": float(partial.statistic),
                "partial_p": float(partial.pvalue),
                "n": len(merged),
            }
        )

    summary_csv = out_dir / "pc2_difficulty_correlations.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    configure_matplotlib(font_family="sans-serif", font_size=10.5)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), sharey=True)
    axes = axes.ravel()
    for ax, proxy in zip(axes, proxies, strict=True):
        x = np.array([r[proxy] for r in merged], dtype=float)
        x_resid = residualize(x, token_count[:, None])
        for t in type_order:
            idx = [i for i, tt in enumerate(types) if tt == t]
            ax.scatter(
                x_resid[idx],
                pc2_resid[idx],
                s=14,
                alpha=0.72,
                color=type_colors[t],
                linewidths=0,
            )
        row = next(r for r in results if r["proxy"] == proxy)
        ax.set_title(f"{proxy}\npartial ρ={row['partial_rho']:.3f}, raw ρ={row['raw_rho']:.3f}", fontsize=10)
        ax.set_xlabel(f"{proxy} residual")
        ax.grid(False)
    axes[0].set_ylabel("PC2 residual (controls: PC1, token count)")
    axes[2].set_ylabel("PC2 residual (controls: PC1, token count)")
    fig.tight_layout()
    fig.savefig(out_dir / "pc2_difficulty_probe_residualized.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = [
        "PC2 difficulty probe summary",
        "",
        "Schema and baseline rows",
        "- Figure 6 main PCA uses raw baseline head-entropy features; it was not length-residualized at input time.",
        "- Baseline rows in verbose.jsonl are those with g_attention_scales == [1.0]*8.",
        f"- Matched baseline prompt rows: {len(merged)}.",
        "",
        "Correlations",
    ]
    for row in results:
        lines.append(
            f"- {row['proxy']}: raw Spearman rho={row['raw_rho']:.4f} (p={row['raw_p']:.3g}), "
            f"PC1-only residual rho={row['pc1_resid_rho']:.4f} (p={row['pc1_resid_p']:.3g}), "
            f"partial rho controlling PC1 and token count={row['partial_rho']:.4f} "
            f"(p={row['partial_p']:.3g}), n={row['n']}."
        )
    lines.append("")
    lines.append("Plot note")
    lines.append("- `pc2_difficulty_probe_residualized.png` plots residualized PC2 against residualized difficulty proxies.")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "n": len(merged),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
