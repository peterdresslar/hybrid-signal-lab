from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_battery_meta() -> dict[str, dict]:
    rows = json.loads((REPO_ROOT / "battery" / "data" / "battery_4" / "all_candidates.json").read_text())
    return {row["id"]: row for row in rows}


def load_baseline_vectors(verbose_path: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    with verbose_path.open() as f:
        for line in f:
            row = json.loads(line)
            if not all(s == 1.0 for s in row["g_attention_scales"]):
                continue
            flat: list[float] = []
            for layer in row["attn_entropy_per_head_final"]:
                flat.extend(layer)
            vectors[row["prompt_id"]] = np.array(flat, dtype=np.float64)
    return vectors


def compute_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0)
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ vt.T
    explained = (s**2) / (s**2).sum()
    return coords, explained


def residualize_on_scalar(X: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(scalar), scalar])
    beta = np.linalg.lstsq(design, X, rcond=None)[0]
    return X - design @ beta


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


def main() -> None:
    out_dir = REPO_ROOT / "docs" / "analysis" / "pca" / "outputs" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    battery_meta = load_battery_meta()
    rows_out: list[dict] = []

    for pca_json in sorted(Path(REPO_ROOT / "data").glob("*/**/analysis/analysis_baseline_attn_pca.json")):
        run_family = pca_json.parents[2].name
        model_key = pca_json.parents[1].name
        model_dir = pca_json.parents[1]
        verbose_path = model_dir / "verbose.jsonl"
        if not verbose_path.exists():
            continue

        obj = json.loads(pca_json.read_text())
        vectors = load_baseline_vectors(verbose_path)
        prompt_ids = sorted(pid for pid in vectors if pid in battery_meta)
        X = np.stack([vectors[pid] for pid in prompt_ids], axis=0)
        types = [battery_meta[pid]["type"] for pid in prompt_ids]
        length = np.array([float(battery_meta[pid].get("tokens_approx", 0.0)) for pid in prompt_ids], dtype=float)

        coords, explained = compute_pca(X)
        X_len_resid = residualize_on_scalar(X, length)
        coords_len_resid, explained_len_resid = compute_pca(X_len_resid)

        rows_out.append(
            {
                "run_family": run_family,
                "model_key": model_key,
                "n_prompts": len(prompt_ids),
                "n_features": int(X.shape[1]),
                "pc1_var": round(float(explained[0]), 6),
                "pc2_var": round(float(explained[1]), 6),
                "pc3_var": round(float(explained[2]), 6),
                "corr_pc1_tokens_approx": round(float(np.corrcoef(coords[:, 0], length)[0, 1]), 6),
                "corr_pc2_tokens_approx": round(float(np.corrcoef(coords[:, 1], length)[0, 1]), 6),
                "length_resid_pc1_var": round(float(explained_len_resid[0]), 6),
                "length_resid_pc2_var": round(float(explained_len_resid[1]), 6),
                "length_resid_pc3_var": round(float(explained_len_resid[2]), 6),
                "eta_type_pc1": round(float(eta_squared(coords[:, 0], types)), 6),
                "eta_type_pc2": round(float(eta_squared(coords[:, 1], types)), 6),
                "eta_type_pc3": round(float(eta_squared(coords[:, 2], types)), 6),
                "eta_type_len_resid_pc1": round(float(eta_squared(coords_len_resid[:, 0], types)), 6),
                "eta_type_len_resid_pc2": round(float(eta_squared(coords_len_resid[:, 1], types)), 6),
                "eta_type_len_resid_pc3": round(float(eta_squared(coords_len_resid[:, 2], types)), 6),
            }
        )

    csv_path = out_dir / "pca_model_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    lines = [
        "# PCA Analysis Summary",
        "",
        "This folder collects reusable PCA-oriented diagnostics across the 022 run families.",
        "",
        "## Notes",
        "",
        "- Baseline PCA is computed from baseline `attn_entropy_per_head_final` vectors only.",
        "- The two hybrid families (`022-balanced-attention-hybrid` and `022-balanced-block-hybrid`) share identical baseline PCA within a model, because the baseline forward pass is the same and only the intervention mode differs.",
        "- `tokens_approx` from Battery 4 is used as the prompt-length proxy for the length-residualized summaries.",
        "",
        "## Files",
        "",
        "- `pca_model_summary.csv`: per-run summary of explained variance, prompt-length loading, and type-structure persistence before and after prompt-length residualization.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"output_csv": str(csv_path), "n_rows": len(rows_out)}, indent=2))


if __name__ == "__main__":
    main()
