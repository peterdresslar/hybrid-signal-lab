from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.analysis.head_entropy.scripts.head_entropy_diagnostics import (
    _safe_auc,
    build_dataset,
    residualize_dataset_on_scalar,
)
from docs.figurelib.head_entropy_diagnostics import plot_shared_vs_specialist_auc


SELECTED_PROFILES = ["constant_2.6", "edges_narrow_bal_0.55", "late_boost_bal_0.60", "triad_odd_bal_0.45"]


def _load_all_profiles(joined_path: Path) -> list[str]:
    profiles = set()
    with joined_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gp = row["g_profile"]
            if gp == "baseline":
                continue
            try:
                float(row["delta_target_prob"])
            except (TypeError, ValueError):
                continue
            profiles.add(gp)
    return sorted(profiles)


def _load_full_library_positive_labels(
    joined_path: Path,
    prompt_ids: list[str],
    profiles: list[str],
) -> np.ndarray:
    profile_set = set(profiles)
    by_prompt: dict[str, float] = {}
    with joined_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gp = row["g_profile"]
            if gp not in profile_set:
                continue
            try:
                delta = float(row["delta_target_prob"])
            except (TypeError, ValueError):
                continue
            pid = row["prompt_id"]
            prev = by_prompt.get(pid, float("-inf"))
            if delta > prev:
                by_prompt[pid] = delta
    return np.array([1 if by_prompt.get(pid, float("-inf")) > 0.0 else 0 for pid in prompt_ids], dtype=int)


def _shared_auc_matrix(X_3d: np.ndarray, y_binary: np.ndarray) -> np.ndarray:
    n_layers = X_3d.shape[1]
    n_heads = X_3d.shape[2]
    out = np.zeros((n_layers, n_heads), dtype=float)
    for layer_pos in range(n_layers):
        for head_idx in range(n_heads):
            out[layer_pos, head_idx] = _safe_auc(y_binary, X_3d[:, layer_pos, head_idx])
    return out


def main() -> None:
    data_dir = REPO_ROOT / "data" / "022-balanced-attention-hybrid"
    joined_path = data_dir / "9B" / "analysis" / "analysis_joined_long.csv"
    battery_json = REPO_ROOT / "battery" / "data" / "battery_4" / "all_candidates.json"
    out_dir = REPO_ROOT / "docs" / "analysis" / "head_entropy" / "outputs" / "qwen9b" / "full_library_left"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = REPO_ROOT / "docs" / "figures" / "diagnostics" / "figure_head_entropy_specialist_auc_qwen9b_full_left.png"

    selected_dataset = build_dataset(data_dir=data_dir, model_key="9B", profiles=SELECTED_PROFILES)

    battery_rows = json.loads(battery_json.read_text())
    token_map = {row["id"]: float(row.get("tokens_approx", 0.0)) for row in battery_rows}
    tokens = np.array([token_map[pid] for pid in selected_dataset.prompt_ids], dtype=float)

    all_profiles = _load_all_profiles(joined_path)
    full_positive = _load_full_library_positive_labels(joined_path, selected_dataset.prompt_ids, all_profiles)

    resid_dataset = residualize_dataset_on_scalar(selected_dataset, tokens)
    shared_auc = _shared_auc_matrix(resid_dataset.X_3d, full_positive)

    shared_summary = {
        "model_key": "9B",
        "data_dir": str(data_dir),
        "n_prompts": len(selected_dataset.prompt_ids),
        "layer_indices": selected_dataset.layer_indices,
        "length_covariate": "tokens_approx",
        "shared_target": "full_library_any_positive_delta",
        "n_full_library_profiles": len(all_profiles),
        "positive_rate": float(full_positive.mean()),
        "auc": {
            "max_auc_matrix": shared_auc.tolist(),
        },
    }
    shared_path = out_dir / "metrics_summary.json"
    shared_path.write_text(json.dumps(shared_summary, indent=2))

    specialist_path = REPO_ROOT / "docs" / "analysis" / "head_entropy" / "outputs" / "qwen9b" / "raw" / "specialist_vs_anchor_summary.json"
    plot_shared_vs_specialist_auc(
        shared_metrics_path=shared_path,
        specialist_metrics_path=specialist_path,
        output_path=out_fig,
    )

    print(
        json.dumps(
            {
                "n_full_library_profiles": len(all_profiles),
                "positive_rate": float(full_positive.mean()),
                "shared_metrics_path": str(shared_path),
                "figure_path": str(out_fig),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
