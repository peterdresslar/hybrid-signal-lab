from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "data" / "022-pure-mimic"


def _rankdata(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            ranks[ordered[k][1]] = rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def summarize_model(model_dir: Path) -> dict:
    overall_rows = load_csv(model_dir / "analysis" / "analysis_overall_profile_summary.csv")
    valid = []
    for row in overall_rows:
        profile = row["g_profile"]
        if profile == "baseline":
            continue
        try:
            mean_delta = float(row["mean_delta_target_prob"])
        except (TypeError, ValueError):
            continue
        valid.append(
            {
                "profile": profile,
                "family": "constant" if profile.startswith("constant_") else "shaped",
                "mean_delta_target_prob": mean_delta,
                "prompt_best_wins": int(row.get("prompt_best_wins") or 0),
                "top_8_mean_delta_target_prob": float(row.get("top_8_mean_delta_target_prob") or 0.0),
            }
        )

    best_overall = max(valid, key=lambda row: row["mean_delta_target_prob"])
    best_constant = max((row for row in valid if row["family"] == "constant"), key=lambda row: row["mean_delta_target_prob"])
    best_shaped = max((row for row in valid if row["family"] == "shaped"), key=lambda row: row["mean_delta_target_prob"])
    top8 = sorted(valid, key=lambda row: row["mean_delta_target_prob"], reverse=True)[:8]

    prompt_winners = load_csv(model_dir / "analysis" / "analysis_prompt_winners.csv")
    oracle_values = [float(row["best_delta_target_prob"]) for row in prompt_winners]
    winner_profiles = Counter(row["best_g_profile"] for row in prompt_winners)
    bookend_rows = [row for row in prompt_winners if row["best_g_profile"] == "bookend_high_bal_0.40"]
    bookend_type_counts = Counter(row["type"] for row in bookend_rows)

    type_rows = load_csv(model_dir / "analysis" / "analysis_type_gain_summary.csv")
    type_means: dict[str, list[float]] = {}
    best_type_profiles: dict[str, tuple[str, float]] = {}
    for row in type_rows:
        if row["g_profile"] == "baseline":
            continue
        try:
            value = float(row["mean_delta_target_prob"])
        except (TypeError, ValueError):
            continue
        prompt_type = row["type"]
        type_means.setdefault(prompt_type, []).append(value)
        if prompt_type not in best_type_profiles or value > best_type_profiles[prompt_type][1]:
            best_type_profiles[prompt_type] = (row["g_profile"], value)
    averaged_type_means = {key: sum(vals) / len(vals) for key, vals in type_means.items()}

    pca = json.loads((model_dir / "analysis" / "analysis_baseline_attn_pca.json").read_text())
    return {
        "model": model_dir.name,
        "best_overall": best_overall,
        "best_constant": best_constant,
        "best_shaped": best_shaped,
        "mean_constant_delta": sum(row["mean_delta_target_prob"] for row in valid if row["family"] == "constant")
        / sum(1 for row in valid if row["family"] == "constant"),
        "mean_shaped_delta": sum(row["mean_delta_target_prob"] for row in valid if row["family"] == "shaped")
        / sum(1 for row in valid if row["family"] == "shaped"),
        "top8_constant_count": sum(1 for row in top8 if row["family"] == "constant"),
        "oracle_mean": sum(oracle_values) / len(oracle_values),
        "oracle_median": statistics.median(oracle_values),
        "oracle_positive_rate": sum(value > 0 for value in oracle_values) / len(oracle_values),
        "winner_profiles_top12": winner_profiles.most_common(12),
        "bookend_high_wins": len(bookend_rows),
        "bookend_high_type_counts": bookend_type_counts.most_common(),
        "averaged_type_means": averaged_type_means,
        "best_type_profiles": best_type_profiles,
        "pca_explained_variance_ratio": pca["explained_variance_ratio"][:3],
    }


def summarize_experiment() -> dict:
    model_dirs = sorted(path for path in EXPERIMENT_DIR.iterdir() if path.is_dir())
    per_model = [summarize_model(model_dir) for model_dir in model_dirs]
    lookup = {row["model"]: row for row in per_model}
    prompt_types = sorted(next(iter(lookup.values()))["averaged_type_means"])
    rank_correlations = []
    model_names = [row["model"] for row in per_model]
    for i, left in enumerate(model_names):
        for right in model_names[i + 1 :]:
            xs = [lookup[left]["averaged_type_means"][prompt_type] for prompt_type in prompt_types]
            ys = [lookup[right]["averaged_type_means"][prompt_type] for prompt_type in prompt_types]
            rank_correlations.append(
                {
                    "left": left,
                    "right": right,
                    "spearman_type_mean_order": _spearman(xs, ys),
                }
            )
    return {
        "experiment": EXPERIMENT_DIR.name,
        "models": per_model,
        "rank_correlations": rank_correlations,
    }


def main() -> None:
    print(json.dumps(summarize_experiment(), indent=2))


if __name__ == "__main__":
    main()
