"""
bistate_router.py — Simple binary routing baseline from baseline-attention PCA.

This experiment answers a narrow question:
    given only PC1 and PC2 from the baseline attention-entropy PCA, how much
    value can a simple intervene/off switch recover relative to always-on and
    oracle binary routing for a single fixed profile?

Usage examples:
    python -m router.experiments.bistate_router \
        --model-key 9B \
        --data-dir data/022-balanced-attention-hybrid \
        --constant-profile constant_2.3

    python -m router.experiments.bistate_router \
        --model-key OLMO \
        --data-dir data/022-balanced-block-hybrid \
        --constant-profile auto
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.backend import InterventionMode, normalize_intervention_mode
from .train_router import build_feature_matrix, load_baseline_entropy_vectors


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt_type: str
    pc1: float
    pc2: float
    delta_single_profile: float
    oracle_full_delta: float


def load_pca_points(path: Path) -> tuple[dict[str, tuple[float, float]], dict[str, str], list[float]]:
    obj = json.loads(path.read_text())
    coords: dict[str, tuple[float, float]] = {}
    prompt_types: dict[str, str] = {}
    for point in obj["points"]:
        coords[point["prompt_id"]] = (float(point["pc1"]), float(point["pc2"]))
        prompt_types[point["prompt_id"]] = point["type"]
    return coords, prompt_types, obj["explained_variance_ratio"]


def auto_select_best_constant(overall_summary_path: Path) -> str:
    rows = list(csv.DictReader(overall_summary_path.open()))
    best_profile = None
    best_mean = None
    for row in rows:
        profile = row["g_profile"]
        if not profile.startswith("constant_"):
            continue
        try:
            mean_delta = float(row["mean_delta_target_prob"])
        except (TypeError, ValueError):
            continue
        if best_mean is None or mean_delta > best_mean:
            best_profile = profile
            best_mean = mean_delta
    if best_profile is None:
        raise ValueError(f"No constant profiles found in {overall_summary_path}")
    return best_profile


def load_prompt_records(analysis_dir: Path, constant_profile: str) -> list[PromptRecord]:
    pca_path = analysis_dir / "analysis_baseline_attn_pca.json"
    joined_path = analysis_dir / "analysis_joined_long.csv"

    pca_coords, pca_types, _ = load_pca_points(pca_path)
    single_profile_delta: dict[str, float] = {}
    full_oracle_delta: dict[str, float] = {}

    with joined_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["g_profile"] == "baseline":
                continue
            prompt_id = row["prompt_id"]
            try:
                delta = float(row["delta_target_prob"])
            except (TypeError, ValueError):
                continue
            if row["g_profile"] == constant_profile:
                single_profile_delta[prompt_id] = delta
            if prompt_id not in full_oracle_delta or delta > full_oracle_delta[prompt_id]:
                full_oracle_delta[prompt_id] = delta

    prompt_ids = sorted(set(pca_coords) & set(single_profile_delta) & set(full_oracle_delta))
    records = [
        PromptRecord(
            prompt_id=prompt_id,
            prompt_type=pca_types[prompt_id],
            pc1=pca_coords[prompt_id][0],
            pc2=pca_coords[prompt_id][1],
            delta_single_profile=single_profile_delta[prompt_id],
            oracle_full_delta=max(0.0, full_oracle_delta[prompt_id]),
        )
        for prompt_id in prompt_ids
    ]
    return records


def build_runtime_feature_matrix(data_dir: Path, model_key: str, records: list[PromptRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompt_ids = [r.prompt_id for r in records]
    entropy_vectors = load_baseline_entropy_vectors(data_dir, model_key)
    X, _, pca_components, pca_mean = build_feature_matrix(
        prompt_ids,
        entropy_vectors,
        scalar_features={},
        n_pca=2,
        feature_set="pca",
    )
    if pca_components is None or pca_mean is None:
        raise ValueError("Failed to build PCA features for bistate router")
    return X, pca_components, pca_mean


def evaluate_bistate_router(records: list[PromptRecord], X: np.ndarray, n_folds: int, random_state: int) -> dict:
    delta = np.array([r.delta_single_profile for r in records], dtype=np.float64)
    oracle_full = np.array([r.oracle_full_delta for r in records], dtype=np.float64)
    y = (delta > 0.0).astype(int)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    pred_labels = np.zeros(len(records), dtype=int)
    pred_probs = np.zeros(len(records), dtype=np.float64)
    fold_results: list[dict] = []

    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000)),
            ]
        )
        clf.fit(X[train_idx], y[train_idx])
        fold_pred = clf.predict(X[test_idx])
        fold_prob = clf.predict_proba(X[test_idx])[:, 1]

        pred_labels[test_idx] = fold_pred
        pred_probs[test_idx] = fold_prob

        fold_router_delta = float(np.mean(delta[test_idx] * fold_pred))
        fold_results.append(
            {
                "fold": fold_index,
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], fold_pred)),
                "routed_mean_delta_p": fold_router_delta,
                "always_on_mean_delta_p": float(np.mean(delta[test_idx])),
                "oracle_bistate_mean_delta_p": float(np.mean(np.maximum(delta[test_idx], 0.0))),
                "oracle_full_mean_delta_p": float(np.mean(oracle_full[test_idx])),
                "positive_predictions": int(fold_pred.sum()),
            }
        )

    always_off = 0.0
    always_on = float(np.mean(delta))
    bistate_router = float(np.mean(delta * pred_labels))
    oracle_bistate = float(np.mean(np.maximum(delta, 0.0)))
    oracle_full_ceiling = float(np.mean(oracle_full))

    return {
        "n_prompts": int(len(records)),
        "class_balance": {
            "off": int((y == 0).sum()),
            "intervene": int((y == 1).sum()),
            "intervene_rate": float(y.mean()),
        },
        "performance_ladder": {
            "always_off": always_off,
            "always_on": always_on,
            "bistate_router": bistate_router,
            "oracle_bistate": oracle_bistate,
            "oracle_full": oracle_full_ceiling,
        },
        "router_vs_always_on_gain": float(bistate_router - always_on),
        "router_capture_of_oracle_bistate": float(bistate_router / oracle_bistate) if oracle_bistate > 0 else 0.0,
        "router_capture_of_oracle_full": float(bistate_router / oracle_full_ceiling) if oracle_full_ceiling > 0 else 0.0,
        "overall_accuracy": float(accuracy_score(y, pred_labels)),
        "mean_predicted_intervene_rate": float(pred_labels.mean()),
        "fold_results": fold_results,
        "predictions": [
            {
                "prompt_id": record.prompt_id,
                "type": record.prompt_type,
                "pc1": record.pc1,
                "pc2": record.pc2,
                "delta_target_prob": record.delta_single_profile,
                "oracle_full_delta_target_prob": record.oracle_full_delta,
                "oracle_label": int(y[idx]),
                "predicted_label": int(pred_labels[idx]),
                "predicted_intervene_prob": float(pred_probs[idx]),
            }
            for idx, record in enumerate(records)
        ],
    }


def fit_full_router_model(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_std, y)
    # Convert sklearn's binary decision function z into a two-logit softmax form
    # [z, 0], so class_names can remain [intervene_profile, off].
    z_weights = clf.coef_[0].astype(np.float64)
    z_bias = float(clf.intercept_[0])
    weights = np.column_stack([z_weights, np.zeros_like(z_weights)])
    bias = np.array([z_bias, 0.0], dtype=np.float64)
    return weights, bias, scaler.mean_.astype(np.float64), scaler.scale_.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a simple bistate router baseline.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--constant-profile", required=True,
                        help='Fixed profile to route, or "auto" to choose best constant by raw mean.')
    parser.add_argument(
        "--intervention-mode",
        default="auto",
        help='Intervention mode for runtime artifact: "attention_contribution", "block_output", or "auto".',
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = data_dir / args.model_key / "analysis"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / args.model_key / "router-bistate"
    output_dir.mkdir(parents=True, exist_ok=True)

    constant_profile = args.constant_profile
    if constant_profile == "auto":
        constant_profile = auto_select_best_constant(analysis_dir / "analysis_overall_profile_summary.csv")

    if args.intervention_mode == "auto":
        inferred_mode = (
            InterventionMode.BLOCK_OUTPUT.value
            if "block" in str(data_dir).lower()
            else InterventionMode.ATTENTION_CONTRIBUTION.value
        )
    else:
        inferred_mode = args.intervention_mode
    intervention_mode = normalize_intervention_mode(inferred_mode)

    records = load_prompt_records(analysis_dir, constant_profile)
    _, _, explained_variance_ratio = load_pca_points(analysis_dir / "analysis_baseline_attn_pca.json")
    X_runtime, pca_components, pca_mean = build_runtime_feature_matrix(data_dir, args.model_key, records)
    results = evaluate_bistate_router(records, X_runtime, n_folds=args.n_folds, random_state=args.random_state)
    y = np.array([int(r.delta_single_profile > 0.0) for r in records], dtype=np.int64)
    weights, bias, standardization_mean, standardization_std = fit_full_router_model(X_runtime, y)

    payload = {
        "model_key": args.model_key,
        "data_dir": str(data_dir),
        "analysis_dir": str(analysis_dir),
        "constant_profile": constant_profile,
        "features": {
            "pca_components_used": ["pc1", "pc2"],
            "explained_variance_ratio": explained_variance_ratio[:2],
        },
        "n_folds": args.n_folds,
        "random_state": args.random_state,
        **results,
    }

    json_path = output_dir / "bistate_router_results.json"
    json_path.write_text(json.dumps(payload, indent=2))

    router_model = {
        "model_key": args.model_key,
        "profiles": [constant_profile],
        "class_names": [constant_profile, "off"],
        "feature_set": "pca",
        "feature_names": ["pc1", "pc2"],
        "n_pca": 2,
        "intervention_mode": intervention_mode.value,
        "standardization_mean": standardization_mean.tolist(),
        "standardization_std": standardization_std.tolist(),
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "pca_components": pca_components.tolist(),
        "pca_mean": pca_mean.tolist(),
        "profile_specs": {
            constant_profile: {
                "g_function": "constant",
                "g_params": {"value": float(constant_profile.split("_", 1)[1])},
            }
        },
    }
    router_model_path = output_dir / "router_model.json"
    router_model_path.write_text(json.dumps(router_model, indent=2))

    report_path = output_dir / "bistate_router_report.txt"
    ladder = results["performance_ladder"]
    with report_path.open("w") as f:
        f.write(f"Bistate Router Baseline: {args.model_key}\n")
        f.write(f"{'=' * 72}\n\n")
        f.write(f"Data dir: {data_dir}\n")
        f.write(f"Analysis dir: {analysis_dir}\n")
        f.write(f"Constant profile: {constant_profile}\n")
        f.write(
            f"Features: PC1 ({explained_variance_ratio[0]:.3%}), "
            f"PC2 ({explained_variance_ratio[1]:.3%})\n"
        )
        f.write(f"Folds: {args.n_folds}\n\n")
        f.write("Class balance:\n")
        f.write(f"  off:       {results['class_balance']['off']}\n")
        f.write(f"  intervene: {results['class_balance']['intervene']}\n")
        f.write(f"  intervene rate: {results['class_balance']['intervene_rate']:.1%}\n\n")
        f.write("Performance ladder (mean delta_target_prob):\n")
        f.write(f"  Always-off:      {ladder['always_off']:.4f}\n")
        f.write(f"  Always-on:       {ladder['always_on']:.4f}\n")
        f.write(f"  Bistate router:  {ladder['bistate_router']:.4f}\n")
        f.write(f"  Oracle bistate:  {ladder['oracle_bistate']:.4f}\n")
        f.write(f"  Oracle full:     {ladder['oracle_full']:.4f}\n\n")
        f.write(f"Router accuracy: {results['overall_accuracy']:.3f}\n")
        f.write(f"Predicted intervene rate: {results['mean_predicted_intervene_rate']:.1%}\n")
        f.write(f"Router minus always-on: {results['router_vs_always_on_gain']:+.4f}\n")
        f.write(f"Capture of oracle bistate: {results['router_capture_of_oracle_bistate']:.1%}\n")
        f.write(f"Capture of oracle full: {results['router_capture_of_oracle_full']:.1%}\n\n")
        f.write("Per-fold results:\n")
        for row in results["fold_results"]:
            f.write(
                f"  Fold {row['fold']}: acc={row['accuracy']:.3f} "
                f"routed={row['routed_mean_delta_p']:.4f} "
                f"always_on={row['always_on_mean_delta_p']:.4f} "
                f"oracle_bi={row['oracle_bistate_mean_delta_p']:.4f} "
                f"oracle_full={row['oracle_full_mean_delta_p']:.4f}\n"
            )

    print(f"Model: {args.model_key}")
    print(f"Constant profile: {constant_profile}")
    print(f"Intervention mode: {intervention_mode.value}")
    print(
        f"Features: PC1 ({explained_variance_ratio[0]:.1%}), "
        f"PC2 ({explained_variance_ratio[1]:.1%})"
    )
    print(f"Class balance: intervene={results['class_balance']['intervene']} / {results['n_prompts']}")
    print("\nPerformance ladder:")
    print(f"  Always-off:      {ladder['always_off']:.4f}")
    print(f"  Always-on:       {ladder['always_on']:.4f}")
    print(f"  Bistate router:  {ladder['bistate_router']:.4f}")
    print(f"  Oracle bistate:  {ladder['oracle_bistate']:.4f}")
    print(f"  Oracle full:     {ladder['oracle_full']:.4f}")
    print(f"\nAccuracy: {results['overall_accuracy']:.3f}")
    print(f"Capture of oracle bistate: {results['router_capture_of_oracle_bistate']:.1%}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved router model: {router_model_path}")


if __name__ == "__main__":
    main()
