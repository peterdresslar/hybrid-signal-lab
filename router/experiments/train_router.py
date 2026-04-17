"""
train_router.py — Train routing classifiers from balanced sweep analysis data.

Given a selected 4-profile set (typically from `select_profiles.py`), train a
multinomial classifier that predicts which profile (or "off") to apply based
on features from a baseline forward pass.

Feature sources:
  - Per-head attention entropy from baseline (g=1.0) verbose.jsonl
    (128 features for 9B, 240 for OLMO)
  - Scalar baseline features from analysis_joined_long.csv
    (final entropy, mean entropy, logit margin, attn entropy mean)
  - PCA projection of the entropy vector (first K components)

The classifier is trained with stratified K-fold cross-validation and
reports accuracy, routed mean delta_p, and confusion between profile
assignments.

Current 030-style usage:
    python -m router.experiments.train_router \
        --model-key 9B \
        --data-dir data/022-balanced-attention-hybrid \
        --profiles constant_2.6 edges_narrow_bal_0.55 late_boost_bal_0.60 triad_odd_bal_0.45 \
        --intervention-mode attention_contribution

    python -m router.experiments.train_router \
        --model-key OLMO \
        --data-dir data/022-balanced-block-hybrid \
        --profiles constant_1.6 edges_narrow_bal_0.40 late_boost_bal_0.30 ramp_up_bal_0.50 \
        --intervention-mode block_output

Optional flags:
    --n-folds         Number of CV folds (default: 5)
    --n-pca           Number of PCA components to use (default: 10)
    --feature-set     Which features: "pca", "raw", "scalar", "pca+scalar" (default: pca+scalar)
    --intervention-mode
                      Runtime intervention mode to embed in the artifact
    --output-dir      Output directory (default: <data-dir>/<model-key>/router/)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from model.backend import InterventionMode, normalize_intervention_mode
from signal_lab.sweep_cartridges import BALANCED_SWEEP_G_SPECS
from signal_lab.sequence_analyze import extract_feature_families


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baseline_entropy_vectors(data_dir: Path, model_key: str) -> dict[str, np.ndarray]:
    """Extract per-head attention entropy from baseline rows in verbose.jsonl.

    Returns dict mapping prompt_id -> flat entropy vector.
    """
    verbose_path = data_dir / model_key / "verbose.jsonl"
    if not verbose_path.exists():
        raise FileNotFoundError(f"Missing {verbose_path}")

    vectors: dict[str, np.ndarray] = {}
    with open(verbose_path) as f:
        for line in f:
            d = json.loads(line)
            # Baseline rows have all g_attention_scales == 1.0
            if not all(s == 1.0 for s in d["g_attention_scales"]):
                continue
            pid = d["prompt_id"]
            # Flatten layers × heads into a single vector
            entropy = []
            for layer_entropies in d["attn_entropy_per_head_final"]:
                entropy.extend(layer_entropies)
            vectors[pid] = np.array(entropy, dtype=np.float64)

    return vectors


def load_scalar_features(analysis_dir: Path) -> dict[str, dict[str, float]]:
    """Load scalar baseline features from analysis_joined_long.csv.

    Returns dict mapping prompt_id -> {feature_name: value}.
    Only reads baseline rows (one per prompt).
    """
    joined_path = analysis_dir / "analysis_joined_long.csv"
    feature_cols = [
        "baseline_final_entropy_bits",
        "baseline_mean_entropy_bits",
        "baseline_top1_top2_logit_margin",
        "baseline_attn_entropy_mean",
        "baseline_target_prob",
    ]

    features: dict[str, dict[str, float]] = {}
    with open(joined_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["g_profile"] != "baseline":
                continue
            pid = row["prompt_id"]
            feat = {}
            for col in feature_cols:
                try:
                    feat[col] = float(row[col])
                except (ValueError, TypeError):
                    feat[col] = 0.0
            features[pid] = feat

    return features


def load_sequence_family_vectors(states_dir: Path, family_name: str) -> dict[str, np.ndarray]:
    """Load raw hidden-state family vectors from a sequence collection states dir."""
    if not states_dir.exists() or not states_dir.is_dir():
        raise FileNotFoundError(f"Missing sequence states directory: {states_dir}")

    state_files = sorted(states_dir.glob("*.pt"))
    if not state_files:
        raise FileNotFoundError(f"No .pt state files found in {states_dir}")

    vectors: dict[str, np.ndarray] = {}
    for state_file in state_files:
        payload = torch.load(state_file, map_location="cpu")
        prompt_id = payload["prompt_id"]
        family_vectors = extract_feature_families(payload)
        if family_name not in family_vectors:
            raise KeyError(f"Unknown sequence family '{family_name}'")
        vectors[prompt_id] = np.asarray(family_vectors[family_name], dtype=np.float64)
    return vectors


def load_delta_matrix(analysis_dir: Path, profile_names: list[str]) -> tuple[list[str], np.ndarray]:
    """Load delta_target_prob for the selected profiles.

    Returns:
        prompt_ids: ordered list of prompt IDs
        matrix: P × (len(profile_names)) array of delta_p values
    """
    joined_path = analysis_dir / "analysis_joined_long.csv"
    profile_set = set(profile_names)

    # Collect (prompt_id, profile, delta_p)
    data: dict[str, dict[str, float]] = {}
    with open(joined_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gp = row["g_profile"]
            if gp not in profile_set:
                continue
            pid = row["prompt_id"]
            if pid not in data:
                data[pid] = {}
            data[pid][gp] = float(row["delta_target_prob"])

    prompt_ids = sorted(data.keys())
    matrix = np.zeros((len(prompt_ids), len(profile_names)), dtype=np.float64)
    for i, pid in enumerate(prompt_ids):
        for j, pname in enumerate(profile_names):
            matrix[i, j] = data[pid].get(pname, 0.0)

    return prompt_ids, matrix


def load_profile_specs(profile_names: list[str]) -> dict[str, dict]:
    """Resolve exact runtime profile specs by name from the balanced sweep vocabulary."""
    available = {
        spec["name"]: {k: v for k, v in spec.items() if k != "name"}
        for spec in BALANCED_SWEEP_G_SPECS
    }
    missing = [name for name in profile_names if name not in available]
    if missing:
        raise ValueError(f"Missing profile specs for: {missing}")
    return {name: available[name] for name in profile_names}


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    prompt_ids: list[str],
    entropy_vectors: dict[str, np.ndarray],
    scalar_features: dict[str, dict[str, float]],
    sequence_vectors: dict[str, np.ndarray] | None = None,
    n_pca: int = 10,
    sequence_n_pca: int = 10,
    feature_set: str = "pca+scalar",
) -> tuple[
    np.ndarray,
    list[str],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Build the feature matrix for the classifier.

    Returns:
        X:              P × D feature matrix
        feature_names:  list of feature name strings
        pca_components: the PCA projection matrix (for deployment), or None
        pca_mean:       the PCA centering vector, or None
        sequence_pca_components: PCA matrix for sequence family, or None
        sequence_pca_mean:       centering vector for sequence family, or None
    """
    n = len(prompt_ids)

    # Raw entropy vectors
    entropy_dim = None
    raw_matrix = None
    if entropy_vectors:
        entropy_dim = len(next(iter(entropy_vectors.values())))
        raw_matrix = np.zeros((n, entropy_dim), dtype=np.float64)
        for i, pid in enumerate(prompt_ids):
            if pid in entropy_vectors:
                raw_matrix[i] = entropy_vectors[pid]

    sequence_raw_matrix = None
    if sequence_vectors:
        sequence_dim = len(next(iter(sequence_vectors.values())))
        sequence_raw_matrix = np.zeros((n, sequence_dim), dtype=np.float64)
        missing_sequence = []
        for i, pid in enumerate(prompt_ids):
            vec = sequence_vectors.get(pid)
            if vec is None:
                missing_sequence.append(pid)
                continue
            sequence_raw_matrix[i] = vec
        if missing_sequence:
            preview = ", ".join(missing_sequence[:5])
            raise ValueError(
                f"Missing sequence vectors for {len(missing_sequence)} prompts. "
                f"First few: {preview}"
            )

    # PCA of entropy vectors
    pca_matrix = None
    pca_components = None
    pca_mean = None
    if raw_matrix is not None and feature_set in ("pca", "pca+scalar"):
        pca_mean = raw_matrix.mean(axis=0)
        centered = raw_matrix - pca_mean
        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(n_pca, Vt.shape[0])
        pca_components = Vt[:k]  # k × D
        pca_matrix = centered @ pca_components.T  # P × k
        explained = (S[:k] ** 2) / (S ** 2).sum()
        print(f"  PCA: {k} components, explained variance: "
              f"{explained.sum():.3f} ({', '.join(f'{v:.3f}' for v in explained[:5])}...)")

    sequence_pca_matrix = None
    sequence_pca_components = None
    sequence_pca_mean = None
    if sequence_raw_matrix is not None and "sequence_pca" in feature_set:
        sequence_pca_mean = sequence_raw_matrix.mean(axis=0)
        centered = sequence_raw_matrix - sequence_pca_mean
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(sequence_n_pca, vt.shape[0])
        sequence_pca_components = vt[:k]
        sequence_pca_matrix = centered @ sequence_pca_components.T
        explained = (s[:k] ** 2) / (s ** 2).sum()
        print(
            f"  Sequence PCA: {k} components, explained variance: "
            f"{explained.sum():.3f} ({', '.join(f'{v:.3f}' for v in explained[:5])}...)"
        )

    # Scalar features
    scalar_names = sorted(next(iter(scalar_features.values())).keys()) if scalar_features else []
    scalar_matrix = None
    if scalar_features and feature_set in ("scalar", "pca+scalar"):
        scalar_matrix = np.zeros((n, len(scalar_names)), dtype=np.float64)
        for i, pid in enumerate(prompt_ids):
            if pid in scalar_features:
                for j, name in enumerate(scalar_names):
                    scalar_matrix[i, j] = scalar_features[pid].get(name, 0.0)

    # Assemble
    parts = []
    names = []
    if feature_set == "raw":
        parts.append(raw_matrix)
        names.extend([f"entropy_{i}" for i in range(entropy_dim)])
    elif feature_set == "pca":
        parts.append(pca_matrix)
        names.extend([f"pc{i+1}" for i in range(pca_matrix.shape[1])])
    elif feature_set == "scalar":
        parts.append(scalar_matrix)
        names.extend(scalar_names)
    elif feature_set == "pca+scalar":
        parts.append(pca_matrix)
        names.extend([f"pc{i+1}" for i in range(pca_matrix.shape[1])])
        parts.append(scalar_matrix)
        names.extend(scalar_names)
    elif feature_set == "sequence_pca":
        parts.append(sequence_pca_matrix)
        names.extend([f"seq_pc{i+1}" for i in range(sequence_pca_matrix.shape[1])])
    elif feature_set == "pca+sequence_pca":
        parts.append(pca_matrix)
        names.extend([f"pc{i+1}" for i in range(pca_matrix.shape[1])])
        parts.append(sequence_pca_matrix)
        names.extend([f"seq_pc{i+1}" for i in range(sequence_pca_matrix.shape[1])])
    elif feature_set == "scalar+sequence_pca":
        parts.append(scalar_matrix)
        names.extend(scalar_names)
        parts.append(sequence_pca_matrix)
        names.extend([f"seq_pc{i+1}" for i in range(sequence_pca_matrix.shape[1])])
    elif feature_set == "pca+scalar+sequence_pca":
        parts.append(pca_matrix)
        names.extend([f"pc{i+1}" for i in range(pca_matrix.shape[1])])
        parts.append(scalar_matrix)
        names.extend(scalar_names)
        parts.append(sequence_pca_matrix)
        names.extend([f"seq_pc{i+1}" for i in range(sequence_pca_matrix.shape[1])])

    X = np.hstack([p for p in parts if p is not None])
    return X, names, pca_components, pca_mean, sequence_pca_components, sequence_pca_mean


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def assign_labels(
    delta_matrix: np.ndarray,
    profile_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Assign each prompt to its best profile or 'off'.

    Returns:
        labels:      integer label array (P,), where len(profile_names) = "off"
        class_names: list of class name strings (profiles + "off")
    """
    class_names = list(profile_names) + ["off"]
    off_idx = len(profile_names)

    per_prompt_best_col = delta_matrix.argmax(axis=1)
    per_prompt_best_val = delta_matrix.max(axis=1)

    labels = np.where(per_prompt_best_val > 0, per_prompt_best_col, off_idx)
    return labels, class_names


# ---------------------------------------------------------------------------
# Classifier: multinomial logistic regression (manual, no sklearn dependency)
# ---------------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lr: float = 0.1,
    n_iter: int = 1000,
    reg: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Train multinomial logistic regression via gradient descent.

    Returns (W, b) where W is D × C and b is (C,).
    """
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)

    # One-hot encode y
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    for it in range(n_iter):
        logits = X @ W + b
        probs = softmax(logits)
        grad_W = X.T @ (probs - Y) / n + reg * W
        grad_b = (probs - Y).mean(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


def predict_logistic(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Predict class labels."""
    logits = X @ W + b
    return logits.argmax(axis=1)


def predict_proba_logistic(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Predict class probabilities."""
    logits = X @ W + b
    return softmax(logits)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    delta_matrix: np.ndarray,
    n_classes: int,
    class_names: list[str],
    n_folds: int = 5,
    lr: float = 0.1,
    n_iter: int = 2000,
    reg: float = 1e-3,
) -> dict:
    """Stratified K-fold cross-validation.

    Reports both classification accuracy and routed delta_p.
    """
    n = len(y)
    np.random.seed(42)
    perm = np.random.permutation(n)
    folds = np.array_split(perm, n_folds)

    fold_results = []
    all_preds = np.zeros(n, dtype=int)
    all_probs = np.zeros((n, n_classes), dtype=np.float64)

    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])

        # Standardize features (fit on train)
        mu = X[train_idx].mean(axis=0)
        std = X[train_idx].std(axis=0)
        std[std == 0] = 1.0
        X_train = (X[train_idx] - mu) / std
        X_test = (X[test_idx] - mu) / std

        W, b = train_logistic(X_train, y[train_idx], n_classes, lr=lr, n_iter=n_iter, reg=reg)
        preds = predict_logistic(X_test, W, b)
        probs = predict_proba_logistic(X_test, W, b)

        all_preds[test_idx] = preds
        all_probs[test_idx] = probs

        # Accuracy
        acc = (preds == y[test_idx]).mean()

        # Routed delta_p: for each test prompt, use the predicted profile
        routed_dp = []
        oracle_dp = []
        for i, global_i in enumerate(test_idx):
            pred_class = preds[i]
            if pred_class < delta_matrix.shape[1]:
                routed_dp.append(max(0.0, delta_matrix[global_i, pred_class]))
            else:
                routed_dp.append(0.0)  # "off"
            oracle_dp.append(max(0.0, delta_matrix[global_i].max()))

        fold_results.append({
            "fold": fold_idx,
            "n_test": len(test_idx),
            "accuracy": float(acc),
            "routed_mean_delta_p": float(np.mean(routed_dp)),
            "oracle_mean_delta_p": float(np.mean(oracle_dp)),
        })

    # Aggregate
    overall_acc = (all_preds == y).mean()

    # Full-dataset routed performance
    routed_dp_all = []
    oracle_dp_all = []
    off_count = 0
    for i in range(n):
        pred = all_preds[i]
        if pred < delta_matrix.shape[1]:
            routed_dp_all.append(max(0.0, delta_matrix[i, pred]))
        else:
            routed_dp_all.append(0.0)
            off_count += 1
        oracle_dp_all.append(max(0.0, delta_matrix[i].max()))

    # Class distribution
    class_dist = {}
    pred_dist = {}
    for c in range(n_classes):
        class_dist[class_names[c]] = int((y == c).sum())
        pred_dist[class_names[c]] = int((all_preds == c).sum())

    return {
        "n_folds": n_folds,
        "fold_results": fold_results,
        "overall_accuracy": float(overall_acc),
        "routed_mean_delta_p": float(np.mean(routed_dp_all)),
        "oracle_mean_delta_p": float(np.mean(oracle_dp_all)),
        "off_predictions": off_count,
        "true_class_distribution": class_dist,
        "predicted_class_distribution": pred_dist,
        "all_preds": all_preds,
        "all_probs": all_probs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train routing classifier.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--profiles", nargs=4, required=True,
                        help="Exactly 4 profile names")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-pca", type=int, default=10)
    parser.add_argument("--sequence-n-pca", type=int, default=10)
    parser.add_argument("--feature-set", default="pca+scalar",
                        choices=[
                            "pca",
                            "raw",
                            "scalar",
                            "pca+scalar",
                            "sequence_pca",
                            "pca+sequence_pca",
                            "scalar+sequence_pca",
                            "pca+scalar+sequence_pca",
                        ])
    parser.add_argument("--sequence-states-dir", default=None)
    parser.add_argument(
        "--sequence-family",
        default=None,
        choices=[
            "embedding_last_token",
            "final_layer_last_token",
            "embedding_mean_pool",
            "final_layer_mean_pool",
            "all_layers_last_token_concat",
            "all_layers_mean_pool_concat",
        ],
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument(
        "--intervention-mode",
        default="auto",
        help='Intervention mode for runtime artifact: "attention_contribution", "block_output", or "auto".',
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    analysis_dir = data_dir / args.model_key / "analysis"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / args.model_key / "router"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.intervention_mode == "auto":
        inferred_mode = (
            InterventionMode.BLOCK_OUTPUT.value
            if "block" in str(data_dir).lower()
            else InterventionMode.ATTENTION_CONTRIBUTION.value
        )
    else:
        inferred_mode = args.intervention_mode
    intervention_mode = normalize_intervention_mode(inferred_mode)

    profile_names = args.profiles
    print(f"Model: {args.model_key}")
    print(f"Profiles: {profile_names}")
    print(f"Feature set: {args.feature_set}")
    print(f"Intervention mode: {intervention_mode.value}")
    print()

    # Load data
    print("Loading baseline entropy vectors from verbose.jsonl...")
    t0 = time.time()
    entropy_vectors = load_baseline_entropy_vectors(data_dir, args.model_key)
    print(f"  {len(entropy_vectors)} vectors in {time.time()-t0:.1f}s")

    print("Loading scalar features from joined_long...")
    scalar_features = load_scalar_features(analysis_dir)
    print(f"  {len(scalar_features)} prompts")

    sequence_vectors = None
    if "sequence_pca" in args.feature_set:
        if not args.sequence_states_dir or not args.sequence_family:
            raise ValueError(
                "Sequence feature sets require both --sequence-states-dir and --sequence-family."
            )
        sequence_states_dir = Path(args.sequence_states_dir)
        print(f"Loading sequence vectors ({args.sequence_family}) from states dir...")
        t0 = time.time()
        sequence_vectors = load_sequence_family_vectors(sequence_states_dir, args.sequence_family)
        print(f"  {len(sequence_vectors)} vectors in {time.time()-t0:.1f}s")

    print("Loading delta matrix for selected profiles...")
    prompt_ids, delta_matrix = load_delta_matrix(analysis_dir, profile_names)
    print(f"  {delta_matrix.shape[0]} prompts × {delta_matrix.shape[1]} profiles")

    # Assign labels
    labels, class_names = assign_labels(delta_matrix, profile_names)
    print(f"\nLabel distribution:")
    for c, name in enumerate(class_names):
        count = (labels == c).sum()
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")

    # Build features
    print(f"\nBuilding feature matrix ({args.feature_set}, n_pca={args.n_pca})...")
    X, feature_names, pca_components, pca_mean, sequence_pca_components, sequence_pca_mean = build_feature_matrix(
        prompt_ids, entropy_vectors, scalar_features,
        sequence_vectors=sequence_vectors,
        n_pca=args.n_pca,
        sequence_n_pca=args.sequence_n_pca,
        feature_set=args.feature_set,
    )
    print(f"  Feature matrix: {X.shape}")

    # Train and evaluate
    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    results = cross_validate(
        X, labels, delta_matrix,
        n_classes=len(class_names),
        class_names=class_names,
        n_folds=args.n_folds,
        lr=args.lr,
        n_iter=args.n_iter,
        reg=args.reg,
    )

    # Compute reference points
    best_fixed_dp = max(np.maximum(delta_matrix, 0.0).mean(axis=0))

    # Report
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {args.model_key}")
    print(f"{'=' * 60}")
    print(f"  Baseline (no intervention):    0.0000")
    print(f"  Best fixed profile:            {best_fixed_dp:.4f}")
    print(f"  Routed (classifier):           {results['routed_mean_delta_p']:.4f}")
    print(f"  Oracle (perfect routing):      {results['oracle_mean_delta_p']:.4f}")
    print(f"  Classification accuracy:       {results['overall_accuracy']:.3f}")
    print(f"  Routing efficiency:            "
          f"{results['routed_mean_delta_p'] / results['oracle_mean_delta_p']:.1%} of oracle")

    print(f"\nPer-fold results:")
    for fr in results["fold_results"]:
        print(f"  Fold {fr['fold']}: acc={fr['accuracy']:.3f}  "
              f"routed={fr['routed_mean_delta_p']:.4f}  "
              f"oracle={fr['oracle_mean_delta_p']:.4f}")

    print(f"\nPredicted class distribution:")
    for name in class_names:
        true_n = results["true_class_distribution"][name]
        pred_n = results["predicted_class_distribution"][name]
        print(f"  {name:<30} true={true_n:>4}  predicted={pred_n:>4}")

    # Save results
    save_results = {
        "model_key": args.model_key,
        "profiles": profile_names,
        "intervention_mode": intervention_mode.value,
        "feature_set": args.feature_set,
        "n_pca": args.n_pca,
        "sequence_n_pca": args.sequence_n_pca if "sequence_pca" in args.feature_set else 0,
        "sequence_family": args.sequence_family,
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "n_folds": args.n_folds,
        "hyperparameters": {"lr": args.lr, "n_iter": args.n_iter, "reg": args.reg},
        "baseline_delta_p": 0.0,
        "best_fixed_delta_p": float(best_fixed_dp),
        "routed_delta_p": results["routed_mean_delta_p"],
        "oracle_delta_p": results["oracle_mean_delta_p"],
        "routing_efficiency": results["routed_mean_delta_p"] / results["oracle_mean_delta_p"],
        "overall_accuracy": results["overall_accuracy"],
        "fold_results": results["fold_results"],
        "true_class_distribution": results["true_class_distribution"],
        "predicted_class_distribution": results["predicted_class_distribution"],
    }

    output_path = output_dir / "train_router_results.json"
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save the trained model artifacts (train on full data for deployment)
    print("\nTraining final model on full dataset...")
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X - mu) / std
    W, b = train_logistic(X_std, labels, len(class_names),
                          lr=args.lr, n_iter=args.n_iter, reg=args.reg)

    model_artifacts = {
        "model_key": args.model_key,
        "profiles": profile_names,
        "class_names": class_names,
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "n_pca": args.n_pca,
        "sequence_n_pca": args.sequence_n_pca if "sequence_pca" in args.feature_set else 0,
        "sequence_family": args.sequence_family,
        "intervention_mode": intervention_mode.value,
        "standardization_mean": mu.tolist(),
        "standardization_std": std.tolist(),
        "weights": W.tolist(),
        "bias": b.tolist(),
        "profile_specs": load_profile_specs(profile_names),
    }
    if pca_components is not None:
        model_artifacts["pca_components"] = pca_components.tolist()
        model_artifacts["pca_mean"] = pca_mean.tolist()
    if sequence_pca_components is not None:
        model_artifacts["sequence_pca_components"] = sequence_pca_components.tolist()
        model_artifacts["sequence_pca_mean"] = sequence_pca_mean.tolist()

    model_path = output_dir / "router_model.json"
    with open(model_path, "w") as f:
        json.dump(model_artifacts, f)
    print(f"Model artifacts saved to {model_path}")


if __name__ == "__main__":
    main()
