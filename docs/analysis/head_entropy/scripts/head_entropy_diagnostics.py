from __future__ import annotations

"""
head_entropy_diagnostics.py

Diagnostic analysis for whether baseline per-head attention entropy contains
useful scout signal for prompt-level profile selection.

Entropy definition:
- Uses `attn_entropy_per_head_final` from `verbose.jsonl`
- This is the entropy of each attention head's distribution at the final prompt
  position (the last token in the prompt prefix before target scoring), not an
  average over prompt tokens
- In code, this is computed from `attn[0, :, -1, :]`

Current scope:
- 9B balanced attention-contribution sweep
- selected profile set + baseline/off as the oracle-winner target
- Phase 1/2 diagnostics only:
  1. variance screen
  2. per-head one-vs-rest AUC
  3. sparse multinomial logistic regression

Mutual-information diagnostics are intentionally deferred unless these
Phase 2 metrics remain ambiguous.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.experiments.train_router import load_baseline_entropy_vectors


DEFAULT_9B_PROFILES = [
    "constant_2.6",
    "edges_narrow_bal_0.55",
    "late_boost_bal_0.60",
    "triad_odd_bal_0.45",
]


@dataclass(frozen=True)
class DiagnosticDataset:
    prompt_ids: list[str]
    layer_indices: list[int]
    profile_names: list[str]
    class_names: list[str]
    X_flat: np.ndarray
    X_3d: np.ndarray
    y: np.ndarray
    deltas: np.ndarray

    @property
    def n_layers(self) -> int:
        return self.X_3d.shape[1]

    @property
    def n_heads(self) -> int:
        return self.X_3d.shape[2]


def _load_layer_indices(verbose_path: Path) -> list[int]:
    with verbose_path.open() as f:
        for line in f:
            row = json.loads(line)
            layer_indices = row.get("attn_entropy_layer_indices")
            if layer_indices:
                return [int(x) for x in layer_indices]
    raise ValueError(f"No attn_entropy_layer_indices found in {verbose_path}")


def _load_selected_delta_rows(joined_path: Path, profile_names: list[str]) -> dict[str, dict[str, float]]:
    profile_set = set(profile_names)
    rows: dict[str, dict[str, float]] = {}
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
            rows.setdefault(row["prompt_id"], {})[gp] = delta
    return rows


def build_dataset(
    *,
    data_dir: Path,
    model_key: str,
    profiles: list[str],
) -> DiagnosticDataset:
    verbose_path = data_dir / model_key / "verbose.jsonl"
    joined_path = data_dir / model_key / "analysis" / "analysis_joined_long.csv"

    entropy_vectors = load_baseline_entropy_vectors(data_dir, model_key)
    layer_indices = _load_layer_indices(verbose_path)
    deltas_by_prompt = _load_selected_delta_rows(joined_path, profiles)

    prompt_ids = sorted(
        pid
        for pid in set(entropy_vectors) & set(deltas_by_prompt)
        if all(profile in deltas_by_prompt[pid] for profile in profiles)
    )
    if not prompt_ids:
        raise ValueError("No prompts have both entropy vectors and complete selected-profile deltas.")

    x_flat = np.stack([entropy_vectors[pid] for pid in prompt_ids], axis=0)
    n_layers = len(layer_indices)
    if x_flat.shape[1] % n_layers != 0:
        raise ValueError("Entropy vector width does not divide evenly by number of attention layers.")
    n_heads = x_flat.shape[1] // n_layers
    x_3d = x_flat.reshape(len(prompt_ids), n_layers, n_heads)

    class_names = list(profiles) + ["baseline"]
    y = np.zeros(len(prompt_ids), dtype=np.int64)
    baseline_idx = len(profiles)
    deltas = np.zeros((len(prompt_ids), len(profiles)), dtype=float)
    for i, pid in enumerate(prompt_ids):
        values = np.array([deltas_by_prompt[pid][profile] for profile in profiles], dtype=float)
        deltas[i] = values
        best_idx = int(np.argmax(values))
        best_val = float(values[best_idx])
        y[i] = best_idx if best_val > 0.0 else baseline_idx

    return DiagnosticDataset(
        prompt_ids=prompt_ids,
        layer_indices=layer_indices,
        profile_names=profiles,
        class_names=class_names,
        X_flat=x_flat,
        X_3d=x_3d,
        y=y,
        deltas=deltas,
    )


def residualize_dataset_on_scalar(dataset: DiagnosticDataset, scalar: np.ndarray) -> DiagnosticDataset:
    if scalar.shape[0] != dataset.X_flat.shape[0]:
        raise ValueError("Scalar length must match number of prompts in dataset.")
    design = np.column_stack([np.ones_like(scalar, dtype=float), scalar.astype(float)])
    beta = np.linalg.lstsq(design, dataset.X_flat, rcond=None)[0]
    x_flat_resid = dataset.X_flat - design @ beta
    x_3d_resid = x_flat_resid.reshape(dataset.X_3d.shape)
    return DiagnosticDataset(
        prompt_ids=dataset.prompt_ids,
        layer_indices=dataset.layer_indices,
        profile_names=dataset.profile_names,
        class_names=dataset.class_names,
        X_flat=x_flat_resid,
        X_3d=x_3d_resid,
        y=dataset.y.copy(),
        deltas=dataset.deltas.copy(),
    )


def variance_screen(dataset: DiagnosticDataset) -> dict:
    matrix = dataset.X_3d.var(axis=0)
    rankings: list[dict] = []
    for layer_pos, layer_idx in enumerate(dataset.layer_indices):
        for head_idx in range(dataset.n_heads):
            rankings.append(
                {
                    "layer_position": layer_pos,
                    "layer_index": layer_idx,
                    "head_index": head_idx,
                    "variance": float(matrix[layer_pos, head_idx]),
                }
            )
    rankings.sort(key=lambda row: row["variance"], reverse=True)
    return {
        "matrix": matrix,
        "rankings": rankings,
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    auc = roc_auc_score(y_true, scores)
    return float(max(auc, 1.0 - auc))


def one_vs_rest_auc(
    dataset: DiagnosticDataset,
    *,
    n_permutations: int,
    random_state: int,
) -> dict:
    rng = np.random.default_rng(random_state)
    n_classes = len(dataset.class_names)
    auc_by_profile = np.zeros((n_classes, dataset.n_layers, dataset.n_heads), dtype=float)
    null_mean_by_profile = np.zeros_like(auc_by_profile)
    null_q95_by_profile = np.zeros_like(auc_by_profile)
    pvalue_by_profile = np.zeros_like(auc_by_profile)

    labels = dataset.y

    for class_idx in range(n_classes):
        y_binary = (labels == class_idx).astype(int)
        null_samples = np.zeros((n_permutations, dataset.n_layers, dataset.n_heads), dtype=float)
        for layer_pos in range(dataset.n_layers):
            for head_idx in range(dataset.n_heads):
                scores = dataset.X_3d[:, layer_pos, head_idx]
                observed = _safe_auc(y_binary, scores)
                auc_by_profile[class_idx, layer_pos, head_idx] = observed
                for perm_idx in range(n_permutations):
                    shuffled = rng.permutation(y_binary)
                    null_samples[perm_idx, layer_pos, head_idx] = _safe_auc(shuffled, scores)

        null_mean_by_profile[class_idx] = null_samples.mean(axis=0)
        null_q95_by_profile[class_idx] = np.quantile(null_samples, 0.95, axis=0)
        pvalue_by_profile[class_idx] = (
            (null_samples >= auc_by_profile[class_idx][None, :, :]).mean(axis=0)
        )

    max_auc_matrix = auc_by_profile.max(axis=0)
    best_profile_idx = auc_by_profile.argmax(axis=0)
    best_profile_matrix = np.vectorize(dataset.class_names.__getitem__)(best_profile_idx)

    rankings: list[dict] = []
    for class_idx, class_name in enumerate(dataset.class_names):
        for layer_pos, layer_idx in enumerate(dataset.layer_indices):
            for head_idx in range(dataset.n_heads):
                rankings.append(
                    {
                        "profile": class_name,
                        "layer_position": layer_pos,
                        "layer_index": layer_idx,
                        "head_index": head_idx,
                        "auc": float(auc_by_profile[class_idx, layer_pos, head_idx]),
                        "null_mean": float(null_mean_by_profile[class_idx, layer_pos, head_idx]),
                        "null_q95": float(null_q95_by_profile[class_idx, layer_pos, head_idx]),
                        "p_value_perm_ge": float(pvalue_by_profile[class_idx, layer_pos, head_idx]),
                    }
                )
    rankings.sort(key=lambda row: row["auc"], reverse=True)

    return {
        "auc_by_profile": auc_by_profile,
        "null_mean_by_profile": null_mean_by_profile,
        "null_q95_by_profile": null_q95_by_profile,
        "pvalue_by_profile": pvalue_by_profile,
        "max_auc_matrix": max_auc_matrix,
        "best_profile_matrix": best_profile_matrix,
        "rankings": rankings,
    }


def specialist_vs_anchor_auc(
    dataset: DiagnosticDataset,
    *,
    anchor_profile: str,
    n_permutations: int,
    random_state: int,
) -> dict:
    if anchor_profile not in dataset.profile_names:
        raise ValueError(f"Anchor profile {anchor_profile!r} not in selected profile set.")

    rng = np.random.default_rng(random_state)
    anchor_idx = dataset.profile_names.index(anchor_profile)
    specialist_names = [p for p in dataset.profile_names if p != anchor_profile]
    auc_by_specialist = np.zeros((len(specialist_names), dataset.n_layers, dataset.n_heads), dtype=float)
    null_mean = np.zeros_like(auc_by_specialist)
    null_q95 = np.zeros_like(auc_by_specialist)
    pvalue = np.zeros_like(auc_by_specialist)

    anchor_delta = dataset.deltas[:, anchor_idx]
    for spec_idx, spec_name in enumerate(specialist_names):
        profile_idx = dataset.profile_names.index(spec_name)
        y_binary = (dataset.deltas[:, profile_idx] > anchor_delta).astype(int)
        null_samples = np.zeros((n_permutations, dataset.n_layers, dataset.n_heads), dtype=float)
        for layer_pos in range(dataset.n_layers):
            for head_idx in range(dataset.n_heads):
                scores = dataset.X_3d[:, layer_pos, head_idx]
                observed = _safe_auc(y_binary, scores)
                auc_by_specialist[spec_idx, layer_pos, head_idx] = observed
                for perm_idx in range(n_permutations):
                    shuffled = rng.permutation(y_binary)
                    null_samples[perm_idx, layer_pos, head_idx] = _safe_auc(shuffled, scores)
        null_mean[spec_idx] = null_samples.mean(axis=0)
        null_q95[spec_idx] = np.quantile(null_samples, 0.95, axis=0)
        pvalue[spec_idx] = (null_samples >= auc_by_specialist[spec_idx][None, :, :]).mean(axis=0)

    max_auc_matrix = auc_by_specialist.max(axis=0)
    best_specialist_idx = auc_by_specialist.argmax(axis=0)
    best_specialist_matrix = np.vectorize(specialist_names.__getitem__)(best_specialist_idx)

    rankings: list[dict] = []
    for spec_idx, spec_name in enumerate(specialist_names):
        profile_idx = dataset.profile_names.index(spec_name)
        y_binary = (dataset.deltas[:, profile_idx] > anchor_delta).astype(int)
        positive_rate = float(np.mean(y_binary))
        for layer_pos, layer_idx in enumerate(dataset.layer_indices):
            for head_idx in range(dataset.n_heads):
                rankings.append(
                    {
                        "profile": spec_name,
                        "layer_position": layer_pos,
                        "layer_index": layer_idx,
                        "head_index": head_idx,
                        "auc": float(auc_by_specialist[spec_idx, layer_pos, head_idx]),
                        "null_mean": float(null_mean[spec_idx, layer_pos, head_idx]),
                        "null_q95": float(null_q95[spec_idx, layer_pos, head_idx]),
                        "p_value_perm_ge": float(pvalue[spec_idx, layer_pos, head_idx]),
                        "positive_rate_profile_beats_anchor": positive_rate,
                    }
                )
    rankings.sort(key=lambda row: row["auc"], reverse=True)

    return {
        "anchor_profile": anchor_profile,
        "specialist_names": specialist_names,
        "auc_by_specialist": auc_by_specialist,
        "null_mean_by_specialist": null_mean,
        "null_q95_by_specialist": null_q95,
        "pvalue_by_specialist": pvalue,
        "max_auc_matrix": max_auc_matrix,
        "best_specialist_matrix": best_specialist_matrix,
        "rankings": rankings,
    }


def _fit_sparse_multinomial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
    random_state: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        C=C,
        random_state=random_state,
    )
    clf.fit(x_train_std, y_train)
    return scaler, clf


def sparse_logistic_diagnostic(
    dataset: DiagnosticDataset,
    *,
    n_splits: int,
    n_permutations: int,
    random_state: int,
    C: float,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_true = np.zeros(len(dataset.prompt_ids), dtype=np.int64)
    y_pred = np.zeros(len(dataset.prompt_ids), dtype=np.int64)
    selected_counts = np.zeros(dataset.X_flat.shape[1], dtype=int)

    fold_rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(dataset.X_flat, dataset.y)):
        scaler, clf = _fit_sparse_multinomial(
            dataset.X_flat[train_idx],
            dataset.y[train_idx],
            C=C,
            random_state=random_state + fold_idx,
        )
        x_test_std = scaler.transform(dataset.X_flat[test_idx])
        preds = clf.predict(x_test_std)

        y_true[test_idx] = dataset.y[test_idx]
        y_pred[test_idx] = preds

        nonzero = np.any(np.abs(clf.coef_) > 1e-12, axis=0)
        selected_counts += nonzero.astype(int)
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_test": int(len(test_idx)),
                "accuracy": float(np.mean(preds == dataset.y[test_idx])),
                "macro_f1": float(f1_score(dataset.y[test_idx], preds, average="macro")),
                "n_selected_heads": int(nonzero.sum()),
            }
        )

    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(len(dataset.class_names)))

    rng = np.random.default_rng(random_state)
    null_accuracies = []
    null_macro_f1s = []
    for perm_idx in range(n_permutations):
        shuffled_y = rng.permutation(dataset.y)
        perm_pred = np.zeros_like(shuffled_y)
        perm_true = np.zeros_like(shuffled_y)
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(dataset.X_flat, shuffled_y)):
            scaler, clf = _fit_sparse_multinomial(
                dataset.X_flat[train_idx],
                shuffled_y[train_idx],
                C=C,
                random_state=random_state + 1000 + perm_idx * n_splits + fold_idx,
            )
            x_test_std = scaler.transform(dataset.X_flat[test_idx])
            perm_pred[test_idx] = clf.predict(x_test_std)
            perm_true[test_idx] = shuffled_y[test_idx]
        null_accuracies.append(float(np.mean(perm_pred == perm_true)))
        null_macro_f1s.append(float(f1_score(perm_true, perm_pred, average="macro")))

    selection_freq_matrix = (selected_counts / n_splits).reshape(dataset.n_layers, dataset.n_heads)
    rankings: list[dict] = []
    for layer_pos, layer_idx in enumerate(dataset.layer_indices):
        for head_idx in range(dataset.n_heads):
            rankings.append(
                {
                    "layer_position": layer_pos,
                    "layer_index": layer_idx,
                    "head_index": head_idx,
                    "selection_frequency": float(selection_freq_matrix[layer_pos, head_idx]),
                }
            )
    rankings.sort(key=lambda row: row["selection_frequency"], reverse=True)

    chance_accuracy = max(np.bincount(dataset.y)) / len(dataset.y)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "chance_accuracy_majority": float(chance_accuracy),
        "confusion_matrix": confusion,
        "fold_rows": fold_rows,
        "selection_frequency_matrix": selection_freq_matrix,
        "selection_rankings": rankings,
        "null_accuracy_mean": float(np.mean(null_accuracies)),
        "null_accuracy_q95": float(np.quantile(null_accuracies, 0.95)),
        "null_accuracy_p_value_ge": float(np.mean(np.array(null_accuracies) >= accuracy)),
        "null_macro_f1_mean": float(np.mean(null_macro_f1s)),
        "null_macro_f1_q95": float(np.quantile(null_macro_f1s, 0.95)),
        "null_macro_f1_p_value_ge": float(np.mean(np.array(null_macro_f1s) >= macro_f1)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    *,
    data_dir: Path,
    model_key: str,
    profiles: list[str],
    output_dir: Path,
    n_permutations_auc: int,
    n_permutations_logistic: int,
    n_splits: int,
    random_state: int,
    logistic_c: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(data_dir=data_dir, model_key=model_key, profiles=profiles)
    variance = variance_screen(dataset)
    auc = one_vs_rest_auc(dataset, n_permutations=n_permutations_auc, random_state=random_state)
    specialist_auc = specialist_vs_anchor_auc(
        dataset,
        anchor_profile=profiles[0],
        n_permutations=n_permutations_auc,
        random_state=random_state + 1,
    )
    logistic = sparse_logistic_diagnostic(
        dataset,
        n_splits=n_splits,
        n_permutations=n_permutations_logistic,
        random_state=random_state,
        C=logistic_c,
    )

    summary = {
        "model_key": model_key,
        "data_dir": str(data_dir),
        "profiles": profiles,
        "class_names": dataset.class_names,
        "n_prompts": len(dataset.prompt_ids),
        "n_layers": dataset.n_layers,
        "n_heads": dataset.n_heads,
        "layer_indices": dataset.layer_indices,
        "entropy_definition": (
            "Per-head attention entropy at the final prompt position "
            "(`attn[0, :, -1, :]`), not averaged over prompt tokens."
        ),
        "class_balance": {
            class_name: int((dataset.y == idx).sum())
            for idx, class_name in enumerate(dataset.class_names)
        },
        "variance": {
            "matrix": variance["matrix"].tolist(),
            "top_rankings": variance["rankings"][:50],
        },
        "auc": {
            "max_auc_matrix": auc["max_auc_matrix"].tolist(),
            "best_profile_matrix": auc["best_profile_matrix"].tolist(),
            "top_rankings": auc["rankings"][:100],
        },
        "specialist_vs_anchor_auc": {
            "anchor_profile": specialist_auc["anchor_profile"],
            "specialist_names": specialist_auc["specialist_names"],
            "max_auc_matrix": specialist_auc["max_auc_matrix"].tolist(),
            "best_specialist_matrix": specialist_auc["best_specialist_matrix"].tolist(),
            "top_rankings": specialist_auc["rankings"][:100],
        },
        "sparse_logistic": {
            "accuracy": logistic["accuracy"],
            "macro_f1": logistic["macro_f1"],
            "chance_accuracy_majority": logistic["chance_accuracy_majority"],
            "confusion_matrix": logistic["confusion_matrix"].tolist(),
            "null_accuracy_mean": logistic["null_accuracy_mean"],
            "null_accuracy_q95": logistic["null_accuracy_q95"],
            "null_accuracy_p_value_ge": logistic["null_accuracy_p_value_ge"],
            "null_macro_f1_mean": logistic["null_macro_f1_mean"],
            "null_macro_f1_q95": logistic["null_macro_f1_q95"],
            "null_macro_f1_p_value_ge": logistic["null_macro_f1_p_value_ge"],
            "selection_frequency_matrix": logistic["selection_frequency_matrix"].tolist(),
            "top_selected_heads": logistic["selection_rankings"][:50],
            "fold_rows": logistic["fold_rows"],
        },
    }

    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(output_dir / "variance_rankings.csv", variance["rankings"])
    _write_csv(output_dir / "auc_rankings.csv", auc["rankings"])
    _write_csv(output_dir / "specialist_vs_anchor_auc_rankings.csv", specialist_auc["rankings"])
    _write_csv(output_dir / "logistic_selected_heads.csv", logistic["selection_rankings"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run head-entropy scout diagnostics.")
    parser.add_argument("--data-dir", default="data/022-balanced-attention-hybrid")
    parser.add_argument("--model-key", default="9B")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_9B_PROFILES)
    parser.add_argument("--output-dir", default="docs/analysis/head_entropy/outputs/qwen9b/raw")
    parser.add_argument("--n-permutations-auc", type=int, default=50)
    parser.add_argument("--n-permutations-logistic", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--logistic-c", type=float, default=0.2)
    args = parser.parse_args()

    summary = run_analysis(
        data_dir=Path(args.data_dir),
        model_key=args.model_key,
        profiles=list(args.profiles),
        output_dir=Path(args.output_dir),
        n_permutations_auc=args.n_permutations_auc,
        n_permutations_logistic=args.n_permutations_logistic,
        n_splits=args.n_splits,
        random_state=args.random_state,
        logistic_c=args.logistic_c,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
