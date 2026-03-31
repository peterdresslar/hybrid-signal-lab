#!/usr/bin/env python3
"""Create a training/test annotation manifest from calibration analyses.

This script reads one or more ``analysis_item_cross_model.csv`` files, merges
them by battery item ``id``, derives a simple consensus difficulty bucket, and
then assigns each eligible prompt to ``train_prompt`` or ``test_prompt``.

Items that are too easy, too hard, or otherwise outside the useful calibration
band are assigned to ``other``.

Typical usage:

    uv run -m battery.src.annotate_battery \
      --analysis-dir ~/workspace/data/dropzone/calibration/b4 \
      --candidates battery/data/battery_4/all_candidates.json \
      --output ~/workspace/data/dropzone/calibration/b4/annotation_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train/test annotation manifest from calibration analyses."
    )
    parser.add_argument(
        "--analysis-dir",
        "--calibration-dir",
        dest="analysis_dir",
        required=True,
        help="Directory containing per-model analysis_item_cross_model.csv files.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Optional all_candidates.json path for backfilling battery item metadata.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the annotation manifest JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split assignment.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of eligible prompts to assign to test within each type.",
    )
    parser.add_argument(
        "--min-eligible-per-type",
        type=int,
        default=10,
        help="Minimum eligible prompts in a type before any are held out for test.",
    )
    parser.add_argument(
        "--min-test-per-type",
        type=int,
        default=2,
        help="Minimum held-out prompts for sufficiently large eligible type groups.",
    )
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=0.01,
        help="Items with max target prob below this across all runs become too_hard.",
    )
    parser.add_argument(
        "--easy-threshold",
        type=float,
        default=0.85,
        help="Informational label threshold: items with min prob above this are labeled easy.",
    )
    parser.add_argument(
        "--separating-low",
        type=float,
        default=0.05,
        help="Informational label: model-separating when min prob is below this.",
    )
    parser.add_argument(
        "--separating-high",
        type=float,
        default=0.25,
        help="Informational label: model-separating when max prob is above this.",
    )
    parser.add_argument(
        "--sweet-min",
        type=float,
        default=0.05,
        help="Informational label: sweet-spot lower bound on median prob.",
    )
    parser.add_argument(
        "--sweet-max",
        type=float,
        default=0.60,
        help="Informational label: sweet-spot upper bound on median prob.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def load_candidates(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    items = json.loads(path.read_text())
    return {str(item["id"]): item for item in items}


def find_analysis_files(base: Path) -> list[Path]:
    files = sorted(base.rglob("analysis_item_cross_model.csv"))
    if not files:
        raise FileNotFoundError(f"No analysis_item_cross_model.csv files found under {base}")
    return files


def detect_model_prefix(fieldnames: list[str]) -> str:
    prefixes = [name[: -len("__target_prob")] for name in fieldnames if name.endswith("__target_prob")]
    if not prefixes:
        raise ValueError("Could not find a __target_prob column in analysis CSV")
    return prefixes[0]


def merge_analysis(
    analysis_files: list[Path],
    candidate_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    merged: dict[str, dict[str, Any]] = {}
    model_prefixes: list[str] = []

    for path in analysis_files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV {path} has no header")
            model_prefix = detect_model_prefix(reader.fieldnames)
            model_prefixes.append(model_prefix)

            prob_col = f"{model_prefix}__target_prob"
            rank_col = f"{model_prefix}__target_rank"
            entropy_col = f"{model_prefix}__final_entropy"
            seq_entropy_col = f"{model_prefix}__mean_seq_entropy"

            for row in reader:
                item_id = str(row["id"])
                item = merged.setdefault(item_id, {"id": item_id, "models": {}})

                if "type" not in item:
                    item["type"] = normalize_text(row.get("type"))
                    item["tier"] = normalize_text(row.get("tier"))
                    item["family"] = normalize_text(row.get("family"))
                    item["concept"] = normalize_text(row.get("concept"))
                    item["difficulty"] = normalize_text(row.get("difficulty"))
                    item["source"] = normalize_text(row.get("source"))
                    item["prompt"] = row.get("prompt") or None
                    item["target"] = row.get("target") or None
                    item["tokens_approx"] = parse_int(row.get("tokens_approx"))
                    item["target_num_tokens"] = parse_int(row.get("target_num_tokens"))

                item["models"][model_prefix] = {
                    "target_prob": parse_float(row.get(prob_col)),
                    "target_rank": parse_int(row.get(rank_col)),
                    "final_entropy": parse_float(row.get(entropy_col)),
                    "mean_seq_entropy": parse_float(row.get(seq_entropy_col)),
                }

    for item_id, item in merged.items():
        candidate = candidate_map.get(item_id)
        if not candidate:
            continue
        item.setdefault("type", candidate.get("type"))
        item.setdefault("tier", candidate.get("tier"))
        item.setdefault("source", candidate.get("source"))
        item.setdefault("prompt", candidate.get("prompt"))
        item.setdefault("target", candidate.get("target"))
        item.setdefault("tokens_approx", candidate.get("tokens_approx"))
        item.setdefault("target_num_tokens", None)
        metadata = candidate.get("metadata", {}) or {}
        item.setdefault("family", normalize_text(metadata.get("family")))
        item.setdefault("concept", normalize_text(metadata.get("concept")))
        item.setdefault("difficulty", normalize_text(metadata.get("difficulty")))

    return merged, model_prefixes


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def classify_item(item: dict[str, Any], args: argparse.Namespace) -> None:
    probs = [stats["target_prob"] for stats in item["models"].values() if stats["target_prob"] is not None]
    ranks = [float(stats["target_rank"]) for stats in item["models"].values() if stats["target_rank"] is not None]
    if not probs:
        item["bucket"] = "too_hard"
        item["split"] = "other"
        item["selection_reason"] = "missing_probabilities"
        return

    item["prob_min"] = min(probs)
    item["prob_max"] = max(probs)
    item["prob_median"] = median(probs)
    item["prob_mean"] = sum(probs) / len(probs)
    item["rank_median"] = median(ranks) if ranks else None

    # Simple two-bucket classification: is any model able to engage with this
    # prompt at all?  If not, the prompt is too hard for gain interventions to
    # have a measurable effect.  Everything else is eligible for train/test.
    #
    # Informational sub-labels (model_separating, easy, sweet_spot) are stored
    # as metadata but do NOT gate eligibility.
    if item["prob_max"] < args.hard_threshold:
        bucket = "too_hard"
        reason = "max_prob_below_hard_threshold"
    else:
        bucket = "eligible"
        # Attach informational sub-labels for downstream analysis
        if item["prob_min"] > args.easy_threshold:
            reason = "eligible_easy"
        elif item["prob_min"] < args.separating_low and item["prob_max"] > args.separating_high:
            reason = "eligible_model_separating"
        elif args.sweet_min <= item["prob_median"] <= args.sweet_max:
            reason = "eligible_sweet_spot"
        else:
            reason = "eligible"

    item["bucket"] = bucket
    item["split"] = "other"
    item["selection_reason"] = reason


def pick_test_ids(rows: list[dict[str, Any]], target_count: int, rng: random.Random) -> set[str]:
    if target_count <= 0:
        return set()

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = row.get("family") or row.get("concept") or row.get("source") or "unknown"
        strata[(row["bucket"], family)].append(row)

    for items in strata.values():
        rng.shuffle(items)

    selected: list[dict[str, Any]] = []
    while len(selected) < target_count:
        active = [(key, items) for key, items in strata.items() if items]
        if not active:
            break
        active.sort(key=lambda pair: (-len(pair[1]), pair[0][0], pair[0][1]))
        for _, items in active:
            if len(selected) >= target_count:
                break
            selected.append(items.pop())

    return {row["id"] for row in selected}


def assign_splits(items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item["bucket"] == "eligible":
            by_type[item["type"]].append(item)

    for type_name, rows in sorted(by_type.items()):
        if len(rows) < args.min_eligible_per_type:
            for row in rows:
                row["split"] = "train_prompt"
                row["selection_reason"] = f"{row['selection_reason']};insufficient_group_for_holdout"
            continue

        target_test = max(args.min_test_per_type, round(len(rows) * args.test_ratio))
        target_test = min(target_test, len(rows) - 1)

        test_ids = pick_test_ids(rows, target_test, rng)
        for row in rows:
            row["split"] = "test_prompt" if row["id"] in test_ids else "train_prompt"
            row["selection_reason"] = f"{row['selection_reason']};eligible_for_split"


def build_manifest(
    merged: dict[str, dict[str, Any]],
    model_prefixes: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    items = sorted(merged.values(), key=lambda item: item["id"])

    for item in items:
        classify_item(item, args)
    assign_splits(items, args)

    split_counts = Counter(item["split"] for item in items)
    bucket_counts = Counter(item["bucket"] for item in items)

    by_type_split: dict[str, dict[str, int]] = {}
    for item in items:
        type_name = item["type"] or "unknown"
        row = by_type_split.setdefault(type_name, {"train_prompt": 0, "test_prompt": 0, "other": 0})
        row[item["split"]] += 1

    manifest_items = []
    for item in items:
        manifest_items.append(
            {
                "id": item["id"],
                "type": item.get("type"),
                "tier": item.get("tier"),
                "family": item.get("family"),
                "concept": item.get("concept"),
                "difficulty": item.get("difficulty"),
                "source": item.get("source"),
                "split": item["split"],
                "bucket": item["bucket"],
                "selection_reason": item["selection_reason"],
                "prob_min": round(item.get("prob_min", 0.0), 6) if item.get("prob_min") is not None else None,
                "prob_median": round(item.get("prob_median", 0.0), 6) if item.get("prob_median") is not None else None,
                "prob_max": round(item.get("prob_max", 0.0), 6) if item.get("prob_max") is not None else None,
                "prob_mean": round(item.get("prob_mean", 0.0), 6) if item.get("prob_mean") is not None else None,
                "rank_median": item.get("rank_median"),
                "target_num_tokens": item.get("target_num_tokens"),
                "tokens_approx": item.get("tokens_approx"),
                "models": item["models"],
            }
        )

    return {
        "version": "annotation_manifest_v1",
        "analysis_dir": str(Path(args.analysis_dir).resolve()),
        "candidates": str(Path(args.candidates).resolve()) if args.candidates else None,
        "models": model_prefixes,
        "thresholds": {
            "hard_threshold": args.hard_threshold,
            "easy_threshold": args.easy_threshold,
            "sweet_min": args.sweet_min,
            "sweet_max": args.sweet_max,
            "separating_low": args.separating_low,
            "separating_high": args.separating_high,
            "test_ratio": args.test_ratio,
            "min_eligible_per_type": args.min_eligible_per_type,
            "min_test_per_type": args.min_test_per_type,
            "seed": args.seed,
        },
        "summary": {
            "total_items": len(items),
            "split_counts": dict(split_counts),
            "bucket_counts": dict(bucket_counts),
            "per_type_split_counts": by_type_split,
        },
        "items": manifest_items,
    }


def print_summary(manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    print(f"Annotated {summary['total_items']} battery items")
    print("Split counts:")
    for key in ("train_prompt", "test_prompt", "other"):
        print(f"  {key:12s}: {summary['split_counts'].get(key, 0)}")
    print("Bucket counts:")
    for key in ("eligible", "too_hard"):
        print(f"  {key:17s}: {summary['bucket_counts'].get(key, 0)}")


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    candidates_path = Path(args.candidates).expanduser().resolve() if args.candidates else None
    output_path = Path(args.output).expanduser().resolve()

    candidate_map = load_candidates(candidates_path)
    analysis_files = find_analysis_files(analysis_dir)
    merged, model_prefixes = merge_analysis(analysis_files, candidate_map)
    manifest = build_manifest(merged, model_prefixes, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    print_summary(manifest)
    print(f"Wrote annotation manifest to {output_path}")


if __name__ == "__main__":
    main()
