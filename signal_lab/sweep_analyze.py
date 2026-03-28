#!/usr/bin/env python3
"""Analyze a single sweep run directory.

This script reads a sweep output directory containing at least `main.jsonl` and
`_meta.json`, joins rows against battery prompt metadata, computes deltas
relative to the baseline gain profile for each prompt/rep pair, and writes a
bundle of analysis artifacts.

Example:
    uv run -m signal_lab.sweep_analyze --run-dir data/sweep_sample
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from signal_lab.paths import DATA_DIR_ENV_VAR, configure_data_dir, default_analysis_output_dir, resolve_input_path


REPORT_FLOAT_DIGITS = 2
DEFAULT_PREFIX = "analysis"
BASELINE_PROFILE_NAME = "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a single sweep run directory.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Sweep run directory containing main.jsonl and _meta.json.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Optional base directory to use in place of data/. Also supports {DATA_DIR_ENV_VAR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated artifacts (default: <run-dir>/analysis).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help=f"Filename prefix for generated artifacts (default: {DEFAULT_PREFIX}).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write a machine-readable JSON report.",
    )
    parser.add_argument(
        "--no-write-files",
        action="store_true",
        help="Print the report without writing CSV/text artifacts.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_num}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_num}")
            rows.append(obj)
    return rows


def to_float(value: Any) -> float:
    if value in {None, ""}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def flatten_numeric_values(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numeric_values(item))
        return out
    numeric = to_float(value)
    return [numeric] if math.isfinite(numeric) else []


def build_verbose_baseline_lookup(
    verbose_records: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    warnings: list[str] = []
    for record in verbose_records:
        prompt_id = record.get("prompt_id")
        if not isinstance(prompt_id, str):
            continue
        rep = int(record.get("rep", 1))
        g_scales = record.get("g_attention_scales")
        is_baseline = (
            isinstance(g_scales, list)
            and g_scales
            and all(abs(to_float(scale) - 1.0) < 1e-6 for scale in g_scales)
        )
        if not is_baseline:
            continue
        key = (prompt_id, rep)
        if key in lookup:
            warnings.append(f"Duplicate verbose baseline row for prompt_id={prompt_id} rep={rep}")
            continue
        lookup[key] = record
    return lookup, warnings


def compute_verbose_baseline_metrics(verbose_row: dict[str, Any] | None) -> dict[str, float]:
    if verbose_row is None:
        return {
            "baseline_mean_entropy_bits": math.nan,
            "baseline_top1_top2_logit_margin": math.nan,
            "baseline_attn_entropy_mean": math.nan,
        }

    top_k_logits = verbose_row.get("top_k_logits")
    top1_top2_margin = math.nan
    if isinstance(top_k_logits, list) and len(top_k_logits) >= 2:
        top1 = to_float(top_k_logits[0])
        top2 = to_float(top_k_logits[1])
        if math.isfinite(top1) and math.isfinite(top2):
            top1_top2_margin = top1 - top2

    attn_values = flatten_numeric_values(verbose_row.get("attn_entropy_per_head_final"))
    return {
        "baseline_mean_entropy_bits": to_float(verbose_row.get("mean_entropy_bits")),
        "baseline_top1_top2_logit_margin": top1_top2_margin,
        "baseline_attn_entropy_mean": mean(attn_values),
    }


def resolve_run_dir(path_str: str) -> Path:
    run_dir = resolve_input_path(path_str)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory must be a directory: {run_dir}")
    return run_dir


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else math.nan


def percentile(values: list[float], q: float) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return math.nan
    return float(np.percentile(np.array(clean, dtype=np.float64), q))


def mean_top_k(values: list[float], k: int) -> float:
    clean = sorted((value for value in values if math.isfinite(value)), reverse=True)
    if not clean:
        return math.nan
    return mean(clean[: min(k, len(clean))])


def kth_largest(values: list[float], k: int) -> float:
    clean = sorted((value for value in values if math.isfinite(value)), reverse=True)
    if not clean or k <= 0:
        return math.nan
    return clean[min(k, len(clean)) - 1]


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan
    return 100.0 * numerator / denominator


def round_report_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return round(value, REPORT_FLOAT_DIGITS)
    return value


def round_report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: round_report_value(value) for key, value in row.items()}


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{REPORT_FLOAT_DIGITS}f}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    parts = [render_row(headers), render_row(["-" * width for width in widths])]
    parts.extend(render_row(row) for row in rows)
    return "\n".join(parts)


def classify_g_family(g_profile: str, g_scales: list[float] | None) -> str:
    if g_profile == BASELINE_PROFILE_NAME:
        return "baseline"
    family_prefixes = [
        "early_boost",
        "late_boost",
        "middle_bump",
        "early_suppress",
        "late_suppress",
        "ramp_up",
        "ramp_down",
        "edges_high",
        "edges_low",
        "early_high_late_low",
        "late_high_early_low",
        "constant",
    ]
    for prefix in family_prefixes:
        if g_profile.startswith(prefix):
            return prefix

    if g_scales:
        if all(abs(scale - 1.0) < 1e-6 for scale in g_scales):
            return "baseline"
        if len(set(round(scale, 6) for scale in g_scales)) == 1:
            return "constant"
    return "other"


def sign_test_pvalue(deltas: list[float]) -> float:
    """Two-sided exact sign test on non-zero finite deltas."""
    clean = [delta for delta in deltas if math.isfinite(delta) and abs(delta) > 1e-12]
    if not clean:
        return math.nan
    positives = sum(delta > 0 for delta in clean)
    negatives = sum(delta < 0 for delta in clean)
    n = positives + negatives
    k = min(positives, negatives)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def resolve_battery_collection_paths(path_or_name: str | Path) -> list[Path]:
    resolved = Path(path_or_name).expanduser()
    if resolved.is_file():
        if resolved.suffix != ".json":
            raise ValueError(f"Battery collection must be JSON: {resolved}")
        return [resolved]

    if resolved.is_dir():
        combined_path = resolved / "all_candidates.json"
        if combined_path.is_file():
            return [combined_path]

        manifest_path = resolved / "manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            manifest_types = manifest.get("types", {}) if isinstance(manifest, dict) else {}
            paths: list[Path] = []
            if isinstance(manifest_types, dict):
                for spec in manifest_types.values():
                    if not isinstance(spec, dict):
                        continue
                    rel_file = spec.get("file")
                    if isinstance(rel_file, str):
                        candidate = resolved / rel_file
                        if candidate.is_file():
                            paths.append(candidate)
            if paths:
                return paths

        json_paths = [
            path for path in sorted(resolved.glob("*.json"))
            if path.name != "manifest.json"
        ]
        if json_paths:
            return json_paths

    raise FileNotFoundError(f"Could not resolve battery collection: {path_or_name}")


def resolve_battery_path(run_dir: Path, meta: dict[str, Any]) -> Path | None:
    prompt_selection = meta.get("prompt_selection", {})
    if not isinstance(prompt_selection, dict):
        return None
    raw_battery = prompt_selection.get("prompt_battery")
    if not isinstance(raw_battery, str) or not raw_battery.strip():
        return None

    candidates = [
        Path(raw_battery).expanduser(),
        Path.cwd() / raw_battery,
        run_dir / raw_battery,
        run_dir.parent / raw_battery,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(raw_battery).expanduser()


def load_battery_lookup(battery_path: Path | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if battery_path is None:
        return {}, []

    lookup: dict[str, dict[str, Any]] = {}
    collection_paths = resolve_battery_collection_paths(battery_path)
    warnings: list[str] = []
    for json_path in collection_paths:
        data = read_json(json_path)
        if not isinstance(data, list):
            warnings.append(f"Battery collection must be a list: {json_path}")
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            prompt_id = entry.get("id")
            if not isinstance(prompt_id, str):
                continue
            lookup.setdefault(prompt_id, dict(entry))
    return lookup, warnings


def build_joined_rows(
    records: list[dict[str, Any]],
    battery_lookup: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    verbose_baseline_lookup: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    baseline_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    for record in records:
        prompt_id = str(record["prompt_id"])
        rep = int(record.get("rep", 1))
        g_profile = str(record.get("g_profile", ""))
        g_scales = record.get("g_attention_scales")
        is_baseline = (
            g_profile == BASELINE_PROFILE_NAME
            or (
                isinstance(g_scales, list)
                and g_scales
                and all(abs(float(scale) - 1.0) < 1e-6 for scale in g_scales)
            )
        )
        if is_baseline:
            baseline_by_key[(prompt_id, rep)] = record

    joined_rows: list[dict[str, Any]] = []
    missing_baselines: set[tuple[str, int]] = set()
    for record in records:
        prompt_id = str(record["prompt_id"])
        rep = int(record.get("rep", 1))
        baseline = baseline_by_key.get((prompt_id, rep))
        if baseline is None:
            missing_baselines.add((prompt_id, rep))
            continue

        prompt_meta = battery_lookup.get(prompt_id, {})
        row = dict(record)
        row["model"] = meta.get("model")
        row["cartridge"] = meta.get("cartridge")
        row["type"] = prompt_meta.get("type")
        row["tier"] = prompt_meta.get("tier")
        row["source"] = prompt_meta.get("source")
        row["prompt"] = prompt_meta.get("prompt")
        row["target"] = prompt_meta.get("target")
        row["tokens_approx"] = prompt_meta.get("tokens_approx")
        row["g_spec"] = json.dumps(record.get("g_spec", {}), ensure_ascii=False, sort_keys=True)
        row["g_attention_scales"] = json.dumps(record.get("g_attention_scales", []), ensure_ascii=False)
        row["metadata_json"] = json.dumps(
            prompt_meta.get("metadata", {}),
            ensure_ascii=False,
            sort_keys=True,
        )
        g_profile = str(record.get("g_profile", ""))
        g_scales = record.get("g_attention_scales")
        row["g_family"] = classify_g_family(g_profile, g_scales if isinstance(g_scales, list) else None)

        baseline_prob = float(baseline.get("target_prob", math.nan))
        baseline_rank = float(baseline.get("target_rank", math.nan))
        baseline_avg_logprob = float(baseline.get("target_avg_logprob", math.nan))
        baseline_geo_mean_prob = float(baseline.get("target_geo_mean_prob", math.nan))
        baseline_entropy = float(baseline.get("final_entropy_bits", math.nan))
        verbose_metrics = compute_verbose_baseline_metrics(
            (verbose_baseline_lookup or {}).get((prompt_id, rep))
        )

        target_prob = float(record.get("target_prob", math.nan))
        target_rank = float(record.get("target_rank", math.nan))
        target_avg_logprob = float(record.get("target_avg_logprob", math.nan))
        target_geo_mean_prob = float(record.get("target_geo_mean_prob", math.nan))
        target_entropy = float(record.get("final_entropy_bits", math.nan))

        row["baseline_target_prob"] = baseline_prob
        row["baseline_target_rank"] = baseline_rank
        row["baseline_target_avg_logprob"] = baseline_avg_logprob
        row["baseline_target_geo_mean_prob"] = baseline_geo_mean_prob
        row["baseline_final_entropy_bits"] = baseline_entropy
        row.update(verbose_metrics)

        row["delta_target_prob"] = target_prob - baseline_prob
        row["delta_target_rank"] = target_rank - baseline_rank
        row["delta_target_avg_logprob"] = target_avg_logprob - baseline_avg_logprob
        row["delta_target_geo_mean_prob"] = target_geo_mean_prob - baseline_geo_mean_prob
        row["delta_final_entropy_bits"] = target_entropy - baseline_entropy
        joined_rows.append(round_report_row(row))

    for prompt_id, rep in sorted(missing_baselines):
        warnings.append(f"Missing baseline row for prompt_id={prompt_id} rep={rep}")

    joined_rows.sort(key=lambda row: (str(row["prompt_id"]), int(row.get("rep", 1)), str(row["g_profile"])))
    return joined_rows, warnings


def summarize_delta_rows(
    rows: list[dict[str, Any]],
    group_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        delta_p = [float(row["delta_target_prob"]) for row in group_rows]
        delta_rank = [float(row["delta_target_rank"]) for row in group_rows]
        delta_entropy = [float(row["delta_final_entropy_bits"]) for row in group_rows]
        delta_avg_logprob = [float(row["delta_target_avg_logprob"]) for row in group_rows]
        delta_geo_mean_prob = [float(row["delta_target_geo_mean_prob"]) for row in group_rows]
        target_probs = [float(row["target_prob"]) for row in group_rows]
        baseline_probs = [float(row["baseline_target_prob"]) for row in group_rows]

        summary: dict[str, Any] = {
            field: value for field, value in zip(group_fields, key, strict=False)
        }
        summary.update(
            {
                "n": len(group_rows),
                "mean_target_prob": mean(target_probs),
                "mean_baseline_target_prob": mean(baseline_probs),
                "mean_delta_target_prob": mean(delta_p),
                "median_delta_target_prob": median(delta_p),
                "stdev_delta_target_prob": stdev(delta_p),
                "pct_delta_target_prob_positive": pct(sum(value > 0 for value in delta_p), len(delta_p)),
                "sign_test_p_delta_target_prob": sign_test_pvalue(delta_p),
                "mean_delta_target_rank": mean(delta_rank),
                "median_delta_target_rank": median(delta_rank),
                "pct_rank_improved": pct(sum(value < 0 for value in delta_rank), len(delta_rank)),
                "mean_delta_final_entropy_bits": mean(delta_entropy),
                "mean_delta_target_avg_logprob": mean(delta_avg_logprob),
                "mean_delta_target_geo_mean_prob": mean(delta_geo_mean_prob),
            }
        )
        summary_rows.append(round_report_row(summary))

    return summary_rows


def build_best_profile_by_type(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    non_baseline = [row for row in rows if row.get("g_profile") != BASELINE_PROFILE_NAME]
    summary_rows = summarize_delta_rows(non_baseline, ["type", "g_profile", "g_family"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["type"])].append(row)

    result: list[dict[str, Any]] = []
    for prompt_type, entries in sorted(grouped.items()):
        best = max(entries, key=lambda row: float(row["mean_delta_target_prob"]))
        result.append(
            round_report_row(
                {
                    "type": prompt_type,
                    "best_g_profile": best["g_profile"],
                    "best_g_family": best["g_family"],
                    "mean_delta_target_prob": best["mean_delta_target_prob"],
                    "pct_delta_target_prob_positive": best["pct_delta_target_prob_positive"],
                    "sign_test_p_delta_target_prob": best["sign_test_p_delta_target_prob"],
                    "mean_delta_target_rank": best["mean_delta_target_rank"],
                    "n": best["n"],
                }
            )
        )
    return result


def build_overall_profile_summary(
    rows: list[dict[str, Any]],
    type_gain_summary: list[dict[str, Any]],
    prompt_winners: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    non_baseline = [row for row in rows if row.get("g_profile") != BASELINE_PROFILE_NAME]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in non_baseline:
        grouped[str(row["g_profile"])].append(row)

    type_rows_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in type_gain_summary:
        g_profile = str(row.get("g_profile", ""))
        if g_profile == BASELINE_PROFILE_NAME:
            continue
        type_rows_by_profile[g_profile].append(row)

    prompt_best_counts = Counter(
        str(row.get("best_g_profile", ""))
        for row in prompt_winners
        if row.get("best_g_profile")
    )
    prompt_worst_counts = Counter(
        str(row.get("worst_g_profile", ""))
        for row in prompt_winners
        if row.get("worst_g_profile")
    )
    total_prompt_cases = len(prompt_winners)

    summary_rows: list[dict[str, Any]] = []
    for g_profile, profile_rows in sorted(grouped.items()):
        delta_p = [float(row["delta_target_prob"]) for row in profile_rows]
        delta_rank = [float(row["delta_target_rank"]) for row in profile_rows]
        delta_entropy = [float(row["delta_final_entropy_bits"]) for row in profile_rows]
        delta_avg_logprob = [float(row["delta_target_avg_logprob"]) for row in profile_rows]
        delta_geo_mean_prob = [float(row["delta_target_geo_mean_prob"]) for row in profile_rows]
        positive_delta_p = [value for value in delta_p if value > 0]
        negative_delta_p = [value for value in delta_p if value < 0]
        positive_count = len(positive_delta_p)
        positive_mass = sum(positive_delta_p) if positive_delta_p else 0.0
        sorted_delta_p = sorted((value for value in delta_p if math.isfinite(value)), reverse=True)

        type_rows = type_rows_by_profile.get(g_profile, [])
        best_type_row = max(
            type_rows,
            key=lambda row: to_float(row.get("mean_delta_target_prob", -math.inf)),
            default=None,
        )
        worst_type_row = min(
            type_rows,
            key=lambda row: to_float(row.get("mean_delta_target_prob", math.inf)),
            default=None,
        )

        best_type_mean = (
            float(best_type_row["mean_delta_target_prob"])
            if best_type_row is not None
            else math.nan
        )
        worst_type_mean = (
            float(worst_type_row["mean_delta_target_prob"])
            if worst_type_row is not None
            else math.nan
        )
        mean_delta_p = mean(delta_p)
        balanced_score = mean_delta_p
        if math.isfinite(worst_type_mean):
            balanced_score -= max(0.0, -worst_type_mean)

        type_delta_spread = math.nan
        if math.isfinite(best_type_mean) and math.isfinite(worst_type_mean):
            type_delta_spread = best_type_mean - worst_type_mean

        prompt_best_wins = int(prompt_best_counts.get(g_profile, 0))
        prompt_worst_losses = int(prompt_worst_counts.get(g_profile, 0))
        summary_rows.append(
            {
                "g_profile": g_profile,
                "g_family": profile_rows[0].get("g_family", ""),
                "n": len(profile_rows),
                "n_types_covered": len(type_rows),
                "mean_delta_target_prob": mean_delta_p,
                "median_delta_target_prob": median(delta_p),
                "stdev_delta_target_prob": stdev(delta_p),
                "mean_positive_delta_target_prob": mean(positive_delta_p),
                "mean_negative_delta_target_prob": mean(negative_delta_p),
                "pct_delta_target_prob_positive": pct(sum(value > 0 for value in delta_p), len(delta_p)),
                "pct_delta_target_prob_negative": pct(sum(value < 0 for value in delta_p), len(delta_p)),
                "sign_test_p_delta_target_prob": sign_test_pvalue(delta_p),
                "positive_prompt_count": positive_count,
                "positive_prompt_rate": pct(positive_count, len(delta_p)),
                "positive_mass_delta_target_prob": positive_mass,
                "min_delta_target_prob": min(delta_p) if delta_p else math.nan,
                "p10_delta_target_prob": percentile(delta_p, 10),
                "p90_delta_target_prob": percentile(delta_p, 90),
                "max_delta_target_prob": max(delta_p) if delta_p else math.nan,
                "top_1_mean_delta_target_prob": mean_top_k(sorted_delta_p, 1),
                "top_2_mean_delta_target_prob": mean_top_k(sorted_delta_p, 2),
                "top_4_mean_delta_target_prob": mean_top_k(sorted_delta_p, 4),
                "top_8_mean_delta_target_prob": mean_top_k(sorted_delta_p, 8),
                "top_16_mean_delta_target_prob": mean_top_k(sorted_delta_p, 16),
                "top_1_cutoff_delta_target_prob": kth_largest(sorted_delta_p, 1),
                "top_2_cutoff_delta_target_prob": kth_largest(sorted_delta_p, 2),
                "top_4_cutoff_delta_target_prob": kth_largest(sorted_delta_p, 4),
                "top_8_cutoff_delta_target_prob": kth_largest(sorted_delta_p, 8),
                "top_16_cutoff_delta_target_prob": kth_largest(sorted_delta_p, 16),
                "mean_delta_target_rank": mean(delta_rank),
                "mean_delta_final_entropy_bits": mean(delta_entropy),
                "mean_delta_target_avg_logprob": mean(delta_avg_logprob),
                "mean_delta_target_geo_mean_prob": mean(delta_geo_mean_prob),
                "best_type": best_type_row.get("type", "") if best_type_row is not None else "",
                "best_type_mean_delta_target_prob": best_type_mean,
                "best_type_pct_delta_target_prob_positive": (
                    float(best_type_row.get("pct_delta_target_prob_positive", math.nan))
                    if best_type_row is not None
                    else math.nan
                ),
                "best_type_n": int(best_type_row.get("n", 0)) if best_type_row is not None else 0,
                "worst_type": worst_type_row.get("type", "") if worst_type_row is not None else "",
                "worst_type_mean_delta_target_prob": worst_type_mean,
                "worst_type_pct_delta_target_prob_positive": (
                    float(worst_type_row.get("pct_delta_target_prob_positive", math.nan))
                    if worst_type_row is not None
                    else math.nan
                ),
                "worst_type_n": int(worst_type_row.get("n", 0)) if worst_type_row is not None else 0,
                "type_delta_spread": type_delta_spread,
                "balanced_score": balanced_score,
                "prompt_best_wins": prompt_best_wins,
                "prompt_worst_losses": prompt_worst_losses,
                "net_prompt_win_minus_loss": prompt_best_wins - prompt_worst_losses,
                "prompt_best_win_rate": pct(prompt_best_wins, total_prompt_cases),
                "prompt_worst_loss_rate": pct(prompt_worst_losses, total_prompt_cases),
            }
        )

    def assign_rank(rows_to_rank: list[dict[str, Any]], value_field: str, rank_field: str) -> None:
        ordered = sorted(
            rows_to_rank,
            key=lambda row: (
                not math.isfinite(to_float(row.get(value_field, math.nan))),
                -to_float(row.get(value_field, math.nan))
                if math.isfinite(to_float(row.get(value_field, math.nan)))
                else 0.0,
                str(row.get("g_profile", "")),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row[rank_field] = rank

    assign_rank(summary_rows, "mean_delta_target_prob", "rank_overall_mean_delta_target_prob")
    assign_rank(summary_rows, "top_1_mean_delta_target_prob", "rank_top_1_mean_delta_target_prob")
    assign_rank(summary_rows, "top_2_mean_delta_target_prob", "rank_top_2_mean_delta_target_prob")
    assign_rank(summary_rows, "top_4_mean_delta_target_prob", "rank_top_4_mean_delta_target_prob")
    assign_rank(summary_rows, "top_8_mean_delta_target_prob", "rank_top_8_mean_delta_target_prob")
    assign_rank(summary_rows, "top_16_mean_delta_target_prob", "rank_top_16_mean_delta_target_prob")
    assign_rank(summary_rows, "best_type_mean_delta_target_prob", "rank_best_type_mean_delta_target_prob")
    assign_rank(summary_rows, "balanced_score", "rank_balanced_score")
    assign_rank(summary_rows, "type_delta_spread", "rank_type_delta_spread")

    return [round_report_row(row) for row in summary_rows]


def build_prompt_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("g_profile") == BASELINE_PROFILE_NAME:
            continue
        grouped[(str(row["prompt_id"]), int(row.get("rep", 1)))].append(row)

    winners: list[dict[str, Any]] = []
    for (prompt_id, rep), entries in sorted(grouped.items()):
        best = max(entries, key=lambda row: float(row["delta_target_prob"]))
        worst = min(entries, key=lambda row: float(row["delta_target_prob"]))
        winners.append(
            round_report_row(
                {
                    "prompt_id": prompt_id,
                    "rep": rep,
                    "type": best.get("type"),
                    "tier": best.get("tier"),
                    "source": best.get("source"),
                    "target": best.get("target"),
                    "best_g_profile": best.get("g_profile"),
                    "best_g_family": best.get("g_family"),
                    "best_delta_target_prob": best.get("delta_target_prob"),
                    "best_delta_target_rank": best.get("delta_target_rank"),
                    "worst_g_profile": worst.get("g_profile"),
                    "worst_g_family": worst.get("g_family"),
                    "worst_delta_target_prob": worst.get("delta_target_prob"),
                    "worst_delta_target_rank": worst.get("delta_target_rank"),
                }
            )
        )
    return winners


def build_completion_rows(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    unique_prompts = {str(row["prompt_id"]) for row in rows}
    unique_reps = {int(row.get("rep", 1)) for row in rows}
    g_specs = meta.get("g_specs", [])
    expected_runs = len(unique_prompts) * len(unique_reps) * len(g_specs)
    completed_runs = len(rows)
    error_count = len(errors)
    return [
        round_report_row(
            {
                "unique_prompts": len(unique_prompts),
                "unique_reps": len(unique_reps),
                "g_profiles_in_meta": len(g_specs),
                "expected_runs": expected_runs,
                "completed_runs": completed_runs,
                "missing_runs": expected_runs - completed_runs,
                "error_rows": error_count,
            }
        )
    ]


def build_matrix_rows(
    summary_rows: list[dict[str, Any]],
    row_field: str,
    column_field: str,
    value_field: str,
) -> list[dict[str, Any]]:
    row_keys = sorted({str(row[row_field]) for row in summary_rows})
    column_keys = sorted({str(row[column_field]) for row in summary_rows})
    value_lookup = {
        (str(row[row_field]), str(row[column_field])): row.get(value_field)
        for row in summary_rows
    }
    matrix_rows: list[dict[str, Any]] = []
    for row_key in row_keys:
        matrix_row: dict[str, Any] = {row_field: row_key}
        for column_key in column_keys:
            matrix_row[column_key] = value_lookup.get((row_key, column_key), "")
        matrix_rows.append(round_report_row(matrix_row))
    return matrix_rows


def build_warnings(
    joined_rows: list[dict[str, Any]],
    battery_lookup: dict[str, dict[str, Any]],
    battery_path: Path | None,
    extra_warnings: list[str],
) -> list[str]:
    warnings = list(extra_warnings)
    if battery_path is None:
        warnings.append("No prompt_battery path found in _meta.json prompt_selection.")
    if not battery_lookup:
        warnings.append("Battery metadata lookup is empty; prompt join fields may be blank.")

    missing_join_count = sum(1 for row in joined_rows if not row.get("prompt"))
    if missing_join_count:
        warnings.append(f"Prompt metadata missing for {missing_join_count} joined rows.")
    return warnings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def build_file_rows(
    run_dir: Path,
    meta_path: Path,
    main_path: Path,
    errors_path: Path,
    verbose_path: Path,
) -> list[dict[str, Any]]:
    rows = [
        {"kind": "run_dir", "path": str(run_dir)},
        {"kind": "meta", "path": str(meta_path)},
        {"kind": "main", "path": str(main_path)},
        {"kind": "errors", "path": str(errors_path)},
    ]
    if verbose_path.is_file():
        rows.append({"kind": "verbose", "path": str(verbose_path)})
    return rows


def build_report_text(
    run_dir: Path,
    meta: dict[str, Any],
    completion_rows: list[dict[str, Any]],
    overall_profile_summary: list[dict[str, Any]],
    type_gain_summary: list[dict[str, Any]],
    type_family_summary: list[dict[str, Any]],
    best_profile_by_type: list[dict[str, Any]],
    prompt_winners: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    parts = [
        f"Run directory: {run_dir}",
        f"Model: {meta.get('model', 'unknown')}",
        f"Cartridge: {meta.get('cartridge', 'unknown')}",
        "",
    ]

    if completion_rows:
        row = completion_rows[0]
        parts.append("Completion")
        parts.append(
            render_table(
                ["unique_prompts", "unique_reps", "g_profiles", "expected_runs", "completed_runs", "missing_runs", "error_rows"],
                [[
                    str(row.get("unique_prompts", "")),
                    str(row.get("unique_reps", "")),
                    str(row.get("g_profiles_in_meta", "")),
                    str(row.get("expected_runs", "")),
                    str(row.get("completed_runs", "")),
                    str(row.get("missing_runs", "")),
                    str(row.get("error_rows", "")),
                ]],
            )
        )
        parts.append("")

    if warnings:
        parts.append("Warnings")
        parts.extend(f"- {warning}" for warning in warnings)
        parts.append("")

    if overall_profile_summary:
        parts.append("Overall Gain Profile Summary")
        parts.append("Use these top-8 rankings to shortlist candidates for top-1 / top-2 / top-4 / top-8 profile sets.")
        parts.append("")

        top_cluster_profiles = sorted(
            overall_profile_summary,
            key=lambda row: int(row.get("rank_top_8_mean_delta_target_prob", 10**9)),
        )[:8]
        if top_cluster_profiles:
            parts.append("Profiles With Strongest Positive Prompt Clusters")
            parts.append(
                render_table(
                    ["rank8", "g_profile", "top1", "top2", "top4", "top8", "top16", "pos_n", "pos_mass", "p90", "max"],
                    [[
                        str(row.get("rank_top_8_mean_delta_target_prob", "")),
                        str(row.get("g_profile", "")),
                        format_float(to_float(row.get("top_1_mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("top_2_mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("top_4_mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("top_8_mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("top_16_mean_delta_target_prob", math.nan))),
                        str(row.get("positive_prompt_count", "")),
                        format_float(to_float(row.get("positive_mass_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("p90_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("max_delta_target_prob", math.nan))),
                    ] for row in top_cluster_profiles],
                )
            )
            parts.append("")

        sharp_cluster_profiles = sorted(
            overall_profile_summary,
            key=lambda row: int(row.get("rank_top_4_mean_delta_target_prob", 10**9)),
        )[:8]
        if sharp_cluster_profiles:
            parts.append("Sharpest Positive Clusters")
            parts.append(
                render_table(
                    ["rank4", "g_profile", "top1", "top2", "top4", "top8_cut", "pos_n", "pos_rate", "mean", "mean-"],
                    [[
                        str(row.get("rank_top_4_mean_delta_target_prob", "")),
                        str(row.get("g_profile", "")),
                        format_float(to_float(row.get("top_1_mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("top_2_mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("top_4_mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("top_8_cutoff_delta_target_prob", math.nan))),
                        str(row.get("positive_prompt_count", "")),
                        format_float(to_float(row.get("positive_prompt_rate", math.nan))),
                        format_float(to_float(row.get("mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("mean_negative_delta_target_prob", math.nan))),
                    ] for row in sharp_cluster_profiles],
                )
            )
            parts.append("")

        top_overall_profiles = sorted(
            overall_profile_summary,
            key=lambda row: int(row.get("rank_overall_mean_delta_target_prob", 10**9)),
        )[:8]
        if top_overall_profiles:
            parts.append("Top Overall Profiles By Mean Delta P")
            parts.append(
                render_table(
                    ["rank", "g_profile", "mean_delta_p", "p10", "p90", "%pos", "mean+", "mean-", "best_type(mean)", "worst_type(mean)"],
                    [[
                        str(row.get("rank_overall_mean_delta_target_prob", "")),
                        str(row.get("g_profile", "")),
                        format_float(to_float(row.get("mean_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("p10_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("p90_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("pct_delta_target_prob_positive", math.nan))),
                        format_float(to_float(row.get("mean_positive_delta_target_prob", math.nan))),
                        format_float(to_float(row.get("mean_negative_delta_target_prob", math.nan))),
                        f"{row.get('best_type', '')} ({format_float(to_float(row.get('best_type_mean_delta_target_prob', math.nan)))})",
                        f"{row.get('worst_type', '')} ({format_float(to_float(row.get('worst_type_mean_delta_target_prob', math.nan)))})",
                    ] for row in top_overall_profiles],
                )
            )
            parts.append("")

        top_balanced_profiles = sorted(
            overall_profile_summary,
            key=lambda row: int(row.get("rank_balanced_score", 10**9)),
        )[:8]
        if top_balanced_profiles:
            parts.append("Top Balanced Profiles By Conservative Score")
            parts.append(
                render_table(
                    ["rank", "g_profile", "balanced", "overall_mean", "worst_type(mean)", "%pos", "wins", "losses"],
                    [[
                        str(row.get("rank_balanced_score", "")),
                        str(row.get("g_profile", "")),
                        format_float(to_float(row.get("balanced_score", math.nan))),
                        format_float(to_float(row.get("mean_delta_target_prob", math.nan))),
                        f"{row.get('worst_type', '')} ({format_float(to_float(row.get('worst_type_mean_delta_target_prob', math.nan)))})",
                        format_float(to_float(row.get("pct_delta_target_prob_positive", math.nan))),
                        str(row.get("prompt_best_wins", "")),
                        str(row.get("prompt_worst_losses", "")),
                    ] for row in top_balanced_profiles],
                )
            )
            parts.append("")

    top_type_rows = sorted(
        [row for row in type_gain_summary if row.get("g_profile") != BASELINE_PROFILE_NAME],
        key=lambda row: to_float(row.get("mean_delta_target_prob", -math.inf)),
        reverse=True,
    )[:12]
    if top_type_rows:
        parts.append("Top Type x Gain Profiles By Mean Delta P")
        parts.append(
            render_table(
                ["type", "g_profile", "mean_delta_p", "%delta_p_pos", "p_value", "mean_delta_rank", "n"],
                [[
                    str(row.get("type", "")),
                    str(row.get("g_profile", "")),
                    format_float(to_float(row.get("mean_delta_target_prob", math.nan))),
                    format_float(to_float(row.get("pct_delta_target_prob_positive", math.nan))),
                    format_float(to_float(row.get("sign_test_p_delta_target_prob", math.nan))),
                    format_float(to_float(row.get("mean_delta_target_rank", math.nan))),
                    str(row.get("n", "")),
                ] for row in top_type_rows],
            )
        )
        parts.append("")

    worst_type_rows = sorted(
        [row for row in type_gain_summary if row.get("g_profile") != BASELINE_PROFILE_NAME],
        key=lambda row: to_float(row.get("mean_delta_target_prob", math.inf)),
    )[:12]
    if worst_type_rows:
        parts.append("Worst Type x Gain Profiles By Mean Delta P")
        parts.append(
            render_table(
                ["type", "g_profile", "mean_delta_p", "%delta_p_pos", "p_value", "mean_delta_rank", "n"],
                [[
                    str(row.get("type", "")),
                    str(row.get("g_profile", "")),
                    format_float(to_float(row.get("mean_delta_target_prob", math.nan))),
                    format_float(to_float(row.get("pct_delta_target_prob_positive", math.nan))),
                    format_float(to_float(row.get("sign_test_p_delta_target_prob", math.nan))),
                    format_float(to_float(row.get("mean_delta_target_rank", math.nan))),
                    str(row.get("n", "")),
                ] for row in worst_type_rows],
            )
        )
        parts.append("")

    family_rows = sorted(
        [row for row in type_family_summary if row.get("g_family") not in {"baseline", "constant"}],
        key=lambda row: (str(row.get("type", "")), -to_float(row.get("mean_delta_target_prob", -math.inf))),
    )[:16]
    if family_rows:
        parts.append("Type x Gain Family Summary")
        parts.append(
            render_table(
                ["type", "g_family", "mean_delta_p", "%delta_p_pos", "mean_delta_rank", "n"],
                [[
                    str(row.get("type", "")),
                    str(row.get("g_family", "")),
                    format_float(to_float(row.get("mean_delta_target_prob", math.nan))),
                    format_float(to_float(row.get("pct_delta_target_prob_positive", math.nan))),
                    format_float(to_float(row.get("mean_delta_target_rank", math.nan))),
                    str(row.get("n", "")),
                ] for row in family_rows],
            )
        )
        parts.append("")

    if best_profile_by_type:
        parts.append("Best Profile By Type")
        parts.append(
            render_table(
                ["type", "best_g_profile", "best_g_family", "mean_delta_p", "%delta_p_pos", "p_value"],
                [[
                    str(row.get("type", "")),
                    str(row.get("best_g_profile", "")),
                    str(row.get("best_g_family", "")),
                    format_float(to_float(row.get("mean_delta_target_prob", math.nan))),
                    format_float(to_float(row.get("pct_delta_target_prob_positive", math.nan))),
                    format_float(to_float(row.get("sign_test_p_delta_target_prob", math.nan))),
                ] for row in best_profile_by_type],
            )
        )
        parts.append("")

    if prompt_winners:
        top_prompt_winners = sorted(
            prompt_winners,
            key=lambda row: to_float(row.get("best_delta_target_prob", -math.inf)),
            reverse=True,
        )[:12]
        parts.append("Top Prompt Winners")
        parts.append(
            render_table(
                ["prompt_id", "type", "best_g_profile", "best_delta_p", "worst_g_profile", "worst_delta_p"],
                [[
                    str(row.get("prompt_id", "")),
                    str(row.get("type", "")),
                    str(row.get("best_g_profile", "")),
                    format_float(to_float(row.get("best_delta_target_prob", math.nan))),
                    str(row.get("worst_g_profile", "")),
                    format_float(to_float(row.get("worst_delta_target_prob", math.nan))),
                ] for row in top_prompt_winners],
            )
        )
        parts.append("")

        worst_prompt_losers = sorted(
            prompt_winners,
            key=lambda row: to_float(row.get("worst_delta_target_prob", math.inf)),
        )[:12]
        parts.append("Worst Prompt Losers")
        parts.append(
            render_table(
                ["prompt_id", "type", "worst_g_profile", "worst_delta_p", "best_g_profile", "best_delta_p"],
                [[
                    str(row.get("prompt_id", "")),
                    str(row.get("type", "")),
                    str(row.get("worst_g_profile", "")),
                    format_float(to_float(row.get("worst_delta_target_prob", math.nan))),
                    str(row.get("best_g_profile", "")),
                    format_float(to_float(row.get("best_delta_target_prob", math.nan))),
                ] for row in worst_prompt_losers],
            )
        )

    return "\n".join(parts)


def write_analysis_files(
    output_dir: Path,
    prefix: str,
    report_text: str,
    file_rows: list[dict[str, Any]],
    warnings: list[str],
    joined_rows: list[dict[str, Any]],
    overall_profile_summary: list[dict[str, Any]],
    type_gain_summary: list[dict[str, Any]],
    tier_gain_summary: list[dict[str, Any]],
    type_tier_gain_summary: list[dict[str, Any]],
    type_family_summary: list[dict[str, Any]],
    best_profile_by_type: list[dict[str, Any]],
    prompt_winners: list[dict[str, Any]],
    completion_rows: list[dict[str, Any]],
    type_gain_matrix_rows: list[dict[str, Any]],
    type_family_matrix_rows: list[dict[str, Any]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    artifact_specs = [
        (f"{prefix}_files.csv", file_rows),
        (f"{prefix}_warnings.csv", [{"warning": warning} for warning in warnings]),
        (f"{prefix}_joined_long.csv", joined_rows),
        (f"{prefix}_overall_profile_summary.csv", overall_profile_summary),
        (f"{prefix}_type_gain_summary.csv", type_gain_summary),
        (f"{prefix}_tier_gain_summary.csv", tier_gain_summary),
        (f"{prefix}_type_tier_gain_summary.csv", type_tier_gain_summary),
        (f"{prefix}_type_family_summary.csv", type_family_summary),
        (f"{prefix}_best_profile_by_type.csv", best_profile_by_type),
        (f"{prefix}_prompt_winners.csv", prompt_winners),
        (f"{prefix}_completion.csv", completion_rows),
        (f"{prefix}_type_gain_matrix_delta_p.csv", type_gain_matrix_rows),
        (f"{prefix}_type_family_matrix_delta_p.csv", type_family_matrix_rows),
    ]

    for filename, rows in artifact_specs:
        path = output_dir / filename
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = ["value"]
        write_csv(path, rows, fieldnames)
        written.append(path)

    text_path = output_dir / f"{prefix}_report.txt"
    write_text(text_path, report_text)
    written.append(text_path)
    return written


def build_baseline_attn_pca(
    verbose_baseline_lookup: dict[tuple[str, int], dict[str, Any]],
    battery_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Run PCA on baseline per-head attention entropy vectors.

    Returns a JSON-serialisable dict with PCA coordinates, metadata, and
    explained variance, or ``None`` if insufficient data is available.
    """
    if not verbose_baseline_lookup:
        return None

    # Collect flat entropy vectors and associated metadata.
    prompt_ids: list[str] = []
    reps: list[int] = []
    types: list[str] = []
    vectors: list[list[float]] = []
    n_layers: int | None = None
    n_heads: int | None = None
    layer_indices: list[int] | None = None

    for (prompt_id, rep), record in sorted(verbose_baseline_lookup.items()):
        raw = record.get("attn_entropy_per_head_final")
        if not isinstance(raw, list) or not raw:
            continue
        flat = flatten_numeric_values(raw)
        if not flat:
            continue

        # Capture shape info from first valid record.
        if n_layers is None:
            n_layers = len(raw)
            n_heads = len(raw[0]) if isinstance(raw[0], list) else None
            raw_indices = record.get("attn_entropy_layer_indices")
            if isinstance(raw_indices, list):
                layer_indices = [int(i) for i in raw_indices]

        prompt_meta = battery_lookup.get(prompt_id, {})
        prompt_ids.append(prompt_id)
        reps.append(rep)
        types.append(prompt_meta.get("type", "unknown"))
        vectors.append(flat)

    if len(vectors) < 3:
        return None

    # Ensure uniform dimensionality — skip any outlier-length rows.
    expected_dim = len(vectors[0])
    keep = [i for i, v in enumerate(vectors) if len(v) == expected_dim]
    if len(keep) < 3:
        return None

    prompt_ids = [prompt_ids[i] for i in keep]
    reps = [reps[i] for i in keep]
    types = [types[i] for i in keep]
    vectors = [vectors[i] for i in keep]

    # PCA via SVD (mean-centred).
    X = np.array(vectors, dtype=np.float64)
    X_mean = X.mean(axis=0)
    X_centred = X - X_mean
    _U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)
    total_var = float((S**2).sum())
    pc1 = Vt[0]
    pc2 = Vt[1]
    coords = X_centred @ np.vstack([pc1, pc2]).T  # (n, 2)

    explained_variance = [float(s**2 / total_var) for s in S[:2]] if total_var > 0 else [0.0, 0.0]

    points: list[dict[str, Any]] = []
    for i in range(len(prompt_ids)):
        points.append(
            {
                "prompt_id": prompt_ids[i],
                "rep": reps[i],
                "type": types[i],
                "pc1": round(float(coords[i, 0]), 6),
                "pc2": round(float(coords[i, 1]), 6),
            }
        )

    return {
        "description": "PCA of baseline per-head attention entropy vectors",
        "n_prompts": len(points),
        "n_features": expected_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "layer_indices": layer_indices,
        "explained_variance_ratio": [round(v, 6) for v in explained_variance],
        "points": points,
    }


def _pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson correlation coefficient and two-tailed p-value (no scipy)."""
    n = len(x)
    if n < 3:
        return (math.nan, math.nan)
    xm = x - x.mean()
    ym = y - y.mean()
    r_num = float(np.dot(xm, ym))
    r_den = float(np.sqrt(np.dot(xm, xm) * np.dot(ym, ym)))
    if r_den < 1e-15:
        return (0.0, 1.0)
    r = r_num / r_den
    r = max(-1.0, min(1.0, r))
    # t-test for significance
    if abs(r) >= 1.0:
        return (r, 0.0)
    t_stat = r * math.sqrt((n - 2) / (1.0 - r * r))
    # Two-tailed p-value via regularised incomplete beta function approximation.
    # Use the survival-function identity: p = 2 * P(T > |t|) with df = n-2.
    df = n - 2
    x_beta = df / (df + t_stat * t_stat)
    p = _betai(0.5 * df, 0.5, x_beta)
    return (r, p)


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction."""
    if x < 0.0 or x > 1.0:
        return math.nan
    if x == 0.0 or x == 1.0:
        return x
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    else:
        return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta (Lentz's method)."""
    max_iter = 200
    eps = 3e-12
    fpmin = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = max(abs(1.0 - qab * x / qap), fpmin)
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = max(abs(1.0 + aa * d), fpmin)
        d = 1.0 / d
        c = max(abs(1.0 + aa / c), fpmin)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = max(abs(1.0 + aa * d), fpmin)
        d = 1.0 / d
        c = max(abs(1.0 + aa / c), fpmin)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def roc_auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for binary labels; higher score should indicate positive class."""
    if scores.size == 0 or labels.size == 0 or scores.size != labels.size:
        return math.nan
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    comparisons = pos[:, None] - neg[None, :]
    wins = float((comparisons > 0).sum())
    ties = float((comparisons == 0).sum())
    return (wins + 0.5 * ties) / float(comparisons.size)


def best_balanced_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Best balanced accuracy over score thresholds; higher score predicts positive class."""
    if scores.size == 0 or labels.size == 0 or scores.size != labels.size:
        return (math.nan, math.nan)
    pos_count = int(labels.sum())
    neg_count = int((~labels).sum())
    if pos_count == 0 or neg_count == 0:
        return (math.nan, math.nan)

    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return (math.nan, math.nan)

    thresholds: list[float] = [float(unique_scores[0]) - 1e-9]
    thresholds.extend(
        float((unique_scores[i] + unique_scores[i + 1]) / 2.0)
        for i in range(len(unique_scores) - 1)
    )
    thresholds.append(float(unique_scores[-1]) + 1e-9)

    best_score = -math.inf
    best_threshold = math.nan
    for threshold in thresholds:
        predicted = scores >= threshold
        true_positive = int(np.logical_and(predicted, labels).sum())
        true_negative = int(np.logical_and(~predicted, ~labels).sum())
        tpr = true_positive / pos_count
        tnr = true_negative / neg_count
        balanced = 0.5 * (tpr + tnr)
        if balanced > best_score:
            best_score = balanced
            best_threshold = threshold
    return (best_score, best_threshold)


def select_top_positive_cluster_profiles(
    overall_profile_summary: list[dict[str, Any]],
    top_n: int = 4,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in overall_profile_summary
        if math.isfinite(to_float(row.get("rank_top_8_mean_delta_target_prob", math.nan)))
        and math.isfinite(to_float(row.get("top_8_mean_delta_target_prob", math.nan)))
        and to_float(row.get("top_8_mean_delta_target_prob", math.nan)) > 0.0
    ]
    eligible.sort(
        key=lambda row: (
            int(row.get("rank_top_8_mean_delta_target_prob", 10**9)),
            str(row.get("g_profile", "")),
        )
    )
    return eligible[: max(top_n, 0)]


def build_head_correlation_analysis(
    verbose_baseline_lookup: dict[tuple[str, int], dict[str, Any]],
    joined_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Correlate each head's baseline attention entropy with delta_p per gain profile.

    Returns a JSON-serialisable dict with per-profile correlation matrices,
    top-10 tables, and scout-head frequency counts, or ``None`` if insufficient data.
    """
    if not verbose_baseline_lookup:
        return None

    # ---- 1. Build baseline entropy matrix: prompt → flat vector ----
    prompt_ids_ordered: list[str] = []
    reps_ordered: list[int] = []
    vectors: list[list[float]] = []
    n_layers: int | None = None
    n_heads: int | None = None
    layer_indices: list[int] | None = None

    for (prompt_id, rep), record in sorted(verbose_baseline_lookup.items()):
        raw = record.get("attn_entropy_per_head_final")
        if not isinstance(raw, list) or not raw:
            continue
        flat = flatten_numeric_values(raw)
        if not flat:
            continue
        if n_layers is None:
            n_layers = len(raw)
            n_heads = len(raw[0]) if isinstance(raw[0], list) else None
            raw_indices = record.get("attn_entropy_layer_indices")
            if isinstance(raw_indices, list):
                layer_indices = [int(i) for i in raw_indices]
        prompt_ids_ordered.append(prompt_id)
        reps_ordered.append(rep)
        vectors.append(flat)

    if len(vectors) < 10 or n_layers is None or n_heads is None:
        return None

    expected_dim = len(vectors[0])
    keep = [i for i, v in enumerate(vectors) if len(v) == expected_dim]
    if len(keep) < 10:
        return None
    prompt_ids_ordered = [prompt_ids_ordered[i] for i in keep]
    reps_ordered = [reps_ordered[i] for i in keep]
    vectors = [vectors[i] for i in keep]

    X = np.array(vectors, dtype=np.float64)  # (n_prompts, n_features)
    n_prompts, n_features = X.shape

    # Build lookup for fast delta_p access: (prompt_id, rep, g_profile) -> delta_p.
    delta_lookup: dict[tuple[str, int, str], float] = {}
    for row in joined_rows:
        pid = str(row.get("prompt_id", ""))
        rep = int(row.get("rep", 1))
        profile = str(row.get("g_profile", ""))
        dp = row.get("delta_target_prob")
        if dp is not None:
            try:
                dpf = float(dp)
                if math.isfinite(dpf):
                    delta_lookup[(pid, rep, profile)] = dpf
            except (ValueError, TypeError):
                pass

    # Discover non-baseline gain profiles.
    all_profiles = sorted({p for (_, _, p) in delta_lookup if p != "baseline"})
    if not all_profiles:
        return None

    # ---- 2. Compute correlations: for each profile × each head ----
    scout_counts = np.zeros(n_features, dtype=np.int64)
    profile_results: list[dict[str, Any]] = []

    for profile in all_profiles:
        # Gather delta_p vector aligned with prompt order.
        delta_vec = np.full(n_prompts, np.nan, dtype=np.float64)
        for i, (pid, rep) in enumerate(zip(prompt_ids_ordered, reps_ordered)):
            dp = delta_lookup.get((pid, rep, profile))
            if dp is not None:
                delta_vec[i] = dp

        valid_mask = np.isfinite(delta_vec)
        n_valid = int(valid_mask.sum())
        if n_valid < 10:
            continue

        delta_valid = delta_vec[valid_mask]
        X_valid = X[valid_mask]

        correlations = np.zeros(n_features, dtype=np.float64)
        p_values = np.ones(n_features, dtype=np.float64)
        for j in range(n_features):
            r, p = _pearsonr(X_valid[:, j], delta_valid)
            correlations[j] = r
            p_values[j] = p

        # Reshape to (n_layers, n_heads) for the heatmap.
        corr_matrix = correlations.reshape(n_layers, n_heads).tolist()
        pval_matrix = p_values.reshape(n_layers, n_heads).tolist()

        # Top 10 heads by |r|.
        top_indices = np.argsort(-np.abs(correlations))[:10]
        top_heads: list[dict[str, Any]] = []
        for idx in top_indices:
            layer_idx = int(idx // n_heads)
            head_idx = int(idx % n_heads)
            top_heads.append({
                "layer_slot": layer_idx,
                "layer_index": layer_indices[layer_idx] if layer_indices else layer_idx,
                "head": head_idx,
                "correlation": round(float(correlations[idx]), 6),
                "p_value": round(float(p_values[idx]), 6),
            })

        # Accumulate scout counts.
        scout_counts[top_indices] += 1

        profile_results.append({
            "g_profile": profile,
            "n_valid_prompts": n_valid,
            "correlation_matrix": [[round(v, 6) for v in row] for row in corr_matrix],
            "p_value_matrix": [[round(v, 6) for v in row] for row in pval_matrix],
            "top_10_heads": top_heads,
        })

    # Scout-heads aggregate.
    scout_matrix = scout_counts.reshape(n_layers, n_heads).tolist()

    return {
        "description": "Per-head correlation between baseline attention entropy and intervention delta_p",
        "n_prompts": n_prompts,
        "n_features": n_features,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "layer_indices": layer_indices,
        "n_profiles": len(profile_results),
        "profiles": profile_results,
        "scout_heads_matrix": [[int(v) for v in row] for row in scout_matrix],
    }


def build_scout_head_rankings(
    verbose_baseline_lookup: dict[tuple[str, int], dict[str, Any]],
    joined_rows: list[dict[str, Any]],
    overall_profile_summary: list[dict[str, Any]],
    head_corr_result: dict[str, Any] | None,
    *,
    cluster_top_k: int = 8,
    top_n_profiles: int = 4,
) -> list[dict[str, Any]]:
    if head_corr_result is None:
        return []

    n_layers = int(head_corr_result.get("n_layers", 0))
    n_heads = int(head_corr_result.get("n_heads", 0))
    layer_indices = head_corr_result.get("layer_indices")
    expected_dim = n_layers * n_heads
    if expected_dim <= 0:
        return []

    entropy_lookup: dict[tuple[str, int], np.ndarray] = {}
    for (prompt_id, rep), record in sorted(verbose_baseline_lookup.items()):
        raw = record.get("attn_entropy_per_head_final")
        if not isinstance(raw, list) or not raw:
            continue
        flat = flatten_numeric_values(raw)
        if len(flat) != expected_dim:
            continue
        entropy_lookup[(prompt_id, rep)] = np.array(flat, dtype=np.float64)

    if not entropy_lookup:
        return []

    profile_lookup = {
        str(profile.get("g_profile", "")): profile
        for profile in head_corr_result.get("profiles", [])
    }
    selected_profiles = select_top_positive_cluster_profiles(overall_profile_summary, top_n=top_n_profiles)
    if not selected_profiles:
        return []

    rows_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        g_profile = str(row.get("g_profile", ""))
        if g_profile != BASELINE_PROFILE_NAME:
            rows_by_profile[g_profile].append(row)

    output_rows: list[dict[str, Any]] = []
    for profile_summary in selected_profiles:
        g_profile = str(profile_summary.get("g_profile", ""))
        head_profile = profile_lookup.get(g_profile)
        profile_rows = rows_by_profile.get(g_profile, [])
        if head_profile is None or not profile_rows:
            continue

        ranked_rows = sorted(
            (
                row
                for row in profile_rows
                if math.isfinite(to_float(row.get("delta_target_prob", math.nan)))
            ),
            key=lambda row: to_float(row.get("delta_target_prob", -math.inf)),
            reverse=True,
        )
        cluster_rows = ranked_rows[: min(cluster_top_k, len(ranked_rows))]
        if not cluster_rows:
            continue

        cluster_keys = {
            (str(row.get("prompt_id", "")), int(row.get("rep", 1)))
            for row in cluster_rows
        }
        cluster_cutoff = float(cluster_rows[-1].get("delta_target_prob", math.nan))
        cluster_mean_delta = mean([to_float(row.get("delta_target_prob", math.nan)) for row in cluster_rows])

        vectors: list[np.ndarray] = []
        labels: list[bool] = []
        for row in ranked_rows:
            key = (str(row.get("prompt_id", "")), int(row.get("rep", 1)))
            vector = entropy_lookup.get(key)
            if vector is None:
                continue
            vectors.append(vector)
            labels.append(key in cluster_keys)

        if len(vectors) < 10:
            continue

        labels_arr = np.array(labels, dtype=bool)
        if int(labels_arr.sum()) == 0 or int((~labels_arr).sum()) == 0:
            continue

        X = np.vstack(vectors)
        corr_matrix = np.array(head_profile.get("correlation_matrix", []), dtype=np.float64)
        pval_matrix = np.array(head_profile.get("p_value_matrix", []), dtype=np.float64)
        if corr_matrix.size != expected_dim or pval_matrix.size != expected_dim:
            continue
        correlations = corr_matrix.reshape(expected_dim)
        p_values = pval_matrix.reshape(expected_dim)
        top10_set = {
            (int(head["layer_slot"]), int(head["head"]))
            for head in head_profile.get("top_10_heads", [])
            if isinstance(head, dict)
            and isinstance(head.get("layer_slot"), int)
            and isinstance(head.get("head"), int)
        }

        profile_head_rows: list[dict[str, Any]] = []
        for idx in range(expected_dim):
            layer_slot = int(idx // n_heads)
            head_idx = int(idx % n_heads)
            raw_scores = X[:, idx]
            corr = float(correlations[idx])
            aligned_scores = raw_scores if corr >= 0 else -raw_scores
            auc = roc_auc_from_scores(aligned_scores, labels_arr)
            balanced_acc, aligned_threshold = best_balanced_accuracy(aligned_scores, labels_arr)
            raw_threshold = aligned_threshold if corr >= 0 else -aligned_threshold
            cluster_mean_entropy = float(raw_scores[labels_arr].mean())
            noncluster_mean_entropy = float(raw_scores[~labels_arr].mean())
            profile_head_rows.append(
                {
                    "g_profile": g_profile,
                    "profile_rank_top_8_mean_delta_target_prob": int(
                        profile_summary.get("rank_top_8_mean_delta_target_prob", 0)
                    ),
                    "profile_top_8_mean_delta_target_prob": float(
                        profile_summary.get("top_8_mean_delta_target_prob", math.nan)
                    ),
                    "profile_positive_prompt_count": int(
                        float(profile_summary.get("positive_prompt_count", 0))
                    ),
                    "cluster_top_k": cluster_top_k,
                    "cluster_size": len(cluster_rows),
                    "cluster_cutoff_delta_target_prob": cluster_cutoff,
                    "cluster_mean_delta_target_prob": cluster_mean_delta,
                    "layer_slot": layer_slot,
                    "layer_index": layer_indices[layer_slot] if isinstance(layer_indices, list) else layer_slot,
                    "head": head_idx,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                    "p_value": float(p_values[idx]),
                    "correlation_sign": "positive" if corr >= 0 else "negative",
                    "entropy_direction": (
                        "higher_entropy_predicts_help"
                        if corr >= 0
                        else "lower_entropy_predicts_help"
                    ),
                    "directional_auroc": auc,
                    "best_balanced_accuracy": balanced_acc,
                    "raw_threshold": raw_threshold,
                    "threshold_operator": ">=" if corr >= 0 else "<=",
                    "cluster_mean_entropy": cluster_mean_entropy,
                    "noncluster_mean_entropy": noncluster_mean_entropy,
                    "entropy_gap_cluster_minus_noncluster": cluster_mean_entropy - noncluster_mean_entropy,
                    "is_top10_abs_correlation": int((layer_slot, head_idx) in top10_set),
                }
            )

        def assign_profile_rank(
            rows_to_rank: list[dict[str, Any]],
            value_field: str,
            rank_field: str,
        ) -> None:
            ordered = sorted(
                rows_to_rank,
                key=lambda row: (
                    not math.isfinite(to_float(row.get(value_field, math.nan))),
                    -to_float(row.get(value_field, math.nan))
                    if math.isfinite(to_float(row.get(value_field, math.nan)))
                    else 0.0,
                    int(row.get("layer_slot", 0)),
                    int(row.get("head", 0)),
                ),
            )
            for rank, row in enumerate(ordered, start=1):
                row[rank_field] = rank

        assign_profile_rank(profile_head_rows, "abs_correlation", "rank_abs_correlation_within_profile")
        assign_profile_rank(profile_head_rows, "directional_auroc", "rank_directional_auroc_within_profile")
        assign_profile_rank(profile_head_rows, "best_balanced_accuracy", "rank_best_balanced_accuracy_within_profile")
        output_rows.extend(round_report_row(row) for row in profile_head_rows)

    output_rows.sort(
        key=lambda row: (
            int(row.get("profile_rank_top_8_mean_delta_target_prob", 10**9)),
            int(row.get("rank_directional_auroc_within_profile", 10**9)),
            int(row.get("rank_best_balanced_accuracy_within_profile", 10**9)),
            int(row.get("rank_abs_correlation_within_profile", 10**9)),
            int(row.get("layer_slot", 0)),
            int(row.get("head", 0)),
        )
    )
    return output_rows


def main() -> None:
    args = parse_args()
    configure_data_dir(args.data_dir)
    run_dir = resolve_run_dir(args.run_dir)
    main_path = run_dir / "main.jsonl"
    meta_path = run_dir / "_meta.json"
    errors_path = run_dir / "errors.jsonl"
    verbose_path = run_dir / "verbose.jsonl"

    if not main_path.is_file():
        raise FileNotFoundError(f"Missing main.jsonl in {run_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing _meta.json in {run_dir}")

    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"_meta.json must contain a JSON object: {meta_path}")

    records = read_jsonl(main_path)
    errors = read_jsonl(errors_path) if errors_path.is_file() else []
    verbose_records = read_jsonl(verbose_path) if verbose_path.is_file() else []

    battery_path = resolve_battery_path(run_dir, meta)
    battery_lookup, battery_warnings = load_battery_lookup(battery_path)
    verbose_baseline_lookup, verbose_warnings = build_verbose_baseline_lookup(verbose_records)
    joined_rows, join_warnings = build_joined_rows(records, battery_lookup, meta, verbose_baseline_lookup)
    warnings = build_warnings(joined_rows, battery_lookup, battery_path, battery_warnings + join_warnings)
    warnings.extend(verbose_warnings)
    if not verbose_path.is_file():
        warnings.append("No verbose.jsonl found; verbose-derived baseline metrics will be blank.")

    type_gain_summary = summarize_delta_rows(joined_rows, ["type", "g_profile", "g_family"])
    tier_gain_summary = summarize_delta_rows(joined_rows, ["tier", "g_profile", "g_family"])
    type_tier_gain_summary = summarize_delta_rows(joined_rows, ["type", "tier", "g_profile", "g_family"])
    type_family_summary = summarize_delta_rows(joined_rows, ["type", "g_family"])
    best_profile_by_type = build_best_profile_by_type(joined_rows)
    prompt_winners = build_prompt_winners(joined_rows)
    overall_profile_summary = build_overall_profile_summary(joined_rows, type_gain_summary, prompt_winners)
    completion_rows = build_completion_rows(joined_rows, errors, meta)
    type_gain_matrix_rows = build_matrix_rows(type_gain_summary, "type", "g_profile", "mean_delta_target_prob")
    type_family_matrix_rows = build_matrix_rows(type_family_summary, "type", "g_family", "mean_delta_target_prob")

    report_text = build_report_text(
        run_dir=run_dir,
        meta=meta,
        completion_rows=completion_rows,
        overall_profile_summary=overall_profile_summary,
        type_gain_summary=type_gain_summary,
        type_family_summary=type_family_summary,
        best_profile_by_type=best_profile_by_type,
        prompt_winners=prompt_winners,
        warnings=warnings,
    )
    print(report_text)

    file_rows = build_file_rows(run_dir, meta_path, main_path, errors_path, verbose_path)
    pca_result = build_baseline_attn_pca(verbose_baseline_lookup, battery_lookup)
    head_corr_result = build_head_correlation_analysis(verbose_baseline_lookup, joined_rows)
    scout_head_rankings = build_scout_head_rankings(
        verbose_baseline_lookup,
        joined_rows,
        overall_profile_summary,
        head_corr_result,
    )

    if not args.no_write_files:
        output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_analysis_output_dir(run_dir)
        written = write_analysis_files(
            output_dir=output_dir,
            prefix=args.prefix,
            report_text=report_text,
            file_rows=file_rows,
            warnings=warnings,
            joined_rows=joined_rows,
            overall_profile_summary=overall_profile_summary,
            type_gain_summary=type_gain_summary,
            tier_gain_summary=tier_gain_summary,
            type_tier_gain_summary=type_tier_gain_summary,
            type_family_summary=type_family_summary,
            best_profile_by_type=best_profile_by_type,
            prompt_winners=prompt_winners,
            completion_rows=completion_rows,
            type_gain_matrix_rows=type_gain_matrix_rows,
            type_family_matrix_rows=type_family_matrix_rows,
        )
        print()
        print("Wrote analysis files:")
        for path in written:
            print(f"- {path}")

        # Second-order analysis: PCA on baseline attention entropy.
        if pca_result is not None:
            pca_path = output_dir / f"{args.prefix}_baseline_attn_pca.json"
            with open(pca_path, "w", encoding="utf-8") as f:
                json.dump(pca_result, f, indent=2)
            print(f"- {pca_path}")
        else:
            print("(Skipped baseline attention PCA — insufficient verbose baseline data.)")

        # Second-order analysis: head-level correlation with intervention delta_p.
        if head_corr_result is not None:
            head_corr_path = output_dir / f"{args.prefix}_head_correlations.json"
            with open(head_corr_path, "w", encoding="utf-8") as f:
                json.dump(head_corr_result, f, indent=2)
            print(f"- {head_corr_path}")
        else:
            print("(Skipped head correlation analysis — insufficient data.)")

        if scout_head_rankings:
            scout_rankings_path = output_dir / f"{args.prefix}_scout_head_rankings.csv"
            write_csv(scout_rankings_path, scout_head_rankings, list(scout_head_rankings[0].keys()))
            print(f"- {scout_rankings_path}")
        else:
            print("(Skipped scout head ranking summary — insufficient data.)")

    if args.json_out:
        json_report = {
            "run_dir": str(run_dir),
            "battery_path": str(battery_path) if battery_path is not None else None,
            "meta": meta,
            "warnings": warnings,
            "files": file_rows,
            "completion": completion_rows,
            "overall_profile_summary": overall_profile_summary,
            "type_gain_summary": type_gain_summary,
            "tier_gain_summary": tier_gain_summary,
            "type_tier_gain_summary": type_tier_gain_summary,
            "type_family_summary": type_family_summary,
            "best_profile_by_type": best_profile_by_type,
            "prompt_winners": prompt_winners,
            "type_gain_matrix_delta_p": type_gain_matrix_rows,
            "type_family_matrix_delta_p": type_family_matrix_rows,
            "scout_head_rankings": scout_head_rankings,
        }
        output_path = Path(args.json_out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)


if __name__ == "__main__":
    main()
