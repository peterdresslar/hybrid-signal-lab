#!/usr/bin/env python3
"""Pairwise comparison of two analyzed sweep run directories.

This script compares the standardized outputs produced by `signal_lab.sweep_analyze`
for two runs and writes side-by-side prompt, type, and family comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from signal_lab.paths import (
    DATA_DIR_ENV_VAR,
    DEFAULT_ANALYSIS_DIRNAME,
    configure_data_dir,
    default_compare_output_dir,
    ensure_new_output_dir,
    resolve_input_path,
)
from signal_lab.sweep_analyze import _pearsonr


REPORT_FLOAT_DIGITS = 2
DEFAULT_PREFIX = "compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two analyzed sweep run directories.")
    parser.add_argument(
        "--run-a",
        type=str,
        required=True,
        help="First analyzed sweep run directory, or a raw run directory containing analysis/.",
    )
    parser.add_argument(
        "--run-b",
        type=str,
        required=True,
        help="Second analyzed sweep run directory, or a raw run directory containing analysis/.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Optional base directory to use in place of data/. Also supports {DATA_DIR_ENV_VAR}.",
    )
    parser.add_argument("--label-a", type=str, default=None, help="Optional label override for run A.")
    parser.add_argument("--label-b", type=str, default=None, help="Optional label override for run B.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for generated artifacts "
            "(default: [DATA_DIR]/outputs/signal_lab/runs/<run_name>/_comparisons/<model-a>_vs_<model-b>)."
        ),
    )
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX, help=f"Filename prefix (default: {DEFAULT_PREFIX}).")
    parser.add_argument("--json-out", type=str, default=None, help="Optional machine-readable JSON output path.")
    parser.add_argument("--no-write-files", action="store_true", help="Print report without writing artifacts.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_dir(path_str: str) -> Path:
    path = resolve_input_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Run directory must be a directory: {path}")
    return path


def resolve_analysis_dir(path_str: str) -> Path:
    run_dir = resolve_run_dir(path_str)
    joined_path = run_dir / "analysis_joined_long.csv"
    if joined_path.is_file():
        return run_dir

    analysis_dir = run_dir / DEFAULT_ANALYSIS_DIRNAME
    if (analysis_dir / "analysis_joined_long.csv").is_file():
        return analysis_dir

    return run_dir


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


def load_run_analysis(run_dir: Path, label_override: str | None) -> dict[str, Any]:
    files_path = run_dir / "analysis_files.csv"
    joined_path = run_dir / "analysis_joined_long.csv"
    type_gain_path = run_dir / "analysis_type_gain_summary.csv"
    type_family_path = run_dir / "analysis_type_family_summary.csv"
    report_path = run_dir / "analysis_report.txt"

    required = [files_path, joined_path, type_gain_path, type_family_path]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required analysis artifact: {path}")

    joined_rows = read_csv(joined_path)
    type_gain_rows = read_csv(type_gain_path)
    type_family_rows = read_csv(type_family_path)
    files_rows = read_csv(files_path)
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""

    model_name = None
    for row in joined_rows:
        model_name = row.get("model")
        if model_name:
            break
    if model_name is None:
        model_name = label_override or run_dir.name

    label = label_override or model_name
    return {
        "run_dir": run_dir,
        "label": label,
        "model": model_name,
        "files_rows": files_rows,
        "joined_rows": joined_rows,
        "type_gain_rows": type_gain_rows,
        "type_family_rows": type_family_rows,
        "report_text": report_text,
    }


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def to_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def build_prompt_pairwise_rows(run_a: dict[str, Any], run_b: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    key_fields = ["prompt_id", "g_profile", "rep"]
    by_key_a = {
        tuple(row[field] for field in key_fields): row
        for row in run_a["joined_rows"]
    }
    by_key_b = {
        tuple(row[field] for field in key_fields): row
        for row in run_b["joined_rows"]
    }

    keys_a = set(by_key_a)
    keys_b = set(by_key_b)
    missing_in_b = sorted(keys_a - keys_b)
    missing_in_a = sorted(keys_b - keys_a)
    if missing_in_b:
        warnings.append(f"Rows present only in {run_a['label']}: {len(missing_in_b)}")
    if missing_in_a:
        warnings.append(f"Rows present only in {run_b['label']}: {len(missing_in_a)}")

    shared_keys = sorted(keys_a & keys_b)
    rows: list[dict[str, Any]] = []
    for key in shared_keys:
        row_a = by_key_a[key]
        row_b = by_key_b[key]
        out = {
            "prompt_id": row_a["prompt_id"],
            "g_profile": row_a["g_profile"],
            "g_family": row_a.get("g_family"),
            "rep": row_a["rep"],
            "type": row_a.get("type"),
            "tier": row_a.get("tier"),
            "source": row_a.get("source"),
            "target": row_a.get("target"),
            "prompt": row_a.get("prompt"),
        }
        metrics = [
            "target_prob",
            "target_rank",
            "target_avg_logprob",
            "target_geo_mean_prob",
            "final_entropy_bits",
            "delta_target_prob",
            "delta_target_rank",
            "delta_target_avg_logprob",
            "delta_target_geo_mean_prob",
            "delta_final_entropy_bits",
        ]
        for metric in metrics:
            value_a = to_float(row_a.get(metric))
            value_b = to_float(row_b.get(metric))
            out[f"{run_a['label']}__{metric}"] = value_a
            out[f"{run_b['label']}__{metric}"] = value_b
            out[f"diff__{metric}"] = value_a - value_b
        rows.append(round_report_row(out))
    return rows, warnings


def build_group_pairwise_rows(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    group_fields: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    by_key_a = {
        tuple(row[field] for field in group_fields): row
        for row in rows_a
    }
    by_key_b = {
        tuple(row[field] for field in group_fields): row
        for row in rows_b
    }
    keys_a = set(by_key_a)
    keys_b = set(by_key_b)
    if keys_a - keys_b:
        warnings.append(f"Group rows present only in run A for fields {group_fields}: {len(keys_a - keys_b)}")
    if keys_b - keys_a:
        warnings.append(f"Group rows present only in run B for fields {group_fields}: {len(keys_b - keys_a)}")

    shared_keys = sorted(keys_a & keys_b)
    out_rows: list[dict[str, Any]] = []
    compare_metrics = [
        "mean_target_prob",
        "mean_baseline_target_prob",
        "mean_delta_target_prob",
        "median_delta_target_prob",
        "stdev_delta_target_prob",
        "pct_delta_target_prob_positive",
        "sign_test_p_delta_target_prob",
        "mean_delta_target_rank",
        "median_delta_target_rank",
        "pct_rank_improved",
        "mean_delta_final_entropy_bits",
        "mean_delta_target_avg_logprob",
        "mean_delta_target_geo_mean_prob",
    ]
    for key in shared_keys:
        row_a = by_key_a[key]
        row_b = by_key_b[key]
        out = {
            field: value for field, value in zip(group_fields, key, strict=False)
        }
        out["n_a"] = to_int(row_a.get("n"))
        out["n_b"] = to_int(row_b.get("n"))
        for metric in compare_metrics:
            value_a = to_float(row_a.get(metric))
            value_b = to_float(row_b.get(metric))
            out[f"a__{metric}"] = value_a
            out[f"b__{metric}"] = value_b
            out[f"diff__{metric}"] = value_a - value_b
        out_rows.append(round_report_row(out))
    return out_rows, warnings


def build_report_text(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    warnings: list[str],
    prompt_rows: list[dict[str, Any]],
    type_gain_rows: list[dict[str, Any]],
    type_family_rows: list[dict[str, Any]],
) -> str:
    parts = [
        f"Run A: {run_a['run_dir']}",
        f"Label A: {run_a['label']}",
        f"Run B: {run_b['run_dir']}",
        f"Label B: {run_b['label']}",
        "",
    ]

    if warnings:
        parts.append("Warnings")
        parts.extend(f"- {warning}" for warning in warnings)
        parts.append("")

    top_type_gaps = sorted(
        type_gain_rows,
        key=lambda row: abs(float(row.get("diff__mean_delta_target_prob", 0.0))),
        reverse=True,
    )[:15]
    if top_type_gaps:
        parts.append("Largest Type x Gain Profile Gaps")
        parts.append(
            render_table(
                ["type", "g_profile", "a_mean_delta_p", "b_mean_delta_p", "diff(a-b)", "a_%pos", "b_%pos"],
                [[
                    str(row.get("type", "")),
                    str(row.get("g_profile", "")),
                    format_float(float(row.get("a__mean_delta_target_prob", math.nan))),
                    format_float(float(row.get("b__mean_delta_target_prob", math.nan))),
                    format_float(float(row.get("diff__mean_delta_target_prob", math.nan))),
                    format_float(float(row.get("a__pct_delta_target_prob_positive", math.nan))),
                    format_float(float(row.get("b__pct_delta_target_prob_positive", math.nan))),
                ] for row in top_type_gaps],
            )
        )
        parts.append("")

    top_family_gaps = sorted(
        type_family_rows,
        key=lambda row: abs(float(row.get("diff__mean_delta_target_prob", 0.0))),
        reverse=True,
    )[:15]
    if top_family_gaps:
        parts.append("Largest Type x Gain Family Gaps")
        parts.append(
            render_table(
                ["type", "g_family", "a_mean_delta_p", "b_mean_delta_p", "diff(a-b)", "a_%pos", "b_%pos"],
                [[
                    str(row.get("type", "")),
                    str(row.get("g_family", "")),
                    format_float(float(row.get("a__mean_delta_target_prob", math.nan))),
                    format_float(float(row.get("b__mean_delta_target_prob", math.nan))),
                    format_float(float(row.get("diff__mean_delta_target_prob", math.nan))),
                    format_float(float(row.get("a__pct_delta_target_prob_positive", math.nan))),
                    format_float(float(row.get("b__pct_delta_target_prob_positive", math.nan))),
                ] for row in top_family_gaps],
            )
        )
        parts.append("")

    top_prompt_gaps = sorted(
        prompt_rows,
        key=lambda row: abs(float(row.get("diff__delta_target_prob", 0.0))),
        reverse=True,
    )[:15]
    if top_prompt_gaps:
        parts.append("Largest Prompt x Gain Gaps")
        parts.append(
            render_table(
                ["prompt_id", "type", "g_profile", "a_delta_p", "b_delta_p", "diff(a-b)"],
                [[
                    str(row.get("prompt_id", "")),
                    str(row.get("type", "")),
                    str(row.get("g_profile", "")),
                    format_float(float(row.get(f"{run_a['label']}__delta_target_prob", math.nan))),
                    format_float(float(row.get(f"{run_b['label']}__delta_target_prob", math.nan))),
                    format_float(float(row.get("diff__delta_target_prob", math.nan))),
                ] for row in top_prompt_gaps],
            )
        )

    return "\n".join(parts)


def write_artifacts(
    output_dir: Path,
    prefix: str,
    report_text: str,
    file_rows: list[dict[str, Any]],
    warning_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    type_gain_rows: list[dict[str, Any]],
    type_family_rows: list[dict[str, Any]],
) -> list[Path]:
    written: list[Path] = []
    specs = [
        (f"{prefix}_files.csv", file_rows),
        (f"{prefix}_warnings.csv", warning_rows),
        (f"{prefix}_prompt_pairwise.csv", prompt_rows),
        (f"{prefix}_type_gain_pairwise.csv", type_gain_rows),
        (f"{prefix}_type_family_pairwise.csv", type_family_rows),
    ]
    for filename, rows in specs:
        path = output_dir / filename
        fieldnames = list(rows[0].keys()) if rows else ["value"]
        write_csv(path, rows, fieldnames)
        written.append(path)

    report_path = output_dir / f"{prefix}_report.txt"
    write_text(report_path, report_text)
    written.append(report_path)
    return written


def build_cross_model_scout_analysis(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    top_k: int = 15,
) -> dict[str, Any] | None:
    """Cross-model scout head alignment analysis.

    For each model, identify the top-K scout heads (most frequently in top-10
    correlated heads across gain profiles).  Then, for each shared prompt,
    extract those scouts' baseline entropy values.  Finally, measure whether
    model A's scout entropy predicts model B's delta_p and vice versa.

    Returns a JSON-serialisable dict or None if data is insufficient.
    """
    head_corr_a_path = run_a["run_dir"] / "analysis_head_correlations.json"
    head_corr_b_path = run_b["run_dir"] / "analysis_head_correlations.json"
    pca_a_path = run_a["run_dir"] / "analysis_baseline_attn_pca.json"
    pca_b_path = run_b["run_dir"] / "analysis_baseline_attn_pca.json"

    for p in [head_corr_a_path, head_corr_b_path, pca_a_path, pca_b_path]:
        if not p.is_file():
            return None

    with open(head_corr_a_path, "r", encoding="utf-8") as f:
        hc_a = json.load(f)
    with open(head_corr_b_path, "r", encoding="utf-8") as f:
        hc_b = json.load(f)
    with open(pca_a_path, "r", encoding="utf-8") as f:
        pca_a = json.load(f)
    with open(pca_b_path, "r", encoding="utf-8") as f:
        pca_b = json.load(f)

    # --- Identify top-K scout heads for each model ---
    def _top_scout_indices(hc: dict[str, Any], k: int) -> list[int]:
        scout = np.array(hc["scout_heads_matrix"], dtype=np.int64).ravel()
        return list(np.argsort(-scout)[:k])

    scouts_a = _top_scout_indices(hc_a, top_k)
    scouts_b = _top_scout_indices(hc_b, top_k)

    # Decode scout indices to (layer_slot, head) for metadata.
    def _decode_scouts(indices: list[int], n_heads: int, layer_indices: list[int] | None) -> list[dict[str, Any]]:
        result = []
        for idx in indices:
            ls = int(idx) // n_heads
            hd = int(idx) % n_heads
            result.append({
                "flat_index": int(idx),
                "layer_slot": ls,
                "layer_index": int(layer_indices[ls]) if layer_indices else ls,
                "head": hd,
            })
        return result

    scouts_a_meta = _decode_scouts(scouts_a, hc_a["n_heads"], hc_a.get("layer_indices"))
    scouts_b_meta = _decode_scouts(scouts_b, hc_b["n_heads"], hc_b.get("layer_indices"))

    # --- Build per-prompt scout entropy vectors from verbose baseline data ---
    # The PCA JSON has the prompt_ids but not the raw entropy vectors.
    # We need to go back to the verbose baseline data.  Since sweep_compare
    # works from analysis dirs, we look for the verbose data in the run_dir's
    # parent (the actual run directory, not the analysis subdirectory).
    def _load_verbose_baselines(analysis_dir: Path) -> dict[str, dict[str, Any]]:
        """Load verbose baseline records, keyed by prompt_id."""
        # analysis_dir is like .../OLMO/analysis; verbose.jsonl is in .../OLMO/
        run_root = analysis_dir.parent
        verbose_path = run_root / "verbose.jsonl"
        if not verbose_path.is_file():
            return {}
        lookup: dict[str, dict[str, Any]] = {}
        with open(verbose_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                g_scales = record.get("g_attention_scales")
                is_baseline = (
                    isinstance(g_scales, list)
                    and g_scales
                    and all(abs(float(s) - 1.0) < 1e-6 for s in g_scales)
                )
                if is_baseline:
                    pid = str(record["prompt_id"])
                    if pid not in lookup:  # keep first (rep=1)
                        lookup[pid] = record
        return lookup

    verbose_a = _load_verbose_baselines(run_a["run_dir"])
    verbose_b = _load_verbose_baselines(run_b["run_dir"])

    if not verbose_a or not verbose_b:
        return None

    def _flatten_entropy(record: dict[str, Any]) -> list[float] | None:
        raw = record.get("attn_entropy_per_head_final")
        if not isinstance(raw, list):
            return None
        flat: list[float] = []
        for item in raw:
            if isinstance(item, list):
                flat.extend(float(v) for v in item)
            else:
                flat.append(float(item))
        return flat if flat else None

    # Build delta_p lookups: prompt_id -> {profile -> delta_p}
    def _build_delta_lookup(joined_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
        lookup: dict[str, dict[str, float]] = {}
        for row in joined_rows:
            pid = str(row.get("prompt_id", ""))
            profile = str(row.get("g_profile", ""))
            if profile == "baseline":
                continue
            dp = row.get("delta_target_prob")
            if dp is None or dp == "":
                continue
            dpf = float(dp)
            if math.isfinite(dpf):
                lookup.setdefault(pid, {})[profile] = dpf
        return lookup

    delta_a = _build_delta_lookup(run_a["joined_rows"])
    delta_b = _build_delta_lookup(run_b["joined_rows"])

    # Find shared prompts with data in both models.
    shared_pids = sorted(set(verbose_a.keys()) & set(verbose_b.keys()) & set(delta_a.keys()) & set(delta_b.keys()))
    if len(shared_pids) < 20:
        return None

    # Extract scout entropy vectors for shared prompts.
    scout_entropy_a: list[list[float]] = []
    scout_entropy_b: list[list[float]] = []
    valid_pids: list[str] = []

    for pid in shared_pids:
        flat_a = _flatten_entropy(verbose_a[pid])
        flat_b = _flatten_entropy(verbose_b[pid])
        if flat_a is None or flat_b is None:
            continue
        if max(scouts_a) >= len(flat_a) or max(scouts_b) >= len(flat_b):
            continue
        scout_entropy_a.append([flat_a[i] for i in scouts_a])
        scout_entropy_b.append([flat_b[i] for i in scouts_b])
        valid_pids.append(pid)

    if len(valid_pids) < 20:
        return None

    se_a = np.array(scout_entropy_a, dtype=np.float64)  # (n_prompts, top_k)
    se_b = np.array(scout_entropy_b, dtype=np.float64)

    # Compute mean scout entropy per prompt for each model.
    mean_se_a = se_a.mean(axis=1)
    mean_se_b = se_b.mean(axis=1)

    # Correlation between the two models' mean scout entropy.
    r_scout_alignment, p_scout_alignment = _pearsonr(mean_se_a, mean_se_b)

    # Shared gain profiles.
    all_profiles_a = set()
    all_profiles_b = set()
    for pid in valid_pids:
        all_profiles_a.update(delta_a.get(pid, {}).keys())
        all_profiles_b.update(delta_b.get(pid, {}).keys())
    shared_profiles = sorted(all_profiles_a & all_profiles_b)

    # Per-profile cross-model analysis.
    profile_results: list[dict[str, Any]] = []
    for profile in shared_profiles:
        # Get delta_p vectors for this profile.
        dp_a = np.array([delta_a.get(pid, {}).get(profile, np.nan) for pid in valid_pids])
        dp_b = np.array([delta_b.get(pid, {}).get(profile, np.nan) for pid in valid_pids])
        valid_mask = np.isfinite(dp_a) & np.isfinite(dp_b)
        n_valid = int(valid_mask.sum())
        if n_valid < 20:
            continue

        dp_a_v = dp_a[valid_mask]
        dp_b_v = dp_b[valid_mask]
        mse_a_v = mean_se_a[valid_mask]
        mse_b_v = mean_se_b[valid_mask]

        # Cross-model predictions:
        # Does model A's scout entropy predict model B's delta_p?
        r_a_scouts_b_delta, p_a_scouts_b_delta = _pearsonr(mse_a_v, dp_b_v)
        # Does model B's scout entropy predict model A's delta_p?
        r_b_scouts_a_delta, p_b_scouts_a_delta = _pearsonr(mse_b_v, dp_a_v)
        # Self-predictions for comparison.
        r_a_scouts_a_delta, p_a_scouts_a_delta = _pearsonr(mse_a_v, dp_a_v)
        r_b_scouts_b_delta, p_b_scouts_b_delta = _pearsonr(mse_b_v, dp_b_v)
        # Model agreement: correlation of delta_p between models.
        r_delta_agreement, p_delta_agreement = _pearsonr(dp_a_v, dp_b_v)

        profile_results.append({
            "g_profile": profile,
            "n_valid": n_valid,
            "r_delta_agreement": round(float(r_delta_agreement), 6),
            "p_delta_agreement": round(float(p_delta_agreement), 6),
            "r_a_scouts_predict_a_delta": round(float(r_a_scouts_a_delta), 6),
            "r_a_scouts_predict_b_delta": round(float(r_a_scouts_b_delta), 6),
            "p_a_scouts_predict_b_delta": round(float(p_a_scouts_b_delta), 6),
            "r_b_scouts_predict_b_delta": round(float(r_b_scouts_b_delta), 6),
            "r_b_scouts_predict_a_delta": round(float(r_b_scouts_a_delta), 6),
            "p_b_scouts_predict_a_delta": round(float(p_b_scouts_a_delta), 6),
        })

    # Per-prompt summary: mean scout entropy + mean delta_p across profiles.
    prompt_scout_summary: list[dict[str, Any]] = []
    for i, pid in enumerate(valid_pids):
        mean_dp_a = np.nanmean([delta_a.get(pid, {}).get(p, np.nan) for p in shared_profiles])
        mean_dp_b = np.nanmean([delta_b.get(pid, {}).get(p, np.nan) for p in shared_profiles])
        prompt_scout_summary.append({
            "prompt_id": pid,
            "mean_scout_entropy_a": round(float(mean_se_a[i]), 6),
            "mean_scout_entropy_b": round(float(mean_se_b[i]), 6),
            "mean_delta_p_a": round(float(mean_dp_a), 6),
            "mean_delta_p_b": round(float(mean_dp_b), 6),
        })

    return {
        "description": "Cross-model scout head alignment analysis",
        "label_a": run_a["label"],
        "label_b": run_b["label"],
        "model_a": run_a["model"],
        "model_b": run_b["model"],
        "top_k": top_k,
        "n_shared_prompts": len(valid_pids),
        "n_shared_profiles": len(shared_profiles),
        "scouts_a": scouts_a_meta,
        "scouts_b": scouts_b_meta,
        "scout_entropy_alignment": {
            "r": round(float(r_scout_alignment), 6),
            "p": round(float(p_scout_alignment), 6),
            "description": "Pearson r between mean scout entropy of model A vs model B across shared prompts",
        },
        "profile_results": profile_results,
        "prompt_scout_summary": prompt_scout_summary,
    }


def main() -> None:
    args = parse_args()
    configure_data_dir(args.data_dir)
    run_a_dir = resolve_analysis_dir(args.run_a)
    run_b_dir = resolve_analysis_dir(args.run_b)
    run_a = load_run_analysis(run_a_dir, args.label_a)
    run_b = load_run_analysis(run_b_dir, args.label_b)

    prompt_rows, prompt_warnings = build_prompt_pairwise_rows(run_a, run_b)
    type_gain_rows, type_gain_warnings = build_group_pairwise_rows(
        run_a["type_gain_rows"],
        run_b["type_gain_rows"],
        ["type", "g_profile", "g_family"],
    )
    type_family_rows, type_family_warnings = build_group_pairwise_rows(
        run_a["type_family_rows"],
        run_b["type_family_rows"],
        ["type", "g_family"],
    )
    warnings = prompt_warnings + type_gain_warnings + type_family_warnings

    file_rows = [
        {"slot": "a", "label": run_a["label"], "model": run_a["model"], "run_dir": str(run_a["run_dir"])},
        {"slot": "b", "label": run_b["label"], "model": run_b["model"], "run_dir": str(run_b["run_dir"])},
    ]
    warning_rows = [{"warning": warning} for warning in warnings]
    report_text = build_report_text(run_a, run_b, warnings, prompt_rows, type_gain_rows, type_family_rows)
    print(report_text)

    if not args.no_write_files:
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else default_compare_output_dir(run_a_dir, run_b_dir)
        )
        ensure_new_output_dir(output_dir, "comparison output directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        written = write_artifacts(
            output_dir=output_dir,
            prefix=args.prefix,
            report_text=report_text,
            file_rows=file_rows,
            warning_rows=warning_rows,
            prompt_rows=prompt_rows,
            type_gain_rows=type_gain_rows,
            type_family_rows=type_family_rows,
        )
        print()
        print("Wrote comparison files:")
        for path in written:
            print(f"- {path}")

        # Cross-model scout analysis.
        scout_result = build_cross_model_scout_analysis(run_a, run_b)
        if scout_result is not None:
            scout_path = output_dir / f"{args.prefix}_cross_model_scouts.json"
            with open(scout_path, "w", encoding="utf-8") as f:
                json.dump(scout_result, f, indent=2)
            print(f"- {scout_path}")
        else:
            print("(Skipped cross-model scout analysis — insufficient data.)")

    if args.json_out:
        payload = {
            "run_a": {"label": run_a["label"], "model": run_a["model"], "run_dir": str(run_a["run_dir"])},
            "run_b": {"label": run_b["label"], "model": run_b["model"], "run_dir": str(run_b["run_dir"])},
            "warnings": warnings,
            "prompt_pairwise": prompt_rows,
            "type_gain_pairwise": type_gain_rows,
            "type_family_pairwise": type_family_rows,
        }
        output_path = Path(args.json_out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
