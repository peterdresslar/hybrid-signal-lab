#!/usr/bin/env python3
"""Pairwise comparison of two analyzed sweep run directories.

This script compares the standardized outputs produced by `colony.sweep_analyze`
for two runs and writes side-by-side prompt, type, and family comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPORT_FLOAT_DIGITS = 2
DEFAULT_PREFIX = "compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two analyzed sweep run directories.")
    parser.add_argument("--run-a", type=str, required=True, help="First analyzed sweep run directory.")
    parser.add_argument("--run-b", type=str, required=True, help="Second analyzed sweep run directory.")
    parser.add_argument("--label-a", type=str, default=None, help="Optional label override for run A.")
    parser.add_argument("--label-b", type=str, default=None, help="Optional label override for run B.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated artifacts (default: run A directory).")
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
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Run directory must be a directory: {path}")
    return path


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


def main() -> None:
    args = parse_args()
    run_a_dir = resolve_run_dir(args.run_a)
    run_b_dir = resolve_run_dir(args.run_b)
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
        output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_a_dir
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
