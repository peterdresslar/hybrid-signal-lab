#!/usr/bin/env python3
"""Analyze calibration JSONL files for a battery folder.

This script scans a battery directory for calibration result files, loads every
``.jsonl`` file it finds, and prints simple cross-model summaries.

It is intentionally stdlib-only so it is easy to run anywhere:

    uv run -m battery.src.calibration_analyze --calibration path/to/calibration.jsonl
    uv run -m battery.src.calibration_analyze --battery-dir battery/data/battery_3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LOW_PROB_THRESHOLD = 0.05
HIGH_PROB_THRESHOLD = 0.85
REPORT_FLOAT_DIGITS = 2


@dataclass
class SummaryRow:
    group: str
    model: str
    n: int
    mean_prob: float
    median_prob: float
    mean_rank: float
    median_rank: float
    pct_rank1: float
    pct_rank_le_5: float
    pct_rank_le_10: float
    pct_prob_low: float
    pct_prob_mid: float
    pct_prob_high: float
    mean_entropy: float
    mean_time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze calibration JSONL files in a battery folder."
    )
    parser.add_argument(
        "--calibration",
        type=str,
        nargs="+",
        default=None,
        help="One or more calibration JSONL files to analyze. Alternative to --battery-dir.",
    )
    parser.add_argument(
        "--battery-dir",
        type=str,
        default=None,
        help="Directory containing calibration JSONL files (for example battery/data/battery_3). "
             "All *.jsonl files in the directory will be loaded.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Path to all_candidates.json for enriching output with prompt metadata. "
             "Auto-detected from --battery-dir when available.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write a machine-readable JSON report.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated analysis files (default: --battery-dir).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="analysis",
        help="Filename prefix for generated artifacts (default: analysis).",
    )
    parser.add_argument(
        "--no-write-files",
        action="store_true",
        help="Print the report without writing CSV/text analysis files.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_num}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_num}")
            rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan
    return 100.0 * numerator / denominator


def summarize_records(group: str, model: str, records: list[dict[str, Any]]) -> SummaryRow:
    probs = [float(r["target_prob"]) for r in records]
    ranks = [int(r["target_rank"]) for r in records]
    entropies = [float(r["final_entropy"]) for r in records]
    times = [float(r.get("time_s", 0.0)) for r in records]

    low_count = sum(p < LOW_PROB_THRESHOLD for p in probs)
    high_count = sum(p > HIGH_PROB_THRESHOLD for p in probs)
    mid_count = len(probs) - low_count - high_count

    return SummaryRow(
        group=group,
        model=model,
        n=len(records),
        mean_prob=mean(probs),
        median_prob=median(probs),
        mean_rank=mean([float(r) for r in ranks]),
        median_rank=median([float(r) for r in ranks]),
        pct_rank1=pct(sum(r == 1 for r in ranks), len(ranks)),
        pct_rank_le_5=pct(sum(r <= 5 for r in ranks), len(ranks)),
        pct_rank_le_10=pct(sum(r <= 10 for r in ranks), len(ranks)),
        pct_prob_low=pct(low_count, len(probs)),
        pct_prob_mid=pct(mid_count, len(probs)),
        pct_prob_high=pct(high_count, len(probs)),
        mean_entropy=mean(entropies),
        mean_time_s=mean(times),
    )


def group_records(
    model_name: str,
    records: list[dict[str, Any]],
    field: str,
) -> list[SummaryRow]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    return [
        summarize_records(group=group_name, model=model_name, records=grouped[group_name])
        for group_name in sorted(grouped)
    ]


def format_float(value: float, digits: int = 2) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def round_report_value(value: Any) -> Any:
    """Round floats for persisted report artifacts."""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return round(value, REPORT_FLOAT_DIGITS)
    return value


def round_report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: round_report_value(value) for key, value in row.items()}


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


def summarize_runs(
    runs: list[dict[str, Any]],
) -> tuple[list[SummaryRow], list[SummaryRow], list[SummaryRow], list[SummaryRow], list[SummaryRow], list[SummaryRow]]:
    overall: list[SummaryRow] = []
    by_type: list[SummaryRow] = []
    by_tier: list[SummaryRow] = []
    by_family: list[SummaryRow] = []
    by_concept: list[SummaryRow] = []
    by_difficulty: list[SummaryRow] = []

    for run in runs:
        model = run["model"]
        records = run["records"]
        overall.append(summarize_records(group="__overall__", model=model, records=records))
        by_type.extend(group_records(model, records, "type"))
        by_tier.extend(group_records(model, records, "tier"))
        family_records = [record for record in records if record.get("family") is not None]
        concept_records = [record for record in records if record.get("concept") is not None]
        difficulty_records = [record for record in records if record.get("difficulty") is not None]
        if family_records:
            by_family.extend(group_records(model, family_records, "family"))
        if concept_records:
            by_concept.extend(group_records(model, concept_records, "concept"))
        if difficulty_records:
            by_difficulty.extend(group_records(model, difficulty_records, "difficulty"))

    overall.sort(key=lambda row: row.model)
    by_type.sort(key=lambda row: (row.group, row.model))
    by_tier.sort(key=lambda row: (row.group, row.model))
    by_family.sort(key=lambda row: (row.group, row.model))
    by_concept.sort(key=lambda row: (row.group, row.model))
    by_difficulty.sort(key=lambda row: (row.group, row.model))
    return overall, by_type, by_tier, by_family, by_concept, by_difficulty


def collect_run_metadata(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    model_names = sorted({str(record.get("model", "unknown")) for record in records})
    render_versions = sorted({str(record.get("prompt_render_version", "unknown")) for record in records})
    adapters = sorted({str(record.get("adapter", "unknown")) for record in records})
    ids = [str(record["id"]) for record in records]

    return {
        "path": str(path),
        "file": path.name,
        "label": path.stem,
        "records": records,
        "row_count": len(records),
        "models": model_names,
        "model": ", ".join(model_names),
        "render_versions": render_versions,
        "adapters": adapters,
        "ids": ids,
    }


def build_consistency_warnings(runs: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if not runs:
        return warnings

    base = runs[0]
    base_id_set = set(base["ids"])
    base_by_id = {record["id"]: record for record in base["records"]}

    for run in runs:
        if len(run["models"]) != 1:
            warnings.append(
                f"{run['file']} contains multiple model names: {run['models']}"
            )
        if len(run["render_versions"]) != 1:
            warnings.append(
                f"{run['file']} contains multiple prompt render versions: {run['render_versions']}"
            )

    for run in runs[1:]:
        id_set = set(run["ids"])
        missing = sorted(base_id_set - id_set)
        extra = sorted(id_set - base_id_set)
        if missing or extra:
            warnings.append(
                f"{run['file']} has ID-set mismatch versus {base['file']}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
            continue

        mismatch_sha = 0
        mismatch_adapter = 0
        mismatch_render_version = 0
        for record in run["records"]:
            base_record = base_by_id[record["id"]]
            if record.get("rendered_prompt_sha256") != base_record.get("rendered_prompt_sha256"):
                mismatch_sha += 1
            if record.get("adapter") != base_record.get("adapter"):
                mismatch_adapter += 1
            if record.get("prompt_render_version") != base_record.get("prompt_render_version"):
                mismatch_render_version += 1

        if mismatch_sha:
            warnings.append(
                f"{run['file']} differs from {base['file']} on rendered_prompt_sha256 for "
                f"{mismatch_sha} IDs"
            )
        if mismatch_adapter:
            warnings.append(
                f"{run['file']} differs from {base['file']} on adapter for "
                f"{mismatch_adapter} IDs"
            )
        if mismatch_render_version:
            warnings.append(
                f"{run['file']} differs from {base['file']} on prompt_render_version for "
                f"{mismatch_render_version} IDs"
            )

    return warnings


def build_type_deltas(by_type: list[SummaryRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SummaryRow]] = defaultdict(list)
    for row in by_type:
        grouped[row.group].append(row)

    deltas: list[dict[str, Any]] = []
    for group_name, rows in grouped.items():
        if len(rows) < 2:
            continue
        best_rank = min(rows, key=lambda row: row.mean_rank)
        worst_rank = max(rows, key=lambda row: row.mean_rank)
        best_rank1 = max(rows, key=lambda row: row.pct_rank1)
        worst_rank1 = min(rows, key=lambda row: row.pct_rank1)
        deltas.append(
            {
                "group": group_name,
                "n_models": len(rows),
                "best_mean_rank_model": best_rank.model,
                "best_mean_rank": best_rank.mean_rank,
                "worst_mean_rank_model": worst_rank.model,
                "worst_mean_rank": worst_rank.mean_rank,
                "mean_rank_span": worst_rank.mean_rank - best_rank.mean_rank,
                "best_rank1_model": best_rank1.model,
                "best_rank1_pct": best_rank1.pct_rank1,
                "worst_rank1_model": worst_rank1.model,
                "worst_rank1_pct": worst_rank1.pct_rank1,
                "rank1_pct_span": best_rank1.pct_rank1 - worst_rank1.pct_rank1,
            }
        )

    deltas.sort(key=lambda item: (-item["mean_rank_span"], item["group"]))
    return deltas


def load_candidate_lookup(battery_dir: Path) -> dict[str, dict[str, Any]]:
    """Load candidate metadata from all_candidates.json when available."""
    candidate_path = battery_dir / "all_candidates.json"
    if not candidate_path.exists():
        return {}

    with open(candidate_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for item in data:
        if isinstance(item, dict) and "id" in item:
            lookup[str(item["id"])] = item
    return lookup


def build_item_comparison_rows(
    runs: list[dict[str, Any]],
    candidate_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = defaultdict(dict)

    for run in runs:
        for record in run["records"]:
            item_id = str(record["id"])
            by_id[item_id]["id"] = item_id
            by_id[item_id]["type"] = record.get("type")
            by_id[item_id]["tier"] = record.get("tier")
            by_id[item_id]["family"] = record.get("family")
            by_id[item_id]["concept"] = record.get("concept")
            by_id[item_id]["difficulty"] = record.get("difficulty")
            by_id[item_id]["source"] = record.get("source", by_id[item_id].get("source"))
            by_id[item_id]["target_num_tokens"] = record.get("target_num_tokens")
            by_id[item_id]["target_starts_with_space"] = record.get("target_starts_with_space")
            by_id[item_id]["target_first_token_str"] = record.get("target_first_token_str")

            candidate = candidate_lookup.get(item_id, {})
            if candidate:
                by_id[item_id]["source"] = candidate.get("source")
                by_id[item_id]["prompt"] = candidate.get("prompt")
                by_id[item_id]["target"] = candidate.get("target")
                by_id[item_id]["tokens_approx"] = candidate.get("tokens_approx")
                metadata = candidate.get("metadata")
                by_id[item_id]["metadata_json"] = json.dumps(
                    metadata if metadata is not None else {},
                    ensure_ascii=False,
                    sort_keys=True,
                )

            label = run["label"]
            by_id[item_id][f"{label}__model"] = record.get("model")
            by_id[item_id][f"{label}__adapter"] = record.get("adapter")
            by_id[item_id][f"{label}__prompt_render_version"] = record.get("prompt_render_version")
            by_id[item_id][f"{label}__rendered_prompt_sha256"] = record.get("rendered_prompt_sha256")
            by_id[item_id][f"{label}__target_prob"] = record.get("target_prob")
            by_id[item_id][f"{label}__target_rank"] = record.get("target_rank")
            by_id[item_id][f"{label}__final_entropy"] = record.get("final_entropy")
            by_id[item_id][f"{label}__mean_seq_entropy"] = record.get("mean_seq_entropy")
            by_id[item_id][f"{label}__target_avg_logp"] = record.get("target_avg_logp")
            by_id[item_id][f"{label}__time_s"] = record.get("time_s")

    rows = list(by_id.values())
    rows.sort(key=lambda row: str(row["id"]))
    return [round_report_row(row) for row in rows]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def summary_rows_to_dicts(summary_rows: list[SummaryRow]) -> list[dict[str, Any]]:
    return [round_report_row(asdict(row)) for row in summary_rows]


def build_file_summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        rows.append(
            {
                "file": run["file"],
                "label": run["label"],
                "row_count": run["row_count"],
                "model": run["model"],
                "models_json": json.dumps(run["models"]),
                "render_versions_json": json.dumps(run["render_versions"]),
                "adapters_json": json.dumps(run["adapters"]),
                "path": run["path"],
            }
        )
    return rows


def build_warning_rows(warnings: list[str]) -> list[dict[str, Any]]:
    return [{"warning": warning} for warning in warnings]


def build_report_text(
    battery_dir: Path,
    runs: list[dict[str, Any]],
    warnings: list[str],
    overall: list[SummaryRow],
    by_type: list[SummaryRow],
    by_tier: list[SummaryRow],
    by_family: list[SummaryRow],
    by_concept: list[SummaryRow],
    by_difficulty: list[SummaryRow],
    type_deltas: list[dict[str, Any]],
) -> str:
    parts = [f"Battery directory: {battery_dir}", f"Calibration files: {len(runs)}", ""]

    file_rows = []
    for run in runs:
        file_rows.append(
            [
                run["file"],
                str(run["row_count"]),
                run["model"],
                ",".join(run["render_versions"]),
                ",".join(run["adapters"]),
            ]
        )
    parts.append("Files")
    parts.append(
        render_table(
            ["file", "rows", "model", "prompt_render_version", "adapters"],
            file_rows,
        )
    )
    parts.append("")

    if warnings:
        parts.append("Warnings")
        parts.extend(f"- {warning}" for warning in warnings)
        parts.append("")

    parts.append("Overall Summary By Model")
    parts.append(
        render_table(
            [
                "model",
                "n",
                "mean_p",
                "median_p",
                "mean_rank",
                "median_rank",
                "%r1",
                "%r<=5",
                "%r<=10",
                "%p<0.05",
                "%mid",
                "%p>0.85",
                "mean_H",
                "mean_s",
            ],
            rows_for_summary(overall, include_group=False),
        )
    )
    parts.append("")

    parts.append("Type Summary By Model")
    parts.append(
        render_table(
            [
                "type",
                "model",
                "n",
                "mean_p",
                "median_p",
                "mean_rank",
                "median_rank",
                "%r1",
                "%r<=5",
                "%r<=10",
                "%p<0.05",
                "%mid",
                "%p>0.85",
                "mean_H",
                "mean_s",
            ],
            rows_for_summary(by_type, include_group=True),
        )
    )
    parts.append("")

    parts.append("Tier Summary By Model")
    parts.append(
        render_table(
            [
                "tier",
                "model",
                "n",
                "mean_p",
                "median_p",
                "mean_rank",
                "median_rank",
                "%r1",
                "%r<=5",
                "%r<=10",
                "%p<0.05",
                "%mid",
                "%p>0.85",
                "mean_H",
                "mean_s",
            ],
            rows_for_summary(by_tier, include_group=True),
        )
    )
    parts.append("")

    if by_family:
        parts.append("Family Summary By Model")
        parts.append(
            render_table(
                [
                    "family", "model", "n", "mean_p", "median_p", "mean_rank", "median_rank",
                    "%r1", "%r<=5", "%r<=10", "%p<0.05", "%mid", "%p>0.85", "mean_H", "mean_s",
                ],
                rows_for_summary(by_family, include_group=True),
            )
        )
        parts.append("")

    if by_difficulty:
        parts.append("Difficulty Summary By Model")
        parts.append(
            render_table(
                [
                    "difficulty", "model", "n", "mean_p", "median_p", "mean_rank", "median_rank",
                    "%r1", "%r<=5", "%r<=10", "%p<0.05", "%mid", "%p>0.85", "mean_H", "mean_s",
                ],
                rows_for_summary(by_difficulty, include_group=True),
            )
        )
        parts.append("")

    if type_deltas:
        delta_rows = [
            [
                item["group"],
                item["best_mean_rank_model"],
                format_float(item["best_mean_rank"], REPORT_FLOAT_DIGITS),
                item["worst_mean_rank_model"],
                format_float(item["worst_mean_rank"], REPORT_FLOAT_DIGITS),
                format_float(item["mean_rank_span"], REPORT_FLOAT_DIGITS),
                item["best_rank1_model"],
                format_float(item["best_rank1_pct"], REPORT_FLOAT_DIGITS),
                item["worst_rank1_model"],
                format_float(item["worst_rank1_pct"], REPORT_FLOAT_DIGITS),
                format_float(item["rank1_pct_span"], REPORT_FLOAT_DIGITS),
            ]
            for item in type_deltas
        ]
        parts.append("Cross-Model Type Deltas")
        parts.append(
            render_table(
                [
                    "type",
                    "best_rank_model",
                    "best_rank",
                    "worst_rank_model",
                    "worst_rank",
                    "rank_span",
                    "best_%r1_model",
                    "best_%r1",
                    "worst_%r1_model",
                    "worst_%r1",
                    "%r1_span",
                ],
                delta_rows,
            )
        )
    return "\n".join(parts)


def write_analysis_files(
    output_dir: Path,
    prefix: str,
    report_text: str,
    runs: list[dict[str, Any]],
    warnings: list[str],
    overall: list[SummaryRow],
    by_type: list[SummaryRow],
    by_tier: list[SummaryRow],
    by_family: list[SummaryRow],
    by_concept: list[SummaryRow],
    by_difficulty: list[SummaryRow],
    type_deltas: list[dict[str, Any]],
    item_rows: list[dict[str, Any]],
) -> list[Path]:
    written: list[Path] = []

    files_path = output_dir / f"{prefix}_files.csv"
    write_csv(
        files_path,
        build_file_summary_rows(runs),
        [
            "file",
            "label",
            "row_count",
            "model",
            "models_json",
            "render_versions_json",
            "adapters_json",
            "path",
        ],
    )
    written.append(files_path)

    warnings_path = output_dir / f"{prefix}_warnings.csv"
    write_csv(warnings_path, build_warning_rows(warnings), ["warning"])
    written.append(warnings_path)

    overall_path = output_dir / f"{prefix}_overall_summary.csv"
    write_csv(overall_path, summary_rows_to_dicts(overall), list(asdict(overall[0]).keys()))
    written.append(overall_path)

    by_type_path = output_dir / f"{prefix}_type_summary.csv"
    write_csv(by_type_path, summary_rows_to_dicts(by_type), list(asdict(by_type[0]).keys()))
    written.append(by_type_path)

    by_tier_path = output_dir / f"{prefix}_tier_summary.csv"
    write_csv(by_tier_path, summary_rows_to_dicts(by_tier), list(asdict(by_tier[0]).keys()))
    written.append(by_tier_path)

    if by_family:
        by_family_path = output_dir / f"{prefix}_family_summary.csv"
        write_csv(by_family_path, summary_rows_to_dicts(by_family), list(asdict(by_family[0]).keys()))
        written.append(by_family_path)

    if by_concept:
        by_concept_path = output_dir / f"{prefix}_concept_summary.csv"
        write_csv(by_concept_path, summary_rows_to_dicts(by_concept), list(asdict(by_concept[0]).keys()))
        written.append(by_concept_path)

    if by_difficulty:
        by_difficulty_path = output_dir / f"{prefix}_difficulty_summary.csv"
        write_csv(by_difficulty_path, summary_rows_to_dicts(by_difficulty), list(asdict(by_difficulty[0]).keys()))
        written.append(by_difficulty_path)

    deltas_path = output_dir / f"{prefix}_type_deltas.csv"
    rounded_type_deltas = [round_report_row(row) for row in type_deltas]
    write_csv(
        deltas_path,
        rounded_type_deltas,
        list(rounded_type_deltas[0].keys()) if rounded_type_deltas else ["group"],
    )
    written.append(deltas_path)

    item_path = output_dir / f"{prefix}_item_cross_model.csv"
    item_fieldnames = []
    if item_rows:
        fieldnames: set[str] = set()
        for row in item_rows:
            fieldnames.update(row.keys())
        preferred = [
            "id",
            "type",
            "tier",
            "family",
            "concept",
            "difficulty",
            "source",
            "target",
            "target_num_tokens",
            "target_starts_with_space",
            "target_first_token_str",
            "tokens_approx",
            "prompt",
            "metadata_json",
        ]
        item_fieldnames = [name for name in preferred if name in fieldnames] + sorted(
            name for name in fieldnames if name not in preferred
        )
    else:
        item_fieldnames = ["id"]
    write_csv(item_path, item_rows, item_fieldnames)
    written.append(item_path)

    text_path = output_dir / f"{prefix}_report.txt"
    write_text(text_path, report_text)
    written.append(text_path)

    return written


def rows_for_summary(summary_rows: list[SummaryRow], include_group: bool) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in summary_rows:
        rendered = []
        if include_group:
            rendered.append(row.group)
        rendered.extend(
            [
                row.model,
                str(row.n),
                format_float(row.mean_prob, REPORT_FLOAT_DIGITS),
                format_float(row.median_prob, REPORT_FLOAT_DIGITS),
                format_float(row.mean_rank, REPORT_FLOAT_DIGITS),
                format_float(row.median_rank, REPORT_FLOAT_DIGITS),
                format_float(row.pct_rank1, REPORT_FLOAT_DIGITS),
                format_float(row.pct_rank_le_5, REPORT_FLOAT_DIGITS),
                format_float(row.pct_rank_le_10, REPORT_FLOAT_DIGITS),
                format_float(row.pct_prob_low, REPORT_FLOAT_DIGITS),
                format_float(row.pct_prob_mid, REPORT_FLOAT_DIGITS),
                format_float(row.pct_prob_high, REPORT_FLOAT_DIGITS),
                format_float(row.mean_entropy, REPORT_FLOAT_DIGITS),
                format_float(row.mean_time_s, REPORT_FLOAT_DIGITS),
            ]
        )
        rows.append(rendered)
    return rows


def print_report(report_text: str) -> None:
    print(report_text)


def main() -> None:
    args = parse_args()

    if not args.calibration and not args.battery_dir:
        raise SystemExit("Error: provide either --calibration FILE(s) or --battery-dir DIR.")

    # Resolve calibration file paths
    if args.calibration:
        calibration_paths = []
        for p in args.calibration:
            path = Path(p).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Calibration file does not exist: {path}")
            calibration_paths.append(path)
        calibration_paths.sort()
        battery_dir = calibration_paths[0].parent
    else:
        battery_dir = Path(args.battery_dir).expanduser()
        if not battery_dir.exists():
            raise FileNotFoundError(f"Battery directory does not exist: {battery_dir}")
        if not battery_dir.is_dir():
            raise NotADirectoryError(f"Battery directory must be a directory: {battery_dir}")
        calibration_paths = sorted(p for p in battery_dir.glob("*.jsonl") if p.is_file())

    if not calibration_paths:
        raise FileNotFoundError(
            f"No calibration JSONL files found in {battery_dir}"
        )

    runs = []
    for path in calibration_paths:
        records = read_jsonl(path)
        if not records:
            continue
        runs.append(collect_run_metadata(path, records))

    if not runs:
        raise ValueError(f"No non-empty calibration files found in {battery_dir}")

    warnings = build_consistency_warnings(runs)
    overall, by_type, by_tier, by_family, by_concept, by_difficulty = summarize_runs(runs)
    type_deltas = build_type_deltas(by_type)
    # Resolve candidate metadata: explicit --candidates, or auto-detect in battery_dir
    if args.candidates:
        candidates_path = Path(args.candidates).expanduser()
        candidate_lookup = load_candidate_lookup(candidates_path.parent)
    else:
        candidate_lookup = load_candidate_lookup(battery_dir)
    item_rows = build_item_comparison_rows(runs, candidate_lookup)
    report_text = build_report_text(
        battery_dir=battery_dir,
        runs=runs,
        warnings=warnings,
        overall=overall,
        by_type=by_type,
        by_tier=by_tier,
        by_family=by_family,
        by_concept=by_concept,
        by_difficulty=by_difficulty,
        type_deltas=type_deltas,
    )

    print_report(report_text)

    if not args.no_write_files:
        output_dir = Path(args.output_dir).expanduser() if args.output_dir else battery_dir
        written = write_analysis_files(
            output_dir=output_dir,
            prefix=args.prefix,
            report_text=report_text,
            runs=runs,
            warnings=warnings,
            overall=overall,
            by_type=by_type,
            by_tier=by_tier,
            by_family=by_family,
            by_concept=by_concept,
            by_difficulty=by_difficulty,
            type_deltas=type_deltas,
            item_rows=item_rows,
        )
        print()
        print("Wrote analysis files:")
        for path in written:
            print(f"- {path}")

    if args.json_out:
        report = {
            "battery_dir": str(battery_dir),
            "files": [
                {
                    "file": run["file"],
                    "path": run["path"],
                    "row_count": run["row_count"],
                    "models": run["models"],
                    "render_versions": run["render_versions"],
                    "adapters": run["adapters"],
                }
                for run in runs
            ],
            "warnings": warnings,
            "overall": [round_report_row(asdict(row)) for row in overall],
            "by_type": [round_report_row(asdict(row)) for row in by_type],
            "by_tier": [round_report_row(asdict(row)) for row in by_tier],
            "by_family": [round_report_row(asdict(row)) for row in by_family],
            "by_concept": [round_report_row(asdict(row)) for row in by_concept],
            "by_difficulty": [round_report_row(asdict(row)) for row in by_difficulty],
            "type_deltas": [round_report_row(row) for row in type_deltas],
            "item_cross_model": item_rows,
        }
        output_path = Path(args.json_out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
