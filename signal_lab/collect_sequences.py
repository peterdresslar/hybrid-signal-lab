"""Collect baseline sequence-state artifacts for router follow-on work."""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import torch

from model import VALID_MODEL_KEYS
from model.prompt import Prompt
from signal_lab.agent import Agent
from signal_lab.paths import (
    DATA_DIR_ENV_VAR,
    configure_data_dir,
    ensure_new_output_dir,
    render_output_path,
    slugify_path_token,
)
from signal_lab.signal_lab import resolve_device, resolve_prompt_collection

SMOKE_PROMPTS_PER_TIER = 4
VALID_VERBOSITY = (0, 1, 2)
VALID_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _parse_csv_strings(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    return values or None


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def resolve_repo_version_metadata() -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"])
    short_commit = _run_git(["rev-parse", "--short", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    exact_tag = _run_git(["describe", "--tags", "--exact-match"])
    describe = _run_git(["describe", "--tags", "--always", "--dirty"])
    status_porcelain = _run_git(["status", "--porcelain"])

    dirty = None
    if status_porcelain is not None:
        dirty = bool(status_porcelain)

    return {
        "repo_version": exact_tag or describe,
        "repo_exact_tag": exact_tag,
        "repo_describe": describe,
        "repo_commit": commit,
        "repo_commit_short": short_commit,
        "repo_branch": branch,
        "repo_dirty": dirty,
    }


def infer_prompt_tier(prompt: Prompt) -> str:
    """Infer the prompt-length tier from ``tokens_approx``."""
    tokens = prompt.tokens_approx
    if tokens is None:
        return "__unknown__"
    if tokens <= 30:
        return "short"
    if tokens <= 80:
        return "brief"
    if tokens <= 200:
        return "med"
    if tokens <= 500:
        return "long"
    return "extended"


def resolve_prompts(
    *,
    prompt_battery: str,
    prompt_ids: list[str] | None,
    prompt_tiers: list[str] | None,
    prompt_types: list[str] | None,
    smoke: bool,
    max_prompts: int | None,
) -> list[Prompt]:
    prompts = resolve_prompt_collection(
        prompt_battery,
        prompt_ids=prompt_ids,
        prompt_tiers=prompt_tiers,
        prompt_types=prompt_types,
    )
    if smoke:
        kept_counts: dict[str, int] = {}
        smoke_selected: list[Prompt] = []
        for prompt in prompts:
            tier_key = infer_prompt_tier(prompt)
            current = kept_counts.get(tier_key, 0)
            if current >= SMOKE_PROMPTS_PER_TIER:
                continue
            smoke_selected.append(prompt)
            kept_counts[tier_key] = current + 1
        prompts = smoke_selected
    if max_prompts is not None:
        if max_prompts <= 0:
            raise ValueError("--max-prompts must be a positive integer.")
        prompts = prompts[:max_prompts]
    return prompts


def log(message: str, *, verbosity: int, level: int = 1) -> None:
    if verbosity >= level:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{timestamp}] {message}", flush=True)


def build_state_filename(index: int, prompt_id: str) -> str:
    return f"{index:04d}_{slugify_path_token(prompt_id)}.pt"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def update_status(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect all-token hidden states for baseline prompt passes.",
    )
    parser.add_argument(
        "--prompt-battery",
        type=str,
        required=True,
        help="Battery directory or JSON prompt collection to collect from.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        help=(
            "Model selector to use. May be one of the registered keys "
            f"({', '.join(VALID_MODEL_KEYS)}) or a raw Hugging Face model id."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the collection run into.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: auto (default), cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Optional base directory to use in place of data/. Also supports {DATA_DIR_ENV_VAR}.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=sorted(VALID_DTYPES),
        help="Serialization dtype for saved hidden states.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=VALID_VERBOSITY,
        help="0=minimal, 1=basic, 2=verbose progress output.",
    )
    parser.add_argument(
        "--SMOKE",
        action="store_true",
        help=(
            "Shortcut pilot mode: keep up to "
            f"{SMOKE_PROMPTS_PER_TIER} prompts per prompt-length tier."
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap on the number of prompts after filtering.",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Single prompt id within --prompt-battery.",
    )
    parser.add_argument(
        "--prompt-ids",
        type=str,
        default=None,
        help="Comma-separated prompt ids within --prompt-battery.",
    )
    parser.add_argument(
        "--prompt-tiers",
        type=str,
        default=None,
        help="Comma-separated prompt tiers within --prompt-battery.",
    )
    parser.add_argument(
        "--prompt-types",
        type=str,
        default=None,
        help="Comma-separated prompt types within --prompt-battery.",
    )
    parser.add_argument(
        "--no-attn-entropy",
        action="store_true",
        help="Skip derived attention-entropy summaries during collection.",
    )
    args = parser.parse_args()

    if args.prompt_id and args.prompt_ids:
        raise ValueError("Use only one of --prompt-id or --prompt-ids.")

    configure_data_dir(args.data_dir)
    runtime_device = resolve_device(args.device)
    output_dir = render_output_path(args.output_dir)
    ensure_new_output_dir(output_dir, "sequence collection output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    prompt_ids = [args.prompt_id] if args.prompt_id else _parse_csv_strings(args.prompt_ids)
    prompt_tiers = _parse_csv_strings(args.prompt_tiers)
    prompt_types = _parse_csv_strings(args.prompt_types)
    prompts = resolve_prompts(
        prompt_battery=args.prompt_battery,
        prompt_ids=prompt_ids,
        prompt_tiers=prompt_tiers,
        prompt_types=prompt_types,
        smoke=args.SMOKE,
        max_prompts=args.max_prompts,
    )
    if not prompts:
        raise ValueError("Prompt selection resolved to zero prompts.")

    log(
        f"Loading model for sequence collection: {args.model_key}",
        verbosity=args.verbosity,
    )
    agent = Agent.from_model_key(args.model_key, runtime_device)

    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "status.json"
    records_path = output_dir / "records.jsonl"
    errors_path = output_dir / "errors.jsonl"
    save_dtype = VALID_DTYPES[args.dtype]
    capture_attention_entropy = not args.no_attn_entropy

    manifest: dict[str, Any] = {
        "run_kind": "sequence_collection",
        "model": agent.backend.model_name,
        "model_selector": args.model_key,
        "device": runtime_device,
        "config": agent.backend.config_summary,
        "prompt_battery": args.prompt_battery,
        "prompt_selection": {
            "prompt_ids": prompt_ids,
            "prompt_tiers": prompt_tiers,
            "prompt_types": prompt_types,
            "smoke": args.SMOKE,
            "max_prompts": args.max_prompts,
        },
        "num_prompts_planned": len(prompts),
        "output_dir": str(output_dir),
        "state_dir": str(states_dir),
        "state_format": "torch.save",
        "capture_type": "layer_output_hidden_states",
        "includes_embedding_layer": True,
        "includes_attention_entropy": capture_attention_entropy,
        "hidden_state_dtype": args.dtype,
        "completed_prompts": 0,
        "failed_prompts": 0,
        "created_at": iso_now(),
    }
    manifest.update(resolve_repo_version_metadata())
    write_json(manifest_path, manifest)

    status: dict[str, Any] = {
        "run_kind": "sequence_collection_status",
        "stage": "initializing",
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "model_selector": args.model_key,
        "model": agent.backend.model_name,
        "output_dir": str(output_dir),
        "status_file": str(status_path),
        "manifest_file": str(manifest_path),
        "records_file": str(records_path),
        "errors_file": str(errors_path),
        "num_prompts_planned": len(prompts),
        "completed_prompts": 0,
        "failed_prompts": 0,
        "current_prompt_index": None,
        "current_prompt_id": None,
        "last_completed_prompt_id": None,
        "last_completed_state_file": None,
        "last_error_prompt_id": None,
        "last_error": None,
    }
    update_status(status_path, status)

    completed = 0
    failed = 0

    log(
        f"Telemetry files: status={status_path.name}, manifest={manifest_path.name}, "
        f"records={records_path.name}, errors={errors_path.name}",
        verbosity=args.verbosity,
    )

    with open(records_path, "w", encoding="utf-8") as records_file, open(
        errors_path, "w", encoding="utf-8"
    ) as errors_file:
        status["stage"] = "collecting"
        status["updated_at"] = iso_now()
        update_status(status_path, status)
        for index, prompt in enumerate(prompts, start=1):
            log(
                f"[{index}/{len(prompts)}] Collecting {prompt.id}",
                verbosity=args.verbosity,
                level=1,
            )
            status["stage"] = "capturing_prompt"
            status["updated_at"] = iso_now()
            status["current_prompt_index"] = index
            status["current_prompt_id"] = prompt.id
            update_status(status_path, status)
            try:
                captured = agent.capture_sequence_states(
                    prompt.prompt_text,
                    prompt_id=prompt.id,
                    hidden_state_dtype=save_dtype,
                    capture_attention_entropy=capture_attention_entropy,
                )

                state_filename = build_state_filename(index, prompt.id)
                state_path = states_dir / state_filename
                status["stage"] = "saving_state"
                status["updated_at"] = iso_now()
                status["pending_state_file"] = f"states/{state_filename}"
                update_status(status_path, status)
                state_payload = {
                    "prompt_id": prompt.id,
                    "prompt_text": prompt.prompt_text,
                    "prompt_type": prompt.type,
                    "target": prompt.target,
                    "tokens_approx": prompt.tokens_approx,
                    "source": prompt.source,
                    "model": agent.backend.model_name,
                    "model_selector": args.model_key,
                    "input_ids": captured["input_ids"],
                    "attention_mask": captured["attention_mask"],
                    "hidden_states": captured["hidden_states"],
                    "attn_entropy_per_head_final": captured.get("attn_entropy_per_head_final"),
                    "attn_entropy_layer_indices": captured.get("attn_entropy_layer_indices"),
                    "attn_entropy_skipped_layers": captured.get("attn_entropy_skipped_layers"),
                    "final_entropy_bits": captured["final_entropy_bits"],
                    "mean_entropy_bits": captured["mean_entropy_bits"],
                    "elapsed_time": captured["elapsed_time"],
                }
                torch.save(state_payload, state_path)
                state_num_bytes = state_path.stat().st_size

                status["stage"] = "writing_record"
                status["updated_at"] = iso_now()
                update_status(status_path, status)
                record = {
                    "prompt_id": prompt.id,
                    "prompt_type": prompt.type,
                    "target": prompt.target,
                    "tokens_approx": prompt.tokens_approx,
                    "source": prompt.source,
                    "num_tokens": captured["num_tokens"],
                    "hidden_state_shape": captured["hidden_state_shape"],
                    "hidden_state_dtype": captured["hidden_state_dtype"],
                    "num_layers_plus_embedding": captured["num_layers_plus_embedding"],
                    "hidden_size": captured["hidden_size"],
                    "final_entropy_bits": captured["final_entropy_bits"],
                    "mean_entropy_bits": captured["mean_entropy_bits"],
                    "elapsed_time": captured["elapsed_time"],
                    "state_file": f"states/{state_filename}",
                    "state_num_bytes": state_num_bytes,
                }
                if capture_attention_entropy:
                    record["attn_entropy_layer_indices"] = captured.get("attn_entropy_layer_indices")

                records_file.write(json.dumps(record) + "\n")
                records_file.flush()
                completed += 1
                manifest["completed_prompts"] = completed
                manifest["failed_prompts"] = failed
                write_json(manifest_path, manifest)

                status["stage"] = "prompt_complete"
                status["updated_at"] = iso_now()
                status["completed_prompts"] = completed
                status["failed_prompts"] = failed
                status["last_completed_prompt_id"] = prompt.id
                status["last_completed_state_file"] = f"states/{state_filename}"
                status["last_error_prompt_id"] = None
                status["last_error"] = None
                status.pop("pending_state_file", None)
                update_status(status_path, status)

                log(
                    f"  saved {state_filename} | tokens={captured['num_tokens']} | "
                    f"shape={tuple(captured['hidden_state_shape'])} | "
                    f"bytes={state_num_bytes} | time={captured['elapsed_time']:.2f}s",
                    verbosity=args.verbosity,
                    level=2,
                )
            except Exception as exc:
                failed += 1
                error_record = {
                    "prompt_id": prompt.id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors_file.write(json.dumps(error_record) + "\n")
                errors_file.flush()
                manifest["completed_prompts"] = completed
                manifest["failed_prompts"] = failed
                write_json(manifest_path, manifest)
                status["stage"] = "prompt_error"
                status["updated_at"] = iso_now()
                status["completed_prompts"] = completed
                status["failed_prompts"] = failed
                status["last_error_prompt_id"] = prompt.id
                status["last_error"] = repr(exc)
                update_status(status_path, status)
                log(
                    f"  [ERROR] {prompt.id}: {exc}",
                    verbosity=args.verbosity,
                    level=1,
                )

    manifest["completed_prompts"] = completed
    manifest["failed_prompts"] = failed
    manifest["completed_at"] = iso_now()
    write_json(manifest_path, manifest)
    status["stage"] = "complete"
    status["updated_at"] = iso_now()
    status["completed_prompts"] = completed
    status["failed_prompts"] = failed
    status["current_prompt_index"] = None
    status["current_prompt_id"] = None
    status.pop("pending_state_file", None)
    update_status(status_path, status)

    log(
        f"Sequence collection complete: {completed} succeeded, {failed} failed. "
        f"Output: {output_dir}",
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    main()
