"""Signal Lab — CLI entry point and environment / prompt utilities."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import dotenv

from model.prompt import Prompt
from model.g_profile import (
    VALID_G_FUNCTIONS,
    build_attention_scales_from_spec,
)
from model import VALID_MODEL_KEYS
from signal_lab.agent import Agent
from signal_lab.paths import (
    DATA_DIR_ENV_VAR,
    configure_data_dir,
    default_probe_output_path,
    ensure_output_file_available,
    get_data_dir,
    resolve_input_path,
)

dotenv.load_dotenv(".env.development")
dotenv.load_dotenv(".env")

# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(requested_device: str | None = None) -> str:
    """Resolve runtime device.

    Priority: explicit argument > COLONY_DEVICE env var > auto-detect.
    """
    env_device = os.getenv("COLONY_DEVICE")
    raw_value = (requested_device or env_device or "auto").strip().lower()

    if raw_value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    valid = {"cuda", "mps", "cpu"}
    if raw_value not in valid:
        valid_str = ", ".join(sorted(valid | {"auto"}))
        raise ValueError(f"Invalid device '{raw_value}'. Expected one of: {valid_str}.")

    if raw_value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' requested, but CUDA is not available in this environment.")
    if raw_value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' requested, but MPS is not available in this environment.")
    return raw_value


# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------

def _resolve_path(path_or_name: str) -> Path:
    return resolve_input_path(path_or_name)


def _load_prompt_entries(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, list):
        raise ValueError(f"Prompt catalog must be a list: {json_path}")
    return data


def _resolve_prompt_collection_paths(path_or_name: str | Path) -> list[Path]:
    """Resolve a prompt collection to one or more JSON files."""
    resolved = _resolve_path(str(path_or_name))
    if resolved.is_file():
        if resolved.suffix != ".json":
            raise ValueError(f"Prompt collection must be JSON: {resolved}")
        return [resolved]

    if resolved.is_dir():
        combined_path = resolved / "all_candidates.json"
        if combined_path.is_file():
            return [combined_path]

        manifest_path = resolved / "manifest.json"
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as file_handle:
                manifest = json.load(file_handle)
            manifest_types = manifest.get("types", {}) if isinstance(manifest, dict) else {}
            paths: list[Path] = []
            if isinstance(manifest_types, dict):
                for spec in manifest_types.values():
                    if not isinstance(spec, dict):
                        continue
                    rel_file = spec.get("file")
                    if not isinstance(rel_file, str):
                        continue
                    candidate_path = resolved / rel_file
                    if candidate_path.is_file():
                        paths.append(candidate_path)
            if paths:
                return paths

        json_paths = [
            path for path in sorted(resolved.glob("*.json"))
            if path.name != "manifest.json"
        ]
        if json_paths:
            return json_paths

    raise FileNotFoundError(f"Could not resolve prompt collection: {path_or_name}")


def _prompt_from_catalog(json_path: Path, prompt_id: str) -> Prompt:
    entries = _load_prompt_entries(json_path)
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == prompt_id:
            return Prompt.from_dict(entry, data_dir=json_path.parent, source=str(json_path))
    raise ValueError(f"Prompt id '{prompt_id}' not found in {json_path}")


def _prompt_from_collection(path_or_name: str | Path, prompt_id: str) -> Prompt:
    """Resolve a prompt id from a JSON prompt collection or battery directory."""
    collection_paths = _resolve_prompt_collection_paths(path_or_name)
    matches: list[Prompt] = []
    for json_path in collection_paths:
        for entry in _load_prompt_entries(json_path):
            if isinstance(entry, dict) and entry.get("id") == prompt_id:
                matches.append(
                    Prompt.from_dict(entry, data_dir=json_path.parent, source=str(json_path))
                )

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        sources = ", ".join(sorted({match.source for match in matches if match.source}))
        raise ValueError(
            f"Prompt id '{prompt_id}' is ambiguous within collection {path_or_name}: {sources}"
        )
    raise ValueError(f"Prompt id '{prompt_id}' not found in collection {path_or_name}")


def _all_prompt_catalogs() -> list[Path]:
    data_dir = get_data_dir()
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("prompts*.json"))


def resolve_prompt_collection(
    collection_arg: str,
    *,
    prompt_ids: list[str] | None = None,
    prompt_tiers: list[str] | None = None,
    prompt_types: list[str] | None = None,
) -> list[Prompt]:
    """Resolve a prompt collection or battery bundle into Prompt objects."""
    collection_paths = _resolve_prompt_collection_paths(collection_arg)
    records: list[tuple[dict[str, Any], Path]] = []
    for json_path in collection_paths:
        for entry in _load_prompt_entries(json_path):
            if not isinstance(entry, dict):
                continue
            if "prompt" not in entry and "prompt_file" not in entry:
                continue
            records.append((entry, json_path))

    if prompt_tiers is not None:
        allowed_tiers = set(prompt_tiers)
        records = [
            (entry, json_path)
            for entry, json_path in records
            if entry.get("tier") in allowed_tiers
        ]

    if prompt_types is not None:
        allowed_types = set(prompt_types)
        records = [
            (entry, json_path)
            for entry, json_path in records
            if entry.get("type") in allowed_types
        ]

    deduped_by_id: dict[str, Prompt] = {}
    for entry, json_path in records:
        prompt = Prompt.from_dict(entry, data_dir=json_path.parent, source=str(json_path))
        deduped_by_id.setdefault(prompt.id, prompt)

    selected = list(deduped_by_id.values())

    if prompt_ids is not None:
        by_id = {prompt.id: prompt for prompt in selected}
        missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_id]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Prompt ids not found in collection {collection_arg}: {missing_str}"
            )
        selected = [by_id[prompt_id] for prompt_id in prompt_ids]

    return selected


def resolve_prompt(prompt_arg: str) -> Prompt:
    """Resolve a prompt argument into a Prompt object.

    Supported forms:
    - direct string prompt
    - text file path or file name in ``data/``
    - prompt id from any ``data/prompts*.json`` catalog
    - explicit catalog selector ``prompts_short.json:short0``
    """
    if ":" in prompt_arg:
        left, right = prompt_arg.split(":", 1)
        maybe_catalog = _resolve_path(left)
        if (maybe_catalog.is_file() and maybe_catalog.suffix == ".json") or maybe_catalog.is_dir():
            return _prompt_from_collection(maybe_catalog, right)

    resolved_path = _resolve_path(prompt_arg)
    if resolved_path.is_file():
        if resolved_path.suffix == ".json":
            entries = _load_prompt_entries(resolved_path)
            if len(entries) == 1:
                entry = entries[0]
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid prompt entry in {resolved_path}")
                return Prompt.from_dict(entry, data_dir=resolved_path.parent, source=str(resolved_path))
            raise ValueError(
                f"Prompt catalog {resolved_path} contains multiple prompts. "
                "Use '<catalog>:<id>' (for example 'prompts_short.json:short0')."
            )
        with open(resolved_path, "r", encoding="utf-8") as file_handle:
            return Prompt(
                id=resolved_path.name,
                prompt_text=file_handle.read().strip(),
                prompt_file=str(resolved_path),
                source=str(resolved_path),
            )

    prompt_matches = []
    for catalog_path in _all_prompt_catalogs():
        for entry in _load_prompt_entries(catalog_path):
            if isinstance(entry, dict) and entry.get("id") == prompt_arg:
                prompt_matches.append(
                    Prompt.from_dict(entry, data_dir=catalog_path.parent, source=str(catalog_path))
                )

    if len(prompt_matches) == 1:
        return prompt_matches[0]
    if len(prompt_matches) > 1:
        sources = ", ".join(sorted({match.source for match in prompt_matches if match.source}))
        raise ValueError(
            f"Prompt id '{prompt_arg}' is ambiguous across catalogs: {sources}. "
            "Use '<catalog>:<id>' to disambiguate."
        )

    return Prompt.from_text(prompt_arg, prompt_id="direct_prompt")


def read_prompt(prompt_arg: str) -> str:
    """Backwards-compatible helper that returns only the prompt text."""
    return resolve_prompt(prompt_arg).prompt_text


# ---------------------------------------------------------------------------
# Top-level convenience runner
# ---------------------------------------------------------------------------

def _parse_csv_floats(raw_value: str | None) -> list[float] | None:
    if raw_value is None:
        return None
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        return None
    return [float(piece) for piece in values]


def _parse_csv_strings(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    return values or None


def run_model(
    prompt_source: str,
    model_key: str = "0_8B",
    device: str | None = None,
    g_spec: dict[str, Any] | None = None,
    output_path: str | None = None,
) -> Path:
    """One-shot convenience runner used by the CLI."""
    runtime_device = resolve_device(device)
    agent = Agent.from_model_key(model_key, runtime_device)

    attn_layers = agent.get_attention_layer_indices()
    resolved_g_spec = g_spec or {"g_function": "constant", "g_params": {"value": 1.0}}
    g_scales = build_attention_scales_from_spec(resolved_g_spec, attention_slots=len(attn_layers))

    prompt = resolve_prompt(prompt_source)

    summary = agent.run_pass(
        prompt.prompt_text,
        g_scales,
        prompt_id=prompt.id,
        return_verbose=True,
    )

    summary["model"] = agent.backend.model_name
    summary["device"] = runtime_device
    summary["config"] = agent.backend.config_summary
    summary["g_spec"] = resolved_g_spec

    resolved_output_path = Path(output_path).expanduser() if output_path else default_probe_output_path()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_output_file_available(resolved_output_path, "summary output")
    with open(resolved_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {resolved_output_path}")
    print(f"Elapsed time: {summary['elapsed_time']:.3f}s")
    print(
        "Top prediction "
        f"(g_function={resolved_g_spec.get('g_function')}, "
        f"scales={summary['g_attention_scales']}): "
        f"index {summary['top_k_indices'][0]} logit {summary['top_k_logits'][0]:.3f}"
    )
    return resolved_output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal Lab: exploring model internals via transformers.",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="The color with the shortest wavelength is",
        help="Path or filename to a prompt file in [DATA_DIR], or a direct string prompt.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help=f"Optional base directory to use in place of data/. Also supports {DATA_DIR_ENV_VAR}.",
    )
    parser.add_argument(
        "--prompt-battery", type=str, default=None,
        help="Battery directory or JSON file for prompt lookup.",
    )
    parser.add_argument(
        "--prompt-id", type=str, default=None,
        help="Prompt id within --prompt-battery.",
    )
    parser.add_argument(
        "--prompt-ids", type=str, default=None,
        help="Comma-separated prompt ids within --prompt-battery. Requires exactly one resolved prompt.",
    )
    parser.add_argument(
        "--prompt-tier", type=str, default=None,
        help="Single tier filter within --prompt-battery.",
    )
    parser.add_argument(
        "--prompt-type", type=str, default=None,
        help="Single type filter within --prompt-battery.",
    )
    parser.add_argument(
        "--model-key", type=str, default="0_8B",
        help=f"Model to use. One of: {', '.join(VALID_MODEL_KEYS)}.",
    )
    parser.add_argument(
        "--g-function", type=str, default="constant",
        choices=sorted(VALID_G_FUNCTIONS),
        help="g profile function family.",
    )
    parser.add_argument(
        "--g", type=float, default=1.0,
        help="Constant value shortcut when --g-function=constant.",
    )
    parser.add_argument(
        "--g-vector", type=str, default=None,
        help="Comma-separated control-point values for --g-function=control_points.",
    )
    parser.add_argument(
        "--g-params-json", type=str, default=None,
        help="JSON object with extra g function params.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: auto (default), cuda, mps, or cpu. Also supports COLONY_DEVICE env var.",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Optional path for the JSON summary output. Defaults under data/outputs/signal_lab/probes/.",
    )
    args = parser.parse_args()
    configure_data_dir(args.data_dir)

    g_params: dict[str, Any] = json.loads(args.g_params_json) if args.g_params_json else {}
    if not isinstance(g_params, dict):
        raise ValueError("--g-params-json must decode to a JSON object.")

    if args.g_function == "constant" and "value" not in g_params:
        g_params["value"] = args.g

    g_spec: dict[str, Any] = {
        "g_function": args.g_function,
        "g_params": g_params,
    }
    g_vector = _parse_csv_floats(args.g_vector)
    if g_vector is not None:
        g_spec["g_vector"] = g_vector

    prompt_source = args.prompt
    if args.prompt_battery is not None:
        if args.prompt and args.prompt != parser.get_default("prompt"):
            raise ValueError("Use either --prompt or --prompt-battery-based selection, not both.")

        prompt_ids = None
        if args.prompt_id and args.prompt_ids:
            raise ValueError("Use only one of --prompt-id or --prompt-ids.")
        if args.prompt_id:
            prompt_ids = [args.prompt_id]
        elif args.prompt_ids:
            prompt_ids = _parse_csv_strings(args.prompt_ids)

        prompts = resolve_prompt_collection(
            args.prompt_battery,
            prompt_ids=prompt_ids,
            prompt_tiers=[args.prompt_tier] if args.prompt_tier else None,
            prompt_types=[args.prompt_type] if args.prompt_type else None,
        )
        if len(prompts) != 1:
            raise ValueError(
                f"--prompt-battery selection resolved to {len(prompts)} prompts; "
                "refine with --prompt-id, --prompt-tier, or --prompt-type."
            )
        prompt_source = f"{args.prompt_battery}:{prompts[0].id}"

    run_model(
        prompt_source,
        args.model_key,
        device=args.device,
        g_spec=g_spec,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
