"""Signal Lab — CLI entry point and environment / prompt utilities."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import dotenv

from colony.model.prompt import Prompt
from colony.model.g_profile import (
    VALID_G_FUNCTIONS,
    build_attention_scales_from_spec,
)
from colony.model import VALID_MODEL_KEYS
from colony.agent import Agent

dotenv.load_dotenv(".env.development")
dotenv.load_dotenv(".env")

DATA_DIR = Path("data")


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
    path = Path(path_or_name)
    if path.is_file():
        return path
    data_path = DATA_DIR / path_or_name
    if data_path.is_file():
        return data_path
    return path


def _load_prompt_entries(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, list):
        raise ValueError(f"Prompt catalog must be a list: {json_path}")
    return data


def _prompt_from_catalog(json_path: Path, prompt_id: str) -> Prompt:
    entries = _load_prompt_entries(json_path)
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == prompt_id:
            return Prompt.from_dict(entry, data_dir=json_path.parent, source=str(json_path))
    raise ValueError(f"Prompt id '{prompt_id}' not found in {json_path}")


def _all_prompt_catalogs() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(DATA_DIR.glob("prompts*.json"))


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
        if maybe_catalog.is_file() and maybe_catalog.suffix == ".json":
            return _prompt_from_catalog(maybe_catalog, right)

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


def run_model(
    prompt_source: str,
    model_key: str = "0_8B",
    device: str | None = None,
    g_spec: dict[str, Any] | None = None,
):
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

    out_path = "signal_lab_output.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {out_path}")
    print(f"Elapsed time: {summary['elapsed_time']:.3f}s")
    print(
        "Top prediction "
        f"(g_function={resolved_g_spec.get('g_function')}, "
        f"scales={summary['g_attention_scales']}): "
        f"index {summary['top_k_indices'][0]} logit {summary['top_k_logits'][0]:.3f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal Lab: exploring model internals via transformers.",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="The color with the shortest wavelength is",
        help="Path or filename to a prompt file in data/, or a direct string prompt.",
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
    args = parser.parse_args()

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

    run_model(args.prompt, args.model_key, device=args.device, g_spec=g_spec)
