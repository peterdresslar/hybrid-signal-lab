"""Shared path policy for Signal Lab inputs and generated outputs."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
import re


DATA_DIR_ENV_VAR = "DATA_DIR"
DEFAULT_SIGNAL_LAB_INPUT_DATA_DIR = Path("data")
SIGNAL_LAB_INPUT_DATA_DIR = DEFAULT_SIGNAL_LAB_INPUT_DATA_DIR
SIGNAL_LAB_OUTPUTS_DIR = DEFAULT_SIGNAL_LAB_INPUT_DATA_DIR / "outputs" / "signal_lab"

DEFAULT_ANALYSIS_DIRNAME = "analysis"
DEFAULT_PLOTS_DIRNAME = "plots"
DEFAULT_COMPARISONS_DIRNAME = "_comparisons"

_configured_data_dir: Path | None = None


def configure_data_dir(path_or_none: str | Path | None) -> Path:
    global _configured_data_dir
    if path_or_none is None:
        _configured_data_dir = None
        return get_data_dir()

    resolved = Path(path_or_none).expanduser()
    _configured_data_dir = resolved
    return resolved


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_data_dir() -> Path:
    if _configured_data_dir is not None:
        return _configured_data_dir
    env_value = os.getenv(DATA_DIR_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_SIGNAL_LAB_INPUT_DATA_DIR


def get_outputs_dir() -> Path:
    return get_data_dir() / "outputs" / "signal_lab"


def slugify_path_token(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "artifact"


def validate_path_token(text: str, label: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{label} must not be empty.")
    token_path = Path(stripped)
    if len(token_path.parts) != 1 or stripped in {".", ".."}:
        raise ValueError(f"{label} must be a single path token, not a nested path: {text!r}")
    return stripped


def resolve_input_path(path_or_name: str) -> Path:
    path = Path(path_or_name).expanduser()
    if path.exists():
        return path
    data_path = get_data_dir() / path_or_name
    if data_path.exists():
        return data_path
    return path


def all_prompt_catalog_glob() -> Path:
    return get_data_dir()


def default_probe_output_path() -> Path:
    return get_outputs_dir() / "probes" / f"signal_lab_output_{timestamp_tag()}.json"


def default_sweep_out_dir(run_name: str, model_key: str) -> Path:
    run_token = validate_path_token(run_name, "run_name")
    return get_outputs_dir() / "runs" / run_token / validate_path_token(model_key, "model_key")


def render_output_path(path_pattern: str) -> Path:
    rendered = path_pattern.replace("{timestamp}", timestamp_tag())
    return Path(rendered).expanduser()


def default_analysis_output_dir(run_dir: Path) -> Path:
    return run_dir / DEFAULT_ANALYSIS_DIRNAME


def artifact_stem(path: Path) -> str:
    if path.name == DEFAULT_ANALYSIS_DIRNAME and path.parent.name:
        return slugify_path_token(path.parent.name)

    if path.parent.name:
        if path.parent.parent.name == "runs":
            return slugify_path_token(path.name)
    return slugify_path_token(path.name)


def run_collection_dir(path: Path) -> Path | None:
    if path.name == DEFAULT_ANALYSIS_DIRNAME:
        model_dir = path.parent
        if model_dir.parent.parent.name == "runs":
            return model_dir.parent
        return None

    if path.parent.parent.name == "runs":
        return path.parent
    return None


def default_compare_output_dir(run_a_dir: Path, run_b_dir: Path) -> Path:
    run_collection_a = run_collection_dir(run_a_dir)
    run_collection_b = run_collection_dir(run_b_dir)
    if run_collection_a is None or run_collection_b is None:
        raise ValueError("Could not determine enclosing run_name directory for comparison output.")
    if run_collection_a != run_collection_b:
        raise ValueError(
            "Default comparison output requires both runs to belong to the same <run_name> directory."
        )

    run_a_name = artifact_stem(run_a_dir)
    run_b_name = artifact_stem(run_b_dir)
    compare_name = f"{run_a_name}_vs_{run_b_name}"
    return run_collection_a / DEFAULT_COMPARISONS_DIRNAME / compare_name


def ensure_new_output_dir(path: Path, label: str = "output directory") -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError(f"{label} exists and is not a directory: {path}")
    try:
        next(path.iterdir())
    except StopIteration:
        return
    raise FileExistsError(f"{label} already exists and is not empty: {path}")


def ensure_output_file_available(path: Path, label: str = "output file") -> None:
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")
