"""model — model backends and shared data structures."""

from model.backend import ModelBackend
from model.qwen import QwenBackend, QWEN_MODELS
from model.olmo import OlmoBackend, OLMO_MODELS

ALL_MODELS: dict[str, str] = {**QWEN_MODELS, **OLMO_MODELS}

VALID_MODEL_KEYS = sorted(ALL_MODELS.keys())


def create_backend(model_key: str) -> ModelBackend:
    """Create the appropriate backend for a model key."""
    if model_key in QWEN_MODELS:
        return QwenBackend(QWEN_MODELS[model_key])
    if model_key in OLMO_MODELS:
        return OlmoBackend(OLMO_MODELS[model_key])
    available = ", ".join(VALID_MODEL_KEYS)
    raise ValueError(f"Unknown model key '{model_key}'. Available: {available}")
