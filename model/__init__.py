"""model — model backends and shared data structures."""

from model.backend import ModelBackend
from model.qwen import QwenBackend, QWEN_MODELS
from model.olmo import OlmoBackend, OLMO_MODELS
from model.transformer import TRANSFORMER_MODELS, TransformerBackend

ALL_MODELS: dict[str, str] = {**QWEN_MODELS, **OLMO_MODELS, **TRANSFORMER_MODELS}

VALID_MODEL_KEYS = sorted(ALL_MODELS.keys())


def create_backend(model_key: str) -> ModelBackend:
    """Create the appropriate backend for a model key."""
    if model_key in QWEN_MODELS:
        return QwenBackend(QWEN_MODELS[model_key])
    if model_key in OLMO_MODELS:
        return OlmoBackend(OLMO_MODELS[model_key])
    if model_key in TRANSFORMER_MODELS:
        return TransformerBackend(TRANSFORMER_MODELS[model_key])
    if "/" in model_key:
        return TransformerBackend(model_key)
    available = ", ".join(VALID_MODEL_KEYS)
    raise ValueError(
        f"Unknown model key '{model_key}'. Available registered keys: {available}. "
        "You can also pass a raw Hugging Face model id like 'Qwen/Qwen2.5-0.5B'."
    )
