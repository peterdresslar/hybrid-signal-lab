"""ModelBackend ABC — the model-specific contract for hybrid transformer inference."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_hf_token() -> str | None:
    """Resolve a Hugging Face token from common env var names."""
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value:
            return value
    return None


def attention_scaler_hook(scale: float):
    """Forward hook that multiplies a module's output by *scale*.

    Works regardless of whether the module returns a tensor, a tuple
    (common for attention layers), or a mutable sequence.
    """

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            scaled = output[0] * scale
            return (scaled,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return output * scale
        else:
            output[0] = output[0] * scale
            return output

    return hook_fn


class ModelBackend(ABC):
    """Abstract base for model-specific hook and output logic.

    Subclasses implement layer detection, hook-target selection, and
    attention-entropy extraction. Everything else (tokenization,
    forward pass, logit processing) is handled by :class:`Agent`.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    def load(self, device: str) -> None:
        """Load model and tokenizer from the HuggingFace hub.

        Subclasses may override to customize loading kwargs.
        """
        self._device = device
        hf_token = resolve_hf_token()

        print(f"Device: {device}")
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA GPU: {gpu_name}")
        if not hf_token:
            print(
                "Warning: no HF token found in HF_TOKEN/HUGGINGFACE_HUB_TOKEN. "
                "If model download fails, set one of these env vars."
            )

        print(f"Loading {self._model_name}...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                token=hf_token,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                token=hf_token,
                dtype=torch.float16,
                device_map=device,
                attn_implementation="eager",
                output_hidden_states=True,
                output_attentions=True,
            )
        except Exception as exc:
            if not hf_token:
                raise RuntimeError(
                    "Failed to load model and no Hugging Face token was found. "
                    "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN on this machine."
                ) from exc
            raise

        self._model.eval()
        print(
            f"Model loaded: {self._model.config.num_hidden_layers} layers, "
            f"hidden_size={self._model.config.hidden_size}"
        )

    @abstractmethod
    def get_attention_layer_indices(self) -> list[int]:
        """Return the layer indices where full attention lives."""

    @abstractmethod
    def get_hook_module(self, layer_idx: int) -> torch.nn.Module:
        """Return the module to attach the scaling hook to for *layer_idx*."""

    @abstractmethod
    def process_attention_entropy(self, outputs) -> dict[str, Any]:
        """Extract per-head attention entropy from model outputs.

        Returns a dict with at least:
        - ``attn_entropy_per_head_final``: list of per-head entropy lists
        - ``attn_entropy_layer_indices``: aligned layer indices
        Optionally:
        - ``attn_entropy_skipped_layers``: diagnostics for skipped layers
        """

    @property
    def config_summary(self) -> dict[str, Any]:
        """Standard config keys for metadata / result dicts."""
        cfg = self.model.config
        return {
            key: getattr(cfg, key, None)
            for key in [
                "model_type",
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "vocab_size",
            ]
        }
