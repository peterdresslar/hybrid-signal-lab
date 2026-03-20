"""QwenBackend — Qwen3.5 hybrid (GDN + Gated Attention, 3:1 ratio)."""

from __future__ import annotations

from typing import Any

import torch

from model.backend import ModelBackend

QWEN_MODELS: dict[str, str] = {
    "0_8B": "Qwen/Qwen3.5-0.8B-Base",
    "2B": "Qwen/Qwen3.5-2B-Base",
    "4B": "Qwen/Qwen3.5-4B-Base",
    "9B": "Qwen/Qwen3.5-9B-Base",
    "27B": "Qwen/Qwen3.5-27B-Base",
    "35B": "Qwen/Qwen3.5-35B-A3B-Base",
}


class QwenBackend(ModelBackend):
    """Model backend for the Qwen3.5 hybrid family.

    Attention layers appear at every 4th position (layers 3, 7, 11, ...).
    Hooks are registered on the full decoder layer so the scaling
    propagates through the residual stream.
    """

    def get_attention_layer_indices(self) -> list[int]:
        n_layers = len(self.model.model.layers)
        return [i for i in range(n_layers) if (i + 1) % 4 == 0]

    def get_hook_module(self, layer_idx: int) -> torch.nn.Module:
        return self.model.model.layers[layer_idx]

    def process_attention_entropy(self, outputs) -> dict[str, Any]:
        result: dict[str, Any] = {
            "attn_entropy_per_head_final": [],
            "attn_entropy_layer_indices": [],
        }
        if not hasattr(outputs, "attentions") or outputs.attentions is None:
            return result

        attn_layers = self.get_attention_layer_indices()
        for layer_pos, attn in enumerate(outputs.attentions):
            if attn is None or attn.ndim != 4:
                continue
            last_token_attn = attn[0, :, -1, :].float()
            ent = -(last_token_attn * torch.log2(last_token_attn + 1e-10)).sum(dim=-1)
            result["attn_entropy_per_head_final"].append(ent.tolist())
            if layer_pos < len(attn_layers):
                result["attn_entropy_layer_indices"].append(attn_layers[layer_pos])

        return result
