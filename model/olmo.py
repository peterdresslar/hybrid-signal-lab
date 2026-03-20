"""OlmoBackend — OLMo-Hybrid (Linear Attention + Full Attention)."""

from __future__ import annotations

from typing import Any

import torch

from model.backend import ModelBackend

OLMO_MODELS: dict[str, str] = {
    "OLMO": "allenai/Olmo-Hybrid-7B",
}


class OlmoBackend(ModelBackend):
    """Model backend for OLMo-Hybrid.

    Full-attention layers are identified at runtime by the presence of a
    ``self_attn`` submodule (linear-attention blocks use ``linear_attn``
    instead). Hooks target the full decoder layer — not ``self_attn``
    directly — because the post-attention RMSNorm would absorb
    sub-module-level scaling.
    """

    def get_attention_layer_indices(self) -> list[int]:
        return [
            i for i, layer in enumerate(self.model.model.layers) if hasattr(layer, "self_attn")
        ]

    def get_hook_module(self, layer_idx: int) -> torch.nn.Module:
        return self.model.model.layers[layer_idx]

    def process_attention_entropy(self, outputs) -> dict[str, Any]:
        result: dict[str, Any] = {
            "attn_entropy_per_head_final": [],
            "attn_entropy_layer_indices": [],
        }
        skipped: list[dict[str, Any]] = []

        if not hasattr(outputs, "attentions") or outputs.attentions is None:
            return result

        attn_layers = self.get_attention_layer_indices()

        for layer_pos, attn in enumerate(outputs.attentions):
            if attn is None:
                skipped.append({"layer_offset": layer_pos, "reason": "none_attention_tensor"})
                continue

            if attn.ndim != 4:
                skipped.append(
                    {
                        "layer_offset": layer_pos,
                        "reason": "unexpected_rank",
                        "shape": list(attn.shape),
                    }
                )
                continue

            source_len = int(attn.shape[-1])
            if source_len <= 1:
                skipped.append(
                    {
                        "layer_offset": layer_pos,
                        "reason": "degenerate_source_len",
                        "shape": list(attn.shape),
                    }
                )
                continue

            last_token_attn = attn[0, :, -1, :].float()
            ent = -(last_token_attn * torch.log2(last_token_attn + 1e-10)).sum(dim=-1)
            result["attn_entropy_per_head_final"].append(ent.tolist())
            if layer_pos < len(attn_layers):
                result["attn_entropy_layer_indices"].append(attn_layers[layer_pos])

        if skipped:
            result["attn_entropy_skipped_layers"] = skipped

        return result
