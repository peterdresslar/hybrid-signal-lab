"""TransformerBackend — generic full-transformer backend for control runs."""

from __future__ import annotations

from typing import Any

import torch

from model.backend import ModelBackend


class TransformerBackend(ModelBackend):
    """Generic backend for transformer-only causal LMs.

    Unlike the hybrid backends, this hooks each layer's attention submodule
    directly so the gain profile scales the attention branch rather than the
    whole decoder block.
    """

    def get_attention_layer_indices(self) -> list[int]:
        return [
            idx
            for idx, layer in enumerate(self.get_decoder_layers())
            if self.get_layer_attention_module(layer) is not None
        ]

    def get_hook_module(self, layer_idx: int) -> torch.nn.Module:
        layers = self.get_decoder_layers()
        try:
            layer = layers[layer_idx]
        except IndexError as exc:
            raise IndexError(
                f"Attention layer index {layer_idx} is out of range for model '{self.model_name}'."
            ) from exc

        module = self.get_layer_attention_module(layer)
        if module is None:
            raise ValueError(
                f"Layer {layer_idx} in model '{self.model_name}' has no recognized attention module."
            )
        return module

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
