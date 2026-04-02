"""OlmoBackend — OLMo-Hybrid (Linear Attention + Full Attention)."""

from __future__ import annotations

from typing import Any

import torch

from model.backend import InterventionMode, ModelBackend

OLMO_MODELS: dict[str, str] = {
    "OLMO": "allenai/Olmo-Hybrid-7B",
}


class OlmoBackend(ModelBackend):
    """Model backend for OLMo-Hybrid.

    Full-attention layers are identified at runtime by the presence of a
    ``self_attn`` submodule (linear-attention blocks use ``linear_attn``
    instead). The softmax layers are post-norm: the attention branch is
    ``N_att(Attn(x))`` before the residual add. Since ``RMSNorm(g * x) =
    RMSNorm(x)``, attention-only scaling must hook ``post_attention_layernorm``
    so the intervention lands on ``a_tilde = N_att(Attn(x))``.
    """

    def get_attention_layer_indices(self) -> list[int]:
        return [i for i, layer in enumerate(self.get_decoder_layers()) if hasattr(layer, "self_attn")]

    @property
    def default_intervention_mode(self) -> InterventionMode:
        return InterventionMode.BLOCK_OUTPUT

    def get_hook_module(
        self,
        layer_idx: int,
        mode: InterventionMode,
    ) -> torch.nn.Module:
        layer = self.get_decoder_layer(layer_idx)
        if mode == InterventionMode.BLOCK_OUTPUT:
            return layer

        module = getattr(layer, "post_attention_layernorm", None)
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                f"Layer {layer_idx} in model '{self.model_name}' has no post_attention_layernorm module."
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
