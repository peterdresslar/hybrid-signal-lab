"""TransformerBackend — generic full-transformer backend for control runs."""

from __future__ import annotations

from typing import Any

import torch

from model.backend import InterventionMode, ModelBackend

TRANSFORMER_MODELS: dict[str, str] = {
    "Q3_30B": "Qwen/Qwen3-30B-A3B-Base",
    "Q3_8B": "Qwen/Qwen3-8B-Base",
    "OLMO_3": "allenai/Olmo-3-1025-7B",
}


class TransformerBackend(ModelBackend):
    """Generic backend for transformer-only causal LMs.

    This backend serves full-softmax transformer families loaded via raw HF ids.
    The ``block_output`` mode always hooks the full decoder layer.

    For ``attention_contribution``, the hook target is architecture-aware:
    pre-norm stacks such as Qwen3 expose the attention contribution directly at
    ``self_attn``, while post-attention-norm stacks such as OLMo3 require the
    hook to sit on ``post_attention_layernorm`` so the scaled value matches the
    quantity that actually enters the residual stream.
    """

    def get_attention_layer_indices(self) -> list[int]:
        return [
            idx
            for idx, layer in enumerate(self.get_decoder_layers())
            if self.get_layer_attention_module(layer) is not None
        ]

    @property
    def default_intervention_mode(self) -> InterventionMode:
        return InterventionMode.ATTENTION_CONTRIBUTION

    def _get_attention_contribution_module(
        self,
        layer_idx: int,
        layer: Any,
    ) -> torch.nn.Module:
        model_type = getattr(self.model.config, "model_type", None)

        # OLMo3 is post-attention-norm:
        # x -> self_attn(x) -> post_attention_layernorm(.) -> x + attn_branch
        if model_type == "olmo3":
            module = getattr(layer, "post_attention_layernorm", None)
            if isinstance(module, torch.nn.Module):
                return module
            raise ValueError(
                f"Layer {layer_idx} in model '{self.model_name}' has model_type='olmo3' "
                "but no post_attention_layernorm module."
            )

        # Qwen3 and most common full-softmax decoder families are pre-norm:
        # x -> input_layernorm(x) -> self_attn(.) -> x + attn_branch
        module = self.get_layer_attention_module(layer)
        if module is None:
            raise ValueError(
                f"Layer {layer_idx} in model '{self.model_name}' has no recognized attention module."
            )

        if model_type == "qwen3":
            return module

        # Conservative fallback: when a layer has a post-attention norm but no
        # input-layer norm, the norm sits between attention and the residual add.
        # In that case the residual-entry quantity is the normalized branch.
        post_attn_norm = getattr(layer, "post_attention_layernorm", None)
        has_post_attn_norm = isinstance(post_attn_norm, torch.nn.Module)
        has_input_norm = isinstance(getattr(layer, "input_layernorm", None), torch.nn.Module)
        if has_post_attn_norm and not has_input_norm:
            return post_attn_norm

        return module

    def get_hook_module(
        self,
        layer_idx: int,
        mode: InterventionMode,
    ) -> torch.nn.Module:
        layer = self.get_decoder_layer(layer_idx)
        if mode == InterventionMode.BLOCK_OUTPUT:
            return layer
        return self._get_attention_contribution_module(layer_idx, layer)

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
