"""Agent — shared inference pipeline over a ModelBackend."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model.backend import ModelBackend, attention_scaler_hook
from model.g_profile import printable_scales
from model import create_backend


class Agent:
    """Runs forward passes with gain-scaled attention hooks.

    Owns the shared inference logic (tokenization, hook lifecycle,
    logit/entropy computation, result-dict assembly).  Model-specific
    behaviour is delegated to :attr:`backend`.
    """

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend

    @classmethod
    def from_model_key(cls, model_key: str, device: str) -> "Agent":
        """Create an Agent with the right backend, loaded and ready."""
        backend = create_backend(model_key)
        backend.load(device)
        return cls(backend)

    @property
    def model(self):
        return self.backend.model

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    @property
    def device(self) -> str:
        return self.backend.device

    def get_attention_layer_indices(self) -> list[int]:
        return self.backend.get_attention_layer_indices()

    def _resolve_target_attention_layers(
        self,
        target_attention_layer_indices: list[int] | None,
    ) -> list[int]:
        return (
            list(target_attention_layer_indices)
            if target_attention_layer_indices is not None
            else self.get_attention_layer_indices()
        )

    def run_pass(
        self,
        prompt: str,
        g_attention_scales: np.ndarray | list[float],
        *,
        prompt_id: str | None = None,
        target_token_id: int | None = None,
        baseline_logits: torch.Tensor | None = None,
        return_raw_logits: bool = False,
        return_verbose: bool = False,
        target_attention_layer_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run a single forward pass with gain-scaled attention hooks."""
        start_time = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        attn_layers = self._resolve_target_attention_layers(target_attention_layer_indices)
        scales = np.asarray(g_attention_scales, dtype=float)
        if scales.ndim != 1 or scales.size != len(attn_layers):
            raise ValueError(
                f"g_attention_scales length ({scales.size}) must equal "
                f"attention layer count ({len(attn_layers)})."
            )

        hooks = []
        for idx, scale in zip(attn_layers, scales.tolist()):
            module = self.backend.get_hook_module(idx)
            handle = module.register_forward_hook(attention_scaler_hook(scale))
            hooks.append(handle)

        with torch.no_grad():
            outputs = self.model(
                **inputs, output_hidden_states=True, output_attentions=True,
            )

        for h in hooks:
            h.remove()

        elapsed_time = time.perf_counter() - start_time

        final_logits = outputs.logits[0, -1, :].float()
        probs = torch.softmax(final_logits, dim=-1)

        top_logits = top_indices = top_tokens = None
        if return_verbose:
            top_logits, top_indices = torch.topk(final_logits, 15)
            top_tokens = [self.tokenizer.decode(idx) for idx in top_indices]

        target_token = target_rank = target_prob = None
        if target_token_id is not None:
            target_token = self.tokenizer.decode(target_token_id)
            target_prob = probs[target_token_id].item()
            target_rank = (probs > probs[target_token_id]).sum().item() + 1

        final_entropy_bits = -(probs * torch.log2(probs + 1e-10)).sum().item()

        mean_entropy_bits = None
        if return_verbose:
            all_logits = outputs.logits[0, :, :].float()
            all_probs = torch.softmax(all_logits, dim=-1)
            mean_entropy_bits = (
                -(all_probs * torch.log2(all_probs + 1e-10)).sum(dim=-1).mean().item()
            )

        kl_from_baseline = None
        if baseline_logits is not None:
            bl = baseline_logits
            if not isinstance(bl, torch.Tensor):
                bl = torch.tensor(bl).to(self.device)
            baseline_probs = torch.softmax(bl.float(), dim=-1)
            kl_from_baseline = F.kl_div(
                torch.log(probs + 1e-10),
                baseline_probs,
                reduction="sum",
            ).item()

        result: dict[str, Any] = {
            "prompt_id": prompt_id if prompt_id else (prompt[:20] + "..."),
            "model_type": getattr(self.model.config, "model_type", None),
            "g_attention_scales": printable_scales(scales),
            "attention_layer_indices": attn_layers,
            "target_token": target_token,
            "target_rank": target_rank,
            "target_prob": target_prob,
            "final_entropy_bits": final_entropy_bits,
            "kl_from_baseline": kl_from_baseline,
            "elapsed_time": elapsed_time,
        }

        if return_verbose:
            result["top_k_logits"] = top_logits.tolist()
            result["top_k_indices"] = top_indices.tolist()
            result["top_k_tokens"] = top_tokens
            result["mean_entropy_bits"] = mean_entropy_bits
            entropy_info = self.backend.process_attention_entropy(outputs)
            result.update(entropy_info)

        if return_raw_logits:
            result["_raw_logits"] = final_logits

        return result

    def score_target(
        self,
        prompt: str,
        target_text: str,
        g_attention_scales: np.ndarray | list[float],
        *,
        target_attention_layer_indices: list[int] | None = None,
    ) -> dict[str, Any] | None:
        """Score an entire target continuation token-by-token (teacher forcing)."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            return None
        if not prompt_ids:
            raise ValueError("Target sequence scoring requires a non-empty prompt.")

        all_ids = prompt_ids + target_ids
        input_ids = torch.tensor([all_ids], dtype=torch.long, device=self.device)

        attn_layers = self._resolve_target_attention_layers(target_attention_layer_indices)
        scales = np.asarray(g_attention_scales, dtype=float)
        if scales.ndim != 1 or scales.size != len(attn_layers):
            raise ValueError(
                f"g_attention_scales length ({scales.size}) must equal "
                f"attention layer count ({len(attn_layers)})."
            )

        hooks = []
        for idx, scale in zip(attn_layers, scales.tolist()):
            module = self.backend.get_hook_module(idx)
            handle = module.register_forward_hook(attention_scaler_hook(scale))
            hooks.append(handle)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        for handle in hooks:
            handle.remove()

        logits = outputs.logits[0].float()
        prompt_len = len(prompt_ids)

        token_logprobs: list[float] = []
        token_probs: list[float] = []
        token_ranks: list[int] = []
        for offset, token_id in enumerate(target_ids):
            logit_pos = prompt_len + offset - 1
            token_logits = logits[logit_pos]
            token_log_probs = F.log_softmax(token_logits, dim=-1)
            lp = token_log_probs[token_id]
            prob = torch.exp(lp)
            rank = int((token_logits > token_logits[token_id]).sum().item()) + 1
            token_logprobs.append(lp.item())
            token_probs.append(prob.item())
            token_ranks.append(rank)

        seq_logprob = float(np.sum(token_logprobs))
        avg_logprob = float(np.mean(token_logprobs))
        geo_mean_prob = float(np.exp(avg_logprob))

        return {
            "target_text": target_text,
            "target_num_tokens": len(target_ids),
            "target_token_ids": target_ids,
            "target_tokens": [self.tokenizer.decode(tid) for tid in target_ids],
            "target_token_logprobs": token_logprobs,
            "target_token_probs": token_probs,
            "target_token_ranks": token_ranks,
            "target_seq_logprob": seq_logprob,
            "target_avg_logprob": avg_logprob,
            "target_geo_mean_prob": geo_mean_prob,
            "target_first_token_rank": token_ranks[0],
            "target_first_token_prob": token_probs[0],
        }
