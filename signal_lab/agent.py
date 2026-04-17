"""Agent — shared inference pipeline over a ModelBackend."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model.backend import InterventionMode, ModelBackend, attention_scaler_hook
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

    def _register_gain_hooks(
        self,
        attn_layers: list[int],
        scales: np.ndarray,
        intervention_mode: str | InterventionMode | None,
    ) -> tuple[list[Any], InterventionMode]:
        resolved_mode = self.backend.resolve_intervention_mode(intervention_mode)
        hooks = []
        for idx, scale in zip(attn_layers, scales.tolist()):
            module = self.backend.get_hook_module(idx, resolved_mode)
            hooks.append(module.register_forward_hook(attention_scaler_hook(scale)))
        return hooks, resolved_mode

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
        return_hidden_states: bool = False,
        hidden_state_dtype: torch.dtype = torch.float16,
        target_attention_layer_indices: list[int] | None = None,
        intervention_mode: str | InterventionMode = InterventionMode.BACKEND_DEFAULT,
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

        hooks, resolved_mode = self._register_gain_hooks(
            attn_layers,
            scales,
            intervention_mode,
        )

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
            "intervention_mode": resolved_mode.value,
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

        if return_hidden_states:
            raw_hidden_states = getattr(outputs, "hidden_states", None)
            if raw_hidden_states is None:
                raise RuntimeError("Model output did not include hidden_states.")
            hidden_states = torch.stack(
                [
                    hidden[0].detach().to(device="cpu", dtype=hidden_state_dtype)
                    for hidden in raw_hidden_states
                ],
                dim=0,
            ).contiguous()
            input_ids = inputs["input_ids"][0].detach().to(device="cpu")
            attention_mask = None
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"][0].detach().to(device="cpu")

            result["input_ids"] = input_ids
            result["attention_mask"] = attention_mask
            result["hidden_states"] = hidden_states
            result["hidden_state_shape"] = list(hidden_states.shape)
            result["hidden_state_dtype"] = str(hidden_states.dtype)
            result["num_tokens"] = int(input_ids.shape[0])
            result["num_layers_plus_embedding"] = int(hidden_states.shape[0])
            result["hidden_size"] = int(hidden_states.shape[-1])

        if return_raw_logits:
            result["_raw_logits"] = final_logits

        return result

    def capture_sequence_states(
        self,
        prompt: str,
        *,
        prompt_id: str | None = None,
        hidden_state_dtype: torch.dtype = torch.float16,
        capture_attention_entropy: bool = True,
    ) -> dict[str, Any]:
        """Capture all-token layer-output hidden states for one prompt.

        Returns CPU-resident tensors suitable for serialization:
        - ``input_ids``: shape ``(N,)``
        - ``attention_mask``: shape ``(N,)`` when available
        - ``hidden_states``: shape ``(L+1, N, d)``
        """
        start_time = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=capture_attention_entropy,
            )

        final_logits = outputs.logits[0, -1, :].float()
        probs = torch.softmax(final_logits, dim=-1)
        final_entropy_bits = -(probs * torch.log2(probs + 1e-10)).sum().item()

        all_logits = outputs.logits[0, :, :].float()
        all_probs = torch.softmax(all_logits, dim=-1)
        mean_entropy_bits = (
            -(all_probs * torch.log2(all_probs + 1e-10)).sum(dim=-1).mean().item()
        )

        raw_hidden_states = getattr(outputs, "hidden_states", None)
        if raw_hidden_states is None:
            raise RuntimeError("Model output did not include hidden_states.")

        hidden_states = torch.stack(
            [
                hidden[0].detach().to(device="cpu", dtype=hidden_state_dtype)
                for hidden in raw_hidden_states
            ],
            dim=0,
        ).contiguous()

        input_ids = inputs["input_ids"][0].detach().to(device="cpu")
        attention_mask = None
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"][0].detach().to(device="cpu")

        elapsed_time = time.perf_counter() - start_time

        result: dict[str, Any] = {
            "prompt_id": prompt_id if prompt_id else (prompt[:20] + "..."),
            "model_type": getattr(self.model.config, "model_type", None),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hidden_states": hidden_states,
            "hidden_state_shape": list(hidden_states.shape),
            "hidden_state_dtype": str(hidden_states.dtype),
            "num_tokens": int(input_ids.shape[0]),
            "num_layers_plus_embedding": int(hidden_states.shape[0]),
            "hidden_size": int(hidden_states.shape[-1]),
            "final_entropy_bits": final_entropy_bits,
            "mean_entropy_bits": mean_entropy_bits,
            "elapsed_time": elapsed_time,
        }

        if capture_attention_entropy:
            result.update(self.backend.process_attention_entropy(outputs))

        return result

    def score_target(
        self,
        prompt: str,
        target_text: str,
        g_attention_scales: np.ndarray | list[float],
        *,
        target_attention_layer_indices: list[int] | None = None,
        intervention_mode: str | InterventionMode = InterventionMode.BACKEND_DEFAULT,
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

        hooks, resolved_mode = self._register_gain_hooks(
            attn_layers,
            scales,
            intervention_mode,
        )

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
            "intervention_mode": resolved_mode.value,
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

    def generate(
        self,
        prompt: str,
        g_attention_scales: np.ndarray | list[float],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_strings: list[str] | None = None,
        target_attention_layer_indices: list[int] | None = None,
        intervention_mode: str | InterventionMode = InterventionMode.BACKEND_DEFAULT,
    ) -> dict[str, Any]:
        """Generate text autoregressively with gain-scaled attention hooks.

        Uses manual token-by-token decoding (full forward pass per step)
        rather than model.generate(), because hybrid linear-attention
        models may not support the KV-cache loop that generate() expects
        when the flash-linear-attention fast path is unavailable.

        Greedy decoding (temperature=0) by default.

        Args:
            prompt: input text
            g_attention_scales: per-attention-layer gain values
            max_new_tokens: generation budget
            temperature: sampling temperature (0 = greedy)
            stop_strings: optional early-stop strings (e.g. ["\\n\\n"])
            target_attention_layer_indices: override which layers to hook
            intervention_mode: which hook target to scale

        Returns:
            dict with generated_text, num_tokens_generated, elapsed_time
        """
        start_time = time.perf_counter()

        attn_layers = self._resolve_target_attention_layers(target_attention_layer_indices)
        scales = np.asarray(g_attention_scales, dtype=float)
        if scales.ndim != 1 or scales.size != len(attn_layers):
            raise ValueError(
                f"g_attention_scales length ({scales.size}) must equal "
                f"attention layer count ({len(attn_layers)})."
            )

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids: list[int] = []
        eos_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens):
                all_ids = input_ids + generated_ids
                ids_tensor = torch.tensor([all_ids], dtype=torch.long, device=self.device)

                # Register hooks fresh each step (cheap, ensures clean state)
                hooks, resolved_mode = self._register_gain_hooks(
                    attn_layers, scales, intervention_mode,
                )
                try:
                    outputs = self.model(input_ids=ids_tensor)
                finally:
                    for h in hooks:
                        h.remove()

                next_logits = outputs.logits[0, -1, :].float()

                if temperature <= 0:
                    next_id = int(next_logits.argmax())
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_id = int(torch.multinomial(probs, 1)[0])

                generated_ids.append(next_id)

                if next_id == eos_id:
                    break

                # Check stop strings on decoded text so far
                if stop_strings:
                    text_so_far = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True,
                    )
                    if any(ss in text_so_far for ss in stop_strings):
                        break

        elapsed_time = time.perf_counter() - start_time
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Trim at stop string
        if stop_strings:
            earliest = len(generated_text)
            for ss in stop_strings:
                idx = generated_text.find(ss)
                if 0 <= idx < earliest:
                    earliest = idx
            if earliest < len(generated_text):
                generated_text = generated_text[:earliest]

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "num_tokens_generated": len(generated_ids),
            "intervention_mode": resolved_mode.value,
            "g_attention_scales": printable_scales(scales),
            "elapsed_time": elapsed_time,
        }
