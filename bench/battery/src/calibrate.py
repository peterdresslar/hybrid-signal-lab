#!/usr/bin/env python3
"""
Calibration sweep: run each candidate prompt at baseline (g=1.0)
and record p(target_token), entropy, and rank.

This is a thin wrapper that calls the existing sweep infrastructure in signal_lab.py.
Adapt the model loading and forward pass to match signal_lab.py.

Usage:
    # Calibrate a single type file:
    python calibrate.py --battery battery/factual_recall.json \
                        --model Qwen/Qwen3.5-2B-Base \
                        --output calibration/factual_recall.jsonl

    # Or the combined file:
    python calibrate.py --battery battery/all_candidates.json \
                        --model Qwen/Qwen3.5-2B-Base \
                        --output calibration_results.jsonl
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .adapters import adapt_prompt
except ImportError:
    from adapters import adapt_prompt


def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer. Adapt to match your signal_lab.py setup."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {
        "dtype": torch.bfloat16,
    }

    if device == "auto":
        model_kwargs["device_map"] = "auto"
    elif device == "cuda":
        model_kwargs["device_map"] = "auto" if torch.cuda.device_count() > 1 else "cuda"
    else:
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.eval()
    print(f"  Loaded. {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    return model, tokenizer


def get_model_input_device(model, requested_device: str) -> torch.device:
    """Return the device where prompt tokens should be placed."""
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        return input_embeddings.weight.device

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for mapped_device in device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if mapped_device not in {"cpu", "disk"}:
                return torch.device(mapped_device)

    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(requested_device)


def run_baseline(model, tokenizer, prompt: str, target: str, device: str = "cuda"):
    """Run a single forward pass at baseline and extract signals."""

    # Tokenize
    input_device = get_model_input_device(model, device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(input_device)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    if len(target_ids) == 0:
        return None

    # We care about the first token of the target
    target_token_id = target_ids[0]

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, output_hidden_states=True)

    logits = outputs.logits[0, -1, :]  # Last token logits
    logits_log2 = torch.log(torch.tensor(2.0, device=logits.device))
    probs = F.softmax(logits, dim=-1)

    # Target probability and rank
    target_prob = probs[target_token_id].item()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item() + 1

    # Entropy
    log_probs = torch.log(probs + 1e-10)
    entropy = (-(probs * log_probs).sum() / logits_log2).item()

    # Mean sequence entropy (over all positions)
    all_logits = outputs.logits[0]  # [seq_len, vocab]
    all_probs = F.softmax(all_logits, dim=-1)
    all_log_probs = torch.log(all_probs + 1e-10)
    all_entropies = -(all_probs * all_log_probs).sum(dim=-1) / logits_log2
    mean_seq_entropy = all_entropies.mean().item()

    # Attention entropy per layer (if available)
    attn_entropies = []
    if outputs.attentions is not None:
        for layer_attn in outputs.attentions:
            # layer_attn shape: [batch, heads, seq, seq]
            attn = layer_attn[0]  # [heads, seq, seq]
            # Entropy of last token's attention over all positions, averaged over heads
            last_attn = attn[:, -1, :]  # [heads, seq]
            log_attn = torch.log(last_attn + 1e-10)
            attn_log2 = torch.log(torch.tensor(2.0, device=last_attn.device))
            head_entropies = -(last_attn * log_attn).sum(dim=-1) / attn_log2
            attn_entropies.append(head_entropies.mean().item())

    return {
        "target_prob": target_prob,
        "target_rank": rank,
        "final_entropy": entropy,
        "mean_seq_entropy": mean_seq_entropy,
        "attn_entropy_profile": attn_entropies,
        "target_avg_logp": (log_probs[target_token_id] / logits_log2).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="calibration_results.jsonl")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Execution target: cpu, mps, cuda, cuda:N, or auto. 'cuda' auto-shards across multiple GPUs.",
    )
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from this item index (for crash recovery)")
    args = parser.parse_args()

    # Load battery
    with open(args.battery) as f:
        battery = json.load(f)
    print(f"Loaded {len(battery)} candidates")

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Run calibration
    output_path = Path(args.output)
    mode = "a" if args.resume_from > 0 else "w"

    with open(output_path, mode) as f:
        for idx, item in enumerate(battery):
            if idx < args.resume_from:
                continue

            t0 = time.time()
            prompt_for_model, adapter_name, prompt_render_version = adapt_prompt(item)
            result = run_baseline(model, tokenizer, prompt_for_model, item["target"], args.device)
            elapsed = time.time() - t0

            if result is None:
                print(f"  [{idx}/{len(battery)}] {item['id']} — SKIP (empty target)")
                continue

            record = {
                "id": item["id"],
                "type": item["type"],
                "tier": item["tier"],
                "model": args.model,
                "adapter": adapter_name,
                "prompt_render_version": prompt_render_version,
                "rendered_prompt_sha256": hashlib.sha256(prompt_for_model.encode()).hexdigest(),
                **result,
                "time_s": round(elapsed, 3),
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            prob_str = f"{result['target_prob']:.4f}"
            print(f"  [{idx+1}/{len(battery)}] {item['id']:30s} [{item['type']:25s}] "
                  f"p={prob_str}  rank={result['target_rank']}  H={result['final_entropy']:.2f}  "
                  f"({elapsed:.2f}s)")

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()