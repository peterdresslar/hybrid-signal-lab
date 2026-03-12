import argparse
import os
import json
import torch
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any

from colony.signal_lab import (
    load_model_and_tokenizer,
    run_model_pass,
    resolve_prompt,
    resolve_device,
    MODEL_NAME_0_8B,
    MODEL_NAME_2B,
    MODEL_NAME_4B,
    MODEL_NAME_9B,
    generate_g_vector_qwen35,
    g_vec_as_printable_array,
)
from colony.model.prompt import Prompt
from colony.sweep_cartridges import get_cartridge, list_cartridges

TIER_TO_CATALOG = {
    "short": "prompts_short.json",
    "brief": "prompts_brief.json",
    "med": "prompts_med.json",
    "long": "prompts_long.json",
    "extended": "prompts_extended.json",
}


def resolve_out_dir(out_dir_pattern: str) -> Path:
    rendered = out_dir_pattern.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_path = Path(rendered)
    if out_path.is_absolute():
        return out_path
    if out_path.parts and out_path.parts[0] == "results":
        return out_path
    return Path("results") / out_path


def _prompts_from_tiers(prompt_tiers: list[str]) -> list[Prompt]:
    prompts: list[Prompt] = []
    for tier in prompt_tiers:
        if tier not in TIER_TO_CATALOG:
            available = ", ".join(sorted(TIER_TO_CATALOG.keys()))
            raise ValueError(f"Unknown prompt tier '{tier}'. Available tiers: {available}")

        catalog_name = TIER_TO_CATALOG[tier]
        catalog_path = Path("data") / catalog_name
        with open(catalog_path, "r", encoding="utf-8") as file_handle:
            entries = json.load(file_handle)
        if not isinstance(entries, list):
            raise ValueError(f"Prompt catalog must be a list: {catalog_path}")

        for entry in entries:
            if not isinstance(entry, dict) or "id" not in entry:
                continue
            prompts.append(resolve_prompt(f"{catalog_name}:{entry['id']}"))
    return prompts


def get_prompts_from_cartridge(cartridge: dict[str, Any]) -> list[Prompt]:
    if cartridge.get("prompt"):
        prompt_obj = Prompt.from_text(
            cartridge["prompt"],
            prompt_id=f"{cartridge['name']}_prompt",
        )
        if cartridge.get("target") is not None:
            prompt_obj.target = cartridge["target"]
        return [prompt_obj]

    if cartridge.get("prompt_id"):
        return [resolve_prompt(cartridge["prompt_id"])]

    if cartridge.get("prompt_ids"):
        return [resolve_prompt(prompt_id) for prompt_id in cartridge["prompt_ids"]]

    if cartridge.get("prompt_tiers"):
        prompts = _prompts_from_tiers(cartridge["prompt_tiers"])
        # Deduplicate by prompt id while preserving order.
        deduped_prompts: list[Prompt] = []
        seen_ids: set[str] = set()
        for prompt in prompts:
            if prompt.id in seen_ids:
                continue
            deduped_prompts.append(prompt)
            seen_ids.add(prompt.id)
        return deduped_prompts

    raise ValueError(
        "Cartridge must define one of: 'prompt', 'prompt_id', 'prompt_ids', or 'prompt_tiers'."
    )

def main():
    parser = argparse.ArgumentParser(description="Sweep g values over different prompts.")
    parser.add_argument("--cartridge", type=str, required=True, choices=list_cartridges(), help="Named cartridge of g_vec configurations to sweep.")
    parser.add_argument("--model-key", type=str, default="0_8B", choices=["0_8B", "2B", "4B", "9B"], help="Model to use. Defaults to 0_8B.")
    parser.add_argument("--device", type=str, default=None, help="Device override: auto (default), cuda, mps, or cpu. Also supports COLONY_DEVICE env var.")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per prompt/g pair.")
    parser.add_argument("--verbose", action="store_true", help="Log heavier metrics (full top-k, attn. entropy) to a separate file.")
    parser.add_argument("--out-dir", type=str, default="results/sweep_{timestamp}", help="Output directory pattern. Use {timestamp} to inject current time.")
    
    args = parser.parse_args()
    runtime_device = resolve_device(args.device)

    cartridge = get_cartridge(args.cartridge)
    prompts_to_run = get_prompts_from_cartridge(cartridge)
    if not prompts_to_run:
        print("No prompts found to run.")
        return
        
    out_dir = resolve_out_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = out_dir / "main.jsonl"
    verbose_file = (out_dir / "verbose.jsonl") if args.verbose else None
    error_file = out_dir / "errors.jsonl"
    
    model_key = args.model_key
    model_name = {
        "0_8B": MODEL_NAME_0_8B,
        "2B": MODEL_NAME_2B,
        "4B": MODEL_NAME_4B,
        "9B": MODEL_NAME_9B,
    }[model_key]
    print("Loading model for sweep...")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=runtime_device
    )

    meta_file = out_dir / "_meta.json"
    config_dict = {
        key: getattr(model.config, key, None)
        for key in ['model_type', 'num_hidden_layers', 'hidden_size',
                     'intermediate_size', 'num_attention_heads',
                     'num_key_value_heads', 'vocab_size']
    }
    metadata = {
        "model": getattr(model.config, "name_or_path", "unknown"),
        "device": runtime_device,
        "config": config_dict
    }
    with open(meta_file, "w") as f_meta:
        json.dump(metadata, f_meta, indent=2)
    print(f"Saved metadata to {meta_file}")
    
    g_vecs = [generate_g_vector_qwen35(g) for g in cartridge["g_vectors"]]
    print(f"Cartridge '{cartridge['name']}': {cartridge['description']}")
    print(f"  {len(g_vecs)} g_vec configurations to sweep.")
    metadata["cartridge"] = cartridge["name"]
    metadata["cartridge_description"] = cartridge["description"]
    metadata["model_key"] = model_key
    metadata["prompt_selection"] = {
        key: cartridge[key]
        for key in ["prompt", "prompt_id", "prompt_ids", "prompt_tiers", "target"]
        if key in cartridge
    }
    with open(meta_file, "w") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    print(f"Will run {len(prompts_to_run)} prompts over {len(g_vecs)} g configurations ({args.repetitions} repetitions).")
    total_runs = len(prompts_to_run) * len(g_vecs) * args.repetitions
    print(f"Total runs: {total_runs}")

    with open(out_file, "w") as f_out:
        f_verb = open(verbose_file, "w") if verbose_file else None
        f_err = open(error_file, "w")
        for prompt in prompts_to_run:
            prompt_id = prompt.id
            prompt_text = prompt.prompt_text
            print(f"\nEvaluating prompt: {prompt_id} (Type: {prompt.type})")

            try:
                baseline_result = run_model_pass(
                    model, tokenizer, prompt_text, generate_g_vector_qwen35(1.0),
                    device=runtime_device, prompt_id=prompt_id,
                    return_raw_logits=True,
                    return_verbose=args.verbose
                )
                baseline_logits = baseline_result.pop("_raw_logits")
            except Exception as e:
                err = {
                    "prompt_id": prompt_id,
                    "stage": "baseline",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                f_err.write(json.dumps(err) + "\n")
                f_err.flush()
                print(f"  [ERROR] baseline failed for {prompt_id}: {e}")
                continue

            target_str = prompt.target
            if target_str is None and len(prompts_to_run) == 1:
                target_str = cartridge.get("target")
            if target_str is not None:
                encoded_target = tokenizer.encode(target_str, add_special_tokens=False)
                if len(encoded_target) > 0:
                    target_token_idx = encoded_target[0]
                else:
                    target_token_idx = baseline_logits.argmax().item()
            else:
                target_token_idx = baseline_logits.argmax().item()

            for rep in range(args.repetitions):
                for g_vec in g_vecs:
                    printable = g_vec_as_printable_array(g_vec)
                    is_baseline = all(abs(v - 1.0) < 1e-4 for v in printable)

                    try:
                        if rep == 0 and is_baseline:
                            res = baseline_result.copy()
                            baseline_probs = torch.softmax(baseline_logits, dim=-1)
                            if "target_token" not in res:
                                res["target_token"] = tokenizer.decode(target_token_idx)
                            res["target_prob"] = baseline_probs[target_token_idx].item()
                            res["target_rank"] = (baseline_probs > baseline_probs[target_token_idx]).sum().item() + 1
                        else:
                            res = run_model_pass(
                                model, tokenizer, prompt_text, g_vec,
                                device=runtime_device, prompt_id=prompt_id,
                                target_token_id=target_token_idx,
                                baseline_logits=baseline_logits,
                                return_verbose=args.verbose
                            )
                    except Exception as e:
                        err = {
                            "prompt_id": prompt_id,
                            "rep": rep + 1,
                            "g_vector": printable,
                            "stage": "run_model_pass",
                            "error": repr(e),
                            "traceback": traceback.format_exc(),
                        }
                        f_err.write(json.dumps(err) + "\n")
                        f_err.flush()
                        print(f"  [ERROR] Rep {rep+1} g={printable} failed: {e}")
                        continue

                    res["rep"] = rep + 1

                    core_res = {
                        "prompt_id": res["prompt_id"],
                        "g_vector": res["g_vector"],
                        "rep": res["rep"],
                        "target_rank": res["target_rank"],
                        "target_prob": res["target_prob"],
                        "final_entropy_bits": res["final_entropy_bits"],
                        "kl_from_baseline": res["kl_from_baseline"],
                        "elapsed_time": res["elapsed_time"],
                    }

                    f_out.write(json.dumps(core_res) + "\n")
                    f_out.flush()

                    if f_verb and args.verbose:
                        f_verb.write(json.dumps(res) + "\n")
                        f_verb.flush()

                    print(f"  [Rep {rep+1}] g={printable} | Time: {res['elapsed_time']:.2f}s | Final Ent: {res['final_entropy_bits']:.2f} bits | Target Rank: {res.get('target_rank')}")

        if f_verb:
            f_verb.close()
        f_err.close()

    print(f"\nSweep complete! Saved results to directory: {out_dir}")

if __name__ == "__main__":
    main()
