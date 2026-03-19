"""Sweep harness — run g-profile configurations across prompt batteries."""

import argparse
import os
import json
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any

from colony.signal_lab import resolve_prompt, resolve_device, resolve_prompt_collection
from colony.model.g_profile import build_attention_scales_from_spec, printable_scales
from colony.model.prompt import Prompt
from colony.model import VALID_MODEL_KEYS
from colony.agent import Agent
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


def _parse_csv_strings(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    return values or None


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


def _prompts_from_battery(cartridge: dict[str, Any]) -> list[Prompt]:
    battery_path = cartridge["prompt_battery"]
    requested_ids: list[str] | None = None
    if cartridge.get("prompt_id"):
        requested_ids = [cartridge["prompt_id"]]
    elif cartridge.get("prompt_ids"):
        requested_ids = list(cartridge["prompt_ids"])

    return resolve_prompt_collection(
        battery_path,
        prompt_ids=requested_ids,
        prompt_tiers=cartridge.get("prompt_tiers"),
        prompt_types=cartridge.get("prompt_types"),
    )


def get_prompts_from_cartridge(cartridge: dict[str, Any]) -> list[Prompt]:
    if cartridge.get("prompt"):
        prompt_obj = Prompt.from_text(
            cartridge["prompt"],
            prompt_id=f"{cartridge['name']}_prompt",
        )
        if cartridge.get("target") is not None:
            prompt_obj.target = cartridge["target"]
        return [prompt_obj]

    if cartridge.get("prompt_battery"):
        prompts = _prompts_from_battery(cartridge)
        deduped_prompts: list[Prompt] = []
        seen_ids: set[str] = set()
        for prompt in prompts:
            if prompt.id in seen_ids:
                continue
            deduped_prompts.append(prompt)
            seen_ids.add(prompt.id)
        return deduped_prompts

    if cartridge.get("prompt_id"):
        return [resolve_prompt(cartridge["prompt_id"])]

    if cartridge.get("prompt_ids"):
        return [resolve_prompt(prompt_id) for prompt_id in cartridge["prompt_ids"]]

    if cartridge.get("prompt_tiers"):
        prompts = _prompts_from_tiers(cartridge["prompt_tiers"])
        deduped_prompts: list[Prompt] = []
        seen_ids: set[str] = set()
        for prompt in prompts:
            if prompt.id in seen_ids:
                continue
            deduped_prompts.append(prompt)
            seen_ids.add(prompt.id)
        return deduped_prompts

    raise ValueError(
        "Cartridge must define one of: 'prompt', 'prompt_battery', "
        "'prompt_id', 'prompt_ids', or 'prompt_tiers'."
    )


def apply_prompt_selection_overrides(
    cartridge: dict[str, Any],
    *,
    prompt_battery: str | None,
    prompt_id: str | None,
    prompt_ids: list[str] | None,
    prompt_tiers: list[str] | None,
    prompt_types: list[str] | None,
) -> dict[str, Any]:
    """Apply CLI prompt-selection overrides without mutating the base cartridge."""
    overridden = dict(cartridge)
    if prompt_battery is None:
        return overridden

    for key in ["prompt", "prompt_id", "prompt_ids", "prompt_tiers", "prompt_types", "target"]:
        overridden.pop(key, None)

    overridden["prompt_battery"] = prompt_battery
    if prompt_id is not None:
        overridden["prompt_id"] = prompt_id
    if prompt_ids is not None:
        overridden["prompt_ids"] = prompt_ids
    if prompt_tiers is not None:
        overridden["prompt_tiers"] = prompt_tiers
    if prompt_types is not None:
        overridden["prompt_types"] = prompt_types
    return overridden

def main():
    parser = argparse.ArgumentParser(description="Sweep g profiles over different prompts.")
    parser.add_argument("--cartridge", type=str, required=True, choices=list_cartridges(), help="Named cartridge of g profile configurations to sweep.")
    parser.add_argument("--model-key", type=str, default="0_8B", choices=VALID_MODEL_KEYS, help="Model to use. Defaults to 0_8B.")
    parser.add_argument("--device", type=str, default=None, help="Device override: auto (default), cuda, mps, or cpu. Also supports COLONY_DEVICE env var.")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per prompt/g pair.")
    parser.add_argument("--verbose", action="store_true", help="Log heavier metrics (full top-k, attn. entropy) to a separate file.")
    parser.add_argument("--out-dir", type=str, default="results/sweep_{timestamp}", help="Output directory pattern. Use {timestamp} to inject current time.")
    parser.add_argument("--prompt-battery", type=str, default=None, help="Battery directory or JSON file to use instead of the cartridge's built-in prompt selection.")
    parser.add_argument("--prompt-id", type=str, default=None, help="Single prompt id within --prompt-battery.")
    parser.add_argument("--prompt-ids", type=str, default=None, help="Comma-separated prompt ids within --prompt-battery.")
    parser.add_argument("--prompt-tiers", type=str, default=None, help="Comma-separated tiers to select from --prompt-battery.")
    parser.add_argument("--prompt-types", type=str, default=None, help="Comma-separated types to select from --prompt-battery.")
    
    args = parser.parse_args()
    runtime_device = resolve_device(args.device)

    if args.prompt_id and args.prompt_ids:
        raise ValueError("Use only one of --prompt-id or --prompt-ids.")

    cartridge = apply_prompt_selection_overrides(
        get_cartridge(args.cartridge),
        prompt_battery=args.prompt_battery,
        prompt_id=args.prompt_id,
        prompt_ids=_parse_csv_strings(args.prompt_ids),
        prompt_tiers=_parse_csv_strings(args.prompt_tiers),
        prompt_types=_parse_csv_strings(args.prompt_types),
    )
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
    print(f"Loading model for sweep: {model_key}")
    agent = Agent.from_model_key(model_key, runtime_device)

    meta_file = out_dir / "_meta.json"
    metadata: dict[str, Any] = {
        "model": agent.backend.model_name,
        "device": runtime_device,
        "config": agent.backend.config_summary,
    }
    with open(meta_file, "w") as f_meta:
        json.dump(metadata, f_meta, indent=2)
    print(f"Saved metadata to {meta_file}")
    
    attn_layers = agent.get_attention_layer_indices()
    g_specs = cartridge["g_specs"]
    g_runs: list[dict[str, Any]] = []
    for idx, g_spec in enumerate(g_specs):
        g_scales = build_attention_scales_from_spec(g_spec, attention_slots=len(attn_layers))
        g_runs.append(
            {
                "index": idx,
                "name": g_spec.get("name", f"g_spec_{idx}"),
                "g_spec": g_spec,
                "g_scales": g_scales,
                "printable_scales": printable_scales(g_scales),
            }
        )

    print(f"Cartridge '{cartridge['name']}': {cartridge['description']}")
    print(f"  {len(g_runs)} g profile configurations to sweep.")
    metadata["cartridge"] = cartridge["name"]
    metadata["cartridge_description"] = cartridge["description"]
    metadata["model_key"] = model_key
    metadata["attention_layer_indices"] = attn_layers
    metadata["attention_slot_count"] = len(attn_layers)
    metadata["g_specs"] = g_specs
    metadata["prompt_selection"] = {
        key: cartridge[key]
        for key in ["prompt", "prompt_battery", "prompt_id", "prompt_ids", "prompt_tiers", "prompt_types", "target"]
        if key in cartridge
    }
    with open(meta_file, "w") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    print(f"Will run {len(prompts_to_run)} prompts over {len(g_runs)} g configurations ({args.repetitions} repetitions).")
    total_runs = len(prompts_to_run) * len(g_runs) * args.repetitions
    print(f"Total runs: {total_runs}")

    with open(out_file, "w") as f_out:
        f_verb = open(verbose_file, "w") if verbose_file else None
        f_err = open(error_file, "w")
        for prompt in prompts_to_run:
            prompt_id = prompt.id
            prompt_text = prompt.prompt_text
            print(f"\nEvaluating prompt: {prompt_id} (Type: {prompt.type})")

            try:
                baseline_result = agent.run_pass(
                    prompt_text,
                    np.ones(len(attn_layers), dtype=float),
                    prompt_id=prompt_id,
                    return_raw_logits=True,
                    return_verbose=args.verbose,
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

            for rep in range(args.repetitions):
                for g_run in g_runs:
                    g_scales = g_run["g_scales"]
                    printable = g_run["printable_scales"]
                    is_baseline = bool(np.allclose(g_scales, 1.0, atol=1e-4))

                    try:
                        if rep == 0 and is_baseline:
                            res = baseline_result.copy()
                        else:
                            res = agent.run_pass(
                                prompt_text, g_scales,
                                prompt_id=prompt_id,
                                baseline_logits=baseline_logits,
                                return_verbose=args.verbose,
                            )
                    except Exception as e:
                        err = {
                            "prompt_id": prompt_id,
                            "rep": rep + 1,
                            "g_profile": g_run["name"],
                            "g_scales": printable,
                            "stage": "run_pass",
                            "error": repr(e),
                            "traceback": traceback.format_exc(),
                        }
                        f_err.write(json.dumps(err) + "\n")
                        f_err.flush()
                        print(f"  [ERROR] Rep {rep+1} g={printable} failed: {e}")
                        continue

                    if target_str is not None:
                        try:
                            target_metrics = agent.score_target(
                                prompt_text,
                                target_str,
                                g_scales,
                            )
                        except Exception as e:
                            err = {
                                "prompt_id": prompt_id,
                                "rep": rep + 1,
                                "g_profile": g_run["name"],
                                "g_scales": printable,
                                "stage": "score_target",
                                "error": repr(e),
                                "traceback": traceback.format_exc(),
                            }
                            f_err.write(json.dumps(err) + "\n")
                            f_err.flush()
                            print(f"  [ERROR] Rep {rep+1} g={printable} target scoring failed: {e}")
                            continue

                        if target_metrics is not None:
                            res.update(target_metrics)
                            res["target_token"] = target_metrics["target_tokens"][0]
                            res["target_rank"] = target_metrics["target_first_token_rank"]
                            res["target_prob"] = target_metrics["target_first_token_prob"]

                    res["rep"] = rep + 1

                    core_res = {
                        "prompt_id": res["prompt_id"],
                        "g_profile": g_run["name"],
                        "g_function": g_run["g_spec"].get("g_function"),
                        "g_spec": g_run["g_spec"],
                        "g_attention_scales": res["g_attention_scales"],
                        "rep": res["rep"],
                        "target_text": res.get("target_text"),
                        "target_num_tokens": res.get("target_num_tokens"),
                        "target_seq_logprob": res.get("target_seq_logprob"),
                        "target_avg_logprob": res.get("target_avg_logprob"),
                        "target_geo_mean_prob": res.get("target_geo_mean_prob"),
                        "target_first_token_rank": res.get("target_first_token_rank"),
                        "target_first_token_prob": res.get("target_first_token_prob"),
                        "target_rank": res.get("target_rank"),
                        "target_prob": res.get("target_prob"),
                        "final_entropy_bits": res["final_entropy_bits"],
                        "kl_from_baseline": res["kl_from_baseline"],
                        "elapsed_time": res["elapsed_time"],
                    }

                    f_out.write(json.dumps(core_res) + "\n")
                    f_out.flush()

                    if f_verb and args.verbose:
                        f_verb.write(json.dumps(res) + "\n")
                        f_verb.flush()

                    print(
                        f"  [Rep {rep+1}] g_profile={g_run['name']} scales={printable} | "
                        f"Time: {res['elapsed_time']:.2f}s | Final Ent: {res['final_entropy_bits']:.2f} bits | "
                        f"Target Rank: {res.get('target_rank')} | "
                        f"Target p(tok): "
                        f"{(res.get('target_geo_mean_prob') if res.get('target_geo_mean_prob') is not None else float('nan')):.4f} | "
                        f"Target Avg LogP: "
                        f"{(res.get('target_avg_logprob') if res.get('target_avg_logprob') is not None else float('nan')):.4f}"
                    )

        if f_verb:
            f_verb.close()
        f_err.close()

    print(f"\nSweep complete! Saved results to directory: {out_dir}")

if __name__ == "__main__":
    main()
