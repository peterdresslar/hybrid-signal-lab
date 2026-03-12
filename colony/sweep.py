import argparse
import os
import json
import torch
import traceback
from pathlib import Path
from datetime import datetime

from colony.signal_lab import (
    load_model_and_tokenizer,
    run_model_pass,
    read_prompt,
    DEVICE,
    MODEL_NAME_0_8B,
    MODEL_NAME_2B,
    MODEL_NAME_4B,
    generate_g_vector_qwen35,
    g_vec_as_printable_array,
)
from colony.sweep_cartridges import get_cartridge

TARGET_DICTIONARY = {
    "short0.txt": " violet",
    "short1.txt": " twenty-one",
    "short2.txt": " U",
    "short3.txt": " door",
    "short4.txt": " so",
    "short5.txt": " nn",
    "med0.txt": " castle",
    "med1.txt": " Pemb",
    "med2.txt": " Stone",
    "med3.txt": " Th",
    "med4.txt": "5",
    "med5.txt": " rain",
    "long1.txt": " TRUMPET"
}

def get_prompts(args):
    data_dir = Path("data")
    prompt_files = []
    
    if args.use_prompt:
        prompt_files.append(args.use_prompt)
    elif args.short_only:
        prompt_files = [f.name for f in data_dir.glob("short*.txt")]
    elif args.med_only:
        prompt_files = [f.name for f in data_dir.glob("med*.txt")]
    else:
        prompt_files = [f.name for f in data_dir.glob("short*.txt")] + [f.name for f in data_dir.glob("med*.txt")]
        
    return sorted(prompt_files)

def main():
    parser = argparse.ArgumentParser(description="Sweep g values over different prompts.")
    parser.add_argument("--granularity", type=float, default=0.25, help="Distance between scalar sweep values. Default 0.25. Ignored when --cartridge is set.")
    parser.add_argument("--cartridge", type=str, default=None, help="Named cartridge of g_vec configurations to sweep. Mutually exclusive with scalar sweep.")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per prompt/g pair.")
    parser.add_argument("--short-only", action="store_true", help="Run only short*.txt prompts.")
    parser.add_argument("--med-only", action="store_true", help="Run only med*.txt prompts.")
    parser.add_argument("--use-prompt", type=str, help="Specify exactly one prompt file to run.")
    parser.add_argument("--verbose", action="store_true", help="Log heavier metrics (full top-k, attn. entropy) to a separate file.")
    parser.add_argument("--out-dir", type=str, default="results/sweep_{timestamp}", help="Output directory pattern. Use {timestamp} to inject current time.")
    parser.add_argument("--model-key", type=str, default="0_8B", help="Model to use. Please enter 0_8B, 2B, or 4B.")
    
    args = parser.parse_args()
    
    prompts_to_run = get_prompts(args)
    if not prompts_to_run:
        print("No prompts found to run.")
        return
        
    out_dir = args.out_dir.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "main.jsonl")
    verbose_file = os.path.join(out_dir, "verbose.jsonl") if args.verbose else None
    error_file = os.path.join(out_dir, "errors.jsonl")
    
    model_name = {
        "0_8B": MODEL_NAME_0_8B,
        "2B": MODEL_NAME_2B,
        "4B": MODEL_NAME_4B
    }[args.model_key]
    print("Loading model for sweep...")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device=DEVICE
    )

    meta_file = os.path.join(out_dir, "_meta.json")
    config_dict = {
        key: getattr(model.config, key, None)
        for key in ['model_type', 'num_hidden_layers', 'hidden_size',
                     'intermediate_size', 'num_attention_heads',
                     'num_key_value_heads', 'vocab_size']
    }
    metadata = {
        "model": getattr(model.config, "name_or_path", "unknown"),
        "device": DEVICE,
        "config": config_dict
    }
    with open(meta_file, "w") as f_meta:
        json.dump(metadata, f_meta, indent=2)
    print(f"Saved metadata to {meta_file}")
    
    cartridge = None
    if args.cartridge:
        cartridge = get_cartridge(args.cartridge)
        g_vecs = [generate_g_vector_qwen35(g) for g in cartridge["g_vectors"]]
        print(f"Cartridge '{cartridge['name']}': {cartridge['description']}")
        print(f"  {len(g_vecs)} g_vec configurations to sweep.")
        metadata["cartridge"] = cartridge["name"]
        metadata["cartridge_description"] = cartridge["description"]
        with open(meta_file, "w") as f_meta:
            json.dump(metadata, f_meta, indent=2)
    else:
        g_scalars = []
        current_g = 0.0
        while current_g <= 2.0001:
            g_scalars.append(current_g)
            current_g += args.granularity
        g_vecs = [generate_g_vector_qwen35(float(g)) for g in g_scalars]

    print(f"Will run {len(prompts_to_run)} prompts over {len(g_vecs)} g configurations ({args.repetitions} repetitions).")
    total_runs = len(prompts_to_run) * len(g_vecs) * args.repetitions
    print(f"Total runs: {total_runs}")

    with open(out_file, "w") as f_out:
        f_verb = open(verbose_file, "w") if verbose_file else None
        f_err = open(error_file, "w")
        for prompt_id in prompts_to_run:
            prompt_text = read_prompt(prompt_id)
            print(f"\nEvaluating prompt: {prompt_id}")

            try:
                baseline_result = run_model_pass(
                    model, tokenizer, prompt_text, generate_g_vector_qwen35(1.0),
                    device=DEVICE, prompt_id=prompt_id,
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

            target_str = TARGET_DICTIONARY.get(prompt_id)
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
                                device=DEVICE, prompt_id=prompt_id,
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
