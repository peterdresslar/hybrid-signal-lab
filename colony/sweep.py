import argparse
import os
import json
import torch
from pathlib import Path
from datetime import datetime

from colony.signal_lab import load_model_and_tokenizer, run_model_pass, read_prompt, DEVICE

TARGET_DICTIONARY = {
    "short0.txt": " violet",
    "short1.txt": " 34",
    "short2.txt": " U",
    "short3.txt": " door",
    "short4.txt": " so",
    "short5.txt": " nn",
    "med0.txt": " castle",
    "med1.txt": " Pemb",
    "med2.txt": " Stone",
    "med3.txt": " Thes",
    "med4.txt": " 5",
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
    parser.add_argument("--granularity", type=float, default=0.25, help="Distance between sweep values. Default 0.25.")
    parser.add_argument("--repetitions", type=int, default=1, help="Number of repetitions per prompt/g pair.")
    parser.add_argument("--short-only", action="store_true", help="Run only short*.txt prompts.")
    parser.add_argument("--med-only", action="store_true", help="Run only med*.txt prompts.")
    parser.add_argument("--use-prompt", type=str, help="Specify exactly one prompt file to run.")
    parser.add_argument("--out", type=str, default="sweep_results_{timestamp}.jsonl", help="Output JSONL file pattern. Use {timestamp} to inject current time.")
    
    args = parser.parse_args()
    
    prompts_to_run = get_prompts(args)
    if not prompts_to_run:
        print("No prompts found to run.")
        return
        
    out_file = args.out.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    print(f"Loading model for sweep...")
    model, tokenizer = load_model_and_tokenizer()
    
    g_values = []
    current_g = 0.0
    while current_g <= 2.0001:  # small epsilon for floating point
        g_values.append(current_g)
        current_g += args.granularity
        
    print(f"Will run {len(prompts_to_run)} prompts over {len(g_values)} g values ({args.repetitions} repetitions).")
    total_runs = len(prompts_to_run) * len(g_values) * args.repetitions
    print(f"Total runs: {total_runs}")
    
    with open(out_file, "w") as f_out:
        for prompt_id in prompts_to_run:
            prompt_text = read_prompt(prompt_id)
            print(f"\nEvaluating prompt: {prompt_id}")
            
            # Step 1: Run baseline (g=1.0) to get raw logits for KL divergence calculation
            baseline_result = run_model_pass(
                model, tokenizer, prompt_text, 1.0, 
                device=DEVICE, prompt_id=prompt_id, 
                return_raw_logits=True
            )
            baseline_logits = baseline_result.pop("_raw_logits")
            
            # Identify target token from our dictionary
            target_str = TARGET_DICTIONARY.get(prompt_id)
            if target_str is not None:
                # Tokenize the specific target string to get its ID.
                # If it happens to be multiple tokens, just take the first one 
                # (e.g. " 34" -> " ", "3", "4" -> takes " ")
                encoded_target = tokenizer.encode(target_str, add_special_tokens=False)
                if len(encoded_target) > 0:
                    target_token_idx = encoded_target[0]
                else:
                    target_token_idx = baseline_result["top_k_indices"][0]
            else:
                # Fallback to the top prediction of the baseline
                target_token_idx = baseline_result["top_k_indices"][0]
            
            for rep in range(args.repetitions):
                for g in g_values:
                    # skip running baseline again if rep == 0 and g == 1.0
                    if rep == 0 and abs(g - 1.0) < 1e-4:
                        res = baseline_result.copy()
                        # already ran, just add repetition tracking if we want to
                    else:
                        res = run_model_pass(
                            model, tokenizer, prompt_text, g, 
                            device=DEVICE, prompt_id=prompt_id,
                            target_token_id=target_token_idx,
                            baseline_logits=baseline_logits
                        )
                    
                    res["repetition"] = rep + 1
                    
                    # Dump to JSONL
                    f_out.write(json.dumps(res) + "\n")
                    f_out.flush()
                    
                    print(f"  [Rep {rep+1}] g={g:.2f} | Time: {res['elapsed_time']:.2f}s | Final Ent: {res['final_entropy_bits']:.2f} bits | Target Rank: {res.get('target_rank')}")

    print(f"\nSweep complete! Saved to {out_file}")

if __name__ == "__main__":
    main()
