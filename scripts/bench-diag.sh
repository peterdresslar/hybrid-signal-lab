#!/bin/bash

#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH -t 0-3:59:59
#SBATCH -p public
#SBATCH -q public
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="pdressla@asu.edu"
#SBATCH --export=NONE
#SBATCH --mem=80GB

set -eu

# === cache dirs on persistent storage ===
export HF_HOME=/scratch/pdressla/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/pdressla/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/pdressla/.cache/huggingface/transformers
export UV_CACHE_DIR=/scratch/pdressla/.cache/uv
export XDG_CACHE_HOME=/scratch/pdressla/.cache
export OUTDIR=/home/pdressla/workspace/data/sl-runs/011-bench/

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

python - <<'PY'
import os
print("HF token visible:", bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")))
print("HF_HOME:", os.getenv("HF_HOME"))
print("HUGGINGFACE_HUB_CACHE:", os.getenv("HUGGINGFACE_HUB_CACHE"))
print("TRANSFORMERS_CACHE:", os.getenv("TRANSFORMERS_CACHE"))
PY

python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = "allenai/Olmo-Hybrid-7B"
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_id, token=token)
print("tokenizer loaded")

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    dtype="auto",
    device_map="cuda",
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
print("model loaded")
PY

uv run -m bench.run_bench \
    --model-key OLMO \
    --tasks copa storycloze gsm8k \
    --baseline-only \
    --output-dir "$OUTDIR/OLMO"

uv run -m bench.run_bench \
    --model-key 9B \
    --tasks copa storycloze gsm8k \
    --baseline-only \
    --output-dir "$OUTDIR/9B"

uv run -m bench.run_bench \
    --model-key OLMO \
    --tasks copa storycloze gsm8k \
    --router-model /home/pdressla/workspace/hybrid-signal-lab/router/router-OLMO-011/router_model.json \
    --output-dir "$OUTDIR/routed_OLMO"

uv run -m bench.run_bench \
    --model-key 9B \
    --tasks copa storycloze gsm8k \
    --router-model /home/pdressla/workspace/hybrid-signal-lab/router/router-9B-011/router_model.json \
    --output-dir "$OUTDIR/routed_9B"
