#!/bin/bash

#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH -t 0-3:59:59
#SBATCH -p htc
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
export HF_DATASETS_CACHE=/scratch/pdressla/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/scratch/pdressla/.cache/huggingface/transformers
export UV_CACHE_DIR=/scratch/pdressla/.cache/uv
export XDG_CACHE_HOME=/scratch/pdressla/.cache
export OUTDIR=/home/pdressla/workspace/data/sl-runs/020-bench/

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"
export TRANSFORMERS_VERBOSITY=info
export HF_HUB_VERBOSITY=debug
export DATASETS_VERBOSITY=info

echo "=== Hugging Face cache preflight ==="
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "-- model cache dirs --"
ls -ld "$HUGGINGFACE_HUB_CACHE/models--allenai--Olmo-Hybrid-7B" 2>/dev/null || echo "missing: $HUGGINGFACE_HUB_CACHE/models--allenai--Olmo-Hybrid-7B"
ls -ld "$HUGGINGFACE_HUB_CACHE/models--Qwen--Qwen3.5-9B-Base" 2>/dev/null || echo "missing: $HUGGINGFACE_HUB_CACHE/models--Qwen--Qwen3.5-9B-Base"
echo "-- lock files under hub cache --"
find "$HUGGINGFACE_HUB_CACHE" -name '*.lock' -print 2>/dev/null || true
echo "-- datasets cache dir --"
ls -ld "$HF_DATASETS_CACHE" 2>/dev/null || echo "missing: $HF_DATASETS_CACHE"
echo "=== end preflight ==="

uv run -m bench.run_bench \
    --model-key 9B \
    --tasks copa storycloze gsm8k \
    --gsm8k-limit 100 \
    --verbose \
    --baseline-only \
    --output-dir "$OUTDIR/9B"
