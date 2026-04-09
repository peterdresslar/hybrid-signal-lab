#!/bin/bash

#SBATCH -G a100:1
#SBATCH -C a100_80
# #SBATCH -G h100:8    # has to be htc
#SBATCH -C a100_80
#SBATCH --mem=80GB
#SBATCH -t 0-3:59:59
#SBATCH -p htc
#SBATCH -q public
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="pdressla@asu.edu"
#SBATCH --export=NONE

set -eu

# === cache dirs on persistent storage ===
export HF_HOME=/scratch/pdressla/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/pdressla/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/pdressla/.cache/huggingface/transformers
export UV_CACHE_DIR=/scratch/pdressla/.cache/uv
export XDG_CACHE_HOME=/scratch/pdressla/.cache

export OUTDIR=/home/pdressla/workspace/data/sl-runs/022-balanced-attn/

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

uv run -m signal_lab.sweep \
  --cartridge balanced_kitchen_sink \
  --prompt-battery battery/data/battery_4 \
  --model-key OLMO \
  --device cuda \
  --out-dir "${OUTDIR}/OLMO" \
  --intervention-strategy attention_contribution \
  --verbose

uv run -m signal_lab.run_analyze \
  --input-dir "${OUTDIR}/OLMO" \
  --intervention-folders \
  --no-compare
