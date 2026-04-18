#!/bin/bash

#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH -t 0-03:59:56
#SBATCH -p public
#SBATCH -q public
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="pdressla@asu.edu"
#SBATCH --export=NONE
#SBATCH --mem=80GB

set -euo pipefail

export HF_HOME=/scratch/pdressla/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/pdressla/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/pdressla/.cache/huggingface/transformers
export UV_CACHE_DIR=/scratch/pdressla/.cache/uv
export XDG_CACHE_HOME=/scratch/pdressla/.cache
export OUTDIR=/home/pdressla/workspace/data/sl-runs/050-bench

export SAT_VALUES=(0.0 0.01 0.02 0.03 0.05)

ROUTER_MODELS=(
  /home/pdressla/workspace/hybrid-signal-lab/router/router-9B-050-probes-all_layers_mean_pool_concat-attn_resid-pc50/probe_router_model.json
)

ROUTER_LABELS=(
  allmean_attn50
)

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

echo "[bench] saturations=${SAT_VALUES[*]}"
echo "[bench] routers=${ROUTER_LABELS[*]}"
uv run -m bench.run_bench \
  --model-key 9B \
  --tasks arc_challenge mmlu_abstract_algebra mmlu_college_math mmlu_college_cs \
  --router-models "${ROUTER_MODELS[@]}" \
  --router-labels "${ROUTER_LABELS[@]}" \
  --router-decision-thresholds "${SAT_VALUES[@]}" \
  --output-dir "${OUTDIR}/routed_9B_probe_allmean_attn50_a"
