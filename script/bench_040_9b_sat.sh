#!/bin/bash

#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH -t 0-12:34:56
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
export OUTDIR=/home/pdressla/workspace/data/sl-runs/040-bench

export ROUTER_MODEL=/home/pdressla/workspace/hybrid-signal-lab/router/router-9B-040-seq/router_model.json
export SAT_VALUES=(0.35 0.45 0.55 0.65)

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

for sat in "${SAT_VALUES[@]}"; do
  sat_slug="${sat/./p}"
  echo "[bench] saturation=${sat}"
  uv run -m bench.run_bench \
    --model-key 9B \
    --tasks arc_challenge mmlu_abstract_algebra mmlu_college_math mmlu_college_cs \
    --router-model "${ROUTER_MODEL}" \
    --router-decision-threshold "${sat}" \
    --output-dir "${OUTDIR}/routed_9B_sat_${sat_slug}"
done
