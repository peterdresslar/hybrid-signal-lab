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

set -eu

export HF_HOME=/scratch/pdressla/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/pdressla/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/pdressla/.cache/huggingface/transformers
export UV_CACHE_DIR=/scratch/pdressla/.cache/uv
export XDG_CACHE_HOME=/scratch/pdressla/.cache
export OUTDIR=/home/pdressla/workspace/data/sl-runs/030-bench/

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

uv run -m bench.run_bench \
  --model-key OLMO \
  --tasks arc_challenge mmlu_abstract_algebra mmlu_college_math mmlu_college_cs \
  --router-model /home/pdressla/workspace/hybrid-signal-lab/router/router-OLMO-030/router_model.json \
  --output-dir "${OUTDIR}/routed_OLMO"
