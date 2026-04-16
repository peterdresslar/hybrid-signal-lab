#!/bin/bash

#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH -t 0-03:59:59
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
export OUTDIR=/scratch/pdressla/new-runs/040-collect

export MODEL_KEY_QWEN=9B
export MODEL_KEY_OLMO=OLMO

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname


uv run -m signal_lab.sequence_analyze \
    --run-dir ${OUTDIR}/${MODEL_KEY_QWEN} \
    --output-dir ${OUTDIR}/${MODEL_KEY_QWEN}/analysis

uv run -m signal_lab.sequence_analyze \
    --run-dir ${OUTDIR}/${MODEL_KEY_OLMO} \
    --output-dir ${OUTDIR}/${MODEL_KEY_OLMO}/analysis