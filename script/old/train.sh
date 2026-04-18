#!/bin/bash

#SBATCH -G a 1
#SBATCH -C a100_80
#SBATCH -t 0-01:00:00
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
export SEQDIR=states
export DATADIR=/scratch/pdressla/sl-runs/022-balanced-attn

export MODEL_KEY_QWEN=9B
export MODEL_KEY_OLMO=OLMO

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

uv run -m router.experiments.train_router \
    --model-key ${MODEL_KEY_QWEN} \
    --data-dir ${DATADIR} \
    --profiles constant_2.6 constant_1.45 plateau_bal_0.55 bowl_bal_0.40 \
    --feature-set pca+scalar+sequence_pca \
    --sequence-states-dir ${OUTDIR}/${MODEL_KEY_QWEN}/${SEQDIR} \
    --sequence-family embedding_last_token \
    --sequence-n-pca 10 \
    --intervention-mode attention_contribution \
    --output-dir router/router-9B-040-seq
