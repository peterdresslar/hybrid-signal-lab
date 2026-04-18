#!/bin/bash

#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 0-07:59:59
#SBATCH -p public
#SBATCH -q public
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="pdressla@asu.edu"
#SBATCH --export=NONE

set -euo pipefail

export HF_HOME=/scratch/pdressla/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/pdressla/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/pdressla/.cache/huggingface/transformers
export UV_CACHE_DIR=/scratch/pdressla/.cache/uv
export XDG_CACHE_HOME=/scratch/pdressla/.cache

export DATADIR=/scratch/pdressla/sl-runs/022-balanced-attn
export OUTDIR=/scratch/pdressla/new-runs/040-collect
export MODEL_KEY_QWEN=9B
export SEQDIR=states
export ROUTER_ROOT=router
export ROUTER_PREFIX=router-9B-040-seq

PANEL=(
  constant_2.6
  constant_1.45
  plateau_bal_0.55
  bowl_bal_0.40
)

FAMILIES=(
  embedding_last_token
  final_layer_last_token
  embedding_mean_pool
  final_layer_mean_pool
  all_layers_last_token_concat
  all_layers_mean_pool_concat
)

RESIDUALIZATIONS=(
  length_resid
  attn_resid
)

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

for resid in "${RESIDUALIZATIONS[@]}"; do
  for family in "${FAMILIES[@]}"; do
    output_dir="${ROUTER_ROOT}/${ROUTER_PREFIX}-${family}-${resid}"
    echo "[train_router] residualization=${resid} family=${family} output=${output_dir}"
    uv run -m router.experiments.train_router \
      --model-key "${MODEL_KEY_QWEN}" \
      --data-dir "${DATADIR}" \
      --profiles "${PANEL[@]}" \
      --feature-set pca+scalar+sequence_pca \
      --sequence-states-dir "${OUTDIR}/${MODEL_KEY_QWEN}/${SEQDIR}" \
      --sequence-family "${family}" \
      --sequence-residualization "${resid}" \
      --sequence-n-pca 10 \
      --intervention-mode attention_contribution \
      --output-dir "${output_dir}"
  done
done
