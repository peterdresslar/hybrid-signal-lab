#!/bin/bash

#SBATCH -G 1
#SBATCH -C a100_80
#SBATCH --mem=80GB
#SBATCH -t 0-03:59:59
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
export ROUTER_PREFIX=router-9B-050-probes

PANEL=(
  constant_2.6
  constant_1.45
  plateau_bal_0.55
  bowl_bal_0.40
)

FAMILIES=(
  embedding_last_token
  final_layer_mean_pool
  all_layers_mean_pool_concat
)

RESIDUALIZATIONS=(
  raw
  length_resid
  attn_resid
)

PCS=(25 50 100)

cd /home/pdressla/workspace/hybrid-signal-lab

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

for family in "${FAMILIES[@]}"; do
  for resid in "${RESIDUALIZATIONS[@]}"; do
    for seq_pca in "${PCS[@]}"; do
      output_dir="${ROUTER_ROOT}/${ROUTER_PREFIX}-${family}-${resid}-pc${seq_pca}"
      echo "[train_probes] family=${family} resid=${resid} seq_pca=${seq_pca} output=${output_dir}"
      uv run -m router.experiments.train_probes \
        --model-key "${MODEL_KEY_QWEN}" \
        --data-dir "${DATADIR}" \
        --profiles "${PANEL[@]}" \
        --feature-set pca+scalar+sequence_pca \
        --sequence-states-dir "${OUTDIR}/${MODEL_KEY_QWEN}/${SEQDIR}" \
        --sequence-family "${family}" \
        --sequence-residualization "${resid}" \
        --n-pca 10 \
        --sequence-n-pca "${seq_pca}" \
        --ridge-reg 1.0 \
        --thresholds 0.0 0.005 0.01 0.02 0.03 0.05 \
        --intervention-mode attention_contribution \
        --output-dir "${output_dir}"
    done
  done
done
