#!/bin/bash

#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH --mem=80GB
#SBATCH -t 0-01:59:59
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
export MPLCONFIGDIR=/scratch/pdressla/.cache/matplotlib

export REPO_ROOT=/home/pdressla/workspace/hybrid-signal-lab
export MODEL_KEY=2B
export MODEL_LABEL="Qwen 2B"
export PROMPT_BATTERY="${REPO_ROOT}/battery/data/battery_4"
export COLLECT_ROOT=/scratch/pdressla/new-runs/040-collect
export TDA_ROOT=/scratch/pdressla/new-runs/040-tda
export SWEEP_ROOT=/scratch/pdressla/sl-runs/022-balanced-attn

export RUN_DIR="${COLLECT_ROOT}/${MODEL_KEY}"
export ANALYSIS_DIR="${RUN_DIR}/analysis"
export SWEEP_RUN_DIR="${SWEEP_ROOT}/${MODEL_KEY}"
export VERBOSE_JSONL="${SWEEP_RUN_DIR}/verbose.jsonl"

FAMILIES=(
  embedding_last_token
  final_layer_last_token
  embedding_mean_pool
  final_layer_mean_pool
  all_layers_last_token_concat
  all_layers_mean_pool_concat
)

cd "${REPO_ROOT}"

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

mkdir -p "${COLLECT_ROOT}" "${TDA_ROOT}" "${MPLCONFIGDIR}"

uv sync

echo "[collect] ${MODEL_KEY}: ${RUN_DIR}"
uv run -m signal_lab.collect_sequences \
  --verbosity 2 \
  --prompt-battery "${PROMPT_BATTERY}" \
  --model-key "${MODEL_KEY}" \
  --output-dir "${RUN_DIR}" \
  --device cuda

echo "[sequence_analyze] ${MODEL_KEY}: ${ANALYSIS_DIR}"
uv run -m signal_lab.sequence_analyze \
  --run-dir "${RUN_DIR}" \
  --output-dir "${ANALYSIS_DIR}"

echo "[sequence_heads] ${MODEL_KEY}"
uv run -m signal_lab.sequence_heads \
  --sequence-analysis-dir "${ANALYSIS_DIR}" \
  --verbose-jsonl "${VERBOSE_JSONL}"

for mode in raw length_resid attn_resid; do
  echo "[sequence_plot_3d] ${MODEL_KEY}: mode=${mode}"
  args=(--analysis-dir "${ANALYSIS_DIR}" --model-label "${MODEL_LABEL}" --mode "${mode}")
  for family in "${FAMILIES[@]}"; do
    args+=(--family "${family}")
  done
  uv run -m signal_lab.sequence_plot_3d "${args[@]}"
done

echo "[tda entropy] ${MODEL_KEY}"
uv run -m signal_lab.tda_analyze \
  --mode entropy \
  --run-dir "${SWEEP_RUN_DIR}" \
  --output-dir "${TDA_ROOT}/qwen2b_entropy" \
  --max-points 400 \
  --pca-dim 24 \
  --maxdim 2

for family in "${FAMILIES[@]}"; do
  echo "[tda sequence] ${MODEL_KEY}: ${family}"
  uv run -m signal_lab.tda_analyze \
    --mode sequence \
    --run-dir "${RUN_DIR}" \
    --family "${family}" \
    --output-dir "${TDA_ROOT}/qwen2b_${family}" \
    --max-points 400 \
    --pca-dim 24 \
    --maxdim 2
done

echo "[done] 040 omnibus complete for ${MODEL_KEY}"
