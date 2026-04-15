#!/bin/bash

#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH -t 0-11:59:59
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

export REPO_ROOT=/home/pdressla/workspace/hybrid-signal-lab
export COLLECT_ROOT=/scratch/pdressla/new-runs/040-collect
export TDA_ROOT=/scratch/pdressla/new-runs/040-tda

export MODEL_KEY_QWEN=9B
export MODEL_KEY_OLMO=OLMO
export PROMPT_BATTERY=/home/pdressla/workspace/hybrid-signal-lab/battery/data/battery_4

cd "${REPO_ROOT}"

set -a
source .env.slurm
set +a

source .venv/bin/activate

pwd
hostname

# New dependencies for sequence/TDA passes.
uv sync

mkdir -p "${COLLECT_ROOT}"
mkdir -p "${TDA_ROOT}"

run_collect() {
    local model_key="$1"
    local run_dir="${COLLECT_ROOT}/${model_key}"

    if [[ -f "${run_dir}/manifest.json" && -f "${run_dir}/records.jsonl" ]]; then
        echo "[collect] ${model_key}: existing sequence collection found at ${run_dir}; skipping collection"
        return
    fi

    echo "[collect] ${model_key}: starting baseline sequence collection"
    uv run -m signal_lab.collect_sequences \
        --verbosity 2 \
        --prompt-battery "${PROMPT_BATTERY}" \
        --model-key "${model_key}" \
        --output-dir "${run_dir}" \
        --device cuda
}

run_sequence_analysis() {
    local model_key="$1"
    local run_dir="${COLLECT_ROOT}/${model_key}"
    local analysis_dir="${run_dir}/analysis"

    echo "[sequence_analyze] ${model_key}: ${analysis_dir}"
    uv run -m signal_lab.sequence_analyze \
        --run-dir "${run_dir}" \
        --output-dir "${analysis_dir}"
}

run_entropy_tda() {
    local model_key="$1"
    local sweep_dir
    local out_dir

    if [[ "${model_key}" == "${MODEL_KEY_QWEN}" ]]; then
        sweep_dir="data/022-balanced-attention-hybrid/9B"
        out_dir="${TDA_ROOT}/qwen9b_entropy"
    else
        sweep_dir="data/022-balanced-attention-hybrid/OLMO"
        out_dir="${TDA_ROOT}/olmo_entropy"
    fi

    echo "[tda entropy] ${model_key}: ${out_dir}"
    uv run -m signal_lab.tda_analyze \
        --mode entropy \
        --run-dir "${sweep_dir}" \
        --output-dir "${out_dir}" \
        --max-points 400 \
        --pca-dim 24 \
        --maxdim 2
}

run_sequence_tda_family() {
    local model_key="$1"
    local family="$2"
    local model_slug

    if [[ "${model_key}" == "${MODEL_KEY_QWEN}" ]]; then
        model_slug="qwen9b"
    else
        model_slug="olmo"
    fi

    echo "[tda sequence] ${model_key} ${family}"
    uv run -m signal_lab.tda_analyze \
        --mode sequence \
        --run-dir "${COLLECT_ROOT}/${model_key}" \
        --family "${family}" \
        --output-dir "${TDA_ROOT}/${model_slug}_${family}" \
        --max-points 400 \
        --pca-dim 24 \
        --maxdim 2
}

run_all_for_model() {
    local model_key="$1"

    run_collect "${model_key}"
    run_sequence_analysis "${model_key}"
    run_entropy_tda "${model_key}"

    run_sequence_tda_family "${model_key}" "final_layer_last_token"
    run_sequence_tda_family "${model_key}" "all_layers_last_token_concat"
    run_sequence_tda_family "${model_key}" "final_layer_mean_pool"
    run_sequence_tda_family "${model_key}" "embedding_last_token"
    run_sequence_tda_family "${model_key}" "all_layers_mean_pool_concat"
}

run_all_for_model "${MODEL_KEY_QWEN}"
run_all_for_model "${MODEL_KEY_OLMO}"

echo "[done] battery 4 deep analysis bundle complete"
