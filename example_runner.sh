#!/usr/bin/env bash
set -eu pipefail 

# === env vars ===
export HF_TOKEN=[token]

# === cache dirs on persistent storage ===
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export UV_CACHE_DIR=/workspace/.cache/uv
export XDG_CACHE_HOME=/workspace/.cache

# === uncomment these for cluster run ===
export HSL_DEVICE_MAP=balanced
export HSL_MAX_MEMORY_JSON='{0:"80GiB",1:"80GiB",2:"80GiB",3:"80GiB",4:"80GiB",5:"80GiB",6:"80GiB",7:"80GiB",cpu:"1000GiB"}'

 uv run -m signal_lab.sweep \
  --cartridge uniform_check \
  --prompt-battery battery/data/battery_3 \
  --model-key 397B \
  --device cuda \
  --out-dir ~/workspace/data/397_{timestamp} \
  --verbose