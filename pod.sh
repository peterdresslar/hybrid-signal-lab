#!/usr/bin/env bash
set -euo pipefail

# ── env vars (set these in RunPod pod config or export before running) ──
# HF_TOKEN=hf_...                     # Hugging Face token (required)
# GH_TOKEN=ghp_...                    # GitHub PAT for private repo clone
# COLONY_DEVICE=cuda                  # optional, auto-detected if unset

# ── cache dirs on persistent storage ──
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export UV_CACHE_DIR=/workspace/.cache/uv
export XDG_CACHE_HOME=/workspace/.cache
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

# ── install uv ──
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ── install gh ──
mkdir -p /etc/apt/keyrings
wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  > /etc/apt/keyrings/githubcli-archive-keyring.gpg
chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  > /etc/apt/sources.list.d/github-cli.list
apt-get update -qq && apt-get install -y -qq gh
apt-get install nano

# ── clone repo ──
printf '%s' "$GH_TOKEN" | gh auth login --with-token
gh repo clone peterdresslar/cas-capstone-dresslar /workspace/cas-capstone-dresslar
cd /workspace/cas-capstone-dresslar

# ── set up project ──
uv venv .venv
source .venv/bin/activate
uv sync

echo "Ready. Run:"
echo "  cd /workspace/cas-capstone-dresslar && source .venv/bin/activate"
echo "  uv run python -m colony.sweep --cartridge test_cartridge --model-key OLMO --verbose"
