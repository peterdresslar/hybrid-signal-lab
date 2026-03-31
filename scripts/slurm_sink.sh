#!/bin/bash

#SBATCH -G a100:1
#SBATCH -t 0-04:00:00   # time in d-hh:mm:ss
#SBATCH -p public       # htc
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="pdressla@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#SBATCH --mem-per-gpu=80G

# === cache dirs on persistent storage ===
export HF_HOME=/scratch/pdressla/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/pdressla/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/pdressla/.cache/huggingface/transformers
export UV_CACHE_DIR=/scratch/pdressla/.cache/uv
export XDG_CACHE_HOME=/scratch/pdressla/.cache
export OUTDIR=/home/pdressla/workspace/data/sl-runs/b4_ks1/

cd ~/Workspace/hybrid-signal-lab
source .env.slurm
source .venv/bin/activate

uv run -m signal_lab.sweep \
  --cartridge kitchen_sink \
  --prompt-battery battery/data/battery_4 \
  --model-key 35B \
  --device cuda \
  --out-dir ${OUTDIR}/35B \
  --verbose \
&& uv run -m signal_lab.run_analyze \
  --input-dir ${OUTDIR}/35B \
  --intervention-folders