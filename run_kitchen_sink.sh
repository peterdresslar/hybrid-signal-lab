#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="early-late-run.txt"
CARTRIDGE="kitchen_sink"
MODELS=("0_8B" "2B" "9B" "OLMO")
RUN_NAME="${RUN_NAME:-kitchen_sink_$(date +%Y%m%d_%H%M%S)}"

# Start fresh each time.
: > "$LOG_FILE"

for model in "${MODELS[@]}"; do
  echo "============================================================" | tee -a "$LOG_FILE"
  echo "Starting cartridge=$CARTRIDGE model=$model at $(date)" | tee -a "$LOG_FILE"
  echo "============================================================" | tee -a "$LOG_FILE"

  uv run python -m signal_lab.sweep \
    --cartridge "$CARTRIDGE" \
    --run-name "$RUN_NAME" \
    --model-key "$model" \
    --verbose \
    2>&1 | tee -a "$LOG_FILE"

  echo "" | tee -a "$LOG_FILE"
  echo "Finished model=$model at $(date)" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done

echo "All runs complete. Combined log: $LOG_FILE" | tee -a "$LOG_FILE"
