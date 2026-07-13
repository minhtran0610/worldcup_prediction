#!/usr/bin/env bash
# Wraps `train` with a pre-run backup of the checkpoint it's about to overwrite.
# Auto-promotion is unconditional (no RPS gate) — this backup is the only
# rollback safety net if a run produces a bad checkpoint.
set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/neural_v2.pt}"

if [ -f "$CHECKPOINT_PATH" ]; then
    cp "$CHECKPOINT_PATH" "${CHECKPOINT_PATH}.bak"
    echo "[entrypoint-trainer] Backed up existing checkpoint to $(readlink -f "${CHECKPOINT_PATH}.bak")"
fi

echo "[entrypoint-trainer] Running: train --checkpoint $(readlink -f "$CHECKPOINT_PATH") $*"
exec train --checkpoint "$CHECKPOINT_PATH" "$@"
