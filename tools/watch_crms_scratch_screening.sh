#!/usr/bin/env bash
set -euo pipefail

# Lightweight scratch screening for CR-MS (ring_importance_ms_v0) + CE coverage.
# It waits for stable 100k-step checkpoints, then writes one independent
# batch_rollouts output tree per checkpoint. Each run stores capture/mix/coverage
# summaries under a step-specific directory, so repeated launches naturally skip
# completed checkpoints and do not overwrite unrelated results.

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 CONFIG RUN_DIR OUTPUT_ROOT [SEED_BASE] [DEVICE]" >&2
  echo "  env overrides: EPISODES GIF_COUNT STEPS WAIT_SECONDS" >&2
  exit 2
fi

CONFIG="$1"
RUN_DIR="$2"
OUTPUT_ROOT="$3"
SEED_BASE="${4:-2026073300}"
DEVICE="${5:-cuda:0}"
EPISODES="${EPISODES:-10}"
GIF_COUNT="${GIF_COUNT:-0}"
STEPS="${STEPS:-100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"

mkdir -p "$OUTPUT_ROOT"

cat > "$OUTPUT_ROOT/README_screening.txt" <<EOF
CR-MS (ring_importance_ms_v0) + CE coverage scratch screening

config: $CONFIG
run_dir: $RUN_DIR
device: $DEVICE
episodes_per_scenario: $EPISODES
gif_count: $GIF_COUNT
scenarios: capture mix coverage
rollout_limits: max_steps=800, capture_max_steps=600, coverage_max_steps=500, max_gif_frames=500
notes: batch summaries include CE coverage_success_rate (strict) plus coverage_cv015_rate / coverage_cv015_best_rate for loose cv<0.15 tracking.
EOF

for STEP in $STEPS; do
  CKPT="$RUN_DIR/checkpoints/step_${STEP}.pt"
  STEP_OUT="$OUTPUT_ROOT/step_${STEP}"
  DONE_MARKER="$STEP_OUT/DONE"
  if [[ -f "$DONE_MARKER" ]]; then
    echo "[skip] step_${STEP} already screened: $STEP_OUT"
    continue
  fi

  while [[ ! -f "$CKPT" ]]; do
    echo "[wait] $(date +%F\ %T) waiting for $CKPT"
    sleep "$WAIT_SECONDS"
  done

  mkdir -p "$STEP_OUT"
  echo "[run] $(date +%F\ %T) screening step_${STEP} -> $STEP_OUT"
  PYTHONPATH=src python3 tools/batch_rollouts.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output-root "$STEP_OUT" \
    --episodes "$EPISODES" \
    --gif-count "$GIF_COUNT" \
    --scenarios capture mix coverage \
    --seed "$((SEED_BASE + STEP / 1000))" \
    --device "$DEVICE" \
    --max-steps 800 \
    --capture-max-steps 600 \
    --coverage-max-steps 500 \
    --max-gif-frames 500
  date +%F\ %T > "$DONE_MARKER"
done

echo "[done] $(date +%F\ %T) all requested screening checkpoints finished."
