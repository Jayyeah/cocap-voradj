#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 CONFIG RUN_DIR OUTPUT_ROOT [SEED_BASE] [DEVICE]" >&2
  echo "  env overrides: EPISODES GIF_COUNT STEPS WAIT_SECONDS" >&2
  exit 2
fi

CONFIG="$1"
RUN_DIR="$2"
OUTPUT_ROOT="$3"
SEED_BASE="${4:-2026080200}"
DEVICE="${5:-cuda:1}"
EPISODES="${EPISODES:-10}"
GIF_COUNT="${GIF_COUNT:-0}"
STEPS="${STEPS:-100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000 1100000 1200000 1300000 1400000 1500000 1600000 1700000 1800000 1900000 2000000}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"

mkdir -p "$OUTPUT_ROOT"

cat > "$OUTPUT_ROOT/README_screening.txt" <<EOF
CR-MS second batch: ring_importance_ms_v0 direct capture + VCT-LS support approach-only + latest CE

config: $CONFIG
run_dir: $RUN_DIR
device: $DEVICE
episodes_per_scenario: $EPISODES
gif_count: $GIF_COUNT
scenarios: capture coverage mix
checkpoints: every 100k through the configured 2M scratch run
rollout_limits: capture=1000, coverage=1200, mix/max=2200
support reward: 0.5 legacy approach-only progress + 0.5 CE coverage for one-hop support agents
selection: screening is diagnostic; stage1 always trains to 2M before historical-best selection
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
    echo "[wait] $(date '+%F %T') waiting for $CKPT"
    sleep "$WAIT_SECONDS"
  done

  mkdir -p "$STEP_OUT"
  echo "[run] $(date '+%F %T') screening step_${STEP} -> $STEP_OUT"
  PYTHONPATH=src python3 tools/batch_rollouts.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output-root "$STEP_OUT" \
    --episodes "$EPISODES" \
    --gif-count "$GIF_COUNT" \
    --scenarios capture coverage mix \
    --seed "$((SEED_BASE + STEP / 1000))" \
    --device "$DEVICE" \
    --max-steps 2200 \
    --capture-max-steps 1000 \
    --coverage-max-steps 1200 \
    --capture-evaders 1 \
    --max-gif-frames 1
  date '+%F %T' > "$DONE_MARKER"
done

echo "[done] $(date '+%F %T') all requested screening checkpoints finished."
