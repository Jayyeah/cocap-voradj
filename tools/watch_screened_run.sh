#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 9 ]]; then
  echo "usage: $0 CONFIG RUN_DIR OUTPUT_ROOT SEED_BASE DEVICE TOTAL_STEPS CAPTURE_EVADERS COVERAGE_MAX_STEPS MIX_MAX_STEPS" >&2
  echo "  env overrides: EPISODES WORKERS CHECKPOINT_INTERVAL WAIT_SECONDS" >&2
  exit 2
fi

CONFIG="$1"
RUN_DIR="$2"
OUTPUT_ROOT="$3"
SEED_BASE="$4"
DEVICE="$5"
TOTAL_STEPS="$6"
CAPTURE_EVADERS="$7"
COVERAGE_MAX_STEPS="$8"
MIX_MAX_STEPS="$9"
EPISODES="${EPISODES:-10}"
WORKERS="${WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100000}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
MAX_NUMPY_SEED=4294967295

if (( SEED_BASE < 0 || SEED_BASE + TOTAL_STEPS / 1000 + EPISODES > MAX_NUMPY_SEED )); then
  echo "SEED_BASE and generated rollout seeds must stay within NumPy RandomState range 0..4294967295" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

cat > "$OUTPUT_ROOT/README_screening.txt" <<EOF
Independent deterministic checkpoint screening

config: $CONFIG
run_dir: $RUN_DIR
device: $DEVICE
episodes_per_scenario: $EPISODES
workers: $WORKERS
scenarios: capture coverage mix
checkpoint_interval: $CHECKPOINT_INTERVAL
total_steps: $TOTAL_STEPS
rollout_limits: capture=1000, coverage=$COVERAGE_MAX_STEPS, mix=$MIX_MAX_STEPS
policy: epsilon=0 with fixed midpoint IQN quantiles
EOF

STEP="$CHECKPOINT_INTERVAL"
while [[ "$STEP" -le "$TOTAL_STEPS" ]]; do
  CKPT="$RUN_DIR/checkpoints/step_${STEP}.pt"
  STEP_OUT="$OUTPUT_ROOT/step_${STEP}"
  DONE_MARKER="$STEP_OUT/DONE"
  if [[ -f "$DONE_MARKER" && -f "$STEP_OUT/all_summaries.json" ]]; then
    echo "[skip] step_${STEP} already screened: $STEP_OUT"
    STEP="$((STEP + CHECKPOINT_INTERVAL))"
    continue
  fi

  while [[ ! -f "$CKPT" ]]; do
    echo "[wait] $(date '+%F %T') waiting for $CKPT"
    sleep "$WAIT_SECONDS"
  done

  mkdir -p "$STEP_OUT"
  echo "[run] $(date '+%F %T') screening step_${STEP} -> $STEP_OUT"
  PYTHONPATH=src python3 tools/batch_rollouts_parallel.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output-root "$STEP_OUT" \
    --episodes "$EPISODES" \
    --gif-count 0 \
    --scenarios capture coverage mix \
    --seed "$((SEED_BASE + STEP / 1000))" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --max-steps "$MIX_MAX_STEPS" \
    --capture-max-steps 1000 \
    --coverage-max-steps "$COVERAGE_MAX_STEPS" \
    --capture-evaders "$CAPTURE_EVADERS" \
    --max-gif-frames 1
  date '+%F %T' > "$DONE_MARKER"
  STEP="$((STEP + CHECKPOINT_INTERVAL))"
done

echo "[done] $(date '+%F %T') all screening checkpoints finished."
