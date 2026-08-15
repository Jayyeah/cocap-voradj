#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT/artifacts/2026-08-16_capture_debug"
LOG="$OUTPUT_ROOT/capture_debug_cpu_queue.log"

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT"
nohup env \
  CUDA_VISIBLE_DEVICES= \
  PYTHONPATH="$ROOT/src:$ROOT" \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 \
  nice -n 10 python3 tools/run_capture_debug_cpu_queue.py \
  >>"$LOG" 2>&1 </dev/null &
queue_pid=$!
echo "capture-debug CPU queue PID=$queue_pid log=$LOG"
