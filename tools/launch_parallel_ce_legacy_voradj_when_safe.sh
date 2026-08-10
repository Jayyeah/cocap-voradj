#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/yjq/rl/CoCap1/cocap-voradj-speedopt"
ARTIFACT_ROOT="$ROOT_DIR/artifacts/2026-08-10_parallel_ce_legacy_voradj_200k"
QUEUE_LOG="$ARTIFACT_ROOT/launcher.log"
OLD_A_PID=1326505
OLD_C_PID=1326499
OLD_A_TAG=stage4a_speedopt_200k_20260809
OLD_C_TAG=stage4c_speedopt_200k_20260809
MIN_FREE_KIB=$((24 * 1024 * 1024))
MAX_GPU0_TEMP=82
MAX_GPU1_TEMP=85

mkdir -p "$ARTIFACT_ROOT/pure_ce" "$ARTIFACT_ROOT/legacy_voradj"

log() {
  printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >> "$QUEUE_LOG"
}

old_line_alive() {
  local pid="$1"
  local tag="$2"
  local command_line
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$command_line" == *"$tag"* ]]
}

gpu_temperature() {
  local index="$1"
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits \
    | sed -n "$((index + 1))p" \
    | tr -d '[:space:]'
}

log "queue armed; waiting for the protected Stage4A/4C processes"
while old_line_alive "$OLD_A_PID" "$OLD_A_TAG" || old_line_alive "$OLD_C_PID" "$OLD_C_TAG"; do
  sleep 60
done
log "protected Stage4A/4C processes have exited"

while true; do
  free_kib="$(df -Pk "$ROOT_DIR" | awk 'NR == 2 {print $4}')"
  gpu0_temp="$(gpu_temperature 0)"
  gpu1_temp="$(gpu_temperature 1)"
  if [[ "$free_kib" -ge "$MIN_FREE_KIB" \
        && "$gpu0_temp" -le "$MAX_GPU0_TEMP" \
        && "$gpu1_temp" -le "$MAX_GPU1_TEMP" ]]; then
    break
  fi
  log "resource gate waiting: free_kib=$free_kib gpu0=${gpu0_temp}C gpu1=${gpu1_temp}C"
  sleep 60
done

PURE_SESSION=parallel_pure_ce_200k
LEGACY_SESSION=parallel_legacy_voradj_200k

if tmux has-session -t "$PURE_SESSION" 2>/dev/null; then
  log "tmux $PURE_SESSION already exists; refusing a duplicate launch"
else
  tmux new-session -d -s "$PURE_SESSION" -c "$ROOT_DIR" \
    "exec nice -n 10 env PYTHONPATH=src:. python3 tools/run_continuous_ctde_training.py --config configs/experiments/parallel_ce_legacy_voradj_20260809/pure_ce_4p0e1obs_200k_aw.yaml --seed 2026080901 --device cuda:0 --total-steps 200000 --screen-episodes 20 --diagnostic-eval-episodes 4 --tag pure_ce_4p0e1obs_200k_aw_20260810 --artifact-root artifacts/2026-08-10_parallel_ce_legacy_voradj_200k/pure_ce >> artifacts/2026-08-10_parallel_ce_legacy_voradj_200k/pure_ce/train.log 2>&1"
  log "launched $PURE_SESSION on cuda:0"
fi

if tmux has-session -t "$LEGACY_SESSION" 2>/dev/null; then
  log "tmux $LEGACY_SESSION already exists; refusing a duplicate launch"
else
  tmux new-session -d -s "$LEGACY_SESSION" -c "$ROOT_DIR" \
    "exec nice -n 10 env PYTHONPATH=src:. python3 tools/run_continuous_ctde_training.py --config configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_oldmix_4p1e1obs_200k_aw.yaml --seed 2026080902 --device cuda:1 --total-steps 200000 --screen-episodes 20 --diagnostic-eval-episodes 4 --tag legacy_voradj_oldmix_4p1e1obs_200k_aw_20260810 --artifact-root artifacts/2026-08-10_parallel_ce_legacy_voradj_200k/legacy_voradj >> artifacts/2026-08-10_parallel_ce_legacy_voradj_200k/legacy_voradj/train.log 2>&1"
  log "launched $LEGACY_SESSION on cuda:1"
fi

sleep 20
for session in "$PURE_SESSION" "$LEGACY_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    log "startup check PASS: $session is alive"
  else
    log "startup check FAIL: $session exited; inspect its train.log"
  fi
done
