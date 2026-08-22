#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <maddpg|maadpg> <cuda:0|cuda:1>" >&2
  exit 2
fi

variant=$1
device=$2
if [[ "$variant" != "maddpg" && "$variant" != "maadpg" ]]; then
  echo "invalid variant: $variant" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
artifact_root="$repo_dir/artifacts/2026-08-23_maadpg_reproduction/formal"
ledger="$artifact_root/runs.jsonl"

cd "$repo_dir"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked worktree changes detected; formal queue requires a frozen commit" >&2
  exit 3
fi

mkdir -p "$artifact_root"
exec 9>"$artifact_root/.${variant}.lock"
if ! flock -n 9; then
  echo "formal queue already holds lock for $variant" >&2
  exit 4
fi
for seed in 2026082301 2026082302 2026082303; do
  run_id="formal1500ep-${variant}-s${seed}"
  run_dir="$artifact_root/${variant}_seed${seed}"
  mkdir -p "$run_dir"
  resume_args=()
  if [[ -f "$run_dir/rolling-full.pt" ]]; then
    resume_args=(--resume "$run_dir/rolling-full.pt")
  fi
  PYTHONPATH=src python3 -m maadpg_reproduction train \
    --variant "$variant" \
    --run-dir "$run_dir" \
    --run-id "$run_id" \
    --seed "$seed" \
    --device "$device" \
    --step-budget 450000 \
    --episode-budget 1500 \
    --checkpoint-every 25000 \
    --model-every 50000 \
    --diagnostic-every 1000 \
    --ledger "$ledger" \
    "${resume_args[@]}" \
    >> "$run_dir/stdout.log" 2>&1
done
