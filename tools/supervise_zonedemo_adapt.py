#!/usr/bin/env python3
"""Supervisor for ZoneDemo B1 adaptation training.

Policy requested 2026-07-28:
- B1 APF is the temporary ZoneDemo default: force_exponent=1.5,
  velocity_k=1.0, dynamic_position_repulsion_requires_closing=false.
- Warm-start from the selected A3 checkpoint at the same scale.
- Train 500k steps with the mixed VorAdj episode replaced by ZoneDemo outer-ring
  intrusion. The pure coverage supplement remains enabled through
  voradj_mixed_coverage.
- Every 50k checkpoint is screened on three tasks: zone mix, old mix, pure
  coverage. After 500k, select the historical best checkpoint and launch formal
  20-rollout/10-GIF diagnostics for the same three tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
ARTIFACT_ROOT = ROOT / "artifacts" / "2026-07-28_zonedemo_b1_adaptation"
SCREEN_ROOT = ARTIFACT_ROOT / "screening"
BEST_ROOT = ARTIFACT_ROOT / "best_20rollout10gif"
STATUS = LOG_DIR / "zonedemo_b1_adaptation_20260728_status.json"
ROLLOUT = ROOT / "tools" / "batch_rollouts_parallel.py"
DEVICE = "cuda:0"
TOTAL_STEPS = 500_000
CHECK_INTERVAL = 50_000
CHECK_EPISODES = 12
CHECK_WORKERS = 3
FORMAL_EPISODES = 20
FORMAL_GIF_COUNT = 10
FORMAL_WORKERS = 4
POLL_SECONDS = 60

LINES: list[dict[str, Any]] = [
    {
        "label": "8p2e2obs",
        "train_config": "configs/experiments/zonedemo_adapt_20260728/zonedemo_b1_adapt_8p2e2obs_500k.yaml",
        "oldmix_config": "configs/experiments/zonedemo_adapt_20260728/zonedemo_b1_oldmix_eval_8p2e2obs.yaml",
        "run_name": "zonedemo_b1_adapt_8p2e2obs_500k_20260728_run1",
        "evaders": 2,
        "seed": 2026072851,
        "zone_max_steps": 500,
        "oldmix_max_steps": 1200,
        "coverage_max_steps": 1200,
    },
    {
        "label": "12p3e3obs",
        "train_config": "configs/experiments/zonedemo_adapt_20260728/zonedemo_b1_adapt_12p3e3obs_500k.yaml",
        "oldmix_config": "configs/experiments/zonedemo_adapt_20260728/zonedemo_b1_oldmix_eval_12p3e3obs.yaml",
        "run_name": "zonedemo_b1_adapt_12p3e3obs_500k_20260728_run1",
        "evaders": 3,
        "seed": 2026072852,
        "zone_max_steps": 500,
        "oldmix_max_steps": 1600,
        "coverage_max_steps": 1600,
    },
]

TASKS = [
    ("zone_mix", "train_config", ["mix"], "zone_max_steps"),
    ("old_mix", "oldmix_config", ["mix"], "oldmix_max_steps"),
    ("pure_coverage", "oldmix_config", ["coverage"], "coverage_max_steps"),
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_status(state: str, **extra: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": now(), "state": state, **extra}
    STATUS.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_dir(line: dict[str, Any]) -> Path:
    cfg = load_yaml(ROOT / line["train_config"])
    return ROOT / cfg.get("output_root", "runs") / str(cfg.get("run_name", line["run_name"]))


def checkpoint_path(line: dict[str, Any], step: int) -> Path:
    return run_dir(line) / "checkpoints" / f"step_{int(step)}.pt"


def final_checkpoint(line: dict[str, Any]) -> Path:
    return run_dir(line) / "checkpoints" / f"final_step_{TOTAL_STEPS}.pt"


def start_training(line: dict[str, Any]) -> subprocess.Popen[Any] | None:
    if final_checkpoint(line).is_file():
        return None
    cfg_path = ROOT / line["train_config"]
    log_path = LOG_DIR / f"train_zonedemo_b1_{line['label']}_20260728.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"[{now()}] launch training {line['label']}\n")
    handle.flush()
    return subprocess.Popen(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg_path)],
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def scalar(summary: dict[str, Any], key: str) -> float:
    try:
        value = float(summary.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if value == value else 0.0


def run_rollout(line: dict[str, Any], checkpoint: Path, step: int, task_name: str, cfg_key: str, scenarios: list[str], max_key: str, formal: bool) -> dict[str, Any]:
    base = BEST_ROOT if formal else SCREEN_ROOT
    stage_dir = base / line["label"] / f"step_{step:06d}"
    out = stage_dir / task_name
    summary = out / "all_summaries.json"
    if summary.is_file():
        return json.loads(summary.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"rollout_zonedemo_b1_{line['label']}_step{step}_{task_name}_{'formal' if formal else 'check'}_20260728.log"
    cmd = [
        sys.executable,
        str(ROLLOUT),
        "--config", str(ROOT / line[cfg_key]),
        "--checkpoint", str(checkpoint),
        "--output-root", str(out),
        "--episodes", str(FORMAL_EPISODES if formal else CHECK_EPISODES),
        "--gif-count", str(FORMAL_GIF_COUNT if formal else 0),
        "--scenarios", *scenarios,
        "--seed", str(int(line["seed"]) + int(step) + (91000 if formal else 0)),
        "--device", DEVICE,
        "--workers", str(FORMAL_WORKERS if formal else CHECK_WORKERS),
        "--max-steps", str(int(line[max_key])),
        "--capture-max-steps", str(min(int(line[max_key]), 700)),
        "--coverage-max-steps", str(int(line[max_key])),
        "--capture-evaders", str(int(line["evaders"])),
        "--max-gif-frames", "1000" if formal else "1",
        "--frame-duration-ms", "100",
    ]
    write_status("rollout_running", label=line["label"], step=step, task=task_name, formal=formal, output_root=rel(out), command=shlex.join(cmd))
    with log_path.open("w", encoding="utf-8") as handle:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"rollout failed label={line['label']} step={step} task={task_name} rc={rc} log={log_path}")
    return json.loads(summary.read_text(encoding="utf-8"))


def evaluate_checkpoint(line: dict[str, Any], checkpoint: Path, step: int) -> dict[str, Any]:
    stage_dir = SCREEN_ROOT / line["label"] / f"step_{step:06d}"
    payload_path = stage_dir / "selection_payload.json"
    if payload_path.is_file():
        return json.loads(payload_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for task_name, cfg_key, scenarios, max_key in TASKS:
        results[task_name] = run_rollout(line, checkpoint, step, task_name, cfg_key, scenarios, max_key, formal=False)
    zone = results["zone_mix"].get("mix", {}) or {}
    old = results["old_mix"].get("mix", {}) or {}
    cov = results["pure_coverage"].get("coverage", {}) or {}
    metrics = {
        "zone_episode_success_rate": scalar(zone, "episode_success_rate"),
        "zone_capture_success_rate": scalar(zone, "capture_success_rate"),
        "zone_coverage_success_rate": scalar(zone, "coverage_success_rate"),
        "zone_breach_rate": scalar(zone, "zone_breach_rate"),
        "zone_collision_rate": scalar(zone, "collision_rate"),
        "zone_avg_steps": scalar(zone, "avg_steps"),
        "old_mix_episode_success_rate": scalar(old, "episode_success_rate"),
        "old_mix_capture_success_rate": scalar(old, "capture_success_rate"),
        "old_mix_coverage_success_rate": scalar(old, "coverage_success_rate"),
        "old_mix_collision_rate": scalar(old, "collision_rate"),
        "old_mix_avg_steps": scalar(old, "avg_steps"),
        "coverage_success_rate": scalar(cov, "coverage_success_rate"),
        "coverage_geometric_rate": scalar(cov, "coverage_geometric_rate"),
        "coverage_settled_rate": scalar(cov, "coverage_settled_rate"),
        "coverage_collision_rate": scalar(cov, "collision_rate"),
        "coverage_avg_steps": scalar(cov, "avg_steps"),
    }
    score_tuple = [
        metrics["zone_episode_success_rate"],
        1.0 - metrics["zone_breach_rate"],
        metrics["old_mix_episode_success_rate"],
        metrics["coverage_settled_rate"],
        metrics["zone_capture_success_rate"],
        metrics["zone_coverage_success_rate"],
        metrics["coverage_geometric_rate"],
        -(metrics["zone_collision_rate"] + metrics["old_mix_collision_rate"] + metrics["coverage_collision_rate"]),
        -metrics["zone_avg_steps"] / max(int(line["zone_max_steps"]), 1),
        int(step),
    ]
    payload = {
        "checked_at": now(),
        "label": line["label"],
        "checkpoint": rel(checkpoint),
        "checkpoint_step": step,
        "metrics": metrics,
        "score_tuple": score_tuple,
        "selection_policy": "Prefer zone mix success, no breach, old mix preservation, pure coverage settled success; speed is only a late tie-breaker.",
        "outputs": {name: rel(stage_dir / name) for name, *_ in TASKS},
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def best_payload(line: dict[str, Any]) -> dict[str, Any] | None:
    root = SCREEN_ROOT / line["label"]
    payloads = []
    if root.is_dir():
        for path in root.glob("step_*/selection_payload.json"):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if not payloads:
        return None
    return max(payloads, key=lambda item: item.get("score_tuple", []))


def formal_marker(line: dict[str, Any]) -> Path:
    return BEST_ROOT / line["label"] / "formal_launched.json"


def launch_formal(line: dict[str, Any], payload: dict[str, Any]) -> None:
    marker = formal_marker(line)
    if marker.is_file():
        return
    step = int(payload["checkpoint_step"])
    checkpoint = ROOT / payload["checkpoint"]
    outputs = {}
    for task_name, cfg_key, scenarios, max_key in TASKS:
        run_rollout(line, checkpoint, step, task_name, cfg_key, scenarios, max_key, formal=True)
        outputs[task_name] = rel(BEST_ROOT / line["label"] / f"step_{step:06d}" / task_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"launched_at": now(), "best_payload": payload, "formal_outputs": outputs}, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-start", action="store_true", help="Do not launch training processes; only monitor existing checkpoints.")
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    procs: dict[str, subprocess.Popen[Any] | None] = {}
    done: set[str] = set()
    if not args.no_start:
        for line in LINES:
            procs[line["label"]] = start_training(line)
    while len(done) < len(LINES):
        status_lines: list[dict[str, Any]] = []
        for line in LINES:
            label = line["label"]
            checked_steps = []
            for step in range(CHECK_INTERVAL, TOTAL_STEPS + 1, CHECK_INTERVAL):
                ckpt = checkpoint_path(line, step)
                if ckpt.is_file():
                    evaluate_checkpoint(line, ckpt, step)
                    checked_steps.append(step)
            final = final_checkpoint(line)
            proc = procs.get(label)
            train_returncode = None if proc is None else proc.poll()
            best = best_payload(line)
            if final.is_file() or (train_returncode is not None and checkpoint_path(line, TOTAL_STEPS).is_file()):
                if best is not None:
                    launch_formal(line, best)
                done.add(label)
            status_lines.append({
                "label": label,
                "run_dir": rel(run_dir(line)),
                "checked_steps": checked_steps,
                "best_step": None if best is None else best.get("checkpoint_step"),
                "best_metrics": None if best is None else best.get("metrics"),
                "training_returncode": train_returncode,
                "formal_launched": formal_marker(line).is_file(),
            })
        write_status("monitoring", lines=status_lines, done=sorted(done))
        if len(done) >= len(LINES):
            break
        time.sleep(POLL_SECONDS)
    write_status("complete", lines=[{"label": line["label"], "best": best_payload(line)} for line in LINES])


if __name__ == "__main__":
    main()
