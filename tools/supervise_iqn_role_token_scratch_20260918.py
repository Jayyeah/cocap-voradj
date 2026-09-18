#!/usr/bin/env python3
"""Detached milestone supervisor for the 200k IQN ROLE scratch control."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/iqn_role_token_scratch_20260918/role_token.yaml"
RUN_ROOT = ROOT / "artifacts/2026-09-18_iqn_role_token_scratch"
RUN_NAME = "iqn_role_token_scratch_20260918"
RUN_DIR = RUN_ROOT / RUN_NAME
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
# The current IQN trainer resolves the configured full_resume_path relative to
# the project root, while ordinary milestone checkpoints live below run_dir.
RESUME = ROOT / "artifacts/2026-09-18_iqn_role_token_scratch/checkpoints/resume_latest.pt"
STATUS = RUN_ROOT / "status.json"
TREND = RUN_ROOT / "trend_ledger.jsonl"
SUPERVISOR_LOG = RUN_ROOT / "supervisor.log"
MILESTONES = (25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def read_status() -> dict[str, Any]:
    if not STATUS.is_file():
        return {}
    return json.loads(STATUS.read_text(encoding="utf-8"))


def read_last_metric() -> dict[str, Any] | None:
    path = RUN_DIR / "metrics.jsonl"
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def current_step_from_status() -> int:
    status = read_status()
    return int(status.get("current_step", 0))


def update_status(payload: dict[str, Any]) -> dict[str, Any]:
    current = read_status()
    current.update(payload)
    current["last_heartbeat"] = now_iso()
    current["last_metric"] = read_last_metric()
    write_json(STATUS, current)
    return current


def train_command(target: int, device: str, resume: Path | None) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(CONFIG),
        "--total-timesteps",
        str(target),
        "--run-name",
        RUN_NAME,
        "--device",
        device,
    ]
    if resume is not None:
        command.extend(["--resume-path", str(resume)])
    return command


def eval_command(checkpoint: Path, output: Path, episodes: int, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools/evaluate_iqn_role_token_formal_20260918.py"),
        "--config",
        str(CONFIG),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(output),
        "--episodes",
        str(episodes),
        "--seed-base",
        "2026092801",
        "--device",
        device,
    ]


def run_child(command: list[str], log_path: Path, target: int, stage: str, device: str) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now_iso()}] command={' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        update_status({
            "stage": stage,
            "target_step": int(target),
            "device": device,
            "command": command,
            "child_pid": int(process.pid),
            "child_log": str(log_path),
            "process_started_at": now_iso(),
        })
        while process.poll() is None:
            update_status({"child_pid": int(process.pid), "target_step": int(target), "stage": stage})
            time.sleep(30)
        return_code = int(process.wait())
        elapsed = time.monotonic() - started
        log.write(f"[{now_iso()}] return_code={return_code} elapsed_seconds={elapsed:.3f}\n")
        log.flush()
    if return_code != 0:
        update_status({"stage": "failed", "child_pid": None, "error_return_code": return_code})
        raise RuntimeError(f"child command failed with return code {return_code}: {' '.join(command)}")
    update_status({"child_pid": None, "last_child_elapsed_seconds": elapsed})
    return elapsed


def _eta_payload(current_step: int, stable_steps_per_second: float | None) -> dict[str, Any]:
    if not stable_steps_per_second or stable_steps_per_second <= 0:
        return {"steps_per_second": None, "to_milestones": {}, "estimated_completion": None}
    now = time.time()
    estimates = {}
    for milestone in MILESTONES:
        remaining = max(0, milestone - current_step)
        estimates[str(milestone)] = {
            "remaining_steps": remaining,
            "eta_seconds": remaining / stable_steps_per_second,
            "estimated_local_time": dt.datetime.fromtimestamp(now + remaining / stable_steps_per_second).astimezone().isoformat(timespec="seconds"),
        }
    return {"steps_per_second": stable_steps_per_second, "to_milestones": estimates, "estimated_completion": estimates.get("200000", {}).get("estimated_local_time")}


def supervise(device: str, episodes: int, preflight: Path | None) -> dict[str, Any]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if preflight is not None:
        payload = json.loads(preflight.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"preflight is not PASS: {preflight}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    started_at = read_status().get("run_started_at") or now_iso()
    status = update_status({
        "schema": "iqn-role-token-scratch-supervisor-v1",
        "status": "running",
        "branch": branch,
        "head": head,
        "worktree": str(ROOT),
        "config": str(CONFIG),
        "run_dir": str(RUN_DIR),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "gpu": device,
        "tmux_session": os.environ.get("TMUX", "detached-supervisor-required"),
        "supervisor_pid": os.getpid(),
        "run_started_at": started_at,
        "automatic_stop_target": 200000,
        "checkpoint_milestones": list(MILESTONES),
        "formal_eval_episodes_per_scene": int(episodes),
    })
    write_json(
        RUN_ROOT / "launch.json",
        {
            "schema": "iqn-role-token-scratch-launch-v1",
            "status": "running",
            "branch": branch,
            "head": head,
            "worktree": str(ROOT),
            "config": str(CONFIG),
            "gpu": device,
            "supervisor_pid": os.getpid(),
            "tmux_session": os.environ.get("TMUX", "detached-supervisor-required"),
            "automatic_stop_target": 200000,
            "checkpoint_milestones": list(MILESTONES),
            "formal_eval_episodes_per_scene": int(episodes),
            "resume_checkpoint": str(RESUME),
            "status_path": str(STATUS),
            "trend_ledger": str(TREND),
        },
    )
    previous_step = int(status.get("current_step", 0))
    throughput_windows: list[float] = list(status.get("throughput_windows", []))
    train_elapsed_total = float(status.get("train_elapsed_seconds", 0.0))

    for target in MILESTONES:
        if target <= previous_step:
            continue
        checkpoint = CHECKPOINT_DIR / f"step_{target}.pt"
        evaluation = RUN_ROOT / f"eval_step_{target:06d}.json"
        if not checkpoint.is_file():
            resume = RESUME if previous_step > 0 else None
            train_log = RUN_ROOT / f"train_to_{target:06d}.log"
            elapsed = run_child(train_command(target, device, resume), train_log, target, "training", device)
            train_elapsed_total += elapsed
            steps_delta = target - previous_step
            window_rate = steps_delta / max(elapsed, 1e-9)
            throughput_windows.append(float(window_rate))
            throughput_windows = throughput_windows[-5:]
            stable_rate = float(np_median(throughput_windows))
            if not checkpoint.is_file() or not RESUME.is_file():
                raise RuntimeError(f"missing checkpoint or full resume after step {target}")
            previous_step = target
            update_status({
                "status": "evaluating",
                "current_step": target,
                "checkpoint": str(checkpoint),
                "resume_checkpoint": str(RESUME),
                "throughput_windows_steps_per_second": throughput_windows,
                "throughput_steps_per_second": stable_rate,
                "train_elapsed_seconds": train_elapsed_total,
                "eta": _eta_payload(target, stable_rate),
            })
            append_jsonl(TREND, {
                "schema": "iqn-role-token-trend-v1",
                "kind": "training_milestone",
                "step": target,
                "checkpoint": str(checkpoint),
                "resume_checkpoint": str(RESUME),
                "throughput_steps_per_second": stable_rate,
                "window_steps_per_second": window_rate,
                "train_elapsed_seconds": train_elapsed_total,
                "timestamp": now_iso(),
                "last_metric": read_last_metric(),
            })
        else:
            previous_step = target

        if not evaluation.is_file():
            eval_log = RUN_ROOT / f"eval_to_{target:06d}.log"
            run_child(eval_command(checkpoint, evaluation, episodes, device), eval_log, target, "formal_eval", device)
        eval_payload = json.loads(evaluation.read_text(encoding="utf-8"))
        if eval_payload.get("status") != "complete":
            raise RuntimeError(f"formal evaluation is incomplete at step {target}")
        stable_rate = float(np_median(throughput_windows)) if throughput_windows else None
        append_jsonl(TREND, {
            "schema": "iqn-role-token-trend-v1",
            "kind": "formal_evaluation",
            "step": target,
            "evaluation": str(evaluation),
            "summary": eval_payload.get("summary"),
            "action_histogram": eval_payload.get("action_histogram"),
            "throughput_steps_per_second": stable_rate,
            "eta": _eta_payload(target, stable_rate),
            "timestamp": now_iso(),
        })
        update_status({
            "status": "running" if target < MILESTONES[-1] else "complete",
            "current_step": target,
            "checkpoint": str(checkpoint),
            "resume_checkpoint": str(RESUME),
            "formal_evaluation": str(evaluation),
            "throughput_windows_steps_per_second": throughput_windows,
            "throughput_steps_per_second": stable_rate,
            "train_elapsed_seconds": train_elapsed_total,
            "eta": _eta_payload(target, stable_rate),
        })

    final = read_status()
    final["status"] = "complete"
    final["current_step"] = MILESTONES[-1]
    final["automatic_stop_reached"] = True
    final["resume_command"] = f"tmux new-session -d -s iqn_role_token_scratch_20260918 'cd {ROOT} && python3 tools/supervise_iqn_role_token_scratch_20260918.py --device {device} --episodes {episodes} --preflight {preflight or RUN_ROOT / 'preflight.json'}'"
    final["log_command"] = f"tail -f {RUN_ROOT / 'supervisor.log'}"
    write_json(STATUS, final)
    return final


def np_median(values: list[float]) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--preflight", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    result = supervise(args.device, args.episodes, args.preflight.resolve() if args.preflight else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
