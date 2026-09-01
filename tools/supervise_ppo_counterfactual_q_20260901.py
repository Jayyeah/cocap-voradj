#!/usr/bin/env python3
"""Queue and supervise three PPO-CF seeds on GPU1 without touching GPU0."""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/ppo_counterfactual_q_20260901"
ARTIFACT_ROOT = ROOT / "artifacts/2026-09-01_ppo_counterfactual_q"
STATUS_PATH = ARTIFACT_ROOT / "supervisor/status.json"
DECISION_PATH = ARTIFACT_ROOT / "gate_decision.json"
SEEDS = tuple(
    {
        "index": index,
        "config": CONFIG_ROOT / f"seed{index}.yaml",
        "run": ARTIFACT_ROOT / f"ppo_cf_seed{index}",
    }
    for index in (1, 2, 3)
)
PARENT_MEAN_BEST_CAPTURE = 0.2167
PARENT_MEAN_BEST_COLLISION = 0.7833


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def status(state: str, **fields: Any) -> None:
    payload = {"updated_at": now(), "state": state, **fields}
    atomic_json(STATUS_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def launch(name: str, command: list[str], log: Path) -> str:
    if tmux_exists(name):
        return "already_running"
    log.parent.mkdir(parents=True, exist_ok=True)
    shell = f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", name, shell], cwd=ROOT, check=True)
    return "launched"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def recent_metrics(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 262144))
            lines = handle.read().splitlines()[-limit:]
        return [json.loads(line) for line in lines]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def gpu_state(index: int) -> dict[str, Any]:
    fields = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=uuid,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().split(",")
    uuid = fields[0].strip()
    apps: list[dict[str, Any]] = []
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",", 3)]
        if len(pieces) == 4 and pieces[1] == uuid:
            apps.append({
                "pid": int(pieces[0]),
                "process_name": pieces[2],
                "memory_mib": int(pieces[3]),
            })
    return {
        "uuid": uuid,
        "memory_used_mib": int(fields[1]),
        "memory_free_mib": int(fields[2]),
        "memory_total_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
        "temperature_c": int(fields[5]),
        "compute_apps": apps,
    }


def resources(gpu_index: int) -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    payload: dict[str, Any] = {
        "disk_free_gib": round(disk.free / 2**30, 2),
        "disk_used_percent": round(100.0 * disk.used / disk.total, 2),
    }
    try:
        payload["gpu"] = gpu_state(gpu_index)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            mem[key] = int(value.strip().split()[0])
        payload["ram_available_gib"] = round(mem["MemAvailable"] * 1024 / 2**30, 2)
    except (OSError, KeyError, ValueError):
        pass
    return payload


def health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"warn": False, "critical": False, "critical_kl_streak": 0, "finite": True}
    streak = 0
    q_streak = 0
    for row in reversed(rows):
        kl = float(row.get("approx_kl_max", row.get("approx_kl", 0.0)) or 0.0)
        if kl > 0.1:
            streak += 1
        else:
            break
    for row in reversed(rows):
        critic_loss = abs(float(row.get("critic_loss", 0.0) or 0.0))
        critic_grad = abs(float(row.get("critic_grad_norm", 0.0) or 0.0))
        q_chosen = abs(float(row.get("q_chosen_mean", 0.0) or 0.0))
        q_span = abs(float(row.get("q_span_mean", 0.0) or 0.0))
        if critic_loss > 1e6 or critic_grad > 1e5 or q_chosen > 1e4 or q_span > 1e4:
            q_streak += 1
        else:
            break
    latest = rows[-1]
    kl = float(latest.get("approx_kl_max", latest.get("approx_kl", 0.0)) or 0.0)
    clip = float(latest.get("clip_fraction", 0.0) or 0.0)
    finite = all(
        not isinstance(value, (int, float)) or math.isfinite(float(value))
        for value in latest.values()
    )
    return {
        "warn": kl > 0.05 or clip > 0.3,
        "critical": streak >= 3 or q_streak >= 3 or not finite,
        "critical_kl_streak": streak,
        "critical_q_scale_streak": q_streak,
        "latest_kl": kl,
        "latest_clip_fraction": clip,
        "latest_critic_loss": latest.get("critic_loss"),
        "latest_critic_grad_norm": latest.get("critic_grad_norm"),
        "latest_q_span_mean": latest.get("q_span_mean"),
        "latest_cf_advantage_std": latest.get("counterfactual_advantage_std"),
        "finite": finite,
    }


def evaluation_summary(run: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((run / "evaluations").glob("step_*.json")):
        payload = read_json(path) or {}
        capture = ((payload.get("deterministic") or {}).get("capture") or {})
        rows.append({
            "step": int(payload.get("step", 0)),
            "capture_rate": float(capture.get("capture_rate", 0.0)),
            "collision_rate": float(capture.get("collision_rate", 1.0)),
            "visited_3plus": float(capture.get("visited_3plus_ring_rate", 0.0)),
        })
    best = max(
        rows or [{"step": 0, "capture_rate": 0.0, "collision_rate": 1.0, "visited_3plus": 0.0}],
        key=lambda row: (row["capture_rate"], -row["collision_rate"], row["step"]),
    )
    return {**best, "nonzero_checkpoint_count": sum(row["capture_rate"] > 0 for row in rows)}


def final_gate() -> dict[str, Any]:
    evaluations = [evaluation_summary(item["run"]) for item in SEEDS]
    repeated = all(row["nonzero_checkpoint_count"] >= 2 for row in evaluations)
    mean_capture = sum(row["capture_rate"] for row in evaluations) / len(evaluations)
    mean_collision = sum(row["collision_rate"] for row in evaluations) / len(evaluations)
    healthy = all(
        not health(recent_metrics(item["run"] / "learning_metrics.jsonl", 50))["critical"]
        for item in SEEDS
    )
    if (
        repeated
        and mean_capture >= 0.30
        and mean_capture > PARENT_MEAN_BEST_CAPTURE
        and mean_collision < PARENT_MEAN_BEST_COLLISION
    ):
        decision = "PASS_COUNTERFACTUAL_CREDIT_IMPROVEMENT"
        next_stage = "Continuous centralized-Q / FACMAC-like CTDE"
    elif healthy:
        decision = "HEALTHY_NO_CLEAR_IMPROVEMENT_OVER_MAPPO9_V2"
        next_stage = "Do not stack another Q critic; centralized V credit is not the primary bottleneck"
    else:
        decision = "UNHEALTHY_Q_IMPLEMENTATION_OR_SCALE_AUDIT"
        next_stage = "Pause and audit Q/return scale, gradients and masks before interpreting performance"
    return {
        "decided_at": now(),
        "decision": decision,
        "next_stage": next_stage,
        "parent_mappo9_v2": {
            "mean_best_capture": PARENT_MEAN_BEST_CAPTURE,
            "mean_best_collision": PARENT_MEAN_BEST_COLLISION,
        },
        "per_seed_best": evaluations,
        "mean_best_capture": mean_capture,
        "mean_best_collision": mean_collision,
        "repeated_nonzero_all_seeds": repeated,
        "optimization_healthy": healthy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-gpu-mib", type=int, default=40000)
    parser.add_argument("--min-disk-free-gib", type=float, default=27.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args()
    gpu_index = int(str(args.device).split(":")[-1])

    while True:
        preflight = resources(gpu_index)
        gpu = preflight.get("gpu") or {}
        active_seed_sessions = [
            f"cocap_ppo_cf_seed{item['index']}_gpu1"
            for item in SEEDS
            if tmux_exists(f"cocap_ppo_cf_seed{item['index']}_gpu1")
        ]
        if active_seed_sessions or (
            not gpu.get("compute_apps")
            and int(gpu.get("memory_free_mib", 0)) >= args.min_free_gpu_mib
        ):
            break
        status("waiting_for_exclusive_gpu1", resources=preflight)
        if args.check_once:
            return 0
        time.sleep(max(5, args.poll_seconds))

    if args.check_once:
        status("queue_preflight_ready", resources=preflight)
        return 0

    for seed in SEEDS:
        index, config, run = int(seed["index"]), seed["config"], seed["run"]
        session = f"cocap_ppo_cf_seed{index}_gpu1"
        log = ARTIFACT_ROOT / f"supervisor/seed{index}.log"
        launch_count = 0
        while True:
            run_status = read_json(run / "status.json") or {}
            if run_status.get("state") == "complete" and int(run_status.get("step", 0)) >= 400_000:
                break
            if not tmux_exists(session):
                launch_resources = resources(gpu_index)
                launch_gpu = launch_resources.get("gpu") or {}
                if (
                    launch_gpu.get("compute_apps")
                    or int(launch_gpu.get("memory_free_mib", 0)) < args.min_free_gpu_mib
                ):
                    status(
                        "waiting_for_exclusive_gpu1",
                        seed=index,
                        resources=launch_resources,
                    )
                    time.sleep(max(5, args.poll_seconds))
                    continue
            command = [
                "python3",
                "tools/run_small_step_ac_migration.py",
                "--config",
                relative(config),
                "--device",
                args.device,
            ]
            resume = run / "resume_latest.pt"
            if resume.is_file():
                command += ["--resume", relative(resume)]
            action = launch(session, command, log)
            if action == "launched":
                launch_count += 1
            restarts = max(0, launch_count - 1)
            rows = recent_metrics(run / "learning_metrics.jsonl")
            optimization = health(rows)
            resource_state = resources(gpu_index)
            status(
                "seed_active",
                seed=index,
                session=session,
                launch_action=action,
                tmux_running=tmux_exists(session),
                run_status=run_status,
                optimization_health=optimization,
                resources=resource_state,
                restarts=restarts,
                checkpoint_count=len(list((run / "checkpoints").glob("step_*.pt"))),
                rolling_resume=resume.is_file(),
            )
            if args.check_once:
                return 0
            if float(resource_state.get("disk_free_gib", 999.0)) < args.min_disk_free_gib:
                subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], cwd=ROOT, check=False)
                status("paused_disk_guard", seed=index, resources=resource_state)
                return 3
            if optimization["critical"]:
                subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], cwd=ROOT, check=False)
                status("paused_critical_q_health", seed=index, optimization_health=optimization)
                return 2
            if restarts > args.max_restarts:
                status("failed_restart_limit", seed=index, restarts=restarts)
                return 4
            time.sleep(max(5, args.poll_seconds))

    decision = final_gate()
    atomic_json(DECISION_PATH, decision)
    status("three_seed_gate_complete", gate_decision=decision, decision_path=relative(DECISION_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
