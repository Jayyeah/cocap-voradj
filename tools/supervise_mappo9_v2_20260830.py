#!/usr/bin/env python3
"""Queue and supervise three MAPPO-9-v2 seeds on GPU1."""
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
CONFIG_ROOT = ROOT / "configs/experiments/mappo9_v2_20260830"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-30_mappo9_v2"
STATUS_PATH = ARTIFACT_ROOT / "supervisor/status.json"
DECISION_PATH = ARTIFACT_ROOT / "gate_decision.json"
SEEDS = (
    {"index": 1, "config": CONFIG_ROOT / "seed1.yaml", "run": ARTIFACT_ROOT / "mappo9_v2_seed1"},
    {"index": 2, "config": CONFIG_ROOT / "seed2.yaml", "run": ARTIFACT_ROOT / "mappo9_v2_seed2"},
    {"index": 3, "config": CONFIG_ROOT / "seed3.yaml", "run": ARTIFACT_ROOT / "mappo9_v2_seed3"},
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def status(state: str, **fields: Any) -> None:
    payload = {"updated_at": now(), "state": state, **fields}
    atomic_json(STATUS_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def launch(name: str, command: list[str], log: Path) -> str:
    if tmux_exists(name):
        return "already_running"
    log.parent.mkdir(parents=True, exist_ok=True)
    shell = f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", name, shell], cwd=ROOT, check=True)
    return "launched"


def process_matches(fragment: str) -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if fragment in command:
            matches.append(int(entry.name))
    return matches


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def recent_metrics(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 262144))
            lines = handle.read().splitlines()[-limit:]
        return [json.loads(line) for line in lines]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def gpu_state(index: int) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}",
         "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return {"memory_used_mib": int(result[0]), "memory_free_mib": int(result[1]),
            "memory_total_mib": int(result[2]), "utilization_percent": int(result[3]),
            "temperature_c": int(result[4])}


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


def legacy_lane_active() -> list[int]:
    return process_matches("td3_aw_seed3.yaml")


def health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"warn": False, "critical": False, "critical_kl_streak": 0}
    streak = 0
    for row in reversed(rows):
        metrics = row
        kl = float(metrics.get("approx_kl_max", metrics.get("approx_kl", 0.0)) or 0.0)
        if kl > 0.1:
            streak += 1
        else:
            break
    latest = rows[-1]
    kl = float(latest.get("approx_kl_max", latest.get("approx_kl", 0.0)) or 0.0)
    clip = float(latest.get("clip_fraction", 0.0) or 0.0)
    finite = all(
        not isinstance(value, (int, float)) or math.isfinite(float(value))
        for value in latest.values()
    )
    return {"warn": kl > 0.05 or clip > 0.3, "critical": streak >= 3 or not finite,
            "critical_kl_streak": streak, "latest_kl": kl, "latest_clip_fraction": clip,
            "finite": finite}


def best_evaluation(run: Path) -> dict[str, Any]:
    best = {"step": 0, "capture_rate": 0.0, "collision_rate": 1.0, "visited_3plus": 0.0}
    for path in sorted((run / "evaluations").glob("step_*.json")):
        payload = read_json(path) or {}
        capture = ((payload.get("deterministic") or {}).get("capture") or {})
        candidate = {
            "step": int(payload.get("step", 0)),
            "capture_rate": float(capture.get("capture_rate", 0.0)),
            "collision_rate": float(capture.get("collision_rate", 1.0)),
            "visited_3plus": float(capture.get("visited_3plus_ring_rate", 0.0)),
        }
        if (candidate["capture_rate"], -candidate["collision_rate"], candidate["step"]) > (
            best["capture_rate"], -best["collision_rate"], best["step"]
        ):
            best = candidate
    return best


def final_gate() -> dict[str, Any]:
    evaluations = [best_evaluation(item["run"]) for item in SEEDS]
    stable_nonzero = all(row["capture_rate"] > 0.0 for row in evaluations)
    mean_capture = sum(row["capture_rate"] for row in evaluations) / len(evaluations)
    mean_collision = sum(row["collision_rate"] for row in evaluations) / len(evaluations)
    learning_rows = [recent_metrics(item["run"] / "learning_metrics.jsonl", 50) for item in SEEDS]
    healthy = all(not health(rows)["critical"] for rows in learning_rows)
    if stable_nonzero and mean_capture >= 0.10 and mean_collision < 0.80:
        decision = "PASS_TO_MAPPO_AW_V2"
        next_stage = "MAPPO-AW-v2; categorical AW9 -> continuous (a,w), all other contracts fixed"
    elif healthy:
        decision = "HEALTHY_FAIL_TO_DISCRETE_COUNTERFACTUAL_Q"
        next_stage = "Discrete centralized-Q / counterfactual CTDE; do not blind-tune PPO"
    else:
        decision = "UNHEALTHY_PAUSE_FOR_IMPLEMENTATION_AUDIT"
        next_stage = "Pause; inspect KL/value/finite-metric failure before another algorithm"
    return {"decided_at": now(), "decision": decision, "next_stage": next_stage,
            "per_seed_best": evaluations, "mean_best_capture": mean_capture,
            "mean_best_collision": mean_collision, "stable_nonzero_all_seeds": stable_nonzero,
            "optimization_healthy": healthy}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-gpu-mib", type=int, default=26000)
    parser.add_argument("--min-disk-free-gib", type=float, default=20.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args()
    gpu_index = int(str(args.device).split(":")[-1])

    while True:
        legacy = legacy_lane_active()
        resource_state = resources(gpu_index)
        free = int((resource_state.get("gpu") or {}).get("memory_free_mib", 0))
        if not legacy and free >= args.min_free_gpu_mib:
            break
        status("waiting_for_gpu1_legacy_td3", legacy_pids=legacy, resources=resource_state)
        if args.check_once:
            return 0
        time.sleep(max(5, args.poll_seconds))

    for seed in SEEDS:
        index, config, run = int(seed["index"]), seed["config"], seed["run"]
        session = f"cocap_mappo9_v2_seed{index}_gpu1"
        log = ARTIFACT_ROOT / f"supervisor/seed{index}.log"
        total = 400_000
        launch_count = 0
        while True:
            run_status = read_json(run / "status.json") or {}
            if run_status.get("state") == "complete" and int(run_status.get("step", 0)) >= total:
                break
            command = ["python3", "tools/run_small_step_ac_migration.py", "--config", relative(config),
                       "--device", args.device]
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
            status("seed_active", seed=index, session=session, launch_action=action,
                   tmux_running=tmux_exists(session), run_status=run_status,
                   optimization_health=optimization, resources=resource_state,
                   restarts=restarts,
                   checkpoint_count=len(list((run / "checkpoints").glob("step_*.pt"))),
                   rolling_resume=resume.is_file())
            if args.check_once:
                return 0
            if float(resource_state.get("disk_free_gib", 999.0)) < args.min_disk_free_gib:
                subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], cwd=ROOT, check=False)
                status("paused_disk_guard", seed=index, resources=resource_state)
                return 3
            if optimization["critical"]:
                subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], cwd=ROOT, check=False)
                status("paused_critical_optimization", seed=index, optimization_health=optimization)
                return 2
            if restarts > args.max_restarts:
                subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], cwd=ROOT, check=False)
                status("failed_restart_limit", seed=index, restarts=restarts)
                return 4
            time.sleep(max(5, args.poll_seconds))
    decision = final_gate()
    atomic_json(DECISION_PATH, decision)
    status("three_seed_gate_complete", gate_decision=decision, decision_path=relative(DECISION_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
