#!/usr/bin/env python3
"""Run same-BC Direct PPO and critic-warm-up PPO branches on shared GPU1."""
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
ARTIFACT_ROOT = ROOT / "artifacts/2026-09-03_iqn_mappo_bc/ppo_branches"
STATUS_PATH = ARTIFACT_ROOT / "supervisor_status.json"
ACTOR = ROOT / "artifacts/2026-09-03_iqn_mappo_bc/distillation/distilled_actor.pt"
ACTOR_SHA = "bb8f971201f6e0a55af3ccb62bcae117f96afa1b07fad2c95c3da1dc552fa35e"
CONFIG = ROOT / "configs/experiments/mappo9_v2_20260830/seed1.yaml"
BRANCHES = (
    {
        "name": "direct_ppo",
        "session": "cocap_bc_direct_ppo_gpu1",
        "total": 100_000,
        "warmup": (),
    },
    {
        "name": "critic_warmup_ppo",
        "session": "cocap_bc_warmup_ppo_gpu1",
        "total": 110_000,
        "warmup": (
            "--critic-warmup-max-steps", "25000",
            "--critic-warmup-min-steps", "10000",
            "--critic-warmup-ev-threshold", "0.20",
            "--critic-warmup-ev-streak", "2",
        ),
    },
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def tmux_pid(name: str) -> int | None:
    result = subprocess.run(
        ["tmux", "display-message", "-p", "-t", name, "#{pane_pid}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip().isdigit() else None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def last_metrics(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 131072))
            rows = handle.read().splitlines()
        return json.loads(rows[-1]) if rows else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def finite(row: dict[str, Any]) -> bool:
    return all(
        not isinstance(value, (int, float)) or math.isfinite(float(value))
        for value in row.values()
    )


def launch(branch: dict[str, Any], device: str, resume: Path | None) -> None:
    name = str(branch["name"])
    run = ARTIFACT_ROOT / name
    command = [
        "env", "CUDA_VISIBLE_DEVICES=1", "PYTHONPATH=src:.",
        "python3", "tools/run_small_step_ac_migration.py",
        "--config", relative(CONFIG),
        "--actor-init", relative(ACTOR),
        "--actor-init-sha256", ACTOR_SHA,
        "--total-steps", str(branch["total"]),
        "--device", device,
        "--run-dir", relative(run),
        *branch["warmup"],
    ]
    if resume is not None:
        command += ["--resume", relative(resume)]
    log = ARTIFACT_ROOT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    shell = f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", str(branch["session"]), shell], cwd=ROOT, check=True)


def resources() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    payload: dict[str, Any] = {
        "disk_free_gib": round(disk.free / 2**30, 2),
        "disk_used_percent": round(100.0 * disk.used / disk.total, 2),
    }
    try:
        line = subprocess.run(
            ["nvidia-smi", "--id=1", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        ).stdout.strip().split(",")
        payload["gpu1"] = {
            "memory_used_mib": int(line[0]), "memory_free_mib": int(line[1]),
            "utilization_percent": int(line[2]), "temperature_c": int(line[3]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--min-disk-free-gib", type=float, default=20.0)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("poll-seconds must be at least 30")
    if not ACTOR.is_file():
        raise FileNotFoundError(ACTOR)
    restart_counts = {str(branch["name"]): 0 for branch in BRANCHES}
    while True:
        states: dict[str, Any] = {}
        all_complete = True
        for branch in BRANCHES:
            name = str(branch["name"])
            run = ARTIFACT_ROOT / name
            run_status = read_json(run / "status.json")
            complete = run_status.get("state") == "complete" and int(run_status.get("step", 0)) >= int(branch["total"])
            running = tmux_exists(str(branch["session"]))
            metrics = last_metrics(run / "learning_metrics.jsonl")
            if metrics and not finite(metrics):
                atomic_json(STATUS_PATH, {"state": "PAUSED_NONFINITE", "branch": name, "metrics": metrics, "updated_at": now()})
                return 2
            if not complete and not running:
                resume = run / "resume_latest.pt"
                if restart_counts[name] >= int(args.max_restarts):
                    atomic_json(STATUS_PATH, {"state": "FAILED_RESTART_LIMIT", "branch": name, "updated_at": now()})
                    return 3
                launch(branch, str(args.device), resume if resume.is_file() else None)
                restart_counts[name] += 1
                running = True
            all_complete = all_complete and complete
            states[name] = {
                "complete": complete,
                "tmux": str(branch["session"]),
                "pane_pid": tmux_pid(str(branch["session"])) if running else None,
                "restart_count": restart_counts[name],
                "status": run_status,
                "last_metrics": metrics,
                "checkpoint_count": len(list((run / "checkpoints").glob("step_*.pt"))),
            }
        resource_state = resources()
        state = "WAITING_FOR_RESULT" if all_complete else "BRANCHES_ACTIVE"
        atomic_json(STATUS_PATH, {
            "schema": "iqn-mappo-bc-ppo-branches-supervisor-v1",
            "state": state,
            "same_actor_checkpoint": relative(ACTOR),
            "same_actor_sha256": ACTOR_SHA,
            "branches": states,
            "resources": resource_state,
            "updated_at": now(),
        })
        if all_complete:
            return 0
        if float(resource_state.get("disk_free_gib", 999.0)) < float(args.min_disk_free_gib):
            atomic_json(STATUS_PATH, {"state": "PAUSED_DISK_GUARD", "resources": resource_state, "updated_at": now()})
            return 4
        time.sleep(int(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
