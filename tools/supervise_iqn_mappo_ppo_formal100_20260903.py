#!/usr/bin/env python3
"""Queue deterministic formal100 after both same-BC PPO branches finish."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BRANCH_ROOT = ROOT / "artifacts/2026-09-03_iqn_mappo_bc/ppo_branches"
FORMAL_ROOT = BRANCH_ROOT / "formal100"
STATUS_PATH = FORMAL_ROOT / "supervisor_status.json"
ACTOR = ROOT / "artifacts/2026-09-03_iqn_mappo_bc/distillation/distilled_actor.pt"
ACTOR_SHA = "bb8f971201f6e0a55af3ccb62bcae117f96afa1b07fad2c95c3da1dc552fa35e"
CONFIG = ROOT / "configs/experiments/mappo9_v2_20260830/seed1.yaml"
EVAL_SEED = 2026090301
BRANCHES = (
    {
        "name": "direct_ppo",
        "total": 100_000,
        "session": "cocap_bc_direct_formal100_gpu1",
        "warmup": (),
    },
    {
        "name": "critic_warmup_ppo",
        "total": 110_000,
        "session": "cocap_bc_warmup_formal100_gpu1",
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
    return str(path.resolve().relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def launch_formal(branch: dict[str, Any], device: str) -> None:
    name = str(branch["name"])
    resume = BRANCH_ROOT / name / "resume_latest.pt"
    if not resume.is_file():
        raise FileNotFoundError(resume)
    output = FORMAL_ROOT / name
    command = [
        "env", "CUDA_VISIBLE_DEVICES=1", "PYTHONPATH=src:.",
        "python3", "tools/run_small_step_ac_migration.py",
        "--config", relative(CONFIG),
        "--actor-init", relative(ACTOR),
        "--actor-init-sha256", ACTOR_SHA,
        "--total-steps", str(branch["total"]),
        "--resume", relative(resume),
        "--eval-only",
        "--eval-episodes", "100",
        "--eval-seed", str(EVAL_SEED),
        "--device", device,
        "--run-dir", relative(output),
        *branch["warmup"],
    ]
    log = FORMAL_ROOT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    shell = f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", str(branch["session"]), shell],
        cwd=ROOT, check=True,
    )


def resources() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    payload: dict[str, Any] = {"disk_free_gib": round(disk.free / 2**30, 2)}
    try:
        fields = subprocess.run(
            [
                "nvidia-smi", "--id=1",
                "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split(",")
        payload["gpu1"] = {
            "memory_used_mib": int(fields[0]),
            "memory_free_mib": int(fields[1]),
            "utilization_percent": int(fields[2]),
            "temperature_c": int(fields[3]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return payload


def report_complete(branch: dict[str, Any]) -> bool:
    report = read_json(FORMAL_ROOT / str(branch["name"]) / "formal100_report.json")
    return (
        report.get("status") == "complete"
        and int(report.get("episodes", 0)) == 100
        and int(report.get("step", -1)) == int(branch["total"])
        and int(report.get("evaluation_seed", -1)) == EVAL_SEED
        and set((report.get("evaluation") or {}).keys()) == {"deterministic"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--min-disk-free-gib", type=float, default=20.0)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("--poll-seconds must be at least 30")
    if not ACTOR.is_file() or sha256_file(ACTOR) != ACTOR_SHA:
        raise RuntimeError("shared BC Actor is missing or has the wrong SHA")

    restart_counts = {str(branch["name"]): 0 for branch in BRANCHES}
    while True:
        resource_state = resources()
        training = {
            str(branch["name"]): read_json(BRANCH_ROOT / str(branch["name"]) / "status.json")
            for branch in BRANCHES
        }
        training_complete = all(
            status.get("state") == "complete"
            and int(status.get("step", 0)) >= int(branch["total"])
            for branch, status in zip(BRANCHES, training.values())
        )
        formal = {
            str(branch["name"]): {
                "complete": report_complete(branch),
                "session": str(branch["session"]),
                "running": tmux_exists(str(branch["session"])),
                "report": read_json(FORMAL_ROOT / str(branch["name"]) / "formal100_report.json"),
            }
            for branch in BRANCHES
        }

        if training_complete:
            active = next((branch for branch in BRANCHES if formal[str(branch["name"])]["running"]), None)
            if active is None:
                pending = next((branch for branch in BRANCHES if not formal[str(branch["name"])]["complete"]), None)
                if pending is not None:
                    name = str(pending["name"])
                    if restart_counts[name] >= int(args.max_restarts):
                        atomic_json(STATUS_PATH, {
                            "state": "FAILED_RESTART_LIMIT", "branch": name, "updated_at": now(),
                        })
                        return 3
                    launch_formal(pending, str(args.device))
                    restart_counts[name] += 1
                    formal[name]["running"] = True
            all_formal_complete = all(row["complete"] for row in formal.values())
            state = "WAITING_FOR_RESULT" if all_formal_complete else "FORMAL100_ACTIVE"
        else:
            state = "WAITING_FOR_TRAINING"

        atomic_json(STATUS_PATH, {
            "schema": "iqn-mappo-bc-ppo-formal100-supervisor-v1",
            "state": state,
            "same_actor_sha256": ACTOR_SHA,
            "training": training,
            "formal": formal,
            "restart_counts": restart_counts,
            "resources": resource_state,
            "updated_at": now(),
        })
        if state == "WAITING_FOR_RESULT":
            return 0
        if float(resource_state.get("disk_free_gib", 999.0)) < float(args.min_disk_free_gib):
            atomic_json(STATUS_PATH, {
                "state": "PAUSED_DISK_GUARD", "resources": resource_state, "updated_at": now(),
            })
            return 4
        time.sleep(int(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
