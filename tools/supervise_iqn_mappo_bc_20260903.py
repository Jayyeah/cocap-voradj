#!/usr/bin/env python3
"""Queue dataset -> actor distillation -> formal Gate without entering PPO."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] START {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[{now()}] EXIT {result.returncode}\n")
        handle.flush()
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--teacher-formal", required=True)
    parser.add_argument("--dataset-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("poll-seconds must be at least 30")

    root = Path(args.root).resolve()
    dataset = root / "teacher_dataset"
    distillation = root / "distillation"
    gate = root / "formal_gate"
    logs = root / "logs"
    status_path = root / "supervisor_status.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"

    atomic_json(status_path, {
        "schema": "iqn-mappo-bc-supervisor-v1",
        "state": "WAITING_FOR_DATASET",
        "dataset_pid": int(args.dataset_pid),
        "updated_at": now(),
    })
    while not (dataset / "DATASET_DONE").is_file():
        try:
            os.kill(int(args.dataset_pid), 0)
        except ProcessLookupError:
            atomic_json(status_path, {
                "schema": "iqn-mappo-bc-supervisor-v1",
                "state": "FAILED_DATASET_PROCESS_EXITED",
                "dataset_pid": int(args.dataset_pid),
                "updated_at": now(),
            })
            return 1
        time.sleep(int(args.poll_seconds))

    atomic_json(status_path, {"schema": "iqn-mappo-bc-supervisor-v1", "state": "DISTILLING", "updated_at": now()})
    distill_command = [
        sys.executable,
        "tools/distill_mappo_actor_from_iqn_20260903.py",
        "--config", str(Path(args.config).resolve()),
        "--dataset-root", str(dataset),
        "--teacher-checkpoint", str(Path(args.teacher_checkpoint).resolve()),
        "--expected-teacher-sha256", str(args.teacher_sha256),
        "--output-root", str(distillation),
        "--device", str(args.device),
        "--temperature", "1.0",
        "--epochs", "20",
        "--batch-size", "2048",
        "--learning-rate", "0.0003",
        "--validation-modulus", "10",
        "--seed", "2026091401",
    ]
    if run_logged(distill_command, logs / "distillation.log", environment) != 0:
        atomic_json(status_path, {"schema": "iqn-mappo-bc-supervisor-v1", "state": "FAILED_DISTILLATION", "updated_at": now()})
        return 1

    actor = distillation / "distilled_actor.pt"
    atomic_json(status_path, {"schema": "iqn-mappo-bc-supervisor-v1", "state": "FORMAL_GATE", "updated_at": now()})
    gate_command = [
        sys.executable,
        "tools/evaluate_distilled_mappo_actor_20260903.py",
        "--config", str(Path(args.config).resolve()),
        "--teacher-checkpoint", str(Path(args.teacher_checkpoint).resolve()),
        "--expected-teacher-sha256", str(args.teacher_sha256),
        "--actor-checkpoint", str(actor),
        "--teacher-formal", str(Path(args.teacher_formal).resolve()),
        "--output-root", str(gate),
        "--episodes", "100",
        "--seed", "2026090301",
        "--device", str(args.device),
    ]
    gate_code = run_logged(gate_command, logs / "formal_gate.log", environment)
    report_path = gate / "gate_report.json"
    decision = None
    if report_path.is_file():
        decision = json.loads(report_path.read_text(encoding="utf-8")).get("decision")
    state = "WAITING_FOR_RESULT" if gate_code == 0 else "GATE_FAILED_STOP_BEFORE_PPO"
    atomic_json(status_path, {
        "schema": "iqn-mappo-bc-supervisor-v1",
        "state": state,
        "gate_exit_code": gate_code,
        "gate_decision": decision,
        "next_action": "inspect Gate; do not start PPO automatically",
        "updated_at": now(),
    })
    return 0 if gate_code in (0, 2) else gate_code


if __name__ == "__main__":
    raise SystemExit(main())
