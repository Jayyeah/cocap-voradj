#!/usr/bin/env python3
"""At P1 100k, extend to 200k only when sustained positive signals pass."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_p1_capture_first_local_support_maxpool_4p1e1obs_100k_aw.yaml"
TAG = "legacy_voradj_p1_local_support_maxpool_p0fixed_100k_20260813"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/p1_local_support_maxpool"
RUN_DIR = ARTIFACT_ROOT / TAG
REPORT = RUN_DIR / f"{TAG}_report.json"
METRICS = RUN_DIR / "metrics.jsonl"
RESUME = RUN_DIR / "resume_latest"
FROZEN = RUN_DIR / "resume_frozen_p1_step_000100000"
CONTROL = ROOT / "artifacts/2026-08-13_capture_first_controls/control"
GATE_REPORT = CONTROL / "p1_maxpool_100k_gate.json"
REQUIRED = (
    "trainer.pt", "replay.pkl", "runtime_state.pkl", "manifest.json",
    "effective_config.yaml", "metrics.jsonl", "diagnostic_eval.json", "checkpoint_storage.json",
)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def rows() -> list[dict[str, Any]]:
    if not METRICS.is_file():
        return []
    return [json.loads(line) for line in METRICS.read_text(encoding="utf-8").splitlines() if line.strip()]


def complete() -> bool:
    if not REPORT.is_file() or not all((RESUME / name).is_file() for name in REQUIRED):
        return False
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    return bool(
        int(report.get("transition_count", -1)) == 100000
        and int(report.get("replay_size", -1)) == 100000
        and report.get("all_finite", False)
    )


def evaluate_gate(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tail = [row for row in metric_rows if 75000 < int(row.get("step", -1)) <= 100000]
    normal_captures = sum(int(row.get("normal_capture_count", 0) or 0) for row in metric_rows)
    three_plus = [
        int(row["step"]) for row in tail
        if int(row.get("max_num_in_ring", 0) or 0) >= 3
        or float(row.get("fraction_steps_3plus_in_ring", 0.0) or 0.0) > 0.0
    ]
    two_plus = [
        int(row["step"]) for row in tail
        if float(row.get("fraction_steps_2plus_in_ring", 0.0) or 0.0) > 0.0
    ]
    support_upgrades = sum(int(row.get("support_upgraded_to_capture_count", 0) or 0) for row in tail)
    passed = bool(normal_captures >= 1 or len(three_plus) >= 2 or len(two_plus) >= 3)
    return {
        "schema_version": 1,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PASS_EXTEND_200K" if passed else "FAIL_STOP_100K",
        "passed": passed,
        "criteria": "normal capture >=1 OR >=2 independent 3+ windows OR >=3 independent 2+ windows in 75-100k",
        "normal_capture_count_0_100k": normal_captures,
        "three_plus_windows_75_100k": three_plus,
        "two_plus_windows_75_100k": two_plus,
        "support_upgrades_75_100k": support_upgrades,
    }


def freeze() -> Path:
    if FROZEN.is_dir():
        if not all((FROZEN / name).is_file() for name in REQUIRED):
            raise RuntimeError(f"incomplete existing P1 frozen bundle: {FROZEN}")
        return FROZEN
    temporary = FROZEN.with_name(FROZEN.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for source in RESUME.iterdir():
        if source.is_file():
            os.link(source, temporary / source.name)
    if not all((temporary / name).is_file() for name in REQUIRED):
        raise RuntimeError("new P1 frozen bundle incomplete")
    temporary.rename(FROZEN)
    return FROZEN


def continuation_command() -> list[str]:
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(CONFIG), "--seed", "2026081305", "--device", "cuda:0",
        "--total-steps", "200000", "--screen-episodes", "0",
        "--diagnostic-eval-episodes", "4", "--tag", TAG,
        "--artifact-root", str(ARTIFACT_ROOT),
        "--resume-checkpoint", str(FROZEN / "trainer.pt"),
        "--resume-replay", str(FROZEN / "replay.pkl"),
        "--resume-step", "100000",
    ]


def process_ids() -> list[int]:
    marker = f"--tag\x00{TAG}".encode()
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in cmdline and b"run_continuous_ctde_training.py" in cmdline:
            result.append(int(entry.name))
    return sorted(result)


def main() -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    log("armed: wait P1 Local-Max 100k and apply sustained-positive gate")
    while not complete():
        time.sleep(30)
    gate = evaluate_gate(rows())
    gate["continuation_command"] = continuation_command() if gate["passed"] else None
    GATE_REPORT.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"P1 100k gate: {gate['decision']}")
    if not gate["passed"]:
        return 0
    deadline = time.monotonic() + 120.0
    while process_ids() and time.monotonic() < deadline:
        time.sleep(2)
    if process_ids():
        raise RuntimeError(f"P1 100k process did not exit before extension: {process_ids()}")
    freeze()
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    env["PYTHONUNBUFFERED"] = "1"
    command = continuation_command()
    log("exec P1 exact continuation 100k -> 200k on cuda:0")
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
