#!/usr/bin/env python3
"""At P1 100k, extend to 200k only when sustained positive signals pass."""
from __future__ import annotations

import json
import math
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
CF3_METRICS = ROOT / (
    "artifacts/2026-08-13_capture_first_controls/cf3_local_support_full_p0fixed_recovery/"
    "legacy_voradj_cf3_local_support_p0fixed_recovery25k_to200k_20260813/metrics.jsonl"
)
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


def rows(path: Path = METRICS) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def complete() -> bool:
    if not REPORT.is_file() or not all((RESUME / name).is_file() for name in REQUIRED):
        return False
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    return bool(
        int(report.get("transition_count", -1)) == 100000
        and int(report.get("replay_size", -1)) == 100000
        and report.get("all_finite", False)
    )


def _summary(metric_rows: list[dict[str, Any]], *, lower_step: int, upper_step: int) -> dict[str, Any]:
    matched = [row for row in metric_rows if lower_step < int(row.get("step", -1)) <= upper_step]
    normal_steps = [
        int(row["step"]) for row in matched if int(row.get("normal_capture_count", 0) or 0) > 0
    ]
    normal_captures = sum(int(row.get("normal_capture_count", 0) or 0) for row in matched)
    distinct_normal = sum(int(row.get("distinct_normal_capture_episodes", 0) or 0) for row in matched)
    three_plus = [
        int(row["step"]) for row in matched
        if int(row.get("max_num_in_ring", 0) or 0) >= 3
        or float(row.get("fraction_steps_3plus_in_ring", 0.0) or 0.0) > 0.0
    ]
    two_plus = [
        int(row["step"]) for row in matched
        if float(row.get("fraction_steps_2plus_in_ring", 0.0) or 0.0) > 0.0
    ]
    return {
        "lower_step_exclusive": lower_step,
        "upper_step_inclusive": upper_step,
        "metric_windows": len(matched),
        "normal_capture_count": normal_captures,
        "distinct_normal_capture_episodes": distinct_normal,
        "normal_capture_steps": normal_steps,
        "two_plus_window_count": len(two_plus),
        "two_plus_window_steps": two_plus,
        "three_plus_window_count": len(three_plus),
        "three_plus_window_steps": three_plus,
        "support_upgrades": sum(int(row.get("support_upgraded_to_capture_count", 0) or 0) for row in matched),
    }


def evaluate_gate(p1_rows: list[dict[str, Any]], cf3_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # CF3's current artifact starts after the corrected-contract switch at 25k.
    # Compare the same 25-100k interval and never let distance-only gains pass.
    p1 = _summary(p1_rows, lower_step=25000, upper_step=100000)
    cf3 = _summary(cf3_rows, lower_step=25000, upper_step=100000)
    required_two_plus = max(2, math.ceil(0.75 * cf3["two_plus_window_count"]))
    required_three_plus = math.ceil(0.75 * cf3["three_plus_window_count"])
    repeated_geometry_near_cf3 = bool(
        p1["two_plus_window_count"] >= required_two_plus
        and p1["three_plus_window_count"] >= required_three_plus
    )
    multiple_independent_normal = p1["distinct_normal_capture_episodes"] >= 2
    normal_and_matched_geometry = bool(
        p1["distinct_normal_capture_episodes"] >= 1 and repeated_geometry_near_cf3
    )
    passed = bool(multiple_independent_normal or normal_and_matched_geometry)
    return {
        "schema_version": 2,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PASS_EXTEND_200K" if passed else "FAIL_STOP_100K",
        "passed": passed,
        "criteria": (
            "PASS only with >=2 independent P1 normal captures, OR >=1 independent P1 normal capture "
            "and P1 2+/3+ window counts each >=75% of CF3 matched 25-100k counts "
            "(2+ floor=2); distance-only or isolated 2+ never passes"
        ),
        "p1_matched_step": p1,
        "cf3_matched_step": cf3,
        "required_p1_two_plus_windows": required_two_plus,
        "required_p1_three_plus_windows": required_three_plus,
        "repeated_geometry_near_cf3": repeated_geometry_near_cf3,
        "multiple_independent_normal": multiple_independent_normal,
        "normal_and_matched_geometry": normal_and_matched_geometry,
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
        "--config", str(CONFIG), "--seed", "2026081305", "--device", "cuda:1",
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
    gate = evaluate_gate(rows(), rows(CF3_METRICS))
    gate["continuation_command"] = continuation_command() if gate["passed"] else None
    GATE_REPORT.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"P1 100k gate: {gate['decision']}")
    freeze()
    if not gate["passed"]:
        return 0
    deadline = time.monotonic() + 120.0
    while process_ids() and time.monotonic() < deadline:
        time.sleep(2)
    if process_ids():
        raise RuntimeError(f"P1 100k process did not exit before extension: {process_ids()}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    env["PYTHONUNBUFFERED"] = "1"
    command = continuation_command()
    log("exec P1 exact continuation 100k -> 200k on cuda:1")
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
