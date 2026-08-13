#!/usr/bin/env python3
"""Wait for P1 release, enforce CF3 stable-capture Gate, then launch safe PC0."""
from __future__ import annotations

import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/2026-08-13_capture_first_controls/control"
SWAP_AUDIT = CONTROL / "cf3_p1_gpu_swap_50k_audit.json"
P1_GATE = CONTROL / "p1_maxpool_100k_gate.json"
CF3_CONFIG = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml"
CF3_TAG = "legacy_voradj_cf3_local_support_p0fixed_recovery25k_to200k_20260813"
CF3_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/cf3_local_support_full_p0fixed_recovery"
CF3_RUN = CF3_ROOT / CF3_TAG
CF3_METRICS = CF3_RUN / "metrics.jsonl"
CF3_RESUME = CF3_RUN / "resume_latest"
CF3_REPORT = CF3_RUN / f"{CF3_TAG}_report.json"
P1_TAG = "legacy_voradj_p1_local_support_maxpool_p0fixed_100k_20260813"
PC0_CONFIG = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_pc0_cf3_actor_warmstart_postcapture300_4p1e1obs_100k_aw.yaml"
PC0_TAG = "legacy_voradj_pc0_cf3_actor_warmstart_postcapture300_4p1e1obs_100k_aw_20260813"
PC0_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/pc0_cf3_actor_postcapture300"
PC0_RUN = PC0_ROOT / PC0_TAG
PC0_INIT = PC0_RUN / "actor_only_init_step_000000000"
STATUS = CONTROL / "cf3_stable_gate_pc0_status.json"
FORMAL_GATE = CONTROL / "cf3_200k_formal_normal_capture_20rollout.json"
COLLISION_AUDIT = CONTROL / "cf3_200k_capture_consolidation_collision_audit.json"
REQUIRED = (
    "trainer.pt", "replay.pkl", "runtime_state.pkl", "manifest.json",
    "effective_config.yaml", "metrics.jsonl", "diagnostic_eval.json", "checkpoint_storage.json",
)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def write_status(payload: dict[str, Any]) -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def process_ids(tag: str) -> list[int]:
    marker = f"--tag\x00{tag}\x00".encode()
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


def metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normal = sum(int(row.get("distinct_normal_capture_episodes", 0) or 0) for row in rows)
    stationary = sum(int(row.get("distinct_stationary_capture_episodes", 0) or 0) for row in rows)
    two_steps = [int(row["step"]) for row in rows if float(row.get("fraction_steps_2plus_in_ring", 0) or 0) > 0]
    three_steps = [int(row["step"]) for row in rows if float(row.get("fraction_steps_3plus_in_ring", 0) or 0) > 0]
    return {
        "through_step": max((int(row.get("step", 0)) for row in rows), default=0),
        "distinct_normal_capture_episodes": normal,
        "distinct_stationary_capture_episodes": stationary,
        "two_plus_window_count": len(two_steps),
        "two_plus_window_steps": two_steps,
        "three_plus_window_count": len(three_steps),
        "three_plus_window_steps": three_steps,
        "max_2plus_hold_steps": max((int(row.get("max_2plus_ring_hold_steps", 0) or 0) for row in rows), default=0),
        "max_3plus_hold_steps": max((int(row.get("max_3plus_ring_hold_steps", 0) or 0) for row in rows), default=0),
        "training_gate_passed": bool(normal >= 3 and normal > stationary),
    }


def swap_verified() -> bool:
    if not SWAP_AUDIT.is_file():
        return False
    return json.loads(SWAP_AUDIT.read_text(encoding="utf-8")).get("status") == "VERIFIED_100_PERCENT_COMPLETE"


def p1_released_after_gate() -> bool:
    if not P1_GATE.is_file() or process_ids(P1_TAG):
        return False
    gate = json.loads(P1_GATE.read_text(encoding="utf-8"))
    if not bool(gate.get("passed", False)):
        return True
    # A passing P1 Gate schedules 100->200k continuation.  Do not interpret
    # the milliseconds between the 100k process exit and continuation launch
    # as GPU release; require the overwritten final 200k report instead.
    report_path = PC0_ROOT.parent / "p1_local_support_maxpool" / P1_TAG / f"{P1_TAG}_report.json"
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return int(report.get("transition_count", -1)) >= 200000


def complete_resume() -> tuple[int, dict[str, Any]] | None:
    if not all((CF3_RESUME / item).is_file() for item in REQUIRED):
        return None
    with (CF3_RESUME / "runtime_state.pkl").open("rb") as handle:
        runtime = dict(pickle.load(handle))
    step = int(runtime.get("transition_count", -1))
    if step <= 0 or step % 25000 != 0:
        return None
    summary = gate_summary(metric_rows(CF3_RESUME / "metrics.jsonl"))
    if summary["through_step"] != step:
        return None
    return step, summary


def freeze_resume(step: int) -> Path:
    destination = CF3_RUN / f"resume_frozen_cf3_stable_gate_step_{step:09d}"
    if destination.is_dir():
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for source in CF3_RESUME.iterdir():
        if source.is_file():
            os.link(source, temporary / source.name)
    if not all((temporary / item).is_file() for item in REQUIRED):
        raise RuntimeError("CF3 stable-gate frozen bundle incomplete")
    temporary.rename(destination)
    return destination


def run_formal_gate(bundle: Path) -> dict[str, Any]:
    command = [
        sys.executable, str(ROOT / "tools/evaluate_cf3_normal_capture_gate.py"),
        "--config", str(CF3_CONFIG), "--checkpoint", str(bundle / "trainer.pt"),
        "--seed", "2026081304", "--episodes", "20", "--device", "cuda:1",
        "--output", str(FORMAL_GATE),
    ]
    subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}, check=True)
    return json.loads(FORMAL_GATE.read_text(encoding="utf-8"))


def build_collision_audit(summary: dict[str, Any]) -> dict[str, Any]:
    rows = metric_rows(CF3_METRICS)
    near = sorted(
        rows,
        key=lambda row: (
            int(row.get("max_num_in_ring", 0) or 0),
            float(row.get("fraction_steps_3plus_in_ring", 0) or 0),
            float(row.get("fraction_steps_2plus_in_ring", 0) or 0),
            int(row.get("max_3plus_ring_hold_steps", 0) or 0),
        ),
        reverse=True,
    )[:12]
    collision_total = sum(int(row.get("collision_count", 0) or 0) for row in rows)
    terminal_total = sum(int(row.get("terminated_count", 0) or 0) for row in rows)
    audit = {
        "schema_version": 1,
        "decision": "CAPTURE_CONSOLIDATION_COLLISION_AUDIT",
        "reason": "CF3 reached 200k without stable normal-capture Gate",
        "gate_summary": summary,
        "collision_count_total": collision_total,
        "terminated_count_total": terminal_total,
        "collision_fraction_of_terminals": collision_total / max(terminal_total, 1),
        "two_to_three_failure_window_count": max(
            summary["two_plus_window_count"] - summary["three_plus_window_count"], 0
        ),
        "near_success_windows": [
            {key: row.get(key) for key in (
                "step", "normal_capture_count", "stationary_capture_count", "collision_count",
                "max_num_in_ring", "fraction_steps_2plus_in_ring", "fraction_steps_3plus_in_ring",
                "max_2plus_ring_hold_steps", "max_3plus_ring_hold_steps", "d1_min", "d1_mean",
                "d2_mean", "d3_mean", "support_enemy_distance_progress_mean",
            )} for row in near
        ],
        "next_required_diagnostics": [
            "replay success/near-success final tens of steps",
            "agent-agent vs obstacle vs boundary collision attribution",
            "2+ to 3+ ring failure sequence",
            "3+ ring hold loss sequence",
            "collision/capture-geometry temporal alignment",
        ],
        "forbidden_next_runs": ["post_capture", "vx_vy", "MATD3", "UTD_1"],
    }
    COLLISION_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def prepare_and_launch_pc0(bundle: Path, source_step: int, gate: dict[str, Any]) -> None:
    PC0_RUN.mkdir(parents=True, exist_ok=True)
    prepare = [
        sys.executable, str(ROOT / "tools/prepare_pc0_actor_warmstart_bundle.py"),
        "--source-checkpoint", str(bundle / "trainer.pt"), "--source-config", str(CF3_CONFIG),
        "--target-config", str(PC0_CONFIG), "--output-dir", str(PC0_INIT),
        "--seed", "2026081306", "--tag", PC0_TAG,
    ]
    subprocess.run(prepare, cwd=ROOT, env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}, check=True)
    command = [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(PC0_CONFIG), "--seed", "2026081306", "--device", "cuda:1",
        "--total-steps", "100000", "--screen-episodes", "20", "--diagnostic-eval-episodes", "4",
        "--tag", PC0_TAG, "--artifact-root", str(PC0_ROOT),
        "--resume-checkpoint", str(PC0_INIT / "trainer.pt"),
        "--resume-replay", str(PC0_INIT / "replay.pkl"), "--resume-step", "0",
    ]
    log_path = PC0_RUN / "pc0_actor_only_launch.log"
    shell = "exec env PYTHONPATH=" + shlex.quote(f"{ROOT / 'src'}:{ROOT}") + " PYTHONUNBUFFERED=1 "
    shell += shlex.join(command) + " >> " + shlex.quote(str(log_path)) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", "pc0_gpu1", "-c", str(ROOT), shell], check=True)
    schedule = {
        "schema_version": 1, "decision": "STABLE_GATE_PASS_LAUNCH_PC0",
        "cf3_source_step": source_step, "cf3_frozen_bundle": str(bundle), "gate": gate,
        "pc0_init_bundle": str(PC0_INIT), "pc0_command": command,
    }
    (CONTROL / "pc0_launch_schedule.json").write_text(
        json.dumps(schedule, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        rows = metric_rows(PC0_RUN / "metrics.jsonl")
        if rows and int(rows[-1].get("step", -1)) >= 1000:
            last = rows[-1]
            if int(last.get("step", -1)) != 1000 or int(last.get("replay_size", -1)) != 1000:
                raise RuntimeError(f"PC0 discontinuous startup: {last.get('step')}/{last.get('replay_size')}")
            if float(last.get("mean_finite", 0.0)) != 1.0:
                raise RuntimeError("PC0 first training window non-finite")
            write_status({"schema_version": 1, "status": "PC0_1K_VERIFIED", "schedule": schedule})
            return
        time.sleep(15)
    raise RuntimeError("PC0 did not produce a finite 1k window")


def main() -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    write_status({"schema_version": 1, "status": "WAITING_SWAP_P1_RELEASE_AND_CF3_GATE"})
    log("armed: CF3 stable Gate -> safe PC0; waits for P1 100k release")
    while not swap_verified():
        time.sleep(30)
    while True:
        current = complete_resume()
        if current is None:
            time.sleep(30)
            continue
        step, frozen_summary = current
        live_summary = gate_summary(metric_rows(CF3_METRICS))
        write_status({
            "schema_version": 1, "status": "WAITING_GATE_OR_P1_RELEASE",
            "cf3_live": live_summary, "cf3_latest_complete_bundle": frozen_summary,
            "p1_gate_written": P1_GATE.is_file(), "p1_processes": process_ids(P1_TAG),
        })
        if not p1_released_after_gate():
            time.sleep(30)
            continue
        bundle = freeze_resume(step)
        gate: dict[str, Any] = {"training": frozen_summary, "formal_20rollout": None}
        passed = bool(frozen_summary["training_gate_passed"])
        if not passed and step >= 200000 and CF3_REPORT.is_file():
            formal = run_formal_gate(bundle)
            gate["formal_20rollout"] = formal
            passed = bool(formal.get("stable_gate_passed", False))
        if passed:
            prepare_and_launch_pc0(bundle, step, gate)
            return 0
        if step >= 200000 and CF3_REPORT.is_file():
            audit = build_collision_audit(frozen_summary)
            write_status({"schema_version": 1, "status": "CF3_200K_UNSTABLE_COLLISION_AUDIT", "audit": audit})
            return 0
        time.sleep(30)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        write_status({
            "schema_version": 1, "status": "FAILED_CLOSED",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
