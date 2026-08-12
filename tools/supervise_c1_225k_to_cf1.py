#!/usr/bin/env python3
"""Stop C1 after its complete 225k milestone and launch CF1 scratch on GPU1."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
C1_TAG = "legacy_voradj_oldmix_allagent_noclip_c1_utd05_300k_aw_20260812"
C1_DIR = ROOT / "artifacts/2026-08-12_p1_extension_controls/c1_utd05" / C1_TAG
C1_METRICS = C1_DIR / "metrics.jsonl"
C1_CHECKPOINT = C1_DIR / "checkpoints/step_000225000"
C1_RESUME = C1_DIR / "resume_latest"
C1_FROZEN = C1_DIR / "resume_frozen_c1_step_000225000"
CONTROL = ROOT / "artifacts/2026-08-13_capture_first_controls/control"
STOP_REPORT = CONTROL / "c1_stop_225k.json"
CF1_CONFIG = ROOT / (
    "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_cf1_capture_first_global_enemy_4p1e1obs_100k_aw.yaml"
)
CF1_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/cf1_global"
CF1_TAG = "legacy_voradj_cf1_capture_first_global_enemy_4p1e1obs_100k_aw_20260813"
REQUIRED_CHECKPOINT = (
    "trainer.pt", "runtime_state.pkl", "manifest.json", "effective_config.yaml",
    "metrics.jsonl", "diagnostic_eval.json", "checkpoint_storage.json",
)
REQUIRED_RESUME = REQUIRED_CHECKPOINT + ("replay.pkl",)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def latest_metric() -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in C1_METRICS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[-1]


def milestone_complete() -> bool:
    if not C1_METRICS.is_file():
        return False
    if int(latest_metric().get("step", -1)) < 225000:
        return False
    return all((C1_CHECKPOINT / name).is_file() for name in REQUIRED_CHECKPOINT) and all(
        (C1_RESUME / name).is_file() for name in REQUIRED_RESUME
    )


def freeze_resume_bundle() -> Path:
    if C1_FROZEN.is_dir():
        if not all((C1_FROZEN / name).is_file() for name in REQUIRED_RESUME):
            raise RuntimeError(f"existing C1 frozen bundle is incomplete: {C1_FROZEN}")
        return C1_FROZEN
    temporary = C1_FROZEN.with_name(C1_FROZEN.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for source in C1_RESUME.iterdir():
        if source.is_file():
            os.link(source, temporary / source.name)
    if not all((temporary / name).is_file() for name in REQUIRED_RESUME):
        raise RuntimeError("new C1 frozen bundle is incomplete")
    temporary.rename(C1_FROZEN)
    return C1_FROZEN


def c1_process_ids() -> list[int]:
    result: list[int] = []
    marker = f"--tag\x00{C1_TAG}".encode()
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


def replay_new_capture_events() -> tuple[int, list[int]]:
    with (C1_FROZEN / "replay.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    ids = [
        int(transition_id)
        for transition_id, transition in payload.get("records", [])
        if int(transition_id) >= 200000
        and "capture" in set((transition.metadata or {}).get("event_ids", []))
    ]
    return len(ids), ids


def build_stop_report(stopped_pid: int, frozen: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in C1_METRICS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    extension = [row for row in rows if 200000 < int(row.get("step", -1)) <= 225000]
    last = max(extension, key=lambda row: int(row["step"]))
    new_capture_count, capture_ids = replay_new_capture_events()
    two_plus_steps = [
        int(row["step"]) for row in extension
        if float(row.get("fraction_steps_2plus_in_ring", 0.0) or 0.0) > 0.0
    ]
    three_plus_steps = [
        int(row["step"]) for row in extension
        if int(row.get("max_num_in_ring", 0) or 0) >= 3
        or float(row.get("fraction_steps_3plus_in_ring", 0.0) or 0.0) > 0.0
    ]
    diagnostic = json.loads((C1_CHECKPOINT / "diagnostic_eval.json").read_text(encoding="utf-8"))
    gate_positive = bool(new_capture_count or three_plus_steps or len(set(two_plus_steps)) >= 2)
    return {
        "schema_version": 1,
        "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_stop_step": 225000,
        "last_metric_step": int(last["step"]),
        "stopped_pid": int(stopped_pid),
        "frozen_bundle": str(frozen),
        "new_real_capture_count_since_200k": new_capture_count,
        "new_capture_transition_ids": capture_ids,
        "three_plus_ring_steps": three_plus_steps,
        "two_plus_ring_steps": two_plus_steps,
        "repeated_two_plus_ring": len(set(two_plus_steps)) >= 2,
        "gate_positive": gate_positive,
        "conclusion": (
            "unexpected new multi-agent signal at 225k; line still stopped per explicit 225k cap"
            if gate_positive
            else "UTD=.5 finite but no new capture benefit; stop at 225k"
        ),
        "last_window": {
            key: last.get(key)
            for key in (
                "step", "update_count", "replay_size", "mean_finite", "collision_count",
                "d1_mean", "d1_min", "max_num_within_8m", "max_num_in_ring",
                "fraction_steps_any_in_ring", "fraction_steps_2plus_in_ring",
                "fraction_steps_3plus_in_ring",
            )
        },
        "diagnostic_eval_4episode": diagnostic,
    }


def cf1_command() -> list[str]:
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(CF1_CONFIG), "--seed", "2026081302", "--device", "cuda:1",
        "--total-steps", "100000", "--screen-episodes", "20",
        "--diagnostic-eval-episodes", "4", "--tag", CF1_TAG,
        "--artifact-root", str(CF1_ROOT),
    ]


def main() -> int:
    if not CF1_CONFIG.is_file():
        raise FileNotFoundError(CF1_CONFIG)
    CONTROL.mkdir(parents=True, exist_ok=True)
    log("armed: waiting for complete C1 step_000225000 + rolling resume_latest")
    while not milestone_complete():
        time.sleep(10)
    frozen = freeze_resume_bundle()
    pids = c1_process_ids()
    if len(pids) != 1:
        raise RuntimeError(f"expected exactly one C1 training PID at stop point, got {pids}")
    stopped_pid = pids[0]
    log(f"C1 225k milestone complete; sending SIGINT to PID {stopped_pid}")
    os.kill(stopped_pid, signal.SIGINT)
    deadline = time.monotonic() + 180.0
    while c1_process_ids() and time.monotonic() < deadline:
        time.sleep(2)
    if c1_process_ids():
        raise RuntimeError(f"C1 PID did not stop after SIGINT: {c1_process_ids()}")
    report = build_stop_report(stopped_pid, frozen)
    STOP_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"C1 stopped and audited: {report['conclusion']}")
    command = cf1_command()
    (CONTROL / "cf1_gpu1_selected_command.json").write_text(
        json.dumps(command, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    CF1_ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    env["PYTHONUNBUFFERED"] = "1"
    os.chdir(ROOT)
    log("C1 GPU allocation released; exec CF1 scratch on cuda:1")
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
