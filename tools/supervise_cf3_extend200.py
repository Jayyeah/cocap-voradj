#!/usr/bin/env python3
"""Freeze CF3 at 100k and resume the exact P0-fixed line to 200k on cuda:1."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml"
TAG = "legacy_voradj_cf3_local_support_p0fixed_recovery25k_to200k_20260813"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/cf3_local_support_full_p0fixed_recovery"
RUN_DIR = ARTIFACT_ROOT / TAG
REPORT = RUN_DIR / f"{TAG}_report.json"
RESUME = RUN_DIR / "resume_latest"
FROZEN = RUN_DIR / "resume_frozen_cf3_p0fixed_step_000100000"
CONTROL = ROOT / "artifacts/2026-08-13_capture_first_controls/control"
REQUIRED = (
    "trainer.pt", "replay.pkl", "runtime_state.pkl", "manifest.json",
    "effective_config.yaml", "metrics.jsonl", "diagnostic_eval.json", "checkpoint_storage.json",
)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def complete() -> bool:
    if not REPORT.is_file() or not all((RESUME / name).is_file() for name in REQUIRED):
        return False
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    return bool(
        int(report.get("transition_count", -1)) == 100000
        and int(report.get("replay_size", -1)) == 100000
        and report.get("all_finite", False)
    )


def freeze() -> Path:
    if FROZEN.is_dir():
        if not all((FROZEN / name).is_file() for name in REQUIRED):
            raise RuntimeError(f"incomplete existing CF3 frozen bundle: {FROZEN}")
        return FROZEN
    temporary = FROZEN.with_name(FROZEN.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for source in RESUME.iterdir():
        if source.is_file():
            os.link(source, temporary / source.name)
    if not all((temporary / name).is_file() for name in REQUIRED):
        raise RuntimeError("new CF3 frozen bundle incomplete")
    temporary.rename(FROZEN)
    return FROZEN


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


def continuation_command() -> list[str]:
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(CONFIG), "--seed", "2026081304", "--device", "cuda:1",
        "--total-steps", "200000", "--screen-episodes", "20",
        "--diagnostic-eval-episodes", "4", "--tag", TAG,
        "--artifact-root", str(ARTIFACT_ROOT),
        "--resume-checkpoint", str(FROZEN / "trainer.pt"),
        "--resume-replay", str(FROZEN / "replay.pkl"),
        "--resume-step", "100000",
    ]


def main() -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    log("armed: CF3 P0-fixed 100k freeze -> exact continuation to 200k on cuda:1")
    while not complete():
        time.sleep(30)
    deadline = time.monotonic() + 180.0
    while process_ids() and time.monotonic() < deadline:
        time.sleep(2)
    if process_ids():
        raise RuntimeError(f"CF3 100k trainer did not exit normally: {process_ids()}")
    frozen = freeze()
    command = continuation_command()
    audit: dict[str, Any] = {
        "schema_version": 1,
        "armed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_bundle": str(frozen),
        "resume_step": 100000,
        "target_step": 200000,
        "continuation": "exact_same_P0_fixed_local_support_contract",
        "command": command,
    }
    (CONTROL / "cf3_100k_to_200k_schedule.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    env["PYTHONUNBUFFERED"] = "1"
    log("exec CF3 exact continuation 100k -> 200k on cuda:1")
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
