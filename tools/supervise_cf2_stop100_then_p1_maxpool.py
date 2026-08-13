#!/usr/bin/env python3
"""Freeze/stop CF2 at 100k, then launch P1 Local-Max on cuda:0."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809"
P1_CONFIG = CONFIG_DIR / "legacy_voradj_p1_capture_first_local_support_maxpool_4p1e1obs_100k_aw.yaml"
CONTROL = ROOT / "artifacts/2026-08-13_capture_first_controls/control"

CF2_TAG = "legacy_voradj_cf2_global_support_p0fixed_cont75k_to200k_20260813"
CF2_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/cf2_global_support_full_p0fixed"
CF2_DIR = CF2_ROOT / CF2_TAG
CF2_METRICS = CF2_DIR / "metrics.jsonl"
CF2_CHECKPOINT = CF2_DIR / "checkpoints/step_000100000"
CF2_RESUME = CF2_DIR / "resume_latest"
CF2_FROZEN = CF2_DIR / "resume_frozen_cf2_p0fixed_step_000100000"

P1_TAG = "legacy_voradj_p1_local_support_maxpool_p0fixed_100k_20260813"
P1_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/p1_local_support_maxpool"
P1_SMOKE_TAG = "legacy_voradj_p1_local_support_maxpool_cuda32_smoke_20260813"
P1_SMOKE_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/p1_local_support_maxpool_smoke"

REQUIRED_CHECKPOINT = (
    "trainer.pt", "runtime_state.pkl", "manifest.json", "effective_config.yaml",
    "metrics.jsonl", "diagnostic_eval.json", "checkpoint_storage.json",
)
REQUIRED_RESUME = REQUIRED_CHECKPOINT + ("replay.pkl",)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def _runtime_step(bundle: Path) -> int:
    with (bundle / "runtime_state.pkl").open("rb") as handle:
        return int(pickle.load(handle).get("transition_count", -1))


def _latest_metric_step() -> int:
    if not CF2_METRICS.is_file():
        return -1
    rows = [line for line in CF2_METRICS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return int(json.loads(rows[-1]).get("step", -1)) if rows else -1


def cf2_100k_complete() -> bool:
    return bool(
        _latest_metric_step() >= 100000
        and all((CF2_CHECKPOINT / name).is_file() for name in REQUIRED_CHECKPOINT)
        and all((CF2_RESUME / name).is_file() for name in REQUIRED_RESUME)
        and _runtime_step(CF2_RESUME) == 100000
    )


def freeze_cf2() -> Path:
    if CF2_FROZEN.is_dir():
        if not all((CF2_FROZEN / name).is_file() for name in REQUIRED_RESUME):
            raise RuntimeError(f"incomplete existing CF2 frozen bundle: {CF2_FROZEN}")
        return CF2_FROZEN
    temporary = CF2_FROZEN.with_name(CF2_FROZEN.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for source in CF2_RESUME.iterdir():
        if source.is_file():
            os.link(source, temporary / source.name)
    if not all((temporary / name).is_file() for name in REQUIRED_RESUME):
        raise RuntimeError("new CF2 frozen bundle is incomplete")
    temporary.rename(CF2_FROZEN)
    return CF2_FROZEN


def process_ids(tag: str) -> list[int]:
    marker = f"--tag\x00{tag}".encode()
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


def p1_command(*, smoke: bool) -> list[str]:
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(P1_CONFIG), "--seed", "2026081305", "--device", "cuda:0",
        "--total-steps", "32" if smoke else "100000",
        "--screen-episodes", "0", "--diagnostic-eval-episodes", "0" if smoke else "4",
        "--tag", P1_SMOKE_TAG if smoke else P1_TAG,
        "--artifact-root", str(P1_SMOKE_ROOT if smoke else P1_ROOT),
    ]


def _finite_report(path: Path, expected_step: int) -> bool:
    if not path.is_file():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        int(report.get("transition_count", -1)) == expected_step
        and int(report.get("replay_size", -1)) == expected_step
        and report.get("all_finite", False)
    )


def main() -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    log("armed: CF2 P0-fixed 100k freeze/stop -> P1 Local-Max scratch on cuda:0")
    while not cf2_100k_complete():
        time.sleep(20)
    frozen = freeze_cf2()
    pids = process_ids(CF2_TAG)
    if len(pids) != 1:
        raise RuntimeError(f"expected one CF2 trainer at 100k handoff, got {pids}")
    log(f"CF2 100k bundle frozen at {frozen}; send SIGINT to trainer {pids[0]}")
    os.kill(pids[0], signal.SIGINT)
    deadline = time.monotonic() + 180.0
    while process_ids(CF2_TAG) and time.monotonic() < deadline:
        time.sleep(2)
    if process_ids(CF2_TAG):
        raise RuntimeError(f"CF2 did not stop after safe 100k handoff: {process_ids(CF2_TAG)}")

    audit: dict[str, Any] = {
        "schema_version": 1,
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "CF2_TRUNCATE_100K_THEN_P1_CUDA0",
        "reason": "CF3 local P0-fixed produced repeated 2+ ring windows and directional support-follow signals",
        "cf2_frozen_bundle": str(frozen),
        "cf2_switch_step": 75000,
        "cf2_stop_step": 100000,
        "p1_smoke_command": p1_command(smoke=True),
        "p1_formal_command": p1_command(smoke=False),
    }
    (CONTROL / "cf2_stop100_then_p1_maxpool_schedule.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    env["PYTHONUNBUFFERED"] = "1"
    log("start P1 Local-Max CUDA32 smoke on cuda:0")
    subprocess.run(p1_command(smoke=True), cwd=ROOT, env=env, check=True)
    smoke_report = P1_SMOKE_ROOT / P1_SMOKE_TAG / f"{P1_SMOKE_TAG}_report.json"
    if not _finite_report(smoke_report, 32):
        raise RuntimeError(f"P1 CUDA32 smoke incomplete/non-finite: {smoke_report}")
    command = p1_command(smoke=False)
    log("P1 smoke passed; exec P1 Local-Max scratch 0 -> 100k on cuda:0")
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
