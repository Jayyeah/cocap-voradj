#!/usr/bin/env python3
"""Switch CF3 to P0 semantics at 25k, finish it, then launch P1 MaxPool."""
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
CF3_CONFIG = CONFIG_DIR / "legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml"
P1_CONFIG = CONFIG_DIR / "legacy_voradj_p1_capture_first_local_support_maxpool_4p1e1obs_100k_aw.yaml"
CONTROL = ROOT / "artifacts/2026-08-13_capture_first_controls/control"

CF3_SOURCE_TAG = "legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw_20260813"
CF3_SOURCE_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/cf3_local_support_full"
CF3_SOURCE_DIR = CF3_SOURCE_ROOT / CF3_SOURCE_TAG
CF3_SOURCE_METRICS = CF3_SOURCE_DIR / "metrics.jsonl"
CF3_SOURCE_CHECKPOINT = CF3_SOURCE_DIR / "checkpoints/step_000025000"
CF3_SOURCE_RESUME = CF3_SOURCE_DIR / "resume_latest"
CF3_SOURCE_FROZEN = CF3_SOURCE_DIR / "resume_frozen_pre_p0_step_000025000"

CF3_FIXED_TAG = "legacy_voradj_cf3_local_support_p0fixed_cont25k_to100k_20260813"
CF3_FIXED_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls/cf3_local_support_full_p0fixed"
CF3_FIXED_REPORT = CF3_FIXED_ROOT / CF3_FIXED_TAG / f"{CF3_FIXED_TAG}_report.json"

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


def _metric_rows() -> list[dict[str, Any]]:
    if not CF3_SOURCE_METRICS.is_file():
        return []
    return [
        json.loads(line)
        for line in CF3_SOURCE_METRICS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _runtime_step(bundle: Path) -> int:
    with (bundle / "runtime_state.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    return int(payload.get("transition_count", -1))


def source_25k_complete() -> bool:
    rows = _metric_rows()
    if not rows or int(rows[-1].get("step", -1)) < 25000:
        return False
    if not all((CF3_SOURCE_CHECKPOINT / name).is_file() for name in REQUIRED_CHECKPOINT):
        return False
    if not all((CF3_SOURCE_RESUME / name).is_file() for name in REQUIRED_RESUME):
        return False
    return _runtime_step(CF3_SOURCE_RESUME) == 25000


def freeze_source_bundle() -> Path:
    if CF3_SOURCE_FROZEN.is_dir():
        if not all((CF3_SOURCE_FROZEN / name).is_file() for name in REQUIRED_RESUME):
            raise RuntimeError(f"incomplete frozen CF3 source: {CF3_SOURCE_FROZEN}")
        return CF3_SOURCE_FROZEN
    temporary = CF3_SOURCE_FROZEN.with_name(CF3_SOURCE_FROZEN.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for source in CF3_SOURCE_RESUME.iterdir():
        if source.is_file():
            os.link(source, temporary / source.name)
    if not all((temporary / name).is_file() for name in REQUIRED_RESUME):
        raise RuntimeError("new frozen CF3 source is incomplete")
    temporary.rename(CF3_SOURCE_FROZEN)
    return CF3_SOURCE_FROZEN


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


def cf3_continuation_command() -> list[str]:
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(CF3_CONFIG), "--seed", "2026081304", "--device", "cuda:1",
        "--total-steps", "100000", "--screen-episodes", "20",
        "--diagnostic-eval-episodes", "4", "--tag", CF3_FIXED_TAG,
        "--artifact-root", str(CF3_FIXED_ROOT),
        "--resume-checkpoint", str(CF3_SOURCE_FROZEN / "trainer.pt"),
        "--resume-replay", str(CF3_SOURCE_FROZEN / "replay.pkl"),
        "--resume-step", "25000", "--resume-fork", "p0_semantics",
    ]


def p1_command(*, smoke: bool) -> list[str]:
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(P1_CONFIG), "--seed", "2026081305", "--device", "cuda:1",
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
    log("armed: CF3 pre-P0 25k -> P0-fixed 100k -> P1 Local-Max scratch")
    while not source_25k_complete():
        time.sleep(20)
    frozen = freeze_source_bundle()
    old_pids = process_ids(CF3_SOURCE_TAG)
    if len(old_pids) != 1:
        raise RuntimeError(f"expected one CF3 source process at handoff, got {old_pids}")
    os.kill(old_pids[0], signal.SIGINT)
    deadline = time.monotonic() + 180.0
    while process_ids(CF3_SOURCE_TAG) and time.monotonic() < deadline:
        time.sleep(2)
    if process_ids(CF3_SOURCE_TAG):
        raise RuntimeError(f"CF3 source did not stop after SIGINT: {process_ids(CF3_SOURCE_TAG)}")

    audit = {
        "schema_version": 1,
        "switched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_tag": CF3_SOURCE_TAG,
        "source_frozen_bundle": str(frozen),
        "switch_step": 25000,
        "old_semantics": "pre-P0",
        "new_semantics": "P0-fixed K10 effective roles; min_active=2; CE min_active=2",
        "cf3_continuation_command": cf3_continuation_command(),
        "p1_smoke_command": p1_command(smoke=True),
        "p1_formal_command": p1_command(smoke=False),
    }
    (CONTROL / "cf3_p0_to_p1_maxpool_schedule.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    env["PYTHONUNBUFFERED"] = "1"
    log("start CF3 P0-fixed continuation 25k -> 100k on cuda:1")
    subprocess.run(cf3_continuation_command(), cwd=ROOT, env=env, check=True)
    if not _finite_report(CF3_FIXED_REPORT, 100000):
        raise RuntimeError(f"CF3 P0-fixed final report is incomplete/non-finite: {CF3_FIXED_REPORT}")
    log("CF3 P0-fixed complete; start P1 Local-Max CUDA32 smoke")
    subprocess.run(p1_command(smoke=True), cwd=ROOT, env=env, check=True)
    smoke_report = P1_SMOKE_ROOT / P1_SMOKE_TAG / f"{P1_SMOKE_TAG}_report.json"
    if not _finite_report(smoke_report, 32):
        raise RuntimeError(f"P1 CUDA32 smoke is incomplete/non-finite: {smoke_report}")
    log("P1 smoke passed; exec P1 Local-Max scratch 0 -> 100k on cuda:1")
    command = p1_command(smoke=False)
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
