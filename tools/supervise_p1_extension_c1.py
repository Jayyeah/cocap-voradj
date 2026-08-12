#!/usr/bin/env python3
"""Queue C1 behind a successful C0 without sharing its GPU."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C0_TAG = "legacy_voradj_oldmix_allagent_noclip_c0_utd025_300k_aw_20260812"
C0_DIR = ROOT / "artifacts/2026-08-12_p1_extension_controls/c0_utd025" / C0_TAG
C0_REPORT = C0_DIR / f"{C0_TAG}_report.json"
P1_FROZEN = (
    ROOT
    / "artifacts/2026-08-11_allagent_noclip_scratch"
    / "legacy_voradj_oldmix_allagent_noclip_scratch_4p1e1obs_200k_aw_20260811"
    / "resume_frozen_p1_step_000200000"
)
C1_CONFIG = ROOT / (
    "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_oldmix_4p1e1obs_300k_aw_allagent_noclip_c1_utd05.yaml"
)
C1_ROOT = ROOT / "artifacts/2026-08-12_p1_extension_controls/c1_utd05"
C1_TAG = "legacy_voradj_oldmix_allagent_noclip_c1_utd05_300k_aw_20260812"


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def gpu0_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_uuid = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == gpu_uuid:
            rows.append(int(parts[1]))
    return rows


def main() -> int:
    required = [C1_CONFIG, P1_FROZEN / "trainer.pt", P1_FROZEN / "replay.pkl"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"C1 prerequisites missing: {missing}")
    log("armed: waiting for successful C0 report")
    while True:
        if C0_REPORT.is_file():
            report = json.loads(C0_REPORT.read_text(encoding="utf-8"))
            if (
                int(report.get("transition_count", -1)) == 300000
                and int(report.get("replay_size", -1)) == 300000
                and bool(report.get("all_finite", False))
                and Path(report.get("final_checkpoint_bundle", "")).is_dir()
            ):
                break
            raise RuntimeError(f"C0 report exists but is not a successful 300k result: {C0_REPORT}")
        time.sleep(60)
    while gpu0_compute_pids():
        log(f"C0 report complete; waiting for GPU0 release: {gpu0_compute_pids()}")
        time.sleep(30)
    C1_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(C1_CONFIG), "--seed", "2026080902", "--device", "cuda:0",
        "--total-steps", "300000", "--screen-episodes", "20",
        "--diagnostic-eval-episodes", "4", "--resume-checkpoint",
        str(P1_FROZEN / "trainer.pt"), "--resume-replay", str(P1_FROZEN / "replay.pkl"),
        "--resume-step", "200000", "--resume-fork", "utd_only",
        "--tag", C1_TAG, "--artifact-root", str(C1_ROOT),
    ]
    log("C0 complete and GPU0 free; exec C1 from frozen P1-200k bundle")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)
