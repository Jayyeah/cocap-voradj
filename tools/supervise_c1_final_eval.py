#!/usr/bin/env python3
"""Launch the C1 final representative rollout sidecar after a valid 300k report."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "legacy_voradj_oldmix_allagent_noclip_c1_utd05_300k_aw_20260812"
RUN_DIR = ROOT / "artifacts/2026-08-12_p1_extension_controls/c1_utd05" / TAG
REPORT = RUN_DIR / f"{TAG}_report.json"
CHECKPOINT = RUN_DIR / "checkpoints/step_000300000/trainer.pt"
CONFIG = RUN_DIR / "effective_config.yaml"
OUTPUT = ROOT / "artifacts/2026-08-13_extension_final_evals/c1_300k_formal20_representative5"
PRESET = ROOT / "configs/evaluation/masac_rollout_gif_20260811/old_mix.yaml"


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def main() -> int:
    log("armed: waiting for successful C1 300k final report")
    while not REPORT.is_file():
        time.sleep(60)
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    if int(payload.get("transition_count", -1)) != 300000 or not bool(payload.get("all_finite", False)):
        raise RuntimeError(f"C1 final report is incomplete or non-finite: {REPORT}")
    for path in (CHECKPOINT, CONFIG, PRESET):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [
        sys.executable,
        str(ROOT / "tools/run_representative_masac_rollouts.py"),
        "--preset-config", str(PRESET),
        "--config", str(CONFIG),
        "--checkpoint", str(CHECKPOINT),
        "--output-root", str(OUTPUT),
        "--episodes", "20",
        "--gif-count", "5",
        "--seed", "2026081201",
        "--workers", "2",
    ]
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    os.chdir(ROOT)
    log("C1 300k valid; exec low-priority CPU evaluation sidecar")
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
