#!/usr/bin/env python3
"""Gate CF0 at 100k and continue CF0 or start CF1 without user intervention."""
from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809"
CF0_CONFIG = CONFIG_DIR / "legacy_voradj_cf0_capture_first_local_4p1e1obs_100k_aw.yaml"
CF1_CONFIG = CONFIG_DIR / "legacy_voradj_cf1_capture_first_global_enemy_4p1e1obs_100k_aw.yaml"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-13_capture_first_controls"
CF0_ROOT = ARTIFACT_ROOT / "cf0_local"
CF1_ROOT = ARTIFACT_ROOT / "cf1_global"
CONTROL_ROOT = ARTIFACT_ROOT / "control"
CF0_TAG = "legacy_voradj_cf0_capture_first_local_4p1e1obs_100k_aw_20260813"
CF1_TAG = "legacy_voradj_cf1_capture_first_global_enemy_4p1e1obs_100k_aw_20260813"
CF0_DIR = CF0_ROOT / CF0_TAG
CF0_REPORT = CF0_DIR / f"{CF0_TAG}_report.json"
CF0_METRICS = CF0_DIR / "metrics.jsonl"
CF0_RESUME = CF0_DIR / "resume_latest"
CF0_FROZEN = CF0_DIR / "resume_frozen_cf0_step_000100000"
GATE_REPORT = CONTROL_ROOT / "cf0_gate_100k.json"


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in CF0_METRICS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def window(rows: list[dict[str, Any]], lower: int, upper: int) -> list[dict[str, Any]]:
    return [row for row in rows if lower < int(row.get("step", -1)) <= upper]


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if finite(row.get(key))]
    return sum(values) / len(values) if values else None


def relative_drop(before: float | None, after: float | None, threshold: float) -> bool:
    return bool(before is not None and after is not None and before > 0.0 and after <= before * (1.0 - threshold))


def evaluate_gate(rows: list[dict[str, Any]], capture_events: int = 0) -> dict[str, Any]:
    early = window(rows, 70000, 75000)
    late = window(rows, 95000, 100000)
    captures = int(capture_events)
    three_plus_rows = [
        int(row["step"]) for row in rows
        if int(row.get("max_num_in_ring", 0)) >= 3
        or float(row.get("fraction_steps_3plus_in_ring", 0.0)) > 0.0
    ]
    two_plus_rows = [
        int(row["step"]) for row in rows
        if float(row.get("fraction_steps_2plus_in_ring", 0.0)) > 0.0
    ]
    strong = {
        "real_capture": captures >= 1,
        "three_plus_ring": bool(three_plus_rows),
        "repeated_two_plus_ring_windows": len(set(two_plus_rows)) >= 2,
        "repeatable_nonzero_two_plus_fraction": sum(
            float(row.get("fraction_steps_2plus_in_ring", 0.0)) >= 0.005 for row in rows
        ) >= 2,
    }

    early_collision = (
        sum(float(row.get("collision_count", 0)) for row in early)
        / max(1.0, sum(float(row.get("window_env_steps", 0)) for row in early))
    )
    late_collision = (
        sum(float(row.get("collision_count", 0)) for row in late)
        / max(1.0, sum(float(row.get("window_env_steps", 0)) for row in late))
    )
    early_any_ring = mean(early, "fraction_steps_any_in_ring")
    late_any_ring = mean(late, "fraction_steps_any_in_ring")
    early_two_ring = mean(early, "fraction_steps_2plus_in_ring")
    late_two_ring = mean(late, "fraction_steps_2plus_in_ring")
    medium = {
        "collision_drop_25pct": early_collision > 0.0 and late_collision <= 0.75 * early_collision,
        "mean_enemy_distance_drop_10pct": relative_drop(mean(early, "d1_mean"), mean(late, "d1_mean"), 0.10),
        "min_enemy_distance_drop_15pct": relative_drop(mean(early, "d1_min"), mean(late, "d1_min"), 0.15),
        "single_ring_fraction_sustained_gain": bool(
            early_any_ring is not None
            and late_any_ring is not None
            and late_any_ring >= max(early_any_ring * 1.5, early_any_ring + 0.002)
        ),
        "two_ring_fraction_75k_to_100k_gain": bool(
            early_two_ring is not None
            and late_two_ring is not None
            and late_two_ring >= max(early_two_ring * 1.5, early_two_ring + 0.001)
        ),
    }
    strong_names = [key for key, passed in strong.items() if passed]
    medium_names = [key for key, passed in medium.items() if passed]
    passed = bool(strong_names or len(medium_names) >= 2)
    return {
        "schema_version": 1,
        "decision": "DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL",
        "controls_training_branch": False,
        "next_action": "CF0_CONTINUES_TO_200K_REGARDLESS_OF_GATE",
        "passed": passed,
        "rule": "any strong signal OR at least two sustained medium signals",
        "distinct_real_capture_events_from_replay": captures,
        "three_plus_ring_steps": three_plus_rows,
        "two_plus_ring_steps": two_plus_rows,
        "strong_signals": strong,
        "medium_signals": medium,
        "strong_passes": strong_names,
        "medium_passes": medium_names,
        "windows": {
            "early_70_75k_records": len(early),
            "late_95_100k_records": len(late),
            "early_collision_fraction": early_collision,
            "late_collision_fraction": late_collision,
            "early_d1_mean": mean(early, "d1_mean"),
            "late_d1_mean": mean(late, "d1_mean"),
            "early_d1_min": mean(early, "d1_min"),
            "late_d1_min": mean(late, "d1_min"),
            "early_any_ring_fraction": early_any_ring,
            "late_any_ring_fraction": late_any_ring,
            "early_two_plus_ring_fraction": early_two_ring,
            "late_two_plus_ring_fraction": late_two_ring,
        },
    }


def replay_capture_event_count() -> int:
    with (CF0_RESUME / "replay.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    return sum(
        "capture" in set((transition.metadata or {}).get("event_ids", []))
        for _transition_id, transition in payload.get("records", [])
    )


def freeze_resume_bundle() -> Path:
    required = ("trainer.pt", "replay.pkl", "runtime_state.pkl", "manifest.json", "effective_config.yaml")
    missing = [name for name in required if not (CF0_RESUME / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CF0 resume bundle missing: {missing}")
    if CF0_FROZEN.is_dir():
        for name in required:
            if not (CF0_FROZEN / name).is_file():
                raise RuntimeError(f"existing CF0 frozen bundle incomplete: {name}")
        return CF0_FROZEN
    temporary = CF0_FROZEN.with_name(CF0_FROZEN.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for source in CF0_RESUME.iterdir():
        if source.is_file():
            os.link(source, temporary / source.name)
    temporary.rename(CF0_FROZEN)
    return CF0_FROZEN


def gpu0_compute_pids() -> list[int]:
    gpu_uuid = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[0] == gpu_uuid:
            pids.append(int(parts[1]))
    return pids


def training_command(_gate: dict[str, Any]) -> list[str]:
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(CF0_CONFIG), "--seed", "2026081301", "--device", "cuda:0",
        "--total-steps", "200000", "--screen-episodes", "20",
        "--diagnostic-eval-episodes", "4",
        "--resume-checkpoint", str(CF0_FROZEN / "trainer.pt"),
        "--resume-replay", str(CF0_FROZEN / "replay.pkl"),
        "--resume-step", "100000", "--tag", CF0_TAG, "--artifact-root", str(CF0_ROOT),
    ]


def main() -> int:
    for path in (CF0_CONFIG,):
        if not path.is_file():
            raise FileNotFoundError(path)
    CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    log("armed: waiting for successful CF0 100k report")
    while not CF0_REPORT.is_file():
        time.sleep(60)
    report = json.loads(CF0_REPORT.read_text(encoding="utf-8"))
    if (
        int(report.get("transition_count", -1)) != 100000
        or not bool(report.get("all_finite", False))
        or not CF0_METRICS.is_file()
        or not all((CF0_RESUME / name).is_file() for name in ("trainer.pt", "replay.pkl", "runtime_state.pkl"))
    ):
        raise RuntimeError(f"CF0 final report/bundle is incomplete or non-finite: {CF0_REPORT}")
    gate = evaluate_gate(load_metrics(), replay_capture_event_count())
    gate["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    gate["cf0_report"] = str(CF0_REPORT)
    frozen = freeze_resume_bundle()
    gate["frozen_bundle"] = str(frozen)
    GATE_REPORT.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"diagnostic gate: {gate['decision']}; frozen={frozen}; CF0 always continues to 200k")
    while gpu0_compute_pids():
        log(f"waiting for GPU0 release: {gpu0_compute_pids()}")
        time.sleep(30)
    command = training_command(gate)
    (CONTROL_ROOT / "cf0_gate_selected_command.json").write_text(
        json.dumps(command, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    env["PYTHONUNBUFFERED"] = "1"
    os.chdir(ROOT)
    log("GPU0 free; exec CF0 100k -> 200k continuation")
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
