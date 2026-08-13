#!/usr/bin/env python3
"""Recover CF3/P1 from their latest complete 75k bundles after host power loss."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/2026-08-13_capture_first_controls/control"
AUDIT = CONTROL / "cf3_p1_power_loss_recovery_20260814.json"
REQUIRED = (
    "trainer.pt", "replay.pkl", "runtime_state.pkl", "manifest.json",
    "effective_config.yaml", "metrics.jsonl", "diagnostic_eval.json", "checkpoint_storage.json",
)
LINES: dict[str, dict[str, Any]] = {
    "cf3": {
        "config": ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml",
        "seed": 2026081304,
        "tag": "legacy_voradj_cf3_local_support_p0fixed_recovery25k_to200k_20260813",
        "artifact_root": ROOT / "artifacts/2026-08-13_capture_first_controls/cf3_local_support_full_p0fixed_recovery",
        "device": "cuda:0", "total_steps": 200000, "screen_episodes": 20,
        "tmux": "cf3_gpu0_power_recovery75k_to200k",
    },
    "p1": {
        "config": ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_p1_capture_first_local_support_maxpool_4p1e1obs_100k_aw.yaml",
        "seed": 2026081305,
        "tag": "legacy_voradj_p1_local_support_maxpool_p0fixed_100k_20260813",
        "artifact_root": ROOT / "artifacts/2026-08-13_capture_first_controls/p1_local_support_maxpool",
        "device": "cuda:1", "total_steps": 100000, "screen_episodes": 0,
        "tmux": "p1_gpu1_power_recovery75k_to100k",
    },
}
for spec in LINES.values():
    spec["run_dir"] = Path(spec["artifact_root"]) / str(spec["tag"])
    spec["resume"] = Path(spec["run_dir"]) / "resume_latest"
    spec["frozen"] = Path(spec["run_dir"]) / "resume_frozen_power_loss_step_000075000"
    spec["metrics"] = Path(spec["run_dir"]) / "metrics.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def validate_and_freeze(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    resume = Path(spec["resume"])
    missing = [item for item in REQUIRED if not (resume / item).is_file()]
    if missing:
        raise RuntimeError(f"{name} resume bundle incomplete: {missing}")
    storage = json.loads((resume / "checkpoint_storage.json").read_text(encoding="utf-8"))
    rows = metric_rows(resume / "metrics.jsonl")
    last = rows[-1]
    if not (
        int(storage.get("step", -1)) == 75000
        and storage.get("contains_replay") is True
        and int(last.get("step", -1)) == 75000
        and int(last.get("replay_size", -1)) == 75000
        and int(last.get("update_count", -1)) == 17501
        and float(last.get("mean_finite", 0.0)) == 1.0
    ):
        raise RuntimeError(f"{name} safe metadata does not describe exact finite 75k bundle")
    frozen = Path(spec["frozen"])
    if not frozen.exists():
        temporary = frozen.with_name(frozen.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        for source in resume.iterdir():
            if source.is_file():
                os.link(source, temporary / source.name)
        if not all((temporary / item).is_file() for item in REQUIRED):
            raise RuntimeError(f"{name} frozen copy incomplete")
        temporary.rename(frozen)
    source_hashes = {
        item: file_sha256(resume / item)
        for item in ("trainer.pt", "replay.pkl", "runtime_state.pkl", "manifest.json", "effective_config.yaml")
    }
    frozen_hashes = {item: file_sha256(frozen / item) for item in source_hashes}
    if source_hashes != frozen_hashes:
        raise RuntimeError(f"{name} frozen hashes do not match resume_latest")
    return {
        "step": 75000, "replay_size": 75000, "update_count": 17501,
        "finite": True, "frozen_bundle": str(frozen), "sha256": source_hashes,
    }


def normalize_metrics(name: str, spec: dict[str, Any], audit: dict[str, Any]) -> None:
    active = metric_rows(Path(spec["metrics"]))
    frozen = metric_rows(Path(spec["frozen"]) / "metrics.jsonl")
    tail = [row for row in active if int(row.get("step", -1)) > 75000]
    if tail:
        archive = Path(spec["run_dir"]) / "metrics_pre_power_loss_uncheckpointed_tail_20260814.jsonl"
        archive.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tail), encoding="utf-8"
        )
        audit["lines"][name]["uncheckpointed_tail_archive"] = str(archive)
    shutil.copy2(Path(spec["frozen"]) / "metrics.jsonl", Path(spec["metrics"]))
    audit["lines"][name]["discarded_unrecoverable_metric_steps"] = [int(row["step"]) for row in tail]
    audit["lines"][name]["metrics_reset_to_step"] = int(frozen[-1]["step"])


def command(spec: dict[str, Any]) -> list[str]:
    frozen = Path(spec["frozen"])
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(spec["config"]), "--seed", str(spec["seed"]),
        "--device", str(spec["device"]), "--total-steps", str(spec["total_steps"]),
        "--screen-episodes", str(spec["screen_episodes"]), "--diagnostic-eval-episodes", "4",
        "--tag", str(spec["tag"]), "--artifact-root", str(spec["artifact_root"]),
        "--resume-checkpoint", str(frozen / "trainer.pt"),
        "--resume-replay", str(frozen / "replay.pkl"), "--resume-step", "75000",
    ]


def launch(name: str, spec: dict[str, Any], audit: dict[str, Any]) -> None:
    if process_ids(str(spec["tag"])):
        raise RuntimeError(f"{name} trainer already exists; refusing duplicate launch")
    cmd = command(spec)
    log_path = Path(spec["run_dir"]) / "power_loss_recovery_75k.log"
    shell = "exec env PYTHONPATH=" + shlex.quote(f"{ROOT / 'src'}:{ROOT}") + " PYTHONUNBUFFERED=1 "
    shell += shlex.join(cmd) + " >> " + shlex.quote(str(log_path)) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", str(spec["tmux"]), "-c", str(ROOT), shell], check=True)
    audit["lines"][name]["command"] = cmd
    audit["lines"][name]["log"] = str(log_path)


def verify(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        pids = process_ids(str(spec["tag"]))
        if len(pids) == 1:
            rows = metric_rows(Path(spec["metrics"]))
            if rows and int(rows[-1].get("step", -1)) >= 76000:
                last = rows[-1]
                if not (
                    int(last.get("step", -1)) == 76000
                    and int(last.get("replay_size", -1)) == 76000
                    and int(last.get("update_count", -1)) == 17751
                    and float(last.get("mean_finite", 0.0)) == 1.0
                ):
                    raise RuntimeError(f"{name} first recovered window discontinuous: {last}")
                return {
                    "pid": pids[0], "tmux": spec["tmux"], "device": spec["device"],
                    "step": 76000, "replay_size": 76000, "update_count": 17751,
                    "finite": True, "env_steps_per_second": last.get("env_steps_per_second"),
                }
        time.sleep(10)
    raise RuntimeError(f"timeout waiting for {name} finite 76k recovery window")


def write_audit(payload: dict[str, Any]) -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(AUDIT)


def main() -> int:
    audit: dict[str, Any] = {
        "schema_version": 1, "status": "VALIDATING_75K",
        "reason": "host restart/power loss; no trainer or supervisor survived",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "validation_policy": "safe JSON metadata + hashes before project runner strict load",
        "lines": {name: {} for name in LINES},
    }
    write_audit(audit)
    for name, spec in LINES.items():
        audit["lines"][name]["source"] = validate_and_freeze(name, spec)
    audit["status"] = "FROZEN_HASH_VERIFIED"
    write_audit(audit)
    for name, spec in LINES.items():
        normalize_metrics(name, spec, audit)
    for name, spec in LINES.items():
        launch(name, spec, audit)
    audit["status"] = "LAUNCHED"
    write_audit(audit)
    for name, spec in LINES.items():
        audit["lines"][name]["post_resume"] = verify(name, spec)
    audit["status"] = "VERIFIED_100_PERCENT_COMPLETE"
    audit["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_audit(audit)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        write_audit({
            "schema_version": 1, "status": "FAILED_CLOSED",
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
