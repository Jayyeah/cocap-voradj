#!/usr/bin/env python3
"""Atomically swap CF3/P1 GPUs after both exact 50k resume bundles exist."""
from __future__ import annotations

import hashlib
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

import torch


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/2026-08-13_capture_first_controls/control"
AUDIT_PATH = CONTROL / "cf3_p1_gpu_swap_50k_audit.json"
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
        "device": "cuda:0",
        "total_steps": 200000,
        "screen_episodes": 20,
        "tmux": "cf3_gpu0_50k_to200k",
    },
    "p1": {
        "config": ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_p1_capture_first_local_support_maxpool_4p1e1obs_100k_aw.yaml",
        "seed": 2026081305,
        "tag": "legacy_voradj_p1_local_support_maxpool_p0fixed_100k_20260813",
        "artifact_root": ROOT / "artifacts/2026-08-13_capture_first_controls/p1_local_support_maxpool",
        "device": "cuda:1",
        "total_steps": 100000,
        "screen_episodes": 0,
        "tmux": "p1_gpu1_50k_to100k",
    },
}
for spec in LINES.values():
    spec["run_dir"] = Path(spec["artifact_root"]) / str(spec["tag"])
    spec["resume"] = Path(spec["run_dir"]) / "resume_latest"
    spec["frozen"] = Path(spec["run_dir"]) / "resume_frozen_gpu_swap_step_000050000"
    spec["metrics"] = Path(spec["run_dir"]) / "metrics.jsonl"


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %z"), message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_runtime(bundle: Path) -> dict[str, Any]:
    with (bundle / "runtime_state.pkl").open("rb") as handle:
        return dict(pickle.load(handle))


def last_metric(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty metrics: {path}")
    return rows[-1]


def inspect_bundle(name: str, bundle: Path) -> dict[str, Any]:
    missing = [item for item in REQUIRED if not (bundle / item).is_file()]
    if missing:
        raise RuntimeError(f"{name} incomplete 50k bundle: missing={missing}")
    runtime = load_runtime(bundle)
    metric = last_metric(bundle / "metrics.jsonl")
    trainer = torch.load(bundle / "trainer.pt", map_location="cpu", weights_only=False)
    replay = torch.load(bundle / "replay.pkl", map_location="cpu", weights_only=False) if False else None
    del replay
    required_trainer = (
        "actor", "critic1", "critic2", "target_critic1", "target_critic2", "log_alpha",
        "actor_optimizer", "critic_optimizer", "alpha_optimizer", "torch_rng_state",
        "python_rng_state", "numpy_rng_state", "runtime_state",
    )
    missing_trainer = [key for key in required_trainer if key not in trainer]
    if missing_trainer:
        raise RuntimeError(f"{name} trainer missing state: {missing_trainer}")
    step = int(runtime.get("transition_count", -1))
    replay_size = int(runtime.get("replay_size", step))
    update_count = int(runtime.get("update_count", -1))
    if step != 50000 or int(metric.get("step", -1)) != 50000:
        raise RuntimeError(f"{name} bundle is not exact 50k: runtime={step}, metric={metric.get('step')}")
    if int(metric.get("replay_size", -1)) != 50000:
        raise RuntimeError(f"{name} 50k metric replay mismatch: {metric.get('replay_size')}")
    if not bool(runtime.get("all_finite", True)) or float(metric.get("mean_finite", 0.0)) != 1.0:
        raise RuntimeError(f"{name} is not finite at 50k")
    optimizer_sizes = {
        key: len(dict(trainer[key]).get("state", {}))
        for key in ("actor_optimizer", "critic_optimizer", "alpha_optimizer")
    }
    if any(size <= 0 for size in optimizer_sizes.values()):
        raise RuntimeError(f"{name} optimizer state empty: {optimizer_sizes}")
    return {
        "step": step,
        "replay_size": int(metric["replay_size"]),
        "runtime_replay_size": replay_size,
        "update_count": update_count,
        "log_alpha": float(trainer["log_alpha"]),
        "optimizer_state_entries": optimizer_sizes,
        "target_critics_present": True,
        "rng_states_present": all(trainer.get(key) is not None for key in ("torch_rng_state", "python_rng_state", "numpy_rng_state"))
        and runtime.get("runner_rng_state") is not None,
        "manifest_sha256": sha256(bundle / "manifest.json"),
        "effective_config_sha256": sha256(bundle / "effective_config.yaml"),
        "trainer_sha256": sha256(bundle / "trainer.pt"),
        "replay_sha256": sha256(bundle / "replay.pkl"),
        "runtime_sha256": sha256(bundle / "runtime_state.pkl"),
    }


def ready(name: str, spec: dict[str, Any]) -> bool:
    bundle = Path(spec["resume"])
    if not all((bundle / item).is_file() for item in REQUIRED):
        return False
    try:
        runtime = load_runtime(bundle)
        metric = last_metric(bundle / "metrics.jsonl")
    except Exception:
        return False
    return int(runtime.get("transition_count", -1)) == 50000 and int(metric.get("step", -1)) == 50000


def freeze(name: str, source: Path, destination: Path) -> None:
    if destination.is_dir():
        if not all((destination / item).is_file() for item in REQUIRED):
            raise RuntimeError(f"{name} existing frozen bundle incomplete: {destination}")
        return
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file():
            os.link(item, temporary / item.name)
    if not all((temporary / item).is_file() for item in REQUIRED):
        raise RuntimeError(f"{name} frozen bundle incomplete")
    temporary.rename(destination)


def continuation_command(spec: dict[str, Any]) -> list[str]:
    frozen = Path(spec["frozen"])
    return [
        sys.executable, str(ROOT / "tools/run_continuous_ctde_training.py"),
        "--config", str(spec["config"]), "--seed", str(spec["seed"]),
        "--device", str(spec["device"]), "--total-steps", str(spec["total_steps"]),
        "--screen-episodes", str(spec["screen_episodes"]), "--diagnostic-eval-episodes", "4",
        "--tag", str(spec["tag"]), "--artifact-root", str(spec["artifact_root"]),
        "--resume-checkpoint", str(frozen / "trainer.pt"),
        "--resume-replay", str(frozen / "replay.pkl"), "--resume-step", "50000",
    ]


def stop_old_trainers(audit: dict[str, Any]) -> None:
    for name, spec in LINES.items():
        pids = process_ids(str(spec["tag"]))
        if len(pids) != 1:
            raise RuntimeError(f"expected exactly one active {name} trainer before swap, got {pids}")
        audit["lines"][name]["pre_swap_pid"] = pids[0]
        os.kill(pids[0], signal.SIGINT)
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        if not any(process_ids(str(spec["tag"])) for spec in LINES.values()):
            return
        time.sleep(2)
    raise RuntimeError("old trainers did not stop after SIGINT")


def normalize_metrics_to_frozen(name: str, spec: dict[str, Any], audit: dict[str, Any]) -> None:
    """Archive any uncheckpointed >50k tail and restart metrics from exact frozen history."""
    active_metrics = Path(spec["metrics"])
    frozen_metrics = Path(spec["frozen"]) / "metrics.jsonl"
    active_rows = [
        json.loads(line) for line in active_metrics.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    frozen_rows = [
        json.loads(line) for line in frozen_metrics.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    discarded = [row for row in active_rows if int(row.get("step", -1)) > 50000]
    if discarded:
        archive = Path(spec["run_dir"]) / "metrics_pre_gpu_swap_uncheckpointed_tail.jsonl"
        archive.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in discarded), encoding="utf-8"
        )
        audit["lines"][name]["uncheckpointed_tail_archive"] = str(archive)
    shutil.copy2(frozen_metrics, active_metrics)
    audit["lines"][name]["uncheckpointed_metric_steps_discarded"] = [
        int(row["step"]) for row in discarded
    ]
    audit["lines"][name]["metrics_reset_to_exact_step"] = int(frozen_rows[-1]["step"])


def launch(name: str, spec: dict[str, Any], audit: dict[str, Any]) -> None:
    command = continuation_command(spec)
    log_path = Path(spec["run_dir"]) / f"gpu_swap_50k_{str(spec['device']).replace(':', '')}.log"
    shell_command = "exec env PYTHONPATH=" + str(ROOT / "src") + ":" + str(ROOT) + " PYTHONUNBUFFERED=1 "
    shell_command += subprocess.list2cmdline(command) + " >> " + subprocess.list2cmdline([str(log_path)]) + " 2>&1"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", str(spec["tmux"]), "-c", str(ROOT), shell_command],
        check=True,
    )
    audit["lines"][name]["continuation_command"] = command
    audit["lines"][name]["log"] = str(log_path)


def verify_started(name: str, spec: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + 1800.0
    while time.monotonic() < deadline:
        pids = process_ids(str(spec["tag"]))
        if len(pids) == 1:
            try:
                metric = last_metric(Path(spec["metrics"]))
            except Exception:
                metric = {}
            step = int(metric.get("step", -1))
            if step >= 51000:
                if step != 51000 or int(metric.get("replay_size", -1)) != 51000:
                    raise RuntimeError(f"{name} post-resume discontinuity: step={step}, replay={metric.get('replay_size')}")
                if int(metric.get("update_count", -1)) <= int(source["update_count"]):
                    raise RuntimeError(f"{name} update_count did not continue")
                if float(metric.get("mean_finite", 0.0)) != 1.0:
                    raise RuntimeError(f"{name} post-resume metric non-finite")
                cmdline = Path(f"/proc/{pids[0]}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
                if str(spec["device"]) not in cmdline or str(spec["frozen"]) not in cmdline:
                    raise RuntimeError(f"{name} resumed with wrong GPU/bundle: {cmdline}")
                return {
                    "post_swap_pid": pids[0], "first_verified_step": step,
                    "replay_size": int(metric["replay_size"]), "update_count": int(metric["update_count"]),
                    "finite": True, "device": spec["device"], "tmux": spec["tmux"],
                    "resume_state_verified": True,
                }
        time.sleep(10)
    raise RuntimeError(f"timeout waiting for {name} first complete post-resume metrics window")


def write_audit(audit: dict[str, Any]) -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(AUDIT_PATH)


def main() -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "armed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "WAITING_BOTH_50K",
        "policy": "exact own-line 50k full-state resume; fail closed on any mismatch",
        "lines": {name: {} for name in LINES},
    }
    write_audit(audit)
    log("armed: wait for both CF3 and P1 exact 50k rolling resume bundles")
    while not all(ready(name, spec) for name, spec in LINES.items()):
        time.sleep(15)
    log("both exact 50k bundles present; deep validation and hard-link freeze")
    for name, spec in LINES.items():
        source = inspect_bundle(name, Path(spec["resume"]))
        freeze(name, Path(spec["resume"]), Path(spec["frozen"]))
        frozen = inspect_bundle(name, Path(spec["frozen"]))
        for key in ("trainer_sha256", "replay_sha256", "runtime_sha256", "manifest_sha256", "effective_config_sha256"):
            if source[key] != frozen[key]:
                raise RuntimeError(f"{name} freeze hash mismatch: {key}")
        audit["lines"][name]["source_50k"] = source
        audit["lines"][name]["frozen_bundle"] = str(spec["frozen"])
    audit["status"] = "BUNDLES_FROZEN_VALIDATED"
    write_audit(audit)
    stop_old_trainers(audit)
    audit["status"] = "OLD_TRAINERS_STOPPED"
    for name, spec in LINES.items():
        normalize_metrics_to_frozen(name, spec, audit)
    write_audit(audit)
    for name, spec in LINES.items():
        launch(name, spec, audit)
    audit["status"] = "RESUME_LAUNCHED"
    write_audit(audit)
    for name, spec in LINES.items():
        audit["lines"][name]["post_resume"] = verify_started(
            name, spec, audit["lines"][name]["source_50k"]
        )
    audit["status"] = "VERIFIED_100_PERCENT_COMPLETE"
    audit["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_audit(audit)
    log("GPU swap complete and both 51k windows verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        CONTROL.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1, "status": "FAILED_CLOSED",
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error": f"{type(exc).__name__}: {exc}",
        }
        (CONTROL / "cf3_p1_gpu_swap_50k_failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise
