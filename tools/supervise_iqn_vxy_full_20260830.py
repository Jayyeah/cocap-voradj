#!/usr/bin/env python3
"""Supervise strict Final-AW -> VXY9 curriculum on GPU0.

The supervisor owns orchestration only: it never mutates algorithm code.  Each
stage trains to its historical cap, screens every 25k checkpoint, applies an
explicit promotion gate, and injects only the selected model into the next
stage.  One rolling full-resume bundle is retained by the trainer itself.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs/iqn_vxy_full_20260830"
RUN_ROOT = ROOT / "artifacts/2026-08-30_iqn_vxy_full"
SCREEN_ROOT = ROOT / "artifacts/2026-08-30_iqn_vxy_full_screening"
BEST_ROOT = ROOT / "artifacts/2026-08-30_iqn_vxy_full_best"
RUNTIME_ROOT = ROOT / "artifacts/2026-08-30_iqn_vxy_full_runtime"
STATUS_PATH = ROOT / "artifacts/2026-08-30_iqn_vxy_full_supervisor/status.json"
INTERVAL = 25_000
GATE = {
    "capture_success_rate": 0.50,
    "mix_capture_rate": 0.50,
    "coverage_ce_strict_rate": 0.10,
    "mix_ce_strict_rate": 0.10,
    "max_collision_rate": 0.50,
}
STAGES: tuple[dict[str, Any], ...] = (
    {"number": 1, "label": "4p1e1obs", "config": "stage1_4p1e1obs_scratch2m.yaml",
     "run": "stage1_4p1e1obs_scratch2m", "steps": 2_000_000, "evaders": 1,
     "coverage_max": 1200, "mix_max": 2200, "screen_seed": 2026083100, "formal_seed": 2026083101},
    {"number": 2, "label": "8p2e2obs", "config": "stage2_8p2e2obs_700k.yaml",
     "run": "stage2_8p2e2obs_700k", "steps": 700_000, "evaders": 2,
     "coverage_max": 1500, "mix_max": 2500, "screen_seed": 2026083200, "formal_seed": 2026083201},
    {"number": 3, "label": "12p3e3obs", "config": "stage3_12p3e3obs_700k.yaml",
     "run": "stage3_12p3e3obs_700k", "steps": 700_000, "evaders": 3,
     "coverage_max": 1800, "mix_max": 2800, "screen_seed": 2026083300, "formal_seed": 2026083301},
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def atomic_status(state: str, **fields: Any) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": now(), "state": state, **fields}
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS_PATH)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def launch_tmux(name: str, command: list[str], log_path: Path) -> str:
    if tmux_exists(name):
        return "already_running"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell = f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)} >> {shlex.quote(str(log_path))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", name, shell], cwd=ROOT, check=True)
    return "launched"


def stage_paths(stage: dict[str, Any]) -> dict[str, Path]:
    number, label = int(stage["number"]), str(stage["label"])
    return {
        "config": ROOT / "configs/experiments/iqn_vxy_full_migration_20260830" / str(stage["config"]),
        "run": RUN_ROOT / str(stage["run"]),
        "screen": SCREEN_ROOT / f"stage{number}_{label}",
        "selection": BEST_ROOT / f"iqn_vxy_full_stage{number}_{label}_selection.json",
        "runtime": RUNTIME_ROOT / f"stage{number}_{label}.yaml",
        "train_log": LOG_ROOT / f"stage{number}_{label}_train.log",
        "screen_log": LOG_ROOT / f"stage{number}_{label}_screen.log",
        "final_log": LOG_ROOT / f"stage{number}_{label}_finalize.log",
    }


def runtime_config(
    base: Path,
    destination: Path,
    pretrained: Path | None,
    full_resume_path: Path | None = None,
) -> Path:
    if pretrained is None and full_resume_path is None:
        return base
    payload: dict[str, Any] = {"extends": str(base.resolve())}
    if pretrained is not None:
        payload["pretrained"] = {
            "path": relative(pretrained),
            "compatibility_mode": "shape_compatible",
        }
        payload["experiment_metadata"] = {
            "selected_from_previous_stage": True,
            "pretrained_checkpoint": relative(pretrained),
        }
    if full_resume_path is not None:
        payload["checkpointing"] = {"full_resume_path": str(full_resume_path.resolve())}
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if destination.is_file() and destination.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"runtime config drift: {destination}")
    if not destination.is_file():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, destination)
    return destination


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def last_jsonl(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            lines = handle.read().splitlines()
            line = lines[-1].decode("utf-8").strip() if lines else ""
        return json.loads(line) if line else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def metrics_are_finite(metrics: dict[str, Any] | None) -> bool:
    if metrics is None:
        return True
    return all(
        not isinstance(value, (int, float)) or math.isfinite(float(value))
        for value in metrics.values()
    )


def resources(device: str) -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    result: dict[str, Any] = {
        "disk_free_gib": round(disk.free / 2**30, 2),
        "disk_used_percent": round(100.0 * disk.used / disk.total, 2),
    }
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.strip().split()[0])
        result["ram_available_gib"] = round(info["MemAvailable"] * 1024 / 2**30, 2)
    except (OSError, KeyError, ValueError):
        pass
    try:
        index = device.split(":")[-1]
        query = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        ).stdout.strip().split(",")
        result["gpu"] = {"memory_used_mib": int(query[0]), "memory_total_mib": int(query[1]),
                         "utilization_percent": int(query[2]), "temperature_c": int(query[3])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return result


def gate_selection(path: Path) -> tuple[bool, dict[str, Any]]:
    selection = read_json(path)
    if selection is None:
        return False, {"reason": "selection_missing"}
    metrics = ((selection.get("selected_screening") or {}).get("metrics") or {})
    checks = {
        "capture": float(metrics.get("capture_success_rate", 0.0)) >= GATE["capture_success_rate"],
        "mix_capture": float(metrics.get("mix_capture_rate", 0.0)) >= GATE["mix_capture_rate"],
        "coverage_ce": float(metrics.get("coverage_ce_strict_rate", 0.0)) >= GATE["coverage_ce_strict_rate"],
        "mix_ce": float(metrics.get("mix_ce_strict_rate", 0.0)) >= GATE["mix_ce_strict_rate"],
        "collision": float(metrics.get("max_collision_rate", 1.0)) <= GATE["max_collision_rate"],
    }
    checkpoint = Path(str(selection.get("selected_checkpoint", "")))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checks["checkpoint"] = checkpoint.is_file()
    return all(checks.values()), {"checks": checks, "metrics": metrics, "checkpoint": relative(checkpoint)}


def can_override_stage_gate(
    stage: dict[str, Any],
    gate: dict[str, Any],
    force_promote_stage2: bool,
) -> bool:
    """Allow only the explicitly authorized Stage-2 gate override.

    The historical metric gate remains failed and unchanged. Promotion is
    possible only when the selected Stage-2 checkpoint still exists.
    """
    checks = gate.get("checks") or {}
    return (
        bool(force_promote_stage2)
        and int(stage["number"]) == 2
        and bool(checks.get("checkpoint", False))
    )


def rolling_resume_path(paths: dict[str, Path], override: Path | None) -> Path:
    return override if override is not None else paths["run"] / "checkpoints/resume_latest.pt"


def ensure_stage(
    stage: dict[str, Any],
    pretrained: Path | None,
    train_device: str,
    eval_device: str,
    *,
    formal_gif_count: int = 0,
    formal_workers: int = 2,
    formal_seed: int | None = None,
    full_resume_path: Path | None = None,
) -> dict[str, str]:
    paths = stage_paths(stage)
    launch_config = runtime_config(
        paths["config"], paths["runtime"], pretrained, full_resume_path
    )
    number, label, total = int(stage["number"]), str(stage["label"]), int(stage["steps"])
    prefix = f"cocap_vxy_full_s{number}_{label}"
    final = paths["run"] / "checkpoints" / f"final_step_{total}.pt"
    resume = rolling_resume_path(paths, full_resume_path)
    train_command = ["python3", "train.py", "--config", relative(launch_config), "--device", train_device]
    if resume.is_file() and not final.is_file():
        train_command += ["--resume-path", relative(resume)]
    actions = {
        "train": "complete" if final.is_file() else launch_tmux(prefix + "_train", train_command, paths["train_log"]),
        "screen": launch_tmux(
            prefix + "_screen",
            ["env", "EPISODES=20", "WORKERS=2", f"CHECKPOINT_INTERVAL={INTERVAL}", "WAIT_SECONDS=60",
             "bash", "tools/watch_screened_run.sh", relative(paths["config"]), relative(paths["run"]),
             relative(paths["screen"]), str(stage["screen_seed"]), eval_device, str(total), str(stage["evaders"]),
             str(stage["coverage_max"]), str(stage["mix_max"])], paths["screen_log"],
        ),
        "finalizer": "ready" if paths["selection"].is_file() else launch_tmux(
            prefix + "_finalize",
            ["python3", "tools/finalize_screened_run.py", "--config", relative(paths["config"]),
             "--run-dir", relative(paths["run"]), "--screening-root", relative(paths["screen"]),
             "--best-root", relative(BEST_ROOT), "--line-label", f"iqn_vxy_full_stage{number}_{label}",
             "--total-steps", str(total), "--checkpoint-interval", str(INTERVAL), "--seed",
             str(stage["formal_seed"] if formal_seed is None else int(formal_seed)),
             "--device", eval_device, "--capture-evaders", str(stage["evaders"]),
             "--coverage-max-steps", str(stage["coverage_max"]), "--max-steps", str(stage["mix_max"]),
             "--formal-episodes", "20", "--gif-count", str(int(formal_gif_count)),
             "--workers", str(int(formal_workers))], paths["final_log"],
        ),
    }
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--min-disk-free-gib", type=float, default=25.0)
    parser.add_argument("--force-promote-stage2", action="store_true")
    parser.add_argument("--stage3-formal-gif-count", type=int, default=0)
    parser.add_argument("--stage3-formal-workers", type=int, default=2)
    parser.add_argument(
        "--stage3-formal-seed",
        type=int,
        default=0,
        help="Override Stage3 formal seed; zero keeps the stage default.",
    )
    parser.add_argument("--stage3-full-resume-path", default="")
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args()
    if args.stage3_formal_gif_count < 0:
        parser.error("--stage3-formal-gif-count must be non-negative")
    if args.stage3_formal_workers < 1:
        parser.error("--stage3-formal-workers must be positive")
    if args.stage3_formal_seed < 0:
        parser.error("--stage3-formal-seed must be non-negative")
    stage3_full_resume_path: Path | None = None
    if str(args.stage3_full_resume_path).strip():
        stage3_full_resume_path = Path(args.stage3_full_resume_path).expanduser()
        if not stage3_full_resume_path.is_absolute():
            stage3_full_resume_path = ROOT / stage3_full_resume_path
    pretrained: Path | None = None
    for stage in STAGES:
        paths = stage_paths(stage)
        number, label = int(stage["number"]), str(stage["label"])
        full_resume_path = stage3_full_resume_path if number == 3 else None
        formal_gif_count = args.stage3_formal_gif_count if number == 3 else 0
        formal_workers = args.stage3_formal_workers if number == 3 else 2
        formal_seed = args.stage3_formal_seed if number == 3 and args.stage3_formal_seed else None
        resume = rolling_resume_path(paths, full_resume_path)
        restarts = 0
        launch_count = 0
        while not paths["selection"].is_file():
            actions = ensure_stage(
                stage,
                pretrained,
                args.train_device,
                args.eval_device,
                formal_gif_count=formal_gif_count,
                formal_workers=formal_workers,
                formal_seed=formal_seed,
                full_resume_path=full_resume_path,
            )
            if actions.get("train") == "launched":
                launch_count += 1
                restarts = max(0, launch_count - 1)
            session = f"cocap_vxy_full_s{number}_{label}_train"
            final = paths["run"] / "checkpoints" / f"final_step_{stage['steps']}.pt"
            latest = last_jsonl(paths["run"] / "metrics.jsonl")
            finite = metrics_are_finite(latest)
            resource_state = resources(args.train_device)
            atomic_status("stage_active", stage=number, label=label, actions=actions,
                          pretrained=relative(pretrained) if pretrained else None,
                          train_session=tmux_exists(session), latest_metrics=latest,
                          finite_metrics=finite,
                          checkpoint_count=len(list((paths["run"] / "checkpoints").glob("step_*.pt"))),
                          rolling_resume=resume.is_file(),
                          rolling_resume_path=relative(resume),
                          stage2_gate_override_enabled=bool(args.force_promote_stage2),
                          formal_rollout={
                              "episodes_per_scenario": 20,
                              "gif_count_per_scenario": int(formal_gif_count),
                              "workers": int(formal_workers),
                              "seed": int(stage["formal_seed"] if formal_seed is None else formal_seed),
                          },
                          resources=resource_state, selection=relative(paths["selection"]))
            if args.check_once:
                return 0
            if float(resource_state.get("disk_free_gib", 999.0)) < args.min_disk_free_gib:
                if tmux_exists(session):
                    subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], cwd=ROOT, check=False)
                atomic_status("paused_disk_guard", stage=number, label=label, resources=resource_state)
                return 3
            if not finite:
                if tmux_exists(session):
                    subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], cwd=ROOT, check=False)
                atomic_status("paused_nonfinite_metrics", stage=number, label=label, latest_metrics=latest)
                return 5
            if restarts > args.max_restarts:
                if tmux_exists(session):
                    subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], cwd=ROOT, check=False)
                atomic_status("failed_restart_limit", stage=number, label=label, restarts=restarts)
                return 4
            time.sleep(max(5, args.poll_seconds))
        passed, gate = gate_selection(paths["selection"])
        if not passed:
            if can_override_stage_gate(stage, gate, args.force_promote_stage2):
                pretrained = ROOT / str(gate["checkpoint"])
                atomic_status(
                    "stage_gate_overridden",
                    stage=number,
                    label=label,
                    reason="user_authorized_20260901",
                    gate=gate,
                    next_stage=3,
                    pretrained=relative(pretrained),
                )
                continue
            atomic_status("stage_gate_failed", stage=number, label=label, gate=gate)
            return 2
        pretrained = ROOT / str(gate["checkpoint"])
        atomic_status("stage_gate_passed", stage=int(stage["number"]), label=str(stage["label"]), gate=gate)
    atomic_status(
        "curriculum_complete",
        final_selected_checkpoint=relative(pretrained) if pretrained else None,
        generalization_contract="historical formal capture/coverage/mix evaluation; no invented fourth training stage",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
