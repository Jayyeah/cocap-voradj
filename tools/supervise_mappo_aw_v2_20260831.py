#!/usr/bin/env python3
"""Supervise the three-seed MAPPO-AW-v2 lane on GPU1.

This lane is intentionally independent from the completed MAPPO-9-v2 lane.
It queues one seed at a time, keeps the runner's rolling full-resume bundle,
and fails closed when GPU1 lacks the configured resource margin.  In particular,
``nvidia-smi`` compute processes are recorded with PID, owner, and command
line.  A process which is not an AW-v2 process is never signalled by this
supervisor.  Shared-GPU launch requires the explicit ``--allow-shared-gpu``
flag and still enforces the free-VRAM guard.

The runner is responsible for writing the 25k checkpoint and deterministic /
stochastic 20-episode evaluation.  This process only orchestrates and audits
those durable outputs; zero capture at 25k or 50k is not a stopping rule.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import numbers
import os
import pwd
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/mappo_aw_v2_20260831"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-31_mappo_aw_v2"
SUPERVISOR_ROOT = ARTIFACT_ROOT / "supervisor"
STATUS_PATH = SUPERVISOR_ROOT / "status.json"
DECISION_PATH = ARTIFACT_ROOT / "gate_decision.json"
LOCK_PATH = SUPERVISOR_ROOT / "gpu1.lock"

GPU_INDEX = 1
DEVICE = f"cuda:{GPU_INDEX}"
TOTAL_STEPS = 400_000
CHECKPOINT_INTERVAL = 25_000
EVAL_EPISODES = 20
POLL_SECONDS = 60
MIN_FREE_GPU_MIB = 26_000
MIN_DISK_FREE_GIB = 20.0
MAX_RESTARTS = 3
SESSION_PREFIX = "cocap_mappo_aw_v2"
EXPERIMENT_TAG = "mappo_aw_v2_20260831"

# Health thresholds are deliberately constants so that status reports and
# tests can identify the exact gate used for a run.
WARN_KL = 0.05
CRITICAL_KL = 0.10
WARN_CLIP = 0.30
CRITICAL_CLIP = 0.30
WARN_SATURATION = 0.50
CRITICAL_SATURATION = 0.95
CRITICAL_PRE_TANH_MEAN = 5.0
CRITICAL_VALUE_LOSS = 1_000.0
CRITICAL_VALUE_GRAD_NORM = 100.0
KL_STREAK_LIMIT = 3
CLIP_STREAK_LIMIT = 5
SATURATION_STREAK_LIMIT = 5
PRE_TANH_STREAK_LIMIT = 5
VALUE_STREAK_LIMIT = 3

# This is the recorded old MAPPO-AW deterministic result.  The promotion
# threshold additionally retains the MAPPO-9-v2 audited threshold, so a tiny
# one-off nonzero result is not called a route-level success.
OLD_MAPPO_AW_MEAN_CAPTURE = 0.05
OLD_MAPPO_AW_MEAN_COLLISION = 0.9333333333333333
ROUTE_MIN_MEAN_CAPTURE = 0.10
ROUTE_MAX_MEAN_COLLISION = 0.90

SEEDS: tuple[dict[str, Any], ...] = tuple(
    {
        "index": index,
        "config": CONFIG_ROOT / f"seed{index}.yaml",
        "run": ARTIFACT_ROOT / f"mappo_aw_v2_seed{index}",
    }
    for index in (1, 2, 3)
)

STOP = False


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a status/report atomically, preserving a complete last record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def status(state: str, **fields: Any) -> dict[str, Any]:
    payload = {"updated_at": now(), "state": state, **fields}
    atomic_json(STATUS_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)
    return payload


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def recent_metrics(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Read the tail without loading an ever-growing learning log."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 512 * 1024))
            lines = handle.read().splitlines()[-limit:]
        rows: list[dict[str, Any]] = []
        for line in lines:
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows
    except (FileNotFoundError, OSError):
        return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted


def _finite(value: Any) -> bool:
    """Recursively reject NaN/Inf in metric payloads, including nested values."""
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return all(_finite(item) for item in value)
    numeric = _number(value)
    return numeric is None or math.isfinite(numeric)


def metrics_are_finite(metrics: Mapping[str, Any] | None) -> bool:
    return metrics is None or _finite(metrics)


def _walk_items(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = f"{prefix}.{key}" if prefix else str(key)
            yield key_text, item
            yield from _walk_items(item, key_text)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            key_text = f"{prefix}[{index}]"
            yield key_text, item
            yield from _walk_items(item, key_text)


def _find_numbers(row: Mapping[str, Any], names: Sequence[str]) -> list[float]:
    """Find numeric aliases in flat or nested metric dictionaries."""
    exact = {name.lower() for name in names}
    values: list[float] = []
    for key, value in _walk_items(row):
        leaf = key.rsplit(".", 1)[-1].lower()
        if leaf in exact:
            numeric = _number(value)
            if numeric is not None:
                values.append(numeric)
            elif isinstance(value, (list, tuple)):
                values.extend(item for item in (_number(v) for v in value) if item is not None)
    return values


def _metric(row: Mapping[str, Any], names: Sequence[str], default: float = 0.0) -> float:
    values = _find_numbers(row, names)
    return values[0] if values else default


def _max_metric(row: Mapping[str, Any], names: Sequence[str], default: float = 0.0) -> float:
    values = _find_numbers(row, names)
    return max(values) if values else default


def _kl(row: Mapping[str, Any]) -> float:
    values = _find_numbers(row, ("approx_kl_max",))
    if values:
        return values[0]
    return _metric(row, ("approx_kl",), 0.0)


def _clip(row: Mapping[str, Any]) -> float:
    return _metric(row, ("clip_fraction", "clip_frac"), 0.0)


def _saturation(row: Mapping[str, Any]) -> float:
    values = _find_numbers(
        row,
        (
            "post_tanh_saturation_ratio",
            "post_tanh_saturation_rate",
            "action_saturation_ratio",
            "action_saturation_rate",
            "saturation_ratio",
            "saturation_rate",
            "saturation",
        ),
    )
    # Some instrumentation emits per-coordinate keys (a/w) rather than a
    # combined ratio.  Include those only when their name clearly denotes a
    # post-tanh/action saturation measurement.
    for key, value in _walk_items(row):
        lowered = key.lower()
        if "saturation" not in lowered or not any(token in lowered for token in ("post", "action", "ratio", "rate")):
            continue
        if lowered.rsplit(".", 1)[-1] in {
            "post_tanh_saturation_ratio",
            "post_tanh_saturation_rate",
            "action_saturation_ratio",
            "action_saturation_rate",
        }:
            continue
        numeric = _number(value)
        if numeric is not None:
            values.append(numeric)
    return max(values) if values else 0.0


def _pre_tanh_mean_abs_max(row: Mapping[str, Any]) -> float:
    values = _find_numbers(
        row,
        (
            "pre_tanh_mean",
            "pre_tanh_mean_abs_max",
            "pre_tanh_mean_max_abs",
            "pre_tanh_mean_a",
            "pre_tanh_mean_w",
            "pre_tanh_mean_omega",
            "gaussian_mean",
            "gaussian_mean_a",
            "gaussian_mean_w",
            "gaussian_mean_omega",
        ),
    )
    return max((abs(value) for value in values), default=0.0)


def _tail_streak(rows: Sequence[Mapping[str, Any]], predicate) -> int:
    streak = 0
    for row in reversed(rows):
        if predicate(row):
            streak += 1
        else:
            break
    return streak


def health(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return warning/critical state with explicit sustained anomaly streaks."""
    rows = list(rows)
    if not rows:
        return {
            "warn": False,
            "critical": False,
            "finite": True,
            "metrics_available": False,
            "critical_kl_streak": 0,
            "critical_clip_streak": 0,
            "critical_saturation_streak": 0,
            "critical_pre_tanh_mean_streak": 0,
            "critical_value_streak": 0,
            "latest_kl": 0.0,
            "latest_clip_fraction": 0.0,
            "latest_saturation": 0.0,
            "latest_pre_tanh_mean_abs_max": 0.0,
            "latest_value_loss": 0.0,
            "latest_value_grad_norm": 0.0,
            "warn_reasons": [],
            "critical_reasons": [],
        }

    finite = all(metrics_are_finite(row) for row in rows)
    latest = rows[-1]
    latest_kl = _kl(latest)
    latest_clip = _clip(latest)
    latest_saturation = _saturation(latest)
    latest_pre_tanh = _pre_tanh_mean_abs_max(latest)
    latest_value_loss = _metric(latest, ("value_loss", "critic_loss", "critic_value_loss"), 0.0)
    latest_value_grad = _metric(latest, ("value_grad_norm", "critic_grad_norm", "critic_value_grad_norm"), 0.0)

    kl_streak = _tail_streak(rows, lambda row: _kl(row) > CRITICAL_KL)
    clip_streak = _tail_streak(rows, lambda row: _clip(row) > CRITICAL_CLIP)
    saturation_streak = _tail_streak(rows, lambda row: _saturation(row) >= CRITICAL_SATURATION)
    pre_tanh_streak = _tail_streak(rows, lambda row: _pre_tanh_mean_abs_max(row) >= CRITICAL_PRE_TANH_MEAN)
    value_streak = _tail_streak(
        rows,
        lambda row: _metric(row, ("value_loss", "critic_loss", "critic_value_loss"), 0.0) > CRITICAL_VALUE_LOSS
        or _metric(row, ("value_grad_norm", "critic_grad_norm", "critic_value_grad_norm"), 0.0) > CRITICAL_VALUE_GRAD_NORM,
    )

    warn_reasons: list[str] = []
    if latest_kl > WARN_KL:
        warn_reasons.append("kl_gt_0.05")
    if latest_clip > WARN_CLIP:
        warn_reasons.append("clip_gt_0.3")
    if latest_saturation > WARN_SATURATION:
        warn_reasons.append("saturation_gt_0.5")
    critical_reasons: list[str] = []
    if not finite:
        critical_reasons.append("nonfinite_metric")
    if kl_streak >= KL_STREAK_LIMIT:
        critical_reasons.append("kl_gt_0.1_for_3")
    if clip_streak >= CLIP_STREAK_LIMIT:
        critical_reasons.append("clip_gt_0.3_for_5")
    if saturation_streak >= SATURATION_STREAK_LIMIT:
        critical_reasons.append("saturation_ge_0.95_for_5")
    if pre_tanh_streak >= PRE_TANH_STREAK_LIMIT:
        critical_reasons.append("pre_tanh_mean_abs_ge_5_for_5")
    if value_streak >= VALUE_STREAK_LIMIT:
        critical_reasons.append("value_loss_gt_1e3_or_grad_gt_1e2_for_3")

    return {
        "warn": bool(warn_reasons),
        "critical": bool(critical_reasons),
        "finite": finite,
        "metrics_available": True,
        "critical_kl_streak": kl_streak,
        "critical_clip_streak": clip_streak,
        "critical_saturation_streak": saturation_streak,
        "critical_pre_tanh_mean_streak": pre_tanh_streak,
        "critical_value_streak": value_streak,
        "latest_kl": latest_kl,
        "latest_clip_fraction": latest_clip,
        "latest_saturation": latest_saturation,
        "latest_pre_tanh_mean_abs_max": latest_pre_tanh,
        "latest_value_loss": latest_value_loss,
        "latest_value_grad_norm": latest_value_grad,
        "warn_reasons": warn_reasons,
        "critical_reasons": critical_reasons,
    }


def tmux_exists(name: str) -> bool:
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", name],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False


def _tmux_text(name: str) -> str:
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", name, "-S", "-120"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout or ""
    except OSError:
        return ""


def session_is_expected(name: str, config: Path | None = None) -> bool:
    """Confirm a pre-existing named session belongs to this experiment."""
    if not tmux_exists(name):
        return False
    text = _tmux_text(name)
    tokens = [EXPERIMENT_TAG, "run_small_step_ac_migration.py"]
    if config is not None:
        tokens.extend((relative(config), config.name, str(config)))
    has_runner = "run_small_step_ac_migration.py" in text
    has_experiment_tag = EXPERIMENT_TAG in text or (config is not None and any(
        token in text for token in (relative(config), config.name, str(config))
    ))
    if has_runner and has_experiment_tag:
        return True
    # The command may have already scrolled out of the pane while the Python
    # child is still running.  Process inspection is read-only and lets us
    # distinguish our own stale session from a same-named foreign session.
    if config is not None:
        return any(config_token in _process_cmdline(pid) for pid in _process_ids() for config_token in tokens[2:])
    return False


def launch_tmux(name: str, command: Sequence[str], log: Path, config: Path | None = None) -> str:
    """Launch only a new, verified AW-v2 tmux session."""
    if tmux_exists(name):
        if session_is_expected(name, config):
            return "already_running"
        return "conflict_existing_session"
    log.parent.mkdir(parents=True, exist_ok=True)
    shell = f"cd {shlex.quote(str(ROOT))} && {shlex.join(list(command))} >> {shlex.quote(str(log))} 2>&1"
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", name, shell], cwd=ROOT, check=True)
    except (OSError, subprocess.SubprocessError):
        return "launch_failed"
    return "launched"


# Keep the shorter name used by the MAPPO-9-v2 supervisor available to callers
# and small contract tests.
launch = launch_tmux


def _process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _process_ids() -> list[int]:
    try:
        return [int(entry.name) for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    except OSError:
        return []


def _process_owner(pid: int) -> str:
    try:
        uid = Path(f"/proc/{pid}").stat().st_uid
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return str(uid)
    except OSError:
        return "unknown"


def _query_compute_rows(gpu_index: int) -> tuple[list[dict[str, Any]], bool, str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], False, repr(exc)
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            memory = int(float(parts[1]))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "pid": pid,
                "owner": _process_owner(pid),
                "cmdline": _process_cmdline(pid),
                "used_memory_mib": memory,
                "gpu_index": gpu_index,
            }
        )
    return rows, True, None


def gpu_compute_processes(gpu_index: int = GPU_INDEX) -> list[dict[str, Any]]:
    """Return compute PID/owner/cmdline records for one GPU."""
    return _query_compute_rows(gpu_index)[0]


def is_aw_v2_process(process: Mapping[str, Any]) -> bool:
    command = str(process.get("cmdline", ""))
    return EXPERIMENT_TAG in command and "run_small_step_ac_migration.py" in command


def foreign_gpu_processes(processes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Identify blockers; this function never signals or mutates processes."""
    return [dict(process) for process in processes if not is_aw_v2_process(process)]


def _gpu_state(gpu_index: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    line = next((item for item in result.stdout.splitlines() if item.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 5:
        raise ValueError(f"unexpected nvidia-smi GPU row: {line!r}")
    return {
        "memory_used_mib": int(float(parts[0])),
        "memory_free_mib": int(float(parts[1])),
        "memory_total_mib": int(float(parts[2])),
        "utilization_percent": int(float(parts[3])),
        "temperature_c": int(float(parts[4])),
    }


def resources(gpu_index: int = GPU_INDEX) -> dict[str, Any]:
    """Capture disk/RAM/GPU and process ownership for status and gating."""
    disk = shutil.disk_usage(ROOT)
    payload: dict[str, Any] = {
        "disk_free_gib": round(disk.free / 2**30, 2),
        "disk_used_percent": round(100.0 * disk.used / disk.total, 2),
        "gpu_index": gpu_index,
    }
    try:
        payload["gpu"] = _gpu_state(gpu_index)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        payload["gpu_query_error"] = repr(exc)
    try:
        mem: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            mem[key] = int(value.strip().split()[0])
        payload["ram_available_gib"] = round(mem["MemAvailable"] * 1024 / 2**30, 2)
    except (OSError, KeyError, ValueError) as exc:
        payload["ram_query_error"] = repr(exc)
    processes, query_ok, query_error = _query_compute_rows(gpu_index)
    payload["gpu_compute_processes"] = processes
    payload["gpu_compute_pids"] = [int(process["pid"]) for process in processes]
    payload["foreign_gpu_processes"] = foreign_gpu_processes(processes)
    payload["compute_process_query_ok"] = query_ok
    if query_error:
        payload["compute_process_query_error"] = query_error
    return payload


def resource_blockers(snapshot: Mapping[str, Any], min_free_gpu_mib: int = MIN_FREE_GPU_MIB,
                      min_disk_free_gib: float = MIN_DISK_FREE_GIB, *,
                      block_foreign_processes: bool = True) -> list[str]:
    blockers: list[str] = []
    if not bool(snapshot.get("compute_process_query_ok", False)):
        blockers.append("gpu_compute_process_query_failed")
    if block_foreign_processes and snapshot.get("foreign_gpu_processes"):
        blockers.append("foreign_gpu_compute_process")
    gpu = snapshot.get("gpu") or {}
    try:
        free = int(gpu.get("memory_free_mib", 0))
    except (TypeError, ValueError):
        free = 0
    if free < int(min_free_gpu_mib):
        blockers.append("free_vram_below_26g")
    try:
        disk_free = float(snapshot.get("disk_free_gib", 0.0))
    except (TypeError, ValueError):
        disk_free = 0.0
    if disk_free < float(min_disk_free_gib):
        blockers.append("disk_free_below_guard")
    return blockers


def stop_owned_session(name: str, config: Path | None = None) -> bool:
    """Interrupt only a verified AW-v2 tmux session; never signal GPU PIDs."""
    if not session_is_expected(name, config):
        return False
    try:
        subprocess.run(["tmux", "send-keys", "-t", name, "C-c"], cwd=ROOT, check=False)
    except OSError:
        return False
    return True


def best_evaluation(run: Path) -> dict[str, Any]:
    best: dict[str, Any] = {
        "step": 0,
        "capture_rate": 0.0,
        "normal_capture_rate": 0.0,
        "stationary_capture_rate": 0.0,
        "collision_rate": 1.0,
        "visited_2plus_ring_rate": 0.0,
        "visited_3plus_ring_rate": 0.0,
        "max_2plus_ring_hold_steps": 0.0,
        "max_3plus_ring_hold_steps": 0.0,
        "mean_largest_angular_gap_rad": 0.0,
        "mean_pairwise_angular_separation_min_rad": 0.0,
        "mean_episode_return": 0.0,
        "evaluation_count": 0,
    }
    evaluation_paths = sorted((run / "evaluations").glob("step_*.json"))
    for path in evaluation_paths:
        payload = read_json(path) or {}
        deterministic = payload.get("deterministic") or {}
        capture = deterministic.get("capture") or {}
        candidate = {
            "step": int(payload.get("step", 0) or 0),
            "capture_rate": float(capture.get("capture_rate", deterministic.get("capture_rate", 0.0)) or 0.0),
            "normal_capture_rate": float(capture.get("normal_capture_rate", deterministic.get("normal_capture_rate", 0.0)) or 0.0),
            "stationary_capture_rate": float(capture.get("stationary_capture_rate", deterministic.get("stationary_capture_rate", 0.0)) or 0.0),
            "collision_rate": float(capture.get("collision_rate", deterministic.get("collision_rate", 1.0)) or 0.0),
            "visited_2plus_ring_rate": float(capture.get("visited_2plus_ring_rate", deterministic.get("visited_2plus_ring_rate", 0.0)) or 0.0),
            "visited_3plus_ring_rate": float(capture.get("visited_3plus_ring_rate", deterministic.get("visited_3plus_ring_rate", 0.0)) or 0.0),
            "max_2plus_ring_hold_steps": float(capture.get("max_2plus_ring_hold_steps", deterministic.get("max_2plus_ring_hold_steps", 0.0)) or 0.0),
            "max_3plus_ring_hold_steps": float(capture.get("max_3plus_ring_hold_steps", deterministic.get("max_3plus_ring_hold_steps", 0.0)) or 0.0),
            "mean_largest_angular_gap_rad": float(capture.get("mean_largest_angular_gap_rad", deterministic.get("mean_largest_angular_gap_rad", 0.0)) or 0.0),
            "mean_pairwise_angular_separation_min_rad": float(capture.get("mean_pairwise_angular_separation_min_rad", deterministic.get("mean_pairwise_angular_separation_min_rad", 0.0)) or 0.0),
            "mean_episode_return": float(capture.get("mean_episode_return", deterministic.get("mean_episode_return", 0.0)) or 0.0),
            "evaluation_count": len(evaluation_paths),
            "evaluation_path": relative(path),
        }
        if (candidate["capture_rate"], -candidate["collision_rate"], candidate["step"]) > (
            best["capture_rate"], -best["collision_rate"], best["step"]
        ):
            best = candidate
    best["evaluation_count"] = len(evaluation_paths)
    return best


def final_gate() -> dict[str, Any]:
    evaluations = [best_evaluation(Path(item["run"])) for item in SEEDS]
    stable_nonzero = bool(evaluations) and all(item["capture_rate"] > 0.0 for item in evaluations)
    mean_capture = sum(item["capture_rate"] for item in evaluations) / max(len(evaluations), 1)
    mean_collision = sum(item["collision_rate"] for item in evaluations) / max(len(evaluations), 1)
    learning = []
    health_rows = []
    for item in SEEDS:
        rows = recent_metrics(Path(item["run"]) / "learning_metrics.jsonl", 50)
        checked = health(rows)
        learning.append({"seed": int(item["index"]), "rows": len(rows), "health": checked})
        health_rows.append(checked)
    optimization_healthy = bool(health_rows) and all(
        item["metrics_available"] and item["finite"] and not item["critical"] for item in health_rows
    )
    better_than_old = (
        mean_capture > OLD_MAPPO_AW_MEAN_CAPTURE
        and mean_collision < OLD_MAPPO_AW_MEAN_COLLISION
    )
    if stable_nonzero and better_than_old and mean_capture >= ROUTE_MIN_MEAN_CAPTURE and mean_collision < ROUTE_MAX_MEAN_COLLISION:
        decision = "CONTINUOUS_AC_ROUTE_ESTABLISHED"
        next_stage = "Continuous Actor-Critic route established; evaluate full capture+coverage restoration."
    elif optimization_healthy:
        decision = "HEALTHY_DISCRETE_TO_CONTINUOUS_GAP"
        next_stage = "Healthy PPO but weaker than MAPPO-9-v2; isolate the discrete-to-continuous action gap."
    else:
        decision = "UNHEALTHY_CONTINUOUS_POLICY_PPO_IMPLEMENTATION"
        next_stage = "Pause and audit continuous policy/PPO implementation; do not jump to another algorithm."
    return {
        "schema": "mappo-aw-v2-three-seed-gate-v1",
        "decided_at": now(),
        "decision": decision,
        "next_stage": next_stage,
        "experiment": EXPERIMENT_TAG,
        "total_env_steps": TOTAL_STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "eval_episodes": EVAL_EPISODES,
        "per_seed_best": evaluations,
        "mean_best_capture": mean_capture,
        "mean_best_collision": mean_collision,
        "stable_nonzero_all_seeds": stable_nonzero,
        "better_than_old_mappo_aw": better_than_old,
        "old_mappo_aw_mean_capture": OLD_MAPPO_AW_MEAN_CAPTURE,
        "old_mappo_aw_mean_collision": OLD_MAPPO_AW_MEAN_COLLISION,
        "optimization_healthy": optimization_healthy,
        "learning_health": learning,
        "gate_thresholds": {
            "route_min_mean_capture": ROUTE_MIN_MEAN_CAPTURE,
            "route_max_mean_collision": ROUTE_MAX_MEAN_COLLISION,
            "old_mappo_aw_mean_capture": OLD_MAPPO_AW_MEAN_CAPTURE,
            "old_mappo_aw_mean_collision": OLD_MAPPO_AW_MEAN_COLLISION,
        },
    }


def _handle_signal(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP
    STOP = True


def acquire_lock(path: Path = LOCK_PATH):
    """Acquire a process-wide non-blocking lock, retaining the handle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        handle.close()
        return None
    return handle


def _complete(run: Path) -> bool:
    payload = read_json(run / "status.json") or {}
    return payload.get("state") == "complete" and int(payload.get("step", 0) or 0) >= TOTAL_STEPS


def _checkpoint_count(run: Path) -> int:
    return len(list((run / "checkpoints").glob("step_*.pt")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--min-free-gpu-mib", type=int, default=MIN_FREE_GPU_MIB)
    parser.add_argument("--min-disk-free-gib", type=float, default=MIN_DISK_FREE_GIB)
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="permit recorded foreign GPU processes when the free-VRAM guard still passes",
    )
    parser.add_argument("--check-once", action="store_true", help="audit preflight without launching a seed")
    args = parser.parse_args(argv)
    try:
        gpu_index = int(str(args.device).split(":")[-1])
    except ValueError as exc:
        parser.error(f"invalid CUDA device: {args.device}")
        raise exc
    if gpu_index != GPU_INDEX:
        parser.error("MAPPO-AW-v2 is pinned to GPU1 (cuda:1)")

    lock_handle = acquire_lock()
    if lock_handle is None:
        status("blocked_duplicate_supervisor", pid=os.getpid(), lock_path=relative(LOCK_PATH))
        return 5
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    status(
        "starting",
        pid=os.getpid(),
        experiment=EXPERIMENT_TAG,
        device=args.device,
        seeds=[int(item["index"]) for item in SEEDS],
        total_env_steps=TOTAL_STEPS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        allow_shared_gpu=bool(args.allow_shared_gpu),
        lock_path=relative(LOCK_PATH),
    )

    try:
        while not STOP:
            snapshot = resources(gpu_index)
            blockers = resource_blockers(
                snapshot,
                args.min_free_gpu_mib,
                args.min_disk_free_gib,
                block_foreign_processes=not args.allow_shared_gpu,
            )
            if not blockers:
                break
            if "foreign_gpu_compute_process" in blockers:
                state = "waiting_for_gpu1_foreign_process"
            elif "free_vram_below_26g" in blockers:
                state = "waiting_for_gpu1_low_vram"
            else:
                state = "waiting_for_gpu1_resource_gate"
            status(state, pid=os.getpid(), blockers=blockers, resources=snapshot)
            if args.check_once:
                return 0
            if "disk_free_below_guard" in blockers:
                status("paused_disk_guard", pid=os.getpid(), blockers=blockers, resources=snapshot)
                return 3
            # Disk exhaustion is not expected to recover quickly and should
            # not launch anything; all current child sessions are checked and
            # interrupted only through the verified AW-v2 tmux path below.
            time.sleep(max(5, int(args.poll_seconds)))
        if STOP:
            status("stopped", pid=os.getpid(), reason="signal")
            return 130
        if args.check_once:
            status("preflight_ready", pid=os.getpid(), resources=resources(gpu_index))
            return 0

        for seed in SEEDS:
            index = int(seed["index"])
            config = Path(seed["config"])
            run = Path(seed["run"])
            session = f"{SESSION_PREFIX}_seed{index}_gpu1"
            log = SUPERVISOR_ROOT / f"seed{index}.log"
            attempts = 0
            while not STOP:
                if _complete(run):
                    status("seed_complete", pid=os.getpid(), seed=index, session=session,
                           step=TOTAL_STEPS, checkpoint_count=_checkpoint_count(run),
                           rolling_resume=(run / "resume_latest.pt").is_file())
                    break

                rows = recent_metrics(run / "learning_metrics.jsonl")
                optimization = health(rows)
                snapshot = resources(gpu_index)
                blockers = resource_blockers(
                    snapshot,
                    args.min_free_gpu_mib,
                    args.min_disk_free_gib,
                    block_foreign_processes=not args.allow_shared_gpu,
                )
                if optimization["critical"]:
                    stopped = stop_owned_session(session, config)
                    status("paused_critical_optimization", pid=os.getpid(), seed=index, session=session,
                           stopped_owned_session=stopped, optimization_health=optimization,
                           last_safe_resume=relative(run / "resume_latest.pt") if (run / "resume_latest.pt").is_file() else None)
                    return 2
                if "disk_free_below_guard" in blockers:
                    stopped = stop_owned_session(session, config)
                    status("paused_disk_guard", pid=os.getpid(), seed=index, session=session,
                           stopped_owned_session=stopped, resources=snapshot)
                    return 3
                if "foreign_gpu_compute_process" in blockers or "gpu_compute_process_query_failed" in blockers or "free_vram_below_26g" in blockers:
                    state = "waiting_for_gpu1_foreign_process" if "foreign_gpu_compute_process" in blockers else (
                        "waiting_for_gpu1_low_vram" if "free_vram_below_26g" in blockers else "waiting_for_gpu1_resource_gate"
                    )
                    status(state, pid=os.getpid(), seed=index,
                           blockers=blockers, resources=snapshot, optimization_health=optimization)
                    if args.check_once:
                        return 0
                    time.sleep(max(5, int(args.poll_seconds)))
                    continue
                if not config.is_file():
                    status("blocked_missing_config", pid=os.getpid(), seed=index, config=relative(config))
                    return 6

                resume = run / "resume_latest.pt"
                # ``max_restarts`` counts retries after the initial launch.
                # Check before creating a new tmux session so the guard can
                # never launch one extra attempt beyond the declared limit.
                if attempts >= int(args.max_restarts) + 1 and not tmux_exists(session):
                    status("failed_restart_limit", pid=os.getpid(), seed=index, session=session,
                           attempts=attempts, restarts=max(0, attempts - 1),
                           last_safe_resume=relative(resume) if resume.is_file() else None)
                    return 4
                command = [
                    sys.executable,
                    "tools/run_small_step_ac_migration.py",
                    "--config",
                    relative(config),
                    "--device",
                    args.device,
                ]
                if resume.is_file():
                    command.extend(("--resume", relative(resume)))
                action = launch_tmux(session, command, log, config)
                if action == "conflict_existing_session":
                    status("paused_existing_session_conflict", pid=os.getpid(), seed=index,
                           session=session, config=relative(config), resources=snapshot)
                    return 7
                if action == "launch_failed":
                    status("paused_launch_failed", pid=os.getpid(), seed=index, session=session,
                           config=relative(config), resources=snapshot)
                    return 8
                if action == "launched":
                    attempts += 1
                restarts = max(0, attempts - 1)
                status(
                    "seed_active",
                    pid=os.getpid(),
                    seed=index,
                    session=session,
                    launch_action=action,
                    run_status=read_json(run / "status.json") or {},
                    optimization_health=optimization,
                    resources=snapshot,
                    attempts=attempts,
                    restarts=restarts,
                    checkpoint_count=_checkpoint_count(run),
                    rolling_resume=resume.is_file(),
                    target_steps=TOTAL_STEPS,
                    note="No capture-based early stop at 25k/50k.",
                )
                if args.check_once:
                    return 0
                time.sleep(max(5, int(args.poll_seconds)))
            if STOP:
                stop_owned_session(session, config)
                status("stopped", pid=os.getpid(), seed=index, reason="signal")
                return 130

        decision = final_gate()
        atomic_json(DECISION_PATH, decision)
        status("three_seed_gate_complete", pid=os.getpid(), gate_decision=decision,
               decision_path=relative(DECISION_PATH))
        return 0
    finally:
        # The descriptor must remain live for the whole supervisor lifetime;
        # releasing it here permits a later independent invocation.
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
