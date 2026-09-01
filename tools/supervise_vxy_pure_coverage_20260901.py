#!/usr/bin/env python3
"""Observe Stage3, then run VXY pure-coverage warm-start -> scratch on GPU0."""
from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/iqn_vxy_pure_coverage_20260901"
ARTIFACT_ROOT = ROOT / "artifacts/2026-09-01_iqn_vxy_pure_coverage"
SCREEN_ROOT = ROOT / "artifacts/2026-09-01_iqn_vxy_pure_coverage_screening"
BEST_ROOT = ROOT / "artifacts/2026-09-01_iqn_vxy_pure_coverage_best"
STATUS_PATH = ARTIFACT_ROOT / "supervisor/status.json"
STAGE3_SELECTION = (
    ROOT
    / "artifacts/2026-08-30_iqn_vxy_full_best"
    / "iqn_vxy_full_stage3_12p3e3obs_selection.json"
)
TOTAL_STEPS = 500_000
INTERVAL = 25_000
ENTRIES = (
    {
        "label": "warm_start",
        "config": CONFIG_ROOT / "warm_start.yaml",
        "run_name": "vxy_pure_coverage_warm_start_8p2obs_500k",
        "seed_base": 2026091100,
        "resume": Path("/dev/shm/cocap_vxy_pure_coverage_warm/resume_latest.pt"),
    },
    {
        "label": "scratch",
        "config": CONFIG_ROOT / "scratch.yaml",
        "run_name": "vxy_pure_coverage_scratch_8p2obs_500k",
        "seed_base": 2026092100,
        "resume": Path("/dev/shm/cocap_vxy_pure_coverage_scratch/resume_latest.pt"),
    },
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def status(state: str, **fields: Any) -> None:
    payload = {"updated_at": now(), "state": state, **fields}
    atomic_json(STATUS_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def launch(name: str, command: list[str], log: Path, env: dict[str, str] | None = None) -> str:
    if tmux_exists(name):
        return "already_running"
    log.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if env:
        prefix = "env " + " ".join(
            f"{shlex.quote(key)}={shlex.quote(value)}" for key, value in env.items()
        ) + " "
    shell = (
        f"cd {shlex.quote(str(ROOT))} && {prefix}{shlex.join(command)} "
        f">> {shlex.quote(str(log))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", name, shell], cwd=ROOT, check=True)
    return "launched"


def process_matches(fragment: str) -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if fragment in command:
            matches.append(int(entry.name))
    return matches


def gpu_state(index: int) -> dict[str, Any]:
    fields = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=uuid,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().split(",")
    uuid = fields[0].strip()
    apps = []
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",", 3)]
        if len(pieces) == 4 and pieces[1] == uuid:
            apps.append({
                "pid": int(pieces[0]),
                "process_name": pieces[2],
                "memory_mib": int(pieces[3]),
            })
    return {
        "uuid": uuid,
        "memory_used_mib": int(fields[1]),
        "memory_free_mib": int(fields[2]),
        "memory_total_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
        "temperature_c": int(fields[5]),
        "compute_apps": apps,
    }


def resources() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    result: dict[str, Any] = {
        "disk_free_gib": round(disk.free / 2**30, 2),
        "disk_used_percent": round(100.0 * disk.used / disk.total, 2),
    }
    try:
        result["gpu0"] = gpu_state(0)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            mem[key] = int(value.strip().split()[0])
        result["ram_available_gib"] = round(mem["MemAvailable"] * 1024 / 2**30, 2)
    except (OSError, KeyError, ValueError):
        pass
    return result


def stage3_formal_done() -> tuple[bool, dict[str, Any]]:
    selection = read_json(STAGE3_SELECTION) or {}
    formal_raw = str(selection.get("formal_output_root") or "").strip()
    formal = Path(formal_raw) if formal_raw else None
    if formal is not None and not formal.is_absolute():
        formal = ROOT / formal
    marker = formal / "FORMAL_DONE" if formal is not None else Path("/nonexistent")
    pids = process_matches("stage3_12p3e3obs")
    gpu = (resources().get("gpu0") or {})
    done = bool(selection and marker.is_file() and not pids and not gpu.get("compute_apps"))
    return done, {
        "selection": relative(STAGE3_SELECTION),
        "formal_output_root": relative(formal) if formal is not None else None,
        "formal_done": marker.is_file(),
        "stage3_pids": pids,
        "gpu0_compute_apps": gpu.get("compute_apps", []),
    }


def latest_training_metrics(run: Path) -> dict[str, Any]:
    path = run / "metrics.jsonl"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 262144))
            rows = [json.loads(line) for line in handle.read().splitlines() if line]
        return rows[-1] if rows else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def select_checkpoint(label: str, run: Path, screen: Path) -> dict[str, Any]:
    candidates = []
    for step in range(INTERVAL, TOTAL_STEPS + 1, INTERVAL):
        summary_path = screen / f"step_{step}" / "all_summaries.json"
        payload = read_json(summary_path) or {}
        coverage = payload.get("coverage") or {}
        if not coverage:
            raise RuntimeError(f"missing coverage summary: {summary_path}")
        candidate = {
            "step": step,
            "checkpoint": relative(run / "checkpoints" / f"step_{step}.pt"),
            "summary": relative(summary_path),
            "metrics": coverage,
        }
        candidate["selection_key"] = [
            float(coverage.get("coverage_success_rate", 0.0)),
            -float(coverage.get("collision_rate", 1.0)),
            -float(coverage.get("boundary_collision_rate", 1.0)),
            -float(coverage.get("avg_final_ce_center_rms", 1e9)),
            -float(coverage.get("avg_final_ce_center_max", 1e9)),
            -float(coverage.get("avg_final_voronoi_cv", 1e9)),
            -float(coverage.get("avg_final_active_speed_rms", 1e9)),
            float(step),
        ]
        candidates.append(candidate)
    selected = max(candidates, key=lambda item: tuple(item["selection_key"]))
    checkpoint = ROOT / selected["checkpoint"]
    selected["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "selected_at": now(),
        "label": label,
        "selection_policy": (
            "max CE success, then min collision/boundary, centroid RMS/max, "
            "final CV, final velocity RMS, then later step"
        ),
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    BEST_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(BEST_ROOT / f"{label}_selection.json", result)
    return result


def run_formal(
    entry: dict[str, Any],
    selection: dict[str, Any],
    device: str,
    workers: int,
) -> Path:
    label = str(entry["label"])
    step = int(selection["selected"]["step"])
    output = BEST_ROOT / f"{label}_step_{step}"
    marker = output / "FORMAL_DONE"
    if marker.is_file():
        return output
    command = [
        "python3",
        "tools/batch_rollouts_parallel.py",
        "--config",
        relative(entry["config"]),
        "--checkpoint",
        str(selection["selected"]["checkpoint"]),
        "--output-root",
        relative(output),
        "--episodes",
        "20",
        "--gif-count",
        "10",
        "--scenarios",
        "coverage",
        "--seed",
        str(int(entry["seed_base"]) + 10000),
        "--device",
        device,
        "--workers",
        str(workers),
        "--max-steps",
        "1500",
        "--capture-max-steps",
        "1000",
        "--coverage-max-steps",
        "1500",
        "--capture-evaders",
        "0",
        "--max-gif-frames",
        "1000",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    summary = read_json(output / "all_summaries.json") or {}
    coverage = summary.get("coverage") or {}
    failures = output / "failures.json"
    gifs = list((output / "coverage").glob("*.gif"))
    if failures.is_file() or int(coverage.get("episodes", 0)) != 20 or len(gifs) != 10:
        raise RuntimeError(
            f"formal coverage validation failed: failures={failures.is_file()} "
            f"episodes={coverage.get('episodes')} gifs={len(gifs)}"
        )
    atomic_json(
        output / "formal_manifest.json",
        {
            "completed_at": now(),
            "label": label,
            "step": step,
            "episodes": 20,
            "gif_count": 10,
            "workers": workers,
            "device": device,
            "summary": coverage,
        },
    )
    marker.write_text(now() + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-disk-free-gib", type=float, default=27.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args()

    while True:
        done, stage3 = stage3_formal_done()
        state = resources()
        status("observing_stage3_read_only", stage3=stage3, resources=state)
        if done:
            break
        if args.check_once:
            return 0
        time.sleep(max(5, args.poll_seconds))

    if args.check_once:
        status("queue_preflight_ready", stage3=stage3, resources=resources())
        return 0

    for entry in ENTRIES:
        label = str(entry["label"])
        run = ARTIFACT_ROOT / str(entry["run_name"])
        screen = SCREEN_ROOT / label
        train_session = f"cocap_vxy_purecov_{label}_train_gpu0"
        screen_session = f"cocap_vxy_purecov_{label}_screen_gpu0"
        train_log = ARTIFACT_ROOT / f"supervisor/{label}_train.log"
        screen_log = ARTIFACT_ROOT / f"supervisor/{label}_screen.log"
        entry["resume"].parent.mkdir(parents=True, exist_ok=True)
        launches = 0
        while True:
            final_checkpoint = run / "checkpoints" / f"final_step_{TOTAL_STEPS}.pt"
            screening_done = (
                screen / f"step_{TOTAL_STEPS}" / "DONE"
            ).is_file() and (
                screen / f"step_{TOTAL_STEPS}" / "all_summaries.json"
            ).is_file()
            if final_checkpoint.is_file() and screening_done:
                break
            if final_checkpoint.is_file():
                action = "complete"
            else:
                if not tmux_exists(train_session):
                    launch_state = resources()
                    launch_gpu = launch_state.get("gpu0") or {}
                    if launch_gpu.get("compute_apps"):
                        status(
                            "waiting_for_exclusive_gpu0",
                            label=label,
                            resources=launch_state,
                        )
                        time.sleep(max(5, args.poll_seconds))
                        continue
                command = [
                    "python3",
                    "train.py",
                    "--config",
                    relative(entry["config"]),
                    "--device",
                    args.device,
                ]
                if entry["resume"].is_file():
                    command += ["--resume-path", str(entry["resume"])]
                action = launch(train_session, command, train_log)
                if action == "launched":
                    launches += 1
            watch_action = launch(
                screen_session,
                [
                    "bash",
                    "tools/watch_screened_run.sh",
                    relative(entry["config"]),
                    relative(run),
                    relative(screen),
                    str(entry["seed_base"]),
                    args.device,
                    str(TOTAL_STEPS),
                    "0",
                    "1500",
                    "1500",
                ],
                screen_log,
                env={
                    "EPISODES": "20",
                    "WORKERS": str(args.workers),
                    "CHECKPOINT_INTERVAL": str(INTERVAL),
                    "WAIT_SECONDS": str(args.poll_seconds),
                    "SCENARIOS": "coverage",
                },
            )
            metric = latest_training_metrics(run)
            state = resources()
            status(
                "pure_coverage_active",
                label=label,
                train_session=train_session,
                screen_session=screen_session,
                train_action=action,
                screen_action=watch_action,
                train_tmux=tmux_exists(train_session),
                screen_tmux=tmux_exists(screen_session),
                step=int(metric.get("global_step", 0)),
                latest_metrics=metric,
                finite_metrics=finite_tree(metric),
                checkpoint_count=len(list((run / "checkpoints").glob("step_*.pt"))),
                screening_done_count=len(list(screen.glob("step_*/DONE"))),
                rolling_resume=entry["resume"].is_file(),
                restarts=max(0, launches - 1),
                resources=state,
            )
            if args.check_once:
                return 0
            if float(state.get("disk_free_gib", 999.0)) < args.min_disk_free_gib:
                subprocess.run(["tmux", "send-keys", "-t", train_session, "C-c"], cwd=ROOT, check=False)
                subprocess.run(["tmux", "send-keys", "-t", screen_session, "C-c"], cwd=ROOT, check=False)
                status("paused_disk_guard", label=label, resources=state)
                return 3
            if metric and not finite_tree(metric):
                subprocess.run(["tmux", "send-keys", "-t", train_session, "C-c"], cwd=ROOT, check=False)
                status("paused_nonfinite_metrics", label=label, latest_metrics=metric)
                return 2
            if max(0, launches - 1) > args.max_restarts:
                status("failed_restart_limit", label=label, restarts=max(0, launches - 1))
                return 4
            time.sleep(max(5, args.poll_seconds))

        selection = select_checkpoint(label, run, screen)
        status("pure_coverage_formal_active", label=label, selection=selection["selected"])
        formal = run_formal(entry, selection, args.device, max(1, args.workers))
        status(
            "pure_coverage_entry_complete",
            label=label,
            selection=selection["selected"],
            formal_output=relative(formal),
            formal_done=(formal / "FORMAL_DONE").is_file(),
            resources=resources(),
        )

    status(
        "queue_complete",
        order=["warm_start", "scratch"],
        selections={
            entry["label"]: read_json(BEST_ROOT / f"{entry['label']}_selection.json")
            for entry in ENTRIES
        },
        resources=resources(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
