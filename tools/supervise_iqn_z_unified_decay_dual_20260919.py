#!/usr/bin/env python3
"""Fail-closed dual-line coordinator for the Z05/Z07 full curriculum."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "tools/iqn_z_unified_decay_curriculum_20260919.py"
SCHEMA = "iqn-z-unified-decay-dual-supervisor-v1"
ARMS = ("z05", "z07")
DEFAULT_OUTPUT = ROOT / "artifacts/2026-09-19_iqn_z_unified_decay_curriculum"
DEFAULT_STORAGE = Path("/data/disk2/home/yjq/cocap-runs/iqn-z-unified-decay-dual-curriculum-20260919")
SESSIONS = {"z05": "iqn_z05_unified_decay_20260919", "z07": "iqn_z07_unified_decay_20260919"}
MIN_FREE_BYTES = 200 * 1024**3


def now_local() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def arm_runtime(storage: Path, arm: str, gpu_pids: dict[int, set[int]]) -> dict[str, Any]:
    train_output = arm_output(storage, arm)
    status = read_json(train_output / "status.json")
    launch = read_json(train_output / "launch.json")
    pid = int(status.get("pid") or launch.get("pid") or 0)
    assigned_raw = str(
        status.get("cuda_visible_devices")
        or launch.get("cuda_visible_devices")
        or ""
    ).strip()
    assigned_gpu = int(assigned_raw) if assigned_raw.isdigit() else None
    observed_gpus = sorted(index for index, pids in gpu_pids.items() if pid in pids)
    return {
        "status": status.get("status") or launch.get("status") or "missing",
        "phase": status.get("phase"),
        "pid": pid or None,
        "pid_alive": pid_alive(pid),
        "tmux_present": tmux_present(SESSIONS[arm]),
        "assigned_gpu": assigned_gpu,
        "observed_gpus": observed_gpus,
        "wrong_gpu": bool(observed_gpus and assigned_gpu is not None and observed_gpus != [assigned_gpu]),
        "current_step": status.get("current_step", 0),
        "stage": status.get("stage"),
        "error": status.get("error"),
    }


def needs_launch(runtime: dict[str, Any]) -> bool:
    if runtime["status"] in {"complete", "failed_closed"}:
        return False
    return not runtime["pid_alive"] and not runtime["tmux_present"]


def tmux_present(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def gpu_processes() -> dict[int, set[int]]:
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {}
    uuid_to_index = {uuid.strip(): int(index.strip()) for index, uuid in (row.split(",", 1) for row in gpu_rows)}
    result = {index: set() for index in uuid_to_index.values()}
    for row in rows:
        pid, uuid = [part.strip() for part in row.split(",", 1)]
        if uuid in uuid_to_index:
            result[uuid_to_index[uuid]].add(int(pid))
    return result


def json_gpu_processes() -> dict[str, list[int]]:
    return {str(index): sorted(pids) for index, pids in gpu_processes().items()}


def storage_snapshot(storage: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(storage)
        return {"path": str(storage), "exists": True, "writable": os.access(storage, os.W_OK), "free": usage.free, "total": usage.total}
    except OSError as exc:
        return {"path": str(storage), "exists": storage.exists(), "writable": False, "free": None, "total": None, "error": repr(exc)}


def arm_output(storage: Path, arm: str) -> Path:
    return storage / arm


def launch_arm(arm: str, output: Path, storage: Path, physical_gpu: int) -> None:
    session = SESSIONS[arm]
    if tmux_present(session):
        return
    train_output = arm_output(storage, arm)
    train_output.mkdir(parents=True, exist_ok=True)
    log = train_output / "supervisor.log"
    command = (
        f"cd {shlex.quote(str(ROOT))} && "
        f"CUDA_VISIBLE_DEVICES={physical_gpu} python3 {RUNNER} supervise --arm {arm} "
        f"--device cuda:0 --output {shlex.quote(str(train_output))} "
        f">> {shlex.quote(str(log))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)


def write_final_comparison(storage: Path) -> None:
    reports = {}
    for arm in ARMS:
        path = arm_output(storage, arm) / f"{arm.upper()}_CURRICULUM_FINAL_REPORT.json"
        if not path.is_file():
            return
        reports[arm] = read_json(path)
    stage_comparison = {}
    for index, stage in enumerate(("stage1", "stage2", "stage3")):
        left = reports["z05"]["stages"][index]
        right = reports["z07"]["stages"][index]
        stage_comparison[stage] = {
            "z05": left["formal_summary"],
            "z07": right["formal_summary"],
            "selected_steps": {"z05": left["selected_step"], "z07": right["selected_step"]},
            "selected_checkpoints": {"z05": left["selected_checkpoint"], "z07": right["selected_checkpoint"]},
        }
    comparison = {
        "schema": SCHEMA,
        "status": "complete",
        "arms": reports,
        "stage_comparison": stage_comparison,
        "scientific_difference": {
            "z05_alpha": 0.5,
            "z07_alpha": 0.7,
            "hard_zero_threshold": 0.10,
            "non_alpha_contract_difference_count": 0,
        },
        "completed_at": now_local(),
    }
    atomic_json(storage / "Z05_VS_Z07_FINAL_COMPARISON.json", comparison)
    lines = [
        "# Z05 vs Z07 Unified-Decay 全课程最终比较",
        "",
        "唯一科学差异：Z05 `alpha=0.5`，Z07 `alpha=0.7`；两线 `hard_zero_threshold=0.10`。",
        "",
    ]
    for stage, payload in stage_comparison.items():
        lines.extend([f"## {stage}", "", "| 指标 | Z05 | Z07 |", "|---|---:|---:|"])
        left, right = payload["z05"], payload["z07"]
        rows = (
            ("Selected step", payload["selected_steps"]["z05"], payload["selected_steps"]["z07"]),
            ("Pure capture", left["capture"]["normal_capture_rate"], right["capture"]["normal_capture_rate"]),
            ("Mixed capture", left["mixed"]["capture_rate"], right["mixed"]["capture_rate"]),
            ("Pure coverage strict CE", left["coverage"]["strict_ce_rate"], right["coverage"]["strict_ce_rate"]),
            ("Mixed post-capture CE", left["mixed"]["post_capture_ce_rate"], right["mixed"]["post_capture_ce_rate"]),
            ("Mixed safe-complete", left["mixed"]["safe_complete_rate"], right["mixed"]["safe_complete_rate"]),
            ("Coverage CE RMS", left["coverage"]["ce_rms"]["mean"], right["coverage"]["ce_rms"]["mean"]),
            ("Coverage area CV", left["coverage"]["area_cv"]["mean"], right["coverage"]["area_cv"]["mean"]),
            ("Z max lineage hop", left["z"]["max_lineage_hop"], right["z"]["max_lineage_hop"]),
            ("Z never-release episodes", left["z"]["post_capture_never_release_episodes"], right["z"]["post_capture_never_release_episodes"]),
        )
        for label, z05, z07 in rows:
            lines.append(f"| {label} | {z05} | {z07} |")
        lines.append("")
    (storage / "Z05_VS_Z07_FINAL_COMPARISON_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def supervise(output: Path, storage: Path, interval: int, min_free_bytes: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    preflight = output / "preflight/startup_sanity.json"
    snapshot = storage_snapshot(storage)
    gpus = gpu_processes()
    if not preflight.is_file() or read_json(preflight).get("status") != "pass":
        status = {"schema": SCHEMA, "status": "failed_closed", "phase": "preflight", "fail_reasons": ["preflight_missing_or_failed"], "updated_at": now_local()}
        atomic_json(output / "status.json", status)
        return 2
    launch = {"schema": SCHEMA, "status": "queued", "pid": os.getpid(), "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "storage": snapshot, "gpus": json_gpu_processes(), "started_at": now_local()}
    atomic_json(output / "launch.json", launch)
    while True:
        snapshot = storage_snapshot(storage)
        gpus = gpu_processes()
        arm_status = {arm: arm_runtime(storage, arm, gpus) for arm in ARMS}
        fail_reasons = []
        if not snapshot["writable"] or snapshot["free"] is None or snapshot["free"] < min_free_bytes:
            atomic_json(output / "status.json", {**launch, "status": "queued", "phase": "waiting_for_storage", "fail_reasons": ["storage_unavailable_or_low"], "storage": snapshot, "gpus": json_gpu_processes(), "updated_at": now_local()})
            time.sleep(interval)
            continue
        current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if current_head != launch["head"]:
            fail_reasons.append("coordinator_head_drift")
        if any(status.get("status") == "failed_closed" for status in arm_status.values()):
            fail_reasons.append("arm_failed_closed")
        if any(status.get("wrong_gpu") for status in arm_status.values()):
            fail_reasons.append("arm_wrong_gpu")
        if fail_reasons:
            atomic_json(output / "status.json", {**launch, "status": "failed_closed", "fail_reasons": sorted(set(fail_reasons)), "storage": snapshot, "gpus": json_gpu_processes(), "arms": arm_status, "updated_at": now_local()})
            return 2
        free_gpus = [index for index in sorted(gpus) if not gpus[index]]
        launchable = [arm for arm in ARMS if needs_launch(arm_status[arm])]
        for arm, gpu in zip(launchable, free_gpus):
            launch_arm(arm, output, storage, gpu)
            arm_status[arm] = {**arm_status[arm], "status": "launch_requested", "assigned_gpu": gpu}
        complete = all(status.get("status") == "complete" for status in arm_status.values())
        if complete:
            write_final_comparison(storage)
        atomic_json(output / "status.json", {**launch, "status": "complete" if complete else "running", "storage": snapshot, "gpus": json_gpu_processes(), "arms": arm_status, "updated_at": now_local()})
        if complete:
            return 0
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "supervise"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--min-free-gib", type=int, default=200)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps({"storage": storage_snapshot(args.storage), "gpus": json_gpu_processes(), "supervisor": read_json(args.output / "status.json")}, indent=2, ensure_ascii=False))
        return 0
    try:
        return supervise(args.output.resolve(), args.storage.resolve(), args.interval, args.min_free_gib * 1024**3)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "status.json", {"schema": SCHEMA, "status": "failed_closed", "error": repr(exc), "traceback": traceback.format_exc(), "updated_at": now_local()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
