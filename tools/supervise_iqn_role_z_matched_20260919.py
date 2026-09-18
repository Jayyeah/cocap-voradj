#!/usr/bin/env python3
"""Detached fail-closed coordinator for matched IQN ROLE/Z scratch runs."""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA = "iqn-role-z-matched-supervisor-v1"
ARM_SCRIPT = "tools/iqn_token_matched_20260919.py"
DEFAULT_ROLE_ROOT = Path("/home/yjq/rl/CoCap1/iqn-role-token-scratch-20260918")
DEFAULT_Z_ROOT = Path("/home/yjq/rl/CoCap1/iqn-z-token-scratch-20260918")
DEFAULT_COORD_OUTPUT = DEFAULT_ROLE_ROOT / "artifacts/2026-09-18_iqn_role_z_matched_supervisor"
ARM_OUTPUTS = {
    "role": "artifacts/2026-09-18_iqn_role_token_scratch",
    "z": "artifacts/2026-09-18_iqn_z_token_scratch",
}
TMUX_SESSIONS = {
    "role": "iqn_role_token_matched_20260919",
    "z": "iqn_z_token_matched_20260919",
}
MILESTONES = tuple(range(25_000, 200_001, 25_000))


def now_local() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def pid_command(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except OSError:
        return ""


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def gpu_processes() -> dict[int, set[int]]:
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    uuid_to_index = {}
    for row in gpu_rows:
        index, uuid = [part.strip() for part in row.split(",", 1)]
        uuid_to_index[uuid] = int(index)
    result = {index: set() for index in uuid_to_index.values()}
    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except subprocess.CalledProcessError:
        return result
    for row in rows:
        pid_text, uuid = [part.strip() for part in row.split(",", 1)]
        if uuid in uuid_to_index:
            result[uuid_to_index[uuid]].add(int(pid_text))
    return result


def replay_sizes(latest: Mapping[str, Any]) -> dict[str, int]:
    return {
        key.removeprefix("replay_size_"): int(value)
        for key, value in latest.items()
        if key.startswith("replay_size_") and isinstance(value, (int, float))
    }


def inspect_resume(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    runtime = payload.get("runtime", {})
    return {
        "present": True,
        "schema": payload.get("schema"),
        "contract_hash": payload.get("contract_hash"),
        "global_step": int(runtime.get("global_step", -1)),
        "bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
    }


def tmux_present(session: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def launch_arm(arm: str, root: Path, output: Path, physical_gpu: int) -> None:
    session = TMUX_SESSIONS[arm]
    if tmux_present(session):
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    log = output / "supervisor.log"
    command = (
        f"cd {shlex.quote(str(root))} && "
        f"CUDA_VISIBLE_DEVICES={physical_gpu} python3 {ARM_SCRIPT} supervise "
        f"--arm {arm} --device cuda:0 --output {shlex.quote(str(output))} "
        f">> {shlex.quote(str(log))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)


def arm_snapshot(
    arm: str,
    root: Path,
    output: Path,
    expected: Mapping[str, Any],
    physical_gpu: int,
    gpu_pids: Mapping[int, set[int]],
) -> dict[str, Any]:
    launch = read_json(output / "launch.json")
    heartbeat = read_json(output / "heartbeat.json")
    status = read_json(output / "status.json")
    preflight = read_json(output / "preflight/startup_sanity.json")
    latest = heartbeat.get("latest_metrics") or status.get("latest_metrics") or read_last_jsonl(output / "training/metrics.jsonl")
    pid = int((launch or status or heartbeat).get("pid", 0) or 0)
    command = pid_command(pid)
    head = git(root, "rev-parse", "HEAD")
    resume_path = output / "training/checkpoints/resume_latest.pt"
    try:
        resume = inspect_resume(resume_path)
        resume_error = None
    except BaseException as exc:
        resume = {"present": True}
        resume_error = repr(exc)
    current_step = int(latest.get("global_step", status.get("current_step", heartbeat.get("current_step", 0))) or 0)
    process_gpus = sorted(index for index, pids in gpu_pids.items() if pid in pids)
    contract = {
        "head": head,
        "resolved_config_hash": launch.get("resolved_config_hash") or preflight.get("contract_hashes", {}).get("resolved_config"),
        "runtime_contract_hash": launch.get("runtime_contract_hash") or preflight.get("contract_hashes", {}).get("runtime_observation"),
        "matched_non_token_contract_hash": launch.get("matched_non_token_contract_hash") or preflight.get("contract_hashes", {}).get("matched_non_token_contract"),
    }
    drift = {key: {"expected": expected.get(key), "actual": value} for key, value in contract.items() if expected.get(key) != value}
    wrong_gpu = bool(process_gpus and process_gpus != [physical_gpu])
    error_text = str(status.get("error", ""))
    fatal_error = any(token in error_text.lower() for token in ("contract", "mismatch", "corrupt", "nan", "inf", "runtime assertion", "wrong gpu"))
    return {
        "arm": arm,
        "root": str(root),
        "output": str(output),
        "branch": git(root, "branch", "--show-current"),
        "head": head,
        "pid": pid or None,
        "pid_alive": pid_alive(pid),
        "pid_command": command,
        "tmux_session": TMUX_SESSIONS[arm],
        "tmux_present": tmux_present(TMUX_SESSIONS[arm]),
        "expected_physical_gpu": physical_gpu,
        "observed_process_gpus": process_gpus,
        "wrong_gpu": wrong_gpu,
        "status": status.get("status") or launch.get("status") or heartbeat.get("status") or "missing",
        "phase": heartbeat.get("phase") or status.get("phase"),
        "current_step": current_step,
        "target_step": heartbeat.get("current_target"),
        "replay_sizes": replay_sizes(latest),
        "optimizer_updates": latest.get("update_steps") or sum(int(value) for key, value in latest.items() if key.startswith("updates_")),
        "loss": latest.get("loss"),
        "loss_ema": latest.get("loss_ema"),
        "metrics_finite": finite_tree(latest),
        "checkpoint": status.get("last_checkpoint"),
        "checkpoint_heartbeat": resume,
        "resume_error": resume_error,
        "evaluation": status.get("last_evaluation"),
        "contract": contract,
        "contract_drift": drift,
        "fatal_error": fatal_error,
        "updated_at": now_local(),
    }


def supervise(
    role_root: Path,
    z_root: Path,
    output: Path,
    role_gpu: int,
    z_gpu: int,
    interval: int,
    auto_resume: bool,
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    roots = {"role": role_root.resolve(), "z": z_root.resolve()}
    arm_outputs = {arm: roots[arm] / ARM_OUTPUTS[arm] for arm in roots}
    gpus = {"role": role_gpu, "z": z_gpu}
    expected: dict[str, dict[str, Any]] = {}
    for arm in ("role", "z"):
        launch = read_json(arm_outputs[arm] / "launch.json")
        preflight = read_json(arm_outputs[arm] / "preflight/startup_sanity.json")
        if preflight.get("status") != "pass":
            raise RuntimeError(f"{arm} preflight is not pass")
        expected[arm] = {
            "head": git(roots[arm], "rev-parse", "HEAD"),
            "resolved_config_hash": launch.get("resolved_config_hash") or preflight["contract_hashes"]["resolved_config"],
            "runtime_contract_hash": launch.get("runtime_contract_hash") or preflight["contract_hashes"]["runtime_observation"],
            "matched_non_token_contract_hash": launch.get("matched_non_token_contract_hash") or preflight["contract_hashes"]["matched_non_token_contract"],
        }
    if expected["role"]["matched_non_token_contract_hash"] != expected["z"]["matched_non_token_contract_hash"]:
        raise RuntimeError("ROLE/Z matched non-token contract hashes differ")

    launch_payload = {
        "schema": SCHEMA,
        "status": "running",
        "pid": os.getpid(),
        "role": {"root": str(roots["role"]), "output": str(arm_outputs["role"]), "gpu": role_gpu, "expected": expected["role"]},
        "z": {"root": str(roots["z"]), "output": str(arm_outputs["z"]), "gpu": z_gpu, "expected": expected["z"]},
        "auto_resume": auto_resume,
        "auto_resume_policy": "only dead process + unchanged HEAD/config/runtime hash + readable exact-resume; all drift/corruption/NaN/wrong-GPU fails closed",
        "started_at": now_local(),
    }
    atomic_json(output / "launch.json", launch_payload)
    seen_milestones = {"role": set(), "z": set()}
    restart_counts = {"role": 0, "z": 0}
    while True:
        gpu_pids = gpu_processes()
        snapshots = {
            arm: arm_snapshot(arm, roots[arm], arm_outputs[arm], expected[arm], gpus[arm], gpu_pids)
            for arm in ("role", "z")
        }
        disk = shutil.disk_usage(role_root)
        fail_reasons = []
        for arm, row in snapshots.items():
            if row["contract_drift"]:
                fail_reasons.append(f"{arm}:contract_drift")
            if row["wrong_gpu"]:
                fail_reasons.append(f"{arm}:wrong_gpu")
            if not row["metrics_finite"]:
                fail_reasons.append(f"{arm}:nan_or_inf")
            if row["resume_error"]:
                fail_reasons.append(f"{arm}:resume_corruption")
            if row["fatal_error"]:
                fail_reasons.append(f"{arm}:fatal_status")
            for milestone in MILESTONES:
                if row["current_step"] >= milestone and milestone not in seen_milestones[arm]:
                    seen_milestones[arm].add(milestone)
                    append_jsonl(output / "matched_ledger.jsonl", {
                        "schema": SCHEMA,
                        "kind": "milestone",
                        "arm": arm,
                        "step": milestone,
                        "snapshot": row,
                        "peer_step": snapshots["z" if arm == "role" else "role"]["current_step"],
                        "timestamp": now_local(),
                    })
        if disk.free < 20 * 1024**3:
            fail_reasons.append("disk_free_below_20GiB")

        for arm, row in snapshots.items():
            if row["status"] == "complete":
                continue
            if row["pid_alive"]:
                continue
            if fail_reasons or not auto_resume:
                fail_reasons.append(f"{arm}:process_dead")
                continue
            resume = row["checkpoint_heartbeat"]
            if not resume.get("present") or int(resume.get("global_step", -1)) < 0:
                fail_reasons.append(f"{arm}:no_valid_resume")
                continue
            launch_arm(arm, roots[arm], arm_outputs[arm], gpus[arm])
            restart_counts[arm] += 1
            append_jsonl(output / "matched_ledger.jsonl", {
                "schema": SCHEMA,
                "kind": "automatic_exact_resume",
                "arm": arm,
                "resume": resume,
                "restart_count": restart_counts[arm],
                "timestamp": now_local(),
            })

        status = {
            **launch_payload,
            "status": "failed_closed" if fail_reasons else ("complete" if all(row["status"] == "complete" for row in snapshots.values()) else "running"),
            "role_runtime": snapshots["role"],
            "z_runtime": snapshots["z"],
            "restart_counts": restart_counts,
            "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
            "fail_reasons": sorted(set(fail_reasons)),
            "updated_at": now_local(),
        }
        atomic_json(output / "status.json", status)
        if fail_reasons:
            append_jsonl(output / "matched_ledger.jsonl", {"schema": SCHEMA, "kind": "failed_closed", "reasons": status["fail_reasons"], "timestamp": now_local()})
            return 2
        if status["status"] == "complete":
            append_jsonl(output / "matched_ledger.jsonl", {"schema": SCHEMA, "kind": "complete", "timestamp": now_local()})
            return 0
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-root", type=Path, default=DEFAULT_ROLE_ROOT)
    parser.add_argument("--z-root", type=Path, default=DEFAULT_Z_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_COORD_OUTPUT)
    parser.add_argument("--role-gpu", type=int, required=True)
    parser.add_argument("--z-gpu", type=int, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()
    try:
        return supervise(args.role_root, args.z_root, args.output.resolve(), args.role_gpu, args.z_gpu, args.interval, not args.no_auto_resume)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "status.json", {"schema": SCHEMA, "status": "failed_closed", "error": repr(exc), "traceback": traceback.format_exc(), "updated_at": now_local()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
