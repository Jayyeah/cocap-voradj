#!/usr/bin/env python3
"""One durable sequential supervisor per GPU for the four-line experiment."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/small_step_ac_migration_20260828"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-28_small_step_ac"
LANES = {
    0: [*(f"mappo9_seed{i}" for i in (1, 2, 3)), *(f"iqn_vxy9_seed{i}" for i in (1, 2, 3))],
    1: [*(f"mappo_aw_seed{i}" for i in (1, 2, 3)), *(f"td3_aw_seed{i}" for i in (1, 2, 3))],
}
STOP = False


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.replace(temporary, path)


def status_file(lane: int) -> Path: return ARTIFACT_ROOT / "supervisor" / f"gpu{lane}_status.json"


def read_status(run: Path) -> dict:
    path = run / "status.json"
    if not path.is_file(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}


def newest_checkpoint(run: Path) -> Path | None:
    rolling = run / "resume_latest.pt"
    if rolling.is_file(): return rolling
    return None


def summarize_group(name: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "tools/summarize_small_step_ac_migration.py"), name], cwd=ROOT, check=True)


def handle_signal(signum, frame):
    del signum, frame
    global STOP; STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--lane", type=int, choices=(0, 1), required=True); parser.add_argument("--max-restarts", type=int, default=3)
    args = parser.parse_args(); lane = args.lane; signal.signal(signal.SIGTERM, handle_signal); signal.signal(signal.SIGINT, handle_signal)
    supervisor_root = ARTIFACT_ROOT / "supervisor"; supervisor_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (supervisor_root / f"gpu{lane}.lock").open("w"); fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(status_file(lane), {"state": "starting", "lane": lane, "pid": os.getpid(), "queue": LANES[lane], "updated": time.time()})
    for queue_index, name in enumerate(LANES[lane]):
        if STOP: break
        algorithm = name.rsplit("_seed", 1)[0]; run = ARTIFACT_ROOT / name; config = CONFIG_ROOT / f"{name}.yaml"
        current = read_status(run)
        if current.get("state") == "complete" and int(current.get("step", 0)) >= 300000:
            atomic_json(status_file(lane), {"state": "skipped_complete", "lane": lane, "pid": os.getpid(), "current": name,
                                                   "queue_index": queue_index, "updated": time.time()})
            continue
        attempts = 0
        while not STOP:
            checkpoint = newest_checkpoint(run); command = [sys.executable, str(ROOT / "tools/run_small_step_ac_migration.py"),
                                                             "--config", str(config), "--device", f"cuda:{lane}"]
            if checkpoint is not None: command += ["--resume", str(checkpoint)]
            log_path = supervisor_root / f"{name}.log"; log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(status_file(lane), {"state": "running", "lane": lane, "pid": os.getpid(), "child_pid": process.pid,
                                                   "current": name, "algorithm": algorithm, "queue_index": queue_index,
                                                   "resume": str(checkpoint) if checkpoint else None, "attempt": attempts + 1,
                                                   "log": str(log_path), "updated": time.time()})
            while process.poll() is None and not STOP:
                time.sleep(20); run_state = read_status(run)
                atomic_json(status_file(lane), {"state": "running", "lane": lane, "pid": os.getpid(), "child_pid": process.pid,
                                                       "current": name, "algorithm": algorithm, "queue_index": queue_index,
                                                       "run_step": run_state.get("step", 0), "run_total": run_state.get("total", 300000),
                                                       "attempt": attempts + 1, "log": str(log_path), "updated": time.time()})
            if STOP and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=30)
            code = process.wait(); log_handle.close()
            if code == 0 and read_status(run).get("state") == "complete": break
            attempts += 1
            if attempts >= int(args.max_restarts):
                atomic_json(status_file(lane), {"state": "failed", "lane": lane, "current": name, "exit_code": code,
                                                       "attempts": attempts, "updated": time.time()})
                return code or 1
            time.sleep(10)
        if name.endswith("seed3"): summarize_group(algorithm)
    final_state = "stopped" if STOP else "complete"
    atomic_json(status_file(lane), {"state": final_state, "lane": lane, "pid": os.getpid(), "updated": time.time()})
    return 0 if not STOP else 130


if __name__ == "__main__": raise SystemExit(main())
