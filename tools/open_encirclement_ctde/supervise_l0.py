#!/usr/bin/env python3
"""Two-lane, fail-closed L0 scheduler for the six formal runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from open_encirclement_ctde.runtime import atomic_json_dump
from open_encirclement_ctde.training import load_config


@dataclass
class ActiveRun:
    name: str
    config_path: Path
    gpu: int
    pid: int
    process: subprocess.Popen | None
    log_handle: object | None
    started_at: str


LANES = {
    0: ["mappo_seed1.json", "maddpg_seed2.json", "mappo_seed3.json"],
    1: ["maddpg_seed1.json", "mappo_seed2.json", "maddpg_seed3.json"],
}


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Supervisor:
    def __init__(self, repo_root: Path, config_dir: Path, state_path: Path) -> None:
        self.repo_root = repo_root
        self.config_dir = config_dir
        self.state_path = state_path
        self.log_dir = state_path.parent / "supervisor_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queues = {gpu: list(names) for gpu, names in LANES.items()}
        self.active: dict[int, ActiveRun] = {}
        self.completed: list[str] = []
        self.failed: list[dict] = []
        self.stop_requested = False

    def validate(self) -> None:
        seen = set()
        for gpu, names in self.queues.items():
            for name in names:
                path = self.config_dir / name
                config = load_config(path)
                if config["run_name"] in seen:
                    raise ValueError(f"duplicate run_name {config['run_name']}")
                seen.add(config["run_name"])
                if config["device"] != f"cuda:{gpu}":
                    raise ValueError(f"{name} device {config['device']} does not match lane cuda:{gpu}")
                run_dir = self.repo_root / config["run_dir"]
                state_file = run_dir / "state.json"
                if state_file.exists():
                    state = json.loads(state_file.read_text(encoding="utf-8"))
                    if state.get("status") == "COMPLETED" and state.get("env_steps") == config["num_env_steps"]:
                        continue
                    if state.get("status") == "RUNNING" and pid_alive(int(state.get("pid", -1))):
                        raise RuntimeError(f"formal run already active: {config['run_name']} pid={state['pid']}")
                    raise FileExistsError(f"formal run directory is not clean/resumable: {run_dir}")

    def _launch_next(self, gpu: int) -> None:
        if gpu in self.active or not self.queues[gpu] or self.failed or self.stop_requested:
            return
        name = self.queues[gpu].pop(0)
        config_path = self.config_dir / name
        config = load_config(config_path)
        run_dir = self.repo_root / config["run_dir"]
        state_file = run_dir / "state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state.get("status") == "COMPLETED" and state.get("env_steps") == config["num_env_steps"]:
                self.completed.append(config["run_name"])
                self._launch_next(gpu)
                return
        log_path = self.log_dir / f"{config['run_name']}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        command = [
            str(self.repo_root / ".venv" / "bin" / "python"),
            str(self.repo_root / "tools" / "open_encirclement_ctde" / "train.py"),
            "--config",
            str(config_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.active[gpu] = ActiveRun(
            name=config["run_name"],
            config_path=config_path,
            gpu=gpu,
            pid=process.pid,
            process=process,
            log_handle=log_handle,
            started_at=datetime.now().astimezone().isoformat(),
        )
        self._write_state("RUNNING")

    def _write_state(self, status: str) -> None:
        active = {}
        for gpu, run in self.active.items():
            config = load_config(run.config_path)
            run_state_path = self.repo_root / config["run_dir"] / "state.json"
            run_state = {}
            if run_state_path.exists():
                try:
                    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    run_state = {"status": "STATE_READ_RETRY"}
            active[str(gpu)] = {
                "name": run.name,
                "pid": run.pid,
                "gpu": gpu,
                "started_at": run.started_at,
                "run_state": run_state,
            }
        atomic_json_dump(
            {
                "schema": "open-encirclement-l0-supervisor-v1",
                "status": status,
                "pid": os.getpid(),
                "updated_at": datetime.now().astimezone().isoformat(),
                "active": active,
                "queues": self.queues,
                "completed": self.completed,
                "failed": self.failed,
            },
            self.state_path,
        )

    def run(self) -> int:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_requested", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        for gpu in sorted(self.queues):
            self._launch_next(gpu)
        while self.active or any(self.queues.values()):
            for gpu, run in list(self.active.items()):
                assert run.process is not None
                return_code = run.process.poll()
                if return_code is None:
                    continue
                if run.log_handle is not None:
                    run.log_handle.close()
                del self.active[gpu]
                if return_code == 0:
                    self.completed.append(run.name)
                    self._launch_next(gpu)
                else:
                    self.failed.append({"name": run.name, "pid": run.pid, "return_code": return_code, "gpu": gpu})
            if self.failed:
                # Existing peer processes are not killed. No further queued
                # work is launched until a human audits the failed run.
                self._write_state("FAILED_CLOSED")
                while self.active:
                    time.sleep(10)
                    for gpu, run in list(self.active.items()):
                        assert run.process is not None
                        if run.process.poll() is not None:
                            if run.log_handle is not None:
                                run.log_handle.close()
                            del self.active[gpu]
                return 1
            if self.stop_requested:
                self._write_state("STOP_REQUESTED_NO_NEW_LAUNCHES")
                while self.active:
                    time.sleep(10)
                    for gpu, run in list(self.active.items()):
                        assert run.process is not None
                        if run.process.poll() is not None:
                            if run.log_handle is not None:
                                run.log_handle.close()
                            del self.active[gpu]
                return 2
            self._write_state("RUNNING")
            time.sleep(10)
        self._write_state("COMPLETED")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs/open_encirclement_ctde"))
    parser.add_argument("--state", type=Path, default=Path("artifacts/open_encirclement_ctde/l0_supervisor/state.json"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = args.config_dir if args.config_dir.is_absolute() else repo_root / args.config_dir
    state_path = args.state if args.state.is_absolute() else repo_root / args.state
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another L0 supervisor holds {lock_path}") from exc
    supervisor = Supervisor(repo_root, config_dir, state_path)
    supervisor.validate()
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_OK", "lanes": LANES, "state": str(state_path)}))
        return
    raise SystemExit(supervisor.run())


if __name__ == "__main__":
    main()
