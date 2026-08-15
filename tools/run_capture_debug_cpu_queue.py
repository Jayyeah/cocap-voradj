#!/usr/bin/env python3
"""Run the fixed CF3 capture-debug sequence once, serially, on CPU.

The queue is deliberately finite: it launches one child process at a time,
writes an atomic status record around every stage, and exits after A4 (or at
the first error).  It never trains or mutates the CF3 checkpoint.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts/2026-08-16_capture_debug"
A1_ROOT = OUTPUT_ROOT / "a1_collision_semantics"
A2_ROOT = OUTPUT_ROOT / "a2_policy_temperature"
A3_ROOT = OUTPUT_ROOT / "a3_iqn_fair_retest"
A4_ROOT = OUTPUT_ROOT / "a4_success_consolidation"
STATUS = OUTPUT_ROOT / "capture_debug_cpu_queue_status.json"
ERROR = OUTPUT_ROOT / "capture_debug_cpu_queue_error.json"
LOCK = OUTPUT_ROOT / "capture_debug_cpu_queue.lock"

CF3_BUNDLE = (
    ROOT
    / "artifacts/2026-08-13_capture_first_controls"
    / "cf3_local_support_full_p0fixed_recovery"
    / "legacy_voradj_cf3_local_support_p0fixed_recovery25k_to200k_20260813"
    / "checkpoints/step_000400000"
)
CF3_CONFIG = CF3_BUNDLE / "effective_config.yaml"
CF3_CHECKPOINT = CF3_BUNDLE / "trainer.pt"
IQN_CHECKPOINT = Path(
    "/home/yjq/rl/CoCap1/cocap-voradj/runs/"
    "iqn_scratch_200k_20260808/checkpoints/step_125000.pt"
)
IQN_SHA256 = "5a0ad1c1400d0004334669908c85db6f7b8496fb2987f36f992040c26bd0344d"

ENV_SEED = 2026081201
NOISE_SEED = 2026081501
A4_ENV_SEED = 2026081601
A4_NOISE_SEED = 2026081651

A1_LEGACY = A1_ROOT / "legacy_end_step.json"
A1_FIXED = A1_ROOT / "synchronized_swept_v1.json"
A1_PREFLIGHT = A1_ROOT / "collision_audit_preflight_passed.json"
A2_OUTPUT = A2_ROOT / "cf3_400k_fixed_t0_t025_t1_20.json"
A3_OUTPUT_20 = A3_ROOT / "iqn_corrected_fixed_20.json"
A3_OUTPUT_100 = A3_ROOT / "iqn_corrected_fixed_100_confirmation.json"
A4_LEGACY = A4_ROOT / "cf3_400k_legacy_t1_up_to300_tail50.json"
A4_FIXED = A4_ROOT / "cf3_400k_fixed_t1_up_to100_tail50.json"

FOCUSED_TESTS = (
    "tests/test_collision_semantics_contract.py",
    "tests/test_cf3_policy_temperature_probe.py",
    "tests/test_iqn_corrected_capture_eval.py",
    "tests/test_cf3_success_consolidation_audit.py",
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def cpu_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    })
    return env


def _base_probe(output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools/probe_cf3_policy_temperature.py"),
        "--config", str(CF3_CONFIG),
        "--checkpoint", str(CF3_CHECKPOINT),
        "--output", str(output),
        "--max-steps", "1000",
        "--seed", str(ENV_SEED),
        "--noise-seed", str(NOISE_SEED),
        "--device", "cpu",
    ]


def a1_command(*, fixed: bool) -> list[str]:
    output = A1_FIXED if fixed else A1_LEGACY
    semantics = "synchronized_swept_v1" if fixed else "legacy_end_step"
    return _base_probe(output) + [
        "--episodes", "20", "--temperatures", "0",
        "--collision-semantics", semantics,
    ]


def a2_command() -> list[str]:
    return _base_probe(A2_OUTPUT) + [
        "--episodes", "20", "--temperatures", "0", "0.25", "1",
        "--collision-semantics", "synchronized_swept_v1",
    ]


def a3_command(*, episodes: int) -> list[str]:
    output = A3_OUTPUT_100 if int(episodes) == 100 else A3_OUTPUT_20
    return [
        sys.executable,
        str(ROOT / "tools/evaluate_iqn_corrected_capture.py"),
        "--config", str(CF3_CONFIG),
        "--checkpoint", str(IQN_CHECKPOINT),
        "--expected-checkpoint-sha256", IQN_SHA256,
        "--output", str(output),
        "--episodes", str(int(episodes)),
        "--max-steps", "1000",
        "--tail-steps", "50",
        "--seed", str(ENV_SEED),
        "--device", "cpu",
        "--collision-semantics", "synchronized_swept_v1",
    ]


def a4_command(*, fixed: bool) -> list[str]:
    output = A4_FIXED if fixed else A4_LEGACY
    semantics = "synchronized_swept_v1" if fixed else "legacy_end_step"
    episodes = 100 if fixed else 300
    return [
        sys.executable,
        str(ROOT / "tools/audit_cf3_success_consolidation.py"),
        "--config", str(CF3_CONFIG),
        "--checkpoint", str(CF3_CHECKPOINT),
        "--output", str(output),
        "--episodes", str(episodes),
        "--max-steps", "1000",
        "--tail-steps", "50",
        "--long-two-plus-steps", "20",
        "--max-cases-per-category", "5",
        "--seed", str(A4_ENV_SEED),
        "--noise-seed", str(A4_NOISE_SEED),
        "--temperature", "1",
        "--device", "cpu",
        "--collision-semantics", semantics,
    ]


def preflight_command() -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS]


def command_invariants(command: Sequence[str]) -> None:
    if "--device" in command:
        index = command.index("--device")
        if index + 1 >= len(command) or command[index + 1] != "cpu":
            raise ValueError(f"non-CPU command rejected: {command}")
    if any(str(token).lower().startswith("cuda:") for token in command):
        raise ValueError(f"accelerator token rejected: {command}")


def _validate_probe(
    payload: Mapping[str, Any], *, episodes: int, temperatures: Sequence[float], semantics: str,
) -> None:
    if payload.get("kind") != "cf3_paired_policy_temperature_probe":
        raise ValueError("wrong policy-probe kind")
    if payload.get("device") != "cpu":
        raise ValueError("policy probe did not run on CPU")
    if int(payload.get("episodes_per_mode", -1)) != int(episodes):
        raise ValueError("policy probe episode count mismatch")
    actual_modes = [float(row["temperature"]) for row in payload.get("temperature_contract", {}).get("modes", [])]
    if actual_modes != [float(value) for value in temperatures]:
        raise ValueError(f"policy probe modes mismatch: {actual_modes}")
    effective = payload.get("collision_semantics_override", {}).get("effective")
    if effective != semantics:
        raise ValueError(f"policy probe semantics mismatch: {effective!r}")
    if Path(str(payload.get("config", ""))).resolve() != CF3_CONFIG.resolve():
        raise ValueError("policy probe config mismatch")
    if Path(str(payload.get("checkpoint", ""))).resolve() != CF3_CHECKPOINT.resolve():
        raise ValueError("policy probe checkpoint mismatch")
    if int(payload.get("environment_seed_base", -1)) != ENV_SEED:
        raise ValueError("policy probe environment seed mismatch")
    if int(payload.get("noise_seed_base", -1)) != NOISE_SEED:
        raise ValueError("policy probe noise seed mismatch")
    if len(payload.get("records", [])) != int(episodes) * len(temperatures):
        raise ValueError("policy probe record count mismatch")


def validate_a1_legacy(payload: Mapping[str, Any]) -> None:
    _validate_probe(payload, episodes=20, temperatures=[0.0], semantics="legacy_end_step")


def validate_a1_fixed(payload: Mapping[str, Any]) -> None:
    _validate_probe(payload, episodes=20, temperatures=[0.0], semantics="synchronized_swept_v1")


def validate_a2(payload: Mapping[str, Any]) -> None:
    _validate_probe(
        payload, episodes=20, temperatures=[0.0, 0.25, 1.0],
        semantics="synchronized_swept_v1",
    )


def _validate_a3(payload: Mapping[str, Any], *, episodes: int) -> None:
    if payload.get("kind") != "iqn_corrected_cf3_capture_eval" or payload.get("device") != "cpu":
        raise ValueError("wrong IQN corrected-eval contract")
    if payload.get("checkpoint_hash_verified") is not True:
        raise ValueError("IQN hash was not verified")
    if payload.get("checkpoint_sha256") != IQN_SHA256:
        raise ValueError("IQN checkpoint hash mismatch")
    if payload.get("collision_semantics") != "synchronized_swept_v1":
        raise ValueError("IQN collision semantics mismatch")
    if int(payload.get("summary", {}).get("episodes", -1)) != int(episodes):
        raise ValueError("IQN episode count mismatch")
    if len(payload.get("records", [])) != int(episodes):
        raise ValueError("IQN record count mismatch")


def validate_a3_20(payload: Mapping[str, Any]) -> None:
    _validate_a3(payload, episodes=20)


def validate_a3_100(payload: Mapping[str, Any]) -> None:
    _validate_a3(payload, episodes=100)


def a3_needs_confirmation(payload: Mapping[str, Any]) -> bool:
    validate_a3_20(payload)
    return int(payload["summary"].get("normal_capture_count", 0)) >= 4


def _validate_a4(payload: Mapping[str, Any], *, max_episodes: int, semantics: str) -> None:
    if payload.get("kind") != "cf3_success_consolidation_audit" or payload.get("device") != "cpu":
        raise ValueError("wrong success-consolidation contract")
    attempted = int(payload.get("episodes_attempted", 0))
    if not 1 <= attempted <= int(max_episodes):
        raise ValueError("success-consolidation episode count mismatch")
    mode = payload.get("mode", {})
    if not (mode.get("name") == "stochastic_t1" and float(mode.get("temperature", -1)) == 1.0):
        raise ValueError("success-consolidation temperature mismatch")
    if payload.get("collision_semantics_override") != semantics:
        raise ValueError("success-consolidation collision semantics mismatch")
    if int(payload.get("tail_steps", -1)) != 50 or int(payload.get("max_cases_per_category", -1)) != 5:
        raise ValueError("success-consolidation tail/case contract mismatch")
    if Path(str(payload.get("config", ""))).resolve() != CF3_CONFIG.resolve():
        raise ValueError("success-consolidation config mismatch")
    if Path(str(payload.get("checkpoint", ""))).resolve() != CF3_CHECKPOINT.resolve():
        raise ValueError("success-consolidation checkpoint mismatch")


def validate_a4_legacy(payload: Mapping[str, Any]) -> None:
    _validate_a4(payload, max_episodes=300, semantics="legacy_end_step")


def validate_a4_fixed(payload: Mapping[str, Any]) -> None:
    _validate_a4(payload, max_episodes=100, semantics="synchronized_swept_v1")


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    output: Path | None
    validator: Callable[[Mapping[str, Any]], None] | None


def planned_stages() -> list[Stage]:
    return [
        Stage("preflight_cpu_tests", preflight_command(), None, None),
        Stage("a1_legacy_t0_20", a1_command(fixed=False), A1_LEGACY, validate_a1_legacy),
        Stage("a1_fixed_t0_20", a1_command(fixed=True), A1_FIXED, validate_a1_fixed),
        Stage("a2_fixed_t0_t025_t1_20", a2_command(), A2_OUTPUT, validate_a2),
        Stage("a3_iqn_fixed_20", a3_command(episodes=20), A3_OUTPUT_20, validate_a3_20),
    ]


class Queue:
    def __init__(self) -> None:
        self.started_at = now()
        self.cf3_checkpoint_sha256 = sha256(CF3_CHECKPOINT) if CF3_CHECKPOINT.is_file() else None
        self.completed: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.conditional: dict[str, Any] = {}

    def payload(self, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "capture_debug_cpu_sequential_queue",
            "status": status,
            "gpu_forbidden": True,
            "queue_pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": now(),
            "cf3_bundle": str(CF3_BUNDLE),
            "cf3_checkpoint_sha256": self.cf3_checkpoint_sha256,
            "outputs": {
                "a1_preflight": str(A1_PREFLIGHT),
                "a1_legacy": str(A1_LEGACY),
                "a1_fixed": str(A1_FIXED),
                "a2_temperature": str(A2_OUTPUT),
                "a3_iqn_20": str(A3_OUTPUT_20),
                "a3_iqn_100_confirmation": str(A3_OUTPUT_100),
                "a4_legacy": str(A4_LEGACY),
                "a4_fixed": str(A4_FIXED),
            },
            "current_stage": self.current,
            "completed_stages": self.completed,
            "conditional": self.conditional,
            **extra,
        }

    def write(self, status: str, **extra: Any) -> None:
        atomic_json(STATUS, self.payload(status, **extra))

    def run_stage(self, stage: Stage) -> Mapping[str, Any] | None:
        command_invariants(stage.command)
        if stage.output is not None and stage.output.exists():
            temporary = stage.output.with_suffix(stage.output.suffix + ".tmp")
            if temporary.exists():
                raise RuntimeError(f"stale temporary output beside completed file: {temporary}")
            payload = read_json(stage.output)
            assert stage.validator is not None
            stage.validator(payload)
            self.completed.append({
                "stage": stage.name, "status": "skipped_exact_completed",
                "output": str(stage.output), "completed_at": now(),
            })
            self.current = None
            self.write("RUNNING")
            return payload

        if stage.output is not None:
            temporary = stage.output.with_suffix(stage.output.suffix + ".tmp")
            if temporary.exists():
                raise RuntimeError(f"stale/incomplete atomic output: {temporary}")
            stage.output.parent.mkdir(parents=True, exist_ok=True)
        log_path = OUTPUT_ROOT / "logs" / f"{stage.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                stage.command,
                cwd=ROOT,
                env=cpu_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            self.current = {
                "stage": stage.name,
                "pid": process.pid,
                "command": stage.command,
                "output": str(stage.output) if stage.output else None,
                "log": str(log_path),
                "started_at": now(),
            }
            self.write("RUNNING")
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"stage {stage.name} exited {return_code}; see {log_path}")

        payload: Mapping[str, Any] | None = None
        if stage.output is not None:
            if not stage.output.is_file():
                raise RuntimeError(f"stage {stage.name} produced no output: {stage.output}")
            payload = read_json(stage.output)
            assert stage.validator is not None
            stage.validator(payload)
        elif stage.name == "preflight_cpu_tests":
            marker = {
                "schema_version": 1,
                "tests_passed": True,
                "device": "cpu",
                "tests": list(FOCUSED_TESTS),
                "completed_at": now(),
            }
            atomic_json(A1_PREFLIGHT, marker)
        self.completed.append({
            "stage": stage.name, "status": "completed",
            "output": str(stage.output) if stage.output else str(A1_PREFLIGHT),
            "log": str(log_path), "completed_at": now(),
        })
        self.current = None
        self.write("RUNNING")
        return payload


def validate_inputs() -> None:
    missing = [path for path in (CF3_CONFIG, CF3_CHECKPOINT, IQN_CHECKPOINT) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing immutable debug input(s): {missing}")
    actual_iqn = sha256(IQN_CHECKPOINT)
    if actual_iqn != IQN_SHA256:
        raise ValueError(f"IQN checkpoint hash mismatch: expected={IQN_SHA256} actual={actual_iqn}")
    if cpu_environment().get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU isolation environment is not active")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"capture-debug CPU queue already owns {LOCK}", file=sys.stderr)
        return 2

    queue = Queue()
    try:
        validate_inputs()
        atomic_json(ERROR, {
            "schema_version": 1, "status": "NO_ERROR", "gpu_forbidden": True,
            "queue_pid": os.getpid(), "updated_at": now(),
        })
        queue.write("RUNNING")
        results: dict[str, Mapping[str, Any] | None] = {}
        for stage in planned_stages():
            results[stage.name] = queue.run_stage(stage)

        a3_20 = results["a3_iqn_fixed_20"]
        assert a3_20 is not None
        confirmation = a3_needs_confirmation(a3_20)
        queue.conditional["a3_iqn_100_confirmation"] = {
            "condition": "normal_capture_count >= 4/20",
            "triggered": confirmation,
            "observed_normal_capture_count": int(a3_20["summary"].get("normal_capture_count", 0)),
        }
        queue.write("RUNNING")
        if confirmation:
            queue.run_stage(Stage(
                "a3_iqn_fixed_100_confirmation",
                a3_command(episodes=100), A3_OUTPUT_100, validate_a3_100,
            ))

        queue.run_stage(Stage(
            "a4_legacy_t1_up_to300_tail50",
            a4_command(fixed=False), A4_LEGACY, validate_a4_legacy,
        ))
        queue.run_stage(Stage(
            "a4_fixed_t1_up_to100_tail50",
            a4_command(fixed=True), A4_FIXED, validate_a4_fixed,
        ))
        queue.write("COMPLETED", completed_at=now())
        atomic_json(ERROR, {
            "schema_version": 1, "status": "NO_ERROR", "gpu_forbidden": True,
            "queue_pid": os.getpid(), "completed_at": now(),
        })
        return 0
    except BaseException as exc:
        error_payload = {
            "schema_version": 1,
            "status": "FAILED_CLOSED",
            "gpu_forbidden": True,
            "queue_pid": os.getpid(),
            "failed_at": now(),
            "current_stage": queue.current,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        atomic_json(ERROR, error_payload)
        queue.write(
            "FAILED_CLOSED",
            failed_at=now(),
            error=error_payload["error"],
        )
        raise
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
