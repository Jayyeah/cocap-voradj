#!/usr/bin/env python3
"""Safely launch and monitor the all-agent old-mix ablation.

The supervisor never stops an existing run. It waits until Stage4A or Stage4C
has a verified finite 200k final bundle, validates that bundle, removes only
obsolete milestone replay files covered by the rolling-latest contract, runs
CUDA/2k preflights on the released GPU, and then launches Baseline B.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer
from tools.run_continuous_ctde_training import (
    _link_bundle_tree_atomic,
    _make_trainer,
    _verify_resume_steps,
)


ROOT = Path(__file__).resolve().parents[1]
SPEED_ROOT = Path("/home/yjq/rl/CoCap1/cocap-voradj-speedopt")
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-10_allagent_oldmix_ablation"
STATUS_PATH = ARTIFACT_ROOT / "supervisor_status.json"
LOG_PATH = ARTIFACT_ROOT / "supervisor.log"
DOC_PATH = ROOT / "docs/ALL_AGENT_MASAC_OLDMIX_ABLATION_20260810_ZH.md"
FORMAL_CONFIG = (
    ROOT
    / "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_oldmix_4p1e1obs_200k_aw_allagent.yaml"
)
PREFLIGHT_CONFIG = (
    ROOT
    / "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_oldmix_4p1e1obs_allagent_preflight_2k.yaml"
)
FORMAL_TAG = "legacy_voradj_oldmix_allagent_4p1e1obs_200k_aw_20260810"
FORMAL_SESSION = "allagent_oldmix_ablation_200k"
SEED = 2026080902
POLL_SECONDS = 60
MIN_DISK_FREE_GIB = 20.0
MIN_GPU_FREE_MIB = 18000
MAX_LAUNCH_TEMPERATURE_C = 90
MAX_POST_PREFLIGHT_TEMPERATURE_C = 92

OLD_LINES = (
    {
        "name": "Stage4A",
        "gpu": 0,
        "tag": "stage4a_speedopt_200k_20260809",
        "run_dir": SPEED_ROOT
        / "artifacts/2026-08-09_200k_speedopt/stage4a/"
        "stage4a_speedopt_200k_20260809",
    },
    {
        "name": "Stage4C",
        "gpu": 1,
        "tag": "stage4c_speedopt_200k_20260809",
        "run_dir": SPEED_ROOT
        / "artifacts/2026-08-09_200k_speedopt/stage4c/"
        "stage4c_speedopt_200k_20260809",
    },
)

PROTECTED_RUNS = (
    {
        "name": "Baseline A",
        "tag": "legacy_voradj_oldmix_4p1e1obs_200k_aw_20260810",
        "report": SPEED_ROOT
        / "artifacts/2026-08-10_parallel_ce_legacy_voradj_200k/legacy_voradj/"
        "legacy_voradj_oldmix_4p1e1obs_200k_aw_20260810/"
        "legacy_voradj_oldmix_4p1e1obs_200k_aw_20260810_report.json",
    },
    {
        "name": "Pure CE",
        "tag": "pure_ce_4p0e1obs_200k_aw_20260810",
        "report": SPEED_ROOT
        / "artifacts/2026-08-10_parallel_ce_legacy_voradj_200k/pure_ce/"
        "pure_ce_4p0e1obs_200k_aw_20260810/"
        "pure_ce_4p0e1obs_200k_aw_20260810_report.json",
    },
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp_{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def log(message: str, **fields: Any) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    record = {"time": now(), "message": message, **fields}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json_atomic(STATUS_PATH, record)


def append_doc(marker: str, body: str) -> None:
    current = DOC_PATH.read_text(encoding="utf-8")
    token = f"<!-- {marker} -->"
    if token in current:
        return
    with DOC_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{token}\n{body.rstrip()}\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def final_report_for(line: dict[str, Any]) -> Path:
    return Path(line["run_dir"]) / f"{line['tag']}_report.json"


def raw_final_bundle(line: dict[str, Any]) -> Path | None:
    """Recognize a complete final bundle before rolling-latest migration.

    Runs started before the storage contract landed do not contain
    ``checkpoint_storage.json``. Absence of that sidecar is therefore not a
    completion failure; the full trainer/replay/runtime load below is the
    authoritative safety check. If a sidecar exists, however, it must agree
    with the 200k full-resume semantics.
    """
    report_path = final_report_for(line)
    if not report_path.is_file():
        return None
    try:
        report = load_json(report_path)
    except (OSError, ValueError, TypeError):
        return None
    if int(report.get("transition_count", -1)) != 200000:
        return None
    if not bool(report.get("all_finite", False)):
        return None
    bundle = Path(line["run_dir"]) / "checkpoints/step_000200000"
    required = (
        "trainer.pt",
        "replay.pkl",
        "runtime_state.pkl",
        "effective_config.yaml",
        "manifest.json",
    )
    if not all((bundle / name).is_file() for name in required):
        return None
    storage_path = bundle / "checkpoint_storage.json"
    if storage_path.is_file():
        try:
            storage = load_json(storage_path)
        except (OSError, ValueError, TypeError):
            return None
        if int(storage.get("step", -1)) != 200000 or not bool(
            storage.get("contains_replay", False)
        ):
            return None
    return bundle


def completion_bundle(line: dict[str, Any]) -> Path | None:
    report_path = final_report_for(line)
    if not report_path.is_file():
        return None
    try:
        report = load_json(report_path)
    except (OSError, ValueError, TypeError):
        return None
    if int(report.get("transition_count", -1)) != 200000:
        return None
    if not bool(report.get("all_finite", False)):
        return None
    bundle = Path(line["run_dir"]) / "checkpoints/step_000200000"
    required = (
        "trainer.pt",
        "replay.pkl",
        "runtime_state.pkl",
        "effective_config.yaml",
        "manifest.json",
        "checkpoint_storage.json",
    )
    if not all((bundle / name).is_file() for name in required):
        return None
    storage = load_json(bundle / "checkpoint_storage.json")
    if int(storage.get("step", -1)) != 200000 or not bool(
        storage.get("contains_replay", False)
    ):
        return None
    resume = Path(line["run_dir"]) / "resume_latest"
    if not all((resume / name).is_file() for name in required):
        return None
    if not os.path.samefile(bundle / "replay.pkl", resume / "replay.pkl"):
        return None
    return bundle


def process_alive(tag: str) -> bool:
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (OSError, PermissionError):
            continue
        if "run_continuous_ctde_training.py" in command and tag in command:
            return True
    return False


def report_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return int(payload.get("transition_count", -1)) == 200000 and bool(
        payload.get("all_finite", False)
    )


def assert_protected_runs_healthy() -> None:
    failures = [
        item["name"]
        for item in PROTECTED_RUNS
        if not process_alive(str(item["tag"]))
        and not report_complete(Path(item["report"]))
    ]
    if failures:
        raise RuntimeError(
            "protected run is neither alive nor cleanly complete: "
            + ", ".join(str(item) for item in failures)
        )


def gpu_snapshots() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    snapshots = []
    for row in output.strip().splitlines():
        index, temperature, utilization, used, total = [
            int(value.strip()) for value in row.split(",")
        ]
        snapshots.append(
            {
                "index": index,
                "temperature_c": temperature,
                "utilization_percent": utilization,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "memory_free_mib": total - used,
            }
        )
    return snapshots


def wait_for_safe_gpu(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    while True:
        assert_protected_runs_healthy()
        free_gib = shutil.disk_usage(ROOT).free / (1024.0**3)
        snapshots = {item["index"]: item for item in gpu_snapshots()}
        eligible = []
        for candidate in candidates:
            snapshot = snapshots[int(candidate["gpu"])]
            if (
                snapshot["memory_free_mib"] >= MIN_GPU_FREE_MIB
                and snapshot["temperature_c"] <= MAX_LAUNCH_TEMPERATURE_C
            ):
                eligible.append((snapshot["temperature_c"], snapshot["memory_used_mib"], candidate))
        if free_gib >= MIN_DISK_FREE_GIB and eligible:
            eligible.sort(key=lambda item: (item[0], item[1]))
            selected = eligible[0][2]
            log(
                "resource gate passed",
                selected_line=selected["name"],
                selected_gpu=selected["gpu"],
                disk_free_gib=free_gib,
                gpu=snapshots[int(selected["gpu"])],
            )
            return selected
        log(
            "waiting for resource gate",
            disk_free_gib=free_gib,
            completed=[item["name"] for item in candidates],
            gpu=list(snapshots.values()),
        )
        time.sleep(POLL_SECONDS)


def validate_full_resume_bundle(line: dict[str, Any], bundle: Path) -> dict[str, Any]:
    manifest = load_json(bundle / "manifest.json")
    config = yaml.safe_load(
        (bundle / "effective_config.yaml").read_text(encoding="utf-8")
    )
    runtime = pickle.loads((bundle / "runtime_state.pkl").read_bytes())
    if int(runtime.get("transition_count", -1)) != 200000:
        raise ValueError("final runtime_state step is not 200000")
    trainer = _make_trainer(config, "cpu")
    trainer.load_checkpoint(bundle / "trainer.pt", manifest)
    replay = JointReplayBuffer.load(bundle / "replay.pkl", manifest)
    _verify_resume_steps(trainer, replay)
    if len(replay) != int(runtime["transition_count"]):
        raise ValueError("final replay cardinality does not match transition step")
    result = {
        "line": line["name"],
        "bundle": str(bundle),
        "transition_count": int(runtime["transition_count"]),
        "replay_size": len(replay),
        "trainer_runtime_step": int(
            trainer.resume_runtime_state.get("transition_count", -1)
        ),
        "replay_runtime_step": int(replay.runtime_state.get("transition_count", -1)),
        "manifest_hash": manifest.get("implementation_hash", ""),
    }
    del replay
    del trainer
    gc.collect()
    return result


def prepare_verified_completion(line: dict[str, Any]) -> dict[str, Any] | None:
    """Validate final, add storage metadata, and atomically expose resume_latest."""
    raw_bundle = raw_final_bundle(line)
    if raw_bundle is None:
        return None
    validation = validate_full_resume_bundle(line, raw_bundle)
    write_json_atomic(
        raw_bundle / "checkpoint_storage.json",
        {
            "schema_version": 1,
            "kind": "full_resume",
            "step": 200000,
            "contains_replay": True,
            "migration": {
                "time": now(),
                "reason": "validated final bundle before rolling-latest migration",
            },
        },
    )
    resume_dir = Path(line["run_dir"]) / "resume_latest"
    _link_bundle_tree_atomic(raw_bundle, resume_dir)
    verified_bundle = completion_bundle(line)
    if verified_bundle is None:
        raise RuntimeError("resume_latest migration did not pass the strict completion Gate")
    return {
        "bundle": verified_bundle,
        "validation": validation,
        "resume_latest": resume_dir,
    }


def prune_obsolete_milestone_replays(
    line: dict[str, Any],
    final_bundle: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(line["run_dir"])
    resume_replay = run_dir / "resume_latest/replay.pkl"
    removed: list[dict[str, Any]] = []
    for replay_path in sorted((run_dir / "checkpoints").glob("step_*/replay.pkl")):
        if replay_path == final_bundle / "replay.pkl":
            continue
        if os.path.samefile(replay_path, resume_replay):
            continue
        size = replay_path.stat().st_size
        milestone = replay_path.parent
        replay_path.unlink()
        storage_path = milestone / "checkpoint_storage.json"
        step = int(milestone.name.removeprefix("step_"))
        write_json_atomic(
            storage_path,
            {
                "schema_version": 1,
                "kind": "evaluation_model_only",
                "step": step,
                "contains_replay": False,
                "migration": {
                    "time": now(),
                    "reason": "verified rolling-latest resume bundle",
                    "verified_final_bundle": str(final_bundle),
                },
            },
        )
        removed.append({"path": str(replay_path), "bytes": size, "step": step})
    audit = {
        "time": now(),
        "validation": validation,
        "protected_resume_replay": str(resume_replay),
        "protected_final_replay": str(final_bundle / "replay.pkl"),
        "removed": removed,
        "removed_bytes": sum(item["bytes"] for item in removed),
    }
    write_json_atomic(
        ARTIFACT_ROOT / f"{str(line['name']).lower()}_storage_cleanup_audit.json",
        audit,
    )
    return audit


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": "src:."},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_logged_profiled(
    command: list[str],
    log_path: Path,
    gpu: int,
) -> dict[str, Any]:
    """Run a preflight while sampling externally visible GPU load."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, int]] = []
    started = time.perf_counter()
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": "src:."},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            try:
                snapshot = {
                    item["index"]: item for item in gpu_snapshots()
                }[int(gpu)]
                samples.append(snapshot)
            except (OSError, subprocess.SubprocessError, KeyError):
                pass
            time.sleep(5)
        if int(process.returncode or 0) != 0:
            raise subprocess.CalledProcessError(int(process.returncode), command)
    elapsed = max(time.perf_counter() - started, 1e-12)
    if not samples:
        return {"elapsed_s": elapsed, "samples": 0}
    return {
        "elapsed_s": elapsed,
        "samples": len(samples),
        "gpu_utilization_mean_percent": float(
            sum(item["utilization_percent"] for item in samples) / len(samples)
        ),
        "gpu_utilization_max_percent": max(
            item["utilization_percent"] for item in samples
        ),
        "temperature_max_c": max(item["temperature_c"] for item in samples),
        "memory_used_max_mib": max(item["memory_used_mib"] for item in samples),
        "memory_free_min_mib": min(item["memory_free_mib"] for item in samples),
    }


def run_preflights(gpu: int) -> dict[str, Any]:
    device = f"cuda:{gpu}"
    cuda32_tag = f"allagent_oldmix_cuda{gpu}_smoke32_20260810"
    run_logged(
        [
            "python3",
            "tools/run_continuous_ctde_training.py",
            "--config",
            str(FORMAL_CONFIG),
            "--seed",
            str(SEED),
            "--device",
            device,
            "--total-steps",
            "32",
            "--screen-episodes",
            "0",
            "--diagnostic-eval-episodes",
            "0",
            "--tag",
            cuda32_tag,
            "--artifact-root",
            str(ARTIFACT_ROOT / "preflight_cuda32"),
        ],
        ARTIFACT_ROOT / "preflight_cuda32.log",
    )
    preflight_tag = f"legacy_voradj_oldmix_allagent_preflight_2k_cuda{gpu}_20260810"
    gpu_profile = run_logged_profiled(
        [
            "python3",
            "tools/run_continuous_ctde_training.py",
            "--config",
            str(PREFLIGHT_CONFIG),
            "--seed",
            str(SEED),
            "--device",
            device,
            "--total-steps",
            "2000",
            "--screen-episodes",
            "0",
            "--diagnostic-eval-episodes",
            "0",
            "--tag",
            preflight_tag,
            "--artifact-root",
            str(ARTIFACT_ROOT / "preflight_2k"),
        ],
        ARTIFACT_ROOT / "preflight_2k.log",
        gpu,
    )
    report_path = (
        ARTIFACT_ROOT
        / "preflight_2k"
        / preflight_tag
        / f"{preflight_tag}_report.json"
    )
    report = load_json(report_path)
    if (
        int(report.get("transition_count", -1)) != 2000
        or int(report.get("updates", 0)) <= 0
        or not bool(report.get("all_finite", False))
    ):
        raise RuntimeError("2k all-agent preflight did not complete finite updates")
    tail = list(report.get("update_metrics_tail", []))
    if not tail:
        raise RuntimeError("2k all-agent preflight has no update metrics")
    final = Path(report["final_checkpoint_bundle"])
    resume = report_path.parent / "resume_latest"
    if not (final / "replay.pkl").is_file() or not (resume / "replay.pkl").is_file():
        raise RuntimeError("2k preflight final/resume replay is incomplete")
    if not os.path.samefile(final / "replay.pkl", resume / "replay.pkl"):
        raise RuntimeError("2k final replay is not hardlinked into resume_latest")
    latest = tail[-1]
    result = {
        "cuda32_tag": cuda32_tag,
        "preflight_tag": preflight_tag,
        "report": str(report_path),
        "updates": int(report["updates"]),
        "env_steps_per_second": float(report.get("env_steps_per_second", 0.0)),
        "update_wall_time_s": float(latest.get("update_wall_time_s", 0.0)),
        "critic_time_s": float(latest.get("critic_time_s", 0.0)),
        "actor_q_time_s": float(latest.get("actor_q_time_s", 0.0)),
        "updates_per_second": float(latest.get("updates_per_second", 0.0)),
        "active_agent_loss_terms_per_update": float(
            latest.get("active_agent_loss_terms_per_update", 0.0)
        ),
        "active_agent_loss_terms_per_second": float(
            latest.get("active_agent_loss_terms_per_second", 0.0)
        ),
        "peak_vram_mib": float(latest.get("peak_vram_mib", 0.0)),
        "gpu_profile": gpu_profile,
        "all_finite": True,
    }
    write_json_atomic(ARTIFACT_ROOT / "preflight_summary.json", result)
    return result


def tmux_has_session(name: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def launch_formal(gpu: int) -> None:
    if tmux_has_session(FORMAL_SESSION):
        raise RuntimeError(f"tmux session already exists: {FORMAL_SESSION}")
    train_log = ARTIFACT_ROOT / "formal_train.log"
    command = [
        "exec",
        "nice",
        "-n",
        "10",
        "env",
        "PYTHONPATH=src:.",
        "python3",
        "tools/run_continuous_ctde_training.py",
        "--config",
        str(FORMAL_CONFIG),
        "--seed",
        str(SEED),
        "--device",
        f"cuda:{gpu}",
        "--total-steps",
        "200000",
        "--screen-episodes",
        "20",
        "--diagnostic-eval-episodes",
        "4",
        "--tag",
        FORMAL_TAG,
        "--artifact-root",
        str(ARTIFACT_ROOT),
    ]
    shell_command = (
        shlex.join(command)
        + " >> "
        + shlex.quote(str(train_log))
        + " 2>&1"
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            FORMAL_SESSION,
            "-c",
            str(ROOT),
            shell_command,
        ],
        check=True,
    )
    time.sleep(30)
    if not tmux_has_session(FORMAL_SESSION) or not process_alive(FORMAL_TAG):
        raise RuntimeError("formal Baseline B exited during startup")
    manifest_path = (
        ARTIFACT_ROOT
        / FORMAL_TAG
        / "checkpoints/step_000000001/manifest.json"
    )
    if not manifest_path.is_file():
        raise RuntimeError("formal Baseline B did not create the step-1 manifest")
    manifest = load_json(manifest_path)
    expected = {
        "optimizer_unit": "joint_transition_all_active_agents",
        "replay_sampling": "uniform_joint",
        "focal_training": False,
        "scene_cycle": ["mixed_crms", "pure_ce"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"formal manifest mismatch for {key}")


def milestone_summary(step_dir: Path) -> dict[str, Any]:
    metrics_path = step_dir / "metrics.jsonl"
    last_metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        lines = [
            line for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if lines:
            last_metrics = json.loads(lines[-1])
    selected_keys = (
        "step",
        "update_count",
        "mean_finite",
        "mean_critic_loss",
        "mean_actor_loss",
        "mean_alpha",
        "mean_update_wall_time_s",
        "mean_updates_per_second",
        "mean_active_agent_loss_terms_per_update",
        "mean_active_agent_loss_terms_per_second",
        "env_steps_per_second",
        "collision_count",
    )
    diagnostic_path = step_dir / "diagnostic_eval.json"
    return {
        "step": int(step_dir.name.removeprefix("step_")),
        "metrics": {key: last_metrics.get(key) for key in selected_keys},
        "sampling_stats": last_metrics.get("sampling_stats", {}),
        "diagnostic_eval_path": str(diagnostic_path),
        "diagnostic_eval": load_json(diagnostic_path)
        if diagnostic_path.is_file()
        else {},
        "storage": load_json(step_dir / "checkpoint_storage.json"),
    }


def monitor_milestones() -> None:
    run_dir = ARTIFACT_ROOT / FORMAL_TAG
    launch_path = ARTIFACT_ROOT / "launch_state.json"
    launch_gpu = int(load_json(launch_path)["gpu"]) if launch_path.is_file() else None
    seen: set[int] = set()
    while True:
        for step_dir in sorted((run_dir / "checkpoints").glob("step_*")):
            try:
                step = int(step_dir.name.removeprefix("step_"))
            except ValueError:
                continue
            if step == 1 or step in seen:
                continue
            summary = milestone_summary(step_dir)
            if launch_gpu is not None:
                summary["gpu_snapshot"] = {
                    item["index"]: item for item in gpu_snapshots()
                }.get(launch_gpu, {})
            seen.add(step)
            write_json_atomic(
                ARTIFACT_ROOT / f"milestone_{step:09d}.json",
                summary,
            )
            append_doc(
                f"AUTO_ALLAGENT_STEP_{step}",
                "\n".join(
                    [
                        f"### Baseline B 自动里程碑 {step:,}",
                        "",
                        f"- 时间：{now()}",
                        f"- checkpoint：`{step_dir}`",
                        f"- diagnostic：`{summary['diagnostic_eval_path']}`",
                        f"- storage：`{summary['storage'].get('kind')}`，"
                        f"contains_replay={summary['storage'].get('contains_replay')}",
                        f"- 指标摘要：`{json.dumps(summary['metrics'], ensure_ascii=False)}`",
                        f"- sampler 摘要：`{json.dumps(summary['sampling_stats'], ensure_ascii=False)}`",
                        f"- GPU 摘要：`{json.dumps(summary.get('gpu_snapshot', {}), ensure_ascii=False)}`",
                    ]
                ),
            )
            log("recorded Baseline B milestone", step=step)
        report_path = run_dir / f"{FORMAL_TAG}_report.json"
        if report_complete(report_path):
            log("Baseline B completed", report=str(report_path))
            return
        if not process_alive(FORMAL_TAG):
            raise RuntimeError("Baseline B stopped before a clean 200k report")
        time.sleep(POLL_SECONDS)


def main() -> int:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    log("supervisor armed", branch="ablation/all-agent-oldmix-20260810")
    if tmux_has_session(FORMAL_SESSION) or process_alive(FORMAL_TAG):
        log("formal Baseline B already active; switching to milestone monitor")
        monitor_milestones()
        return 0

    completed: list[dict[str, Any]] = []
    prepared: dict[str, dict[str, Any]] = {}
    while not completed:
        assert_protected_runs_healthy()
        for line in OLD_LINES:
            if raw_final_bundle(line) is None:
                continue
            prepared_line = prepare_verified_completion(line)
            if prepared_line is None:
                continue
            prepared[str(line["name"])] = prepared_line
            completed.append(line)
        if not completed:
            log(
                "waiting for first clean Stage4 completion",
                old_line_alive={
                    line["name"]: process_alive(str(line["tag"]))
                    for line in OLD_LINES
                },
            )
            time.sleep(POLL_SECONDS)

    selected = wait_for_safe_gpu(completed)
    selected_prepared = prepared[str(selected["name"])]
    final_bundle = Path(selected_prepared["bundle"])
    validation = dict(selected_prepared["validation"])
    cleanup = prune_obsolete_milestone_replays(
        selected,
        final_bundle,
        validation,
    )
    log(
        "completed-line storage validated and old milestone replay pruned",
        validation=validation,
        removed_bytes=cleanup["removed_bytes"],
    )
    assert_protected_runs_healthy()
    preflight = run_preflights(int(selected["gpu"]))
    post_snapshot = {
        item["index"]: item for item in gpu_snapshots()
    }[int(selected["gpu"])]
    if post_snapshot["temperature_c"] > MAX_POST_PREFLIGHT_TEMPERATURE_C:
        raise RuntimeError("GPU temperature exceeded post-preflight safety gate")
    if post_snapshot["memory_free_mib"] < MIN_GPU_FREE_MIB:
        raise RuntimeError("GPU free memory is below formal-launch safety gate")
    assert_protected_runs_healthy()
    launch_formal(int(selected["gpu"]))
    launch_state = {
        "time": now(),
        "released_line": selected["name"],
        "gpu": int(selected["gpu"]),
        "branch": "ablation/all-agent-oldmix-20260810",
        "worktree": str(ROOT),
        "config": str(FORMAL_CONFIG),
        "seed": SEED,
        "tmux": FORMAL_SESSION,
        "artifact_root": str(ARTIFACT_ROOT),
        "preflight": preflight,
        "post_launch_gpu": {
            item["index"]: item for item in gpu_snapshots()
        }[int(selected["gpu"])],
        "storage_cleanup": cleanup,
    }
    write_json_atomic(ARTIFACT_ROOT / "launch_state.json", launch_state)
    append_doc(
        "AUTO_ALLAGENT_FORMAL_LAUNCH",
        "\n".join(
            [
                "## Baseline B — Standard All-Agent Old-Mix 已启动",
                "",
                f"- 时间：{launch_state['time']}",
                f"- 释放资源的旧线：{selected['name']}",
                f"- branch/worktree：`ablation/all-agent-oldmix-20260810` / `{ROOT}`",
                f"- seed/GPU/tmux：`{SEED}` / `cuda:{selected['gpu']}` / `{FORMAL_SESSION}`",
                f"- artifact root：`{ARTIFACT_ROOT}`",
                "- 唯一算法差异：focal role-balanced item update → "
                "uniform joint-transition all-active-agent update。",
                "- 保持不变：mixed:pure=1:1、reward、Legacy-VorAdj、APF、"
                "network、SAC hyperparameters、seed、200k budget。",
                f"- 2k preflight：`{json.dumps(preflight, ensure_ascii=False)}`",
                "- 下一个正式 checkpoint：25k。",
            ]
        ),
    )
    log("formal Baseline B launched", launch_state=launch_state)
    monitor_milestones()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log("supervisor failed", error=repr(exc))
        raise
