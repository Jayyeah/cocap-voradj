#!/usr/bin/env python3
"""Run matched Local-Binary and Global-Oracle curricula beside frozen Z05."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import yaml

from tools import iqn_token_matched_20260919 as matched
from tools.iqn_evidence_comparison_20261006 import (
    CONFIGS,
    STAGES,
    VARIANTS,
    contract_diff_report,
    now,
    resolved,
)
from tools.iqn_z_unified_decay_curriculum_20260919 import MILESTONES, select_balanced
from tools.evaluate_iqn_evidence_checkpoint_20261006 import evaluate, choose_workers, gpu_snapshot


RUNTIME = Path("/home/yjq/rl/CoCap1/iqn-evidence-comparison-20261006-runtime")
ARTIFACTS = ROOT / "artifacts/2026-10-06_iqn_evidence_comparison"
VARIANTS_TO_TRAIN = ("local_binary", "global_oracle")
SCREENING_EPISODES = 20
SCREENING_SEED_BASE = 2026091900
FINAL_EPISODES = 50
FINAL_SEED_BASE = 2026100601
GPU_BY_VARIANT = {"local_binary": 0, "global_oracle": 1}
STORAGE_ESTIMATE_PER_VARIANT_BYTES = 8 * 1024**3
STORAGE_MARGIN_BYTES = 15 * 1024**3
POLL_SECONDS = 5
_STATE_LOCK = threading.Lock()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_resource_gate() -> dict[str, Any]:
    usage = shutil.disk_usage(RUNTIME.parent)
    gpus = gpu_snapshot()
    try:
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).splitlines()
    except Exception:
        processes = []
    required = 2 * STORAGE_ESTIMATE_PER_VARIANT_BYTES + STORAGE_MARGIN_BYTES
    eligible_gpus = len(gpus) >= 2 and all(gpu["util_percent"] < 20 and gpu["free_mib"] > 40_000 for gpu in gpus[:2])
    result = {
        "observed_at": now(),
        "disk_free_bytes": usage.free,
        "disk_free_gib": usage.free / 1024**3,
        "estimated_incremental_outputs_bytes": 2 * STORAGE_ESTIMATE_PER_VARIANT_BYTES,
        "reserved_safety_margin_bytes": STORAGE_MARGIN_BYTES,
        "required_free_bytes": required,
        "storage_gate_pass": usage.free >= required,
        "gpus": gpus,
        "compute_processes": [row.strip() for row in processes],
        "gpu_lease_gate_pass": eligible_gpus and not processes,
        "physical_gpu_assignment": GPU_BY_VARIANT,
        "cuda_mapping": "each process CUDA_VISIBLE_DEVICES=<physical>; process-local cuda:0",
    }
    result["status"] = "PASS" if result["storage_gate_pass"] and result["gpu_lease_gate_pass"] else "BLOCKED"
    return result


def preflight_gate() -> dict[str, Any]:
    static = contract_diff_report()
    cpu = read_json(ARTIFACTS / "preflight/training_smoke_cpu.json", {})
    cuda = read_json(ARTIFACTS / "preflight/training_smoke_cuda.json", {})
    evaluator = read_json(ARTIFACTS / "preflight/independent_evaluator_smoke.json", {})
    runtime = read_json(ARTIFACTS / "preflight/runtime_smoke.json", {})
    resource = assert_resource_gate()
    failures = []
    if static.get("status") != "pass":
        failures.append("static contract diff failed")
    if cpu.get("status") != "pass" or len(cpu.get("smokes", [])) != 2:
        failures.append("CPU optimizer/checkpoint/resume smoke missing or failed")
    if cuda.get("status") != "pass" or len(cuda.get("smokes", [])) != 2:
        failures.append("CUDA optimizer/checkpoint/resume smoke missing or failed")
    if evaluator.get("status") != "pass" or not evaluator.get("parallel_rollout"):
        failures.append("independent parallel evaluator smoke missing or failed")
    if runtime.get("status") != "pass":
        failures.append("Local/Global runtime observation smoke missing or failed")
    if resource.get("status") != "PASS":
        failures.append("resource lease or storage margin gate failed")
    return {
        "schema": "iqn-evidence-phase-a-launch-preflight-v1",
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
        "static_contract_status": static.get("status"),
        "cpu_smoke_status": cpu.get("status"),
        "cuda_smoke_status": cuda.get("status"),
        "independent_evaluator_smoke_status": evaluator.get("status"),
        "runtime_smoke_status": runtime.get("status"),
        "resource_gate": resource,
        "completed_at": now(),
    }


def stable_checkpoint(path: Path, previous: dict[Path, tuple[int, int]]) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    size = int(stat.st_size)
    mtime = int(stat.st_mtime_ns)
    prior = previous.get(path)
    if prior is None or prior != (size, mtime):
        previous[path] = (size, mtime)
        return False
    return size > 1_000_000


def write_effective_config(variant: str, stage: str, stage_dir: Path, gpu: int, warm_start: Path | None) -> Path:
    cfg = copy.deepcopy(resolved(variant, stage))
    cfg.update(output_root=str(stage_dir), run_name="training", device="cuda:0")
    cfg["total_timesteps"] = int(cfg["experiment_metadata"]["stage_timesteps"])
    cfg.setdefault("checkpointing", {})["full_resume"] = True
    if stage == "stage1":
        if warm_start is not None:
            raise AssertionError("Stage1 must start from scratch")
        cfg.setdefault("pretrained", {}).pop("path", None)
    else:
        if warm_start is None or not warm_start.is_file():
            raise FileNotFoundError(f"same-representation parent checkpoint missing for {variant}/{stage}")
        cfg.setdefault("pretrained", {})["path"] = str(warm_start)
        cfg["pretrained"]["compatibility_mode"] = "shape_compatible"
    config_path = stage_dir / "training" / "effective_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def training_step(stage_dir: Path) -> int:
    metrics = stage_dir / "training" / "metrics.jsonl"
    if metrics.is_file():
        try:
            last = metrics.read_text(encoding="utf-8").splitlines()[-1]
            return int(json.loads(last).get("global_step", 0))
        except (IndexError, json.JSONDecodeError, ValueError):
            pass
    resume = stage_dir / "training/checkpoints/resume_latest.pt"
    if resume.is_file():
        try:
            import torch
            payload = torch.load(resume, map_location="cpu", weights_only=False)
            return int(payload.get("global_step", 0))
        except Exception:
            return 0
    return 0


def screening_eval_command(variant: str, stage: str, checkpoint: Path, out: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools/evaluate_iqn_evidence_checkpoint_20261006.py"),
        "--variant", variant,
        "--stage", stage,
        "--checkpoint", str(checkpoint),
        "--output", str(out),
        "--episodes", str(SCREENING_EPISODES),
        "--seed-base", str(SCREENING_SEED_BASE),
    ]


def screening_candidate_pending(step: int, reports: dict, ready: list, running_step: int | None) -> bool:
    return step not in reports and step not in ready and step != running_step


def run_stage(variant: str, stage: str, warm_start: Path | None) -> dict[str, Any]:
    variant_root = RUNTIME / variant
    stage_dir = variant_root / "stages" / stage
    training_dir = stage_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    device = GPU_BY_VARIANT[variant]
    cfg_path = write_effective_config(variant, stage, stage_dir, device, warm_start)
    checkpoint_dir = training_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / "resume_latest.pt"
    final_checkpoint = checkpoint_dir / f"final_step_{max(MILESTONES[stage])}.pt"
    selected_checkpoint = None
    selected_report_path = stage_dir / "selection_report.json"
    existing_selection = read_json(selected_report_path)
    if existing_selection and existing_selection.get("selected", {}).get("checkpoint"):
        candidate = Path(existing_selection["selected"]["checkpoint"])
        if candidate.is_file():
            selected_checkpoint = candidate
    if selected_checkpoint is not None:
        selection = existing_selection["selected"]
        return {
            "variant": variant,
            "stage": stage,
            "status": "selected",
            "selected_step": int(selection["step"]),
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": sha256(selected_checkpoint),
            "selection_report": str(selected_report_path),
            "selection_rule": "balanced_floor",
            "candidate_count": len(MILESTONES[stage]),
            "screening_episodes_per_scene": SCREENING_EPISODES,
            "training_pid": None,
            "physical_gpu": device,
            "resumed_existing_selection": True,
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    if resume_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg.setdefault("checkpointing", {})["resume_path"] = str(resume_path)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log_path = training_dir / "stdout.log"
    log = log_path.open("a", encoding="utf-8", buffering=1)
    command = [sys.executable, str(ROOT / "train.py"), "--config", str(cfg_path)]
    training = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {
        "schema": "iqn-evidence-training-launch-v1",
        "variant": variant,
        "stage": stage,
        "pid": training.pid,
        "physical_gpu": device,
        "cuda_visible_devices": str(device),
        "process_local_device": "cuda:0",
        "config_path": str(cfg_path),
        "config_sha256": sha256(cfg_path),
        "total_timesteps": max(MILESTONES[stage]),
        "seed": int(resolved(variant, stage)["seed"]),
        "warm_start_checkpoint": str(warm_start) if warm_start else None,
        "started_at": now(),
        "command": command,
    }
    atomic_json(training_dir / "launch.json", launch)

    expected = MILESTONES[stage]
    reports: dict[int, dict[str, Any]] = {}
    running_eval: tuple[int, subprocess.Popen, Any, Path] | None = None
    report_pool = (stage_dir / "evaluations")
    report_pool.mkdir(parents=True, exist_ok=True)
    stability: dict[Path, tuple[int, int]] = {}
    ready: list[int] = []
    last_heartbeat = 0.0
    while True:
        for step in expected:
            eval_dir = report_pool / f"step_{step:09d}"
            report_path = eval_dir / "report.json"
            if report_path.is_file():
                report = read_json(report_path, {})
                if report.get("status") == "complete" and report.get("episodes_per_scene") == SCREENING_EPISODES:
                    reports[step] = report
                    continue
            checkpoint = checkpoint_dir / f"step_{step}.pt"
            if screening_candidate_pending(step, reports, ready, running_eval[0] if running_eval else None) and checkpoint.is_file() and stable_checkpoint(checkpoint, stability):
                ready.append(step)

        if running_eval is None and ready:
            step = min(ready)
            ready.remove(step)
            eval_dir = report_pool / f"step_{step:09d}"
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
            checkpoint = checkpoint_dir / f"step_{step}.pt"
            eval_log = (eval_dir.with_suffix(".stdout.log")).open("w", encoding="utf-8", buffering=1)
            proc = subprocess.Popen(
                screening_eval_command(variant, stage, checkpoint, eval_dir),
                cwd=ROOT,
                env={**env, "CUDA_VISIBLE_DEVICES": ""},
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running_eval = (step, proc, eval_log, eval_dir)

        if running_eval is not None and running_eval[1].poll() is not None:
            step, proc, eval_log, eval_dir = running_eval
            eval_log.close()
            if proc.returncode != 0:
                raise RuntimeError(f"{variant}/{stage} screening evaluation at {step} failed; see {eval_dir}.stdout.log")
            report = read_json(eval_dir / "report.json", {})
            if report.get("status") != "complete" or report.get("episodes_per_scene") != SCREENING_EPISODES:
                raise RuntimeError(f"{variant}/{stage} screening evaluation at {step} produced incomplete report")
            reports[step] = report
            running_eval = None

        now_mono = time.monotonic()
        if now_mono - last_heartbeat >= 15:
            progress = {
                "schema": "iqn-evidence-training-status-v1",
                "variant": variant,
                "stage": stage,
                "status": "running" if training.poll() is None else "training_process_exited",
                "phase": "training_and_detached_screening",
                "pid": training.pid,
                "pid_alive": training.poll() is None,
                "physical_gpu": device,
                "current_step": training_step(stage_dir),
                "target_step": max(expected),
                "screening_reports_complete": sorted(reports),
                "screening_reports_pending": [step for step in expected if step not in reports],
                "evaluator_pid": running_eval[1].pid if running_eval else None,
                "evaluator_step": running_eval[0] if running_eval else None,
                "last_checkpoint": str(checkpoint_dir / f"step_{max(reports) if reports else 0}.pt") if reports else None,
                "heartbeat_at": now(),
            }
            atomic_json(stage_dir / "status.json", progress)
            last_heartbeat = now_mono

        if training.poll() is not None and running_eval is None and not ready:
            break
        time.sleep(POLL_SECONDS)

    log.close()
    return_code = training.wait()
    if return_code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-10000:]
        raise RuntimeError(f"{variant}/{stage} training exited {return_code}:\n{tail}")
    missing = sorted(set(expected) - set(reports))
    if missing:
        raise RuntimeError(f"{variant}/{stage} stage complete but screening reports missing: {missing}")
    selection = select_balanced(stage_dir, tuple(expected))
    selected = Path(selection["selected"]["checkpoint"])
    if not selected.is_file():
        raise FileNotFoundError(selected)
    selected_hash = sha256(selected)
    if selected_hash != selection["selected"]["checkpoint_sha256"]:
        raise AssertionError("selected checkpoint hash changed after screening")
    result = {
        "variant": variant,
        "stage": stage,
        "status": "selected",
        "selected_step": int(selection["selected"]["step"]),
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": selected_hash,
        "selection_report": str(selected_report_path),
        "selection_rule": "balanced_floor",
        "candidate_count": len(expected),
        "screening_episodes_per_scene": SCREENING_EPISODES,
        "training_pid": training.pid,
        "physical_gpu": device,
    }
    atomic_json(selected_report_path.with_name("stage_result.json"), result)
    atomic_json(stage_dir / "status.json", {**result, "phase": "stage_selected", "heartbeat_at": now()})
    return result


def run_variant(variant: str) -> dict[str, Any]:
    stages = []
    warm_start = None
    for stage in STAGES:
        result = run_stage(variant, stage, warm_start)
        stages.append(result)
        warm_start = Path(result["selected_checkpoint"])
        atomic_json(RUNTIME / variant / "CURRICULUM_STATUS.json", {"variant": variant, "status": "running", "stages": stages, "updated_at": now()})
    return {"variant": variant, "stages": stages, "selected_stage3": stages[-1]}


def final_eval(variant: str, checkpoint: Path, output: Path, config_stage: str = "stage3") -> dict[str, Any]:
    return evaluate(variant, config_stage, checkpoint, output, FINAL_EPISODES, FINAL_SEED_BASE, workers=None)


def load_completed_curricula() -> dict[str, Any]:
    """Read frozen selections for evaluation only; never enter the trainer path."""
    curricula = {}
    for variant in VARIANTS_TO_TRAIN:
        stages = []
        for stage in STAGES:
            stage_dir = RUNTIME / variant / "stages" / stage
            selection = read_json(stage_dir / "selection_report.json", {})
            selected = selection.get("selected", {})
            if not selected:
                raise RuntimeError(f"selection pending: {variant}/{stage}; no training launched")
            checkpoint = Path(selected["checkpoint"])
            if not checkpoint.is_file() or sha256(checkpoint) != selected["checkpoint_sha256"]:
                raise RuntimeError(f"selected checkpoint hash mismatch: {variant}/{stage}")
            for step in MILESTONES[stage]:
                report = read_json(stage_dir / "evaluations" / f"step_{step:09d}" / "report.json", {})
                if report.get("status") != "complete" or report.get("episodes_per_scene") != SCREENING_EPISODES:
                    raise RuntimeError(f"screening pending: {variant}/{stage}/{step}; no training launched")
            final_checkpoint = stage_dir / "training/checkpoints" / f"final_step_{max(MILESTONES[stage])}.pt"
            if not final_checkpoint.is_file() or training_step(stage_dir) != max(MILESTONES[stage]):
                raise RuntimeError(f"training completion unverified: {variant}/{stage}; no training launched")
            stages.append({
                "variant": variant, "stage": stage, "status": "selected",
                "selected_step": int(selected["step"]),
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": selected["checkpoint_sha256"],
                "selection_report": str(stage_dir / "selection_report.json"),
                "selection_rule": "balanced_floor",
                "candidate_count": len(MILESTONES[stage]),
                "screening_episodes_per_scene": SCREENING_EPISODES,
                "training_pid": None, "physical_gpu": GPU_BY_VARIANT[variant],
            })
        curricula[variant] = {"variant": variant, "stages": stages, "selected_stage3": stages[-1]}
    return curricula


def final_comparison(curricula: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for variant in VARIANTS_TO_TRAIN:
        selected = curricula[variant]["selected_stage3"]
        outputs[variant] = (Path(selected["selected_checkpoint"]), selected["selected_checkpoint_sha256"])
    z05_selection_path = Path("/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/selection_report.json")
    z05_selection = read_json(z05_selection_path, {})
    z05_checkpoint = Path(z05_selection.get("selected", {}).get("checkpoint", ""))
    if not z05_checkpoint.is_file() or sha256(z05_checkpoint) != z05_selection.get("selected", {}).get("checkpoint_sha256"):
        raise AssertionError("frozen Z05 selected checkpoint missing or hash mismatch")
    outputs["z05"] = (z05_checkpoint, z05_selection["selected"]["checkpoint_sha256"])

    # Keep the post-audit evaluator pass separate from any reports generated by
    # a supervisor process that imported an earlier evaluator implementation.
    final_root = RUNTIME / "final_heldout_50_corrected"
    final_root.mkdir(parents=True, exist_ok=True)
    final_reports: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(final_eval, variant, checkpoint, final_root / variant): variant
            for variant, (checkpoint, _digest) in outputs.items()
        }
        for future in concurrent.futures.as_completed(futures):
            variant = futures[future]
            final_reports[variant] = future.result()
    seed_manifests = {variant: report["seed_manifest"] for variant, report in final_reports.items()}
    if len({tuple(values) for values in seed_manifests.values()}) != 1:
        raise AssertionError("final held-out seed manifest differs across variants")
    for scene in ("coverage", "capture", "mixed"):
        fingerprints: dict[str, list[str]] = {}
        for variant, report in final_reports.items():
            fingerprints[variant] = [row["initial_state_fingerprint"] for row in report["records"] if row["scene"] == scene]
        if len({tuple(values) for values in fingerprints.values()}) != 1:
            raise AssertionError(f"paired environment state fingerprints differ in {scene}")

    z05_stage_dirs = Path("/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages")
    z05_stages = []
    for stage in STAGES:
        selection = read_json(z05_stage_dirs / stage / "selection_report.json", {})
        picked = selection.get("selected", {})
        z05_stages.append({
            "variant": "z05",
            "stage": stage,
            "selected_step": picked.get("step"),
            "selected_checkpoint": picked.get("checkpoint"),
            "checkpoint_sha256": picked.get("checkpoint_sha256"),
            "selection_rule": "balanced_floor",
            "retrained_in_phase_a": False,
        })
    selected_rows = list(z05_stages)
    for variant in VARIANTS_TO_TRAIN:
        for row in curricula[variant]["stages"]:
            selected_rows.append({
                "variant": variant,
                "stage": row["stage"],
                "selected_step": row["selected_step"],
                "selected_checkpoint": row["selected_checkpoint"],
                "checkpoint_sha256": row["selected_checkpoint_sha256"],
                "selection_rule": "balanced_floor",
                "retrained_in_phase_a": True,
            })

    contract_table = [
        {"item": "NormSense", "local_binary": "NormSense V2", "z05": "NormSense V2", "global_oracle": "NormSense V2"},
        {"item": "policy enemy evidence", "local_binary": "current direct-visible bit d_i and friend d_j; no memory/propagation", "z05": "synchronous recursive Z, lambda=eta=0.5, hard-zero=0.10", "global_oracle": "all active target tokens + active-target scalar"},
        {"item": "reward/internal roles", "local_binary": "unchanged Z05 contract", "z05": "unchanged Z05 contract", "global_oracle": "unchanged local/support contract"},
        {"item": "friend ordering", "local_binary": "physical_only", "z05": "physical_only", "global_oracle": "physical_only"},
        {"item": "information interpretation", "local_binary": "decentralized local information", "z05": "one-hop temporal/recursive propagated evidence", "global_oracle": "global target-state information upper bound; not a fair decentralized baseline"},
        {"item": "network and token capacities", "local_binary": "same IQN; 9/7 feature dims; 8/8/5 caps; AW9", "z05": "same IQN; 9/7 feature dims; 8/8/5 caps; AW9", "global_oracle": "same IQN; 9/7 feature dims; 8/8/5 caps; AW9"},
        {"item": "training seed family", "local_binary": "2026091901 / 2026091902 / 2026091903", "z05": "2026091901 / 2026091902 / 2026091903", "global_oracle": "2026091901 / 2026091902 / 2026091903"},
        {"item": "curriculum budgets", "local_binary": "2.0M / 0.7M / 0.7M", "z05": "2.0M / 0.7M / 0.7M (existing; no retrain)", "global_oracle": "2.0M / 0.7M / 0.7M"},
    ]
    metric_definitions = {
        "safe_complete_rate": {"meaning": "capture followed by strict coverage recovery without task failure", "better": "↑"},
        "capture_rate": {"meaning": "episode completed capture", "better": "↑"},
        "normal_capture_rate": {"meaning": "normal moving capture; excludes stationary capture", "better": "↑"},
        "post_capture_ce_rate": {"meaning": "strict CE recovery after capture", "better": "↑"},
        "collision_rate": {"meaning": "episode with collision", "better": "↓"},
        "recovery_time_seconds": {"meaning": "capture to strict post-capture CE among episodes where it occurred", "better": "↓"},
        "mission_time_seconds": {"meaning": "episode start to safe-complete event among safe-complete episodes", "better": "↓"},
        "censored_fraction": {"meaning": "episodes not reaching the scene's defined success event", "better": "↓"},
        "strict_ce_rate": {"meaning": "strict Voronoi coverage CE success", "better": "↑"},
        "ce_rms": {"meaning": "coverage center-error RMS", "better": "↓"},
        "ce_max": {"meaning": "coverage maximum center error", "better": "↓"},
        "area_cv": {"meaning": "Voronoi area coefficient of variation", "better": "↓ / diagnostic"},
        "stationary_capture_rate": {"meaning": "capture while stationary", "better": "diagnostic; not a substitute for normal capture"},
    }
    failure_taxonomy = {}
    for variant, report in final_reports.items():
        failure_taxonomy[variant] = {}
        for scene in ("coverage", "capture", "mixed"):
            rows = [row for row in report["records"] if row["scene"] == scene]
            bins: dict[str, int] = {}
            for row in rows:
                if scene == "mixed":
                    if row["safe_complete"]:
                        category = "safe_complete"
                    elif not row["captured"]:
                        category = "capture_not_completed"
                    elif row["collision"]:
                        category = "collision_failure"
                    elif not row["ce_success"]:
                        category = "post_capture_ce_not_reached"
                    else:
                        category = "other_lifecycle_failure"
                elif scene == "capture":
                    if row["normal_capture"]:
                        category = "normal_capture"
                    elif row["stationary_capture"]:
                        category = "stationary_capture_diagnostic"
                    elif row["collision"]:
                        category = "collision_before_capture"
                    else:
                        category = "capture_not_completed"
                else:
                    if row["ce_success"]:
                        category = "strict_ce_completed"
                    elif row["collision"]:
                        category = "collision_before_ce"
                    else:
                        category = "strict_ce_not_reached"
                bins[category] = bins.get(category, 0) + 1
            failure_taxonomy[variant][scene] = bins

    comparison = {
        "schema": "iqn-evidence-phase-a-final-comparison-v1",
        "scientific_question": "Under matched network, reward, curriculum, task, physics, and seed family, does the enemy-evidence representation improve Mixed capture-and-recovery coordination?",
        "scientific_classification": "descriptive matched Phase-A single-training-seed comparison; Local-Binary vs Z05 is the primary comparison; GLOBAL_ORACLE is a global target-state information upper bound, not a fair distributed baseline",
        "contracts": contract_table,
        "stage_selection": selected_rows,
        "heldout_final": {
            "episodes_per_scene": FINAL_EPISODES,
            "seed_base": FINAL_SEED_BASE,
            "diagnostic_implementation": "corrected_global_target_token_occupancy_and_shortfall_v2",
            "paired_seed_rule": "seed_base + scene_index*100000 + episode_index",
            "seed_manifest": next(iter(seed_manifests.values())),
            "initial_environment_fingerprints_paired": True,
            "variants": {
                variant: {
                    "checkpoint_sha256": outputs[variant][1],
                    "summary": report["summary"],
                    "evidence_diagnostics": report["evidence_diagnostics"],
                    "report_path": str(final_root / variant / "report.json"),
                    "resource_policy": report["resource_policy"],
                }
                for variant, report in final_reports.items()
            },
        },
        "metric_definitions": metric_definitions,
        "failure_taxonomy": failure_taxonomy,
        "resource_runtime_notes": {
            "root_free_at_launch_gib": read_json(ARTIFACTS / "preflight/launch_preflight.json", {}).get("resource_gate", {}).get("disk_free_gib"),
            "run_root": str(RUNTIME),
            "training_gpus": GPU_BY_VARIANT,
            "evaluation": "separate CPU processes; worker count dynamically constrained from CPU load, RAM, GPU load/VRAM; evaluator priority lowered",
            "training_evaluation_policy": "checkpoint rollouts run during in-stage training; stage promotion waits for that stage's selection reports",
        },
        "curriculum_reports": curricula,
        "z05_existing_stage_selection": z05_stages,
        "generated_at": now(),
    }
    atomic_json(ARTIFACTS / "final_comparison.json", comparison)
    markdown = render_markdown(comparison)
    (ARTIFACTS / "FINAL_REPORT_ZH.md").write_text(markdown, encoding="utf-8")
    return comparison


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EXP-EVIDENCE-01 Phase-A 最终报告",
        "",
        "## 合同对照",
        "",
        "| 项目 | Local-Binary | Z05 | Global-Oracle |",
        "|---|---|---|---|",
    ]
    for row in report["contracts"]:
        lines.append(f"| {row['item']} | {row['local_binary']} | {row['z05']} | {row['global_oracle']} |")
    lines += ["", "## Stage 选择", "", "| Variant | Stage | selected step | checkpoint SHA256 |", "|---|---|---:|---|"]
    for row in report["stage_selection"]:
        lines.append(f"| {row['variant']} | {row['stage']} | {row['selected_step']} | `{row['checkpoint_sha256']}` |")
    lines += ["", "## Stage3 Pure Coverage", "", "| Variant | strict CE ↑ | CE RMS ↓ | CE max ↓ | area CV ↓ / diagnostic | time-to-CE ↓ | collision ↓ | censored ↓ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, values in report["heldout_final"]["variants"].items():
        m = values["summary"]["coverage"]
        lines.append(f"| {variant} | {m['strict_ce_rate']:.3f} | {m['ce_rms']['mean']} | {m['ce_max']['mean']} | {m['area_cv']['mean']} | {m['time_to_ce_seconds']['mean']} | {m['collision_rate']:.3f} | {m['censored_fraction']:.3f} |")
    lines += ["", "## Stage3 Pure Capture", "", "| Variant | normal capture ↑ | capture ↑ | stationary diagnostic | capture time ↓ | collision ↓ | censored ↓ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, values in report["heldout_final"]["variants"].items():
        m = values["summary"]["capture"]
        lines.append(f"| {variant} | {m['normal_capture_rate']:.3f} | {m['capture_rate']:.3f} | {m['stationary_capture_rate']:.3f} | {m['capture_seconds']['mean']} | {m['collision_rate']:.3f} | {m['censored_fraction']:.3f} |")
    lines += ["", "## Stage3 Mixed（主结果）", "", "| Variant | safe-complete ↑ | capture ↑ | post-capture CE ↑ | collision ↓ | recovery time ↓ | mission time ↓ | censored ↓ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, values in report["heldout_final"]["variants"].items():
        m = values["summary"]["mixed"]
        lines.append(f"| {variant} | {m['safe_complete_rate']:.3f} | {m['capture_rate']:.3f} | {m['post_capture_ce_rate']:.3f} | {m['collision_rate']:.3f} | {m['recovery_time_seconds']['mean']} | {m['mission_time_seconds']['mean']} | {m['censored_fraction']:.3f} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "`GLOBAL_ORACLE = global target-state information upper bound`，不作为公平 decentralized baseline。Z05 不重训；三条 Stage3 selected checkpoint 使用同一独立 held-out 50-episode seed manifest。结果是 Phase-A 单训练 seed 的描述性证据，不预设 Local < Z < Global，也不因结果弱而擅自开启额外 seed family。",
        "",
        "## Evidence diagnostics 与 failure taxonomy",
        "",
        "完整数值见同目录 `final_comparison.json`。机制诊断不计入主性能 score。",
        "",
        f"生成时间：{report['generated_at']}",
        "",
    ]
    return "\n".join(lines)


def coordinator() -> dict[str, Any]:
    gate = preflight_gate()
    atomic_json(ARTIFACTS / "preflight/launch_preflight.json", gate)
    if gate["status"] != "PASS":
        raise RuntimeError(f"Phase-A launch gate not passed: {gate['failures']}")
    state: dict[str, Any] = {
        "schema": "iqn-evidence-phase-a-coordinator-v1",
        "status": "running",
        "variant_status": {},
        "started_at": now(),
        "launch_preflight": str(ARTIFACTS / "preflight/launch_preflight.json"),
    }
    atomic_json(ARTIFACTS / "coordinator_state.json", state)
    failures: dict[str, str] = {}
    curricula: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_variant, variant): variant for variant in VARIANTS_TO_TRAIN}
        for future in concurrent.futures.as_completed(futures):
            variant = futures[future]
            try:
                curricula[variant] = future.result()
                state["variant_status"][variant] = {"status": "curriculum_complete", "stages": curricula[variant]["stages"]}
            except BaseException as exc:
                failures[variant] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                state["variant_status"][variant] = {"status": "failed", "error": failures[variant]}
            state["updated_at"] = now()
            state["failures"] = failures
            atomic_json(ARTIFACTS / "coordinator_state.json", state)
    if failures:
        state["status"] = "training_or_selection_failed"
        atomic_json(ARTIFACTS / "coordinator_state.json", state)
        raise RuntimeError(json.dumps(failures, ensure_ascii=False, indent=2))
    final = final_comparison(curricula)
    state["status"] = "complete"
    state["final_report"] = str(ARTIFACTS / "final_comparison.json")
    state["updated_at"] = now()
    atomic_json(ARTIFACTS / "coordinator_state.json", state)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--final-only", action="store_true", help="evaluate completed selections without launching any training")
    parser.add_argument("--check-final-readiness", action="store_true", help="read-only check of training and selection completion")
    args = parser.parse_args()
    if sum((args.launch, args.preflight_only, args.final_only, args.check_final_readiness)) != 1:
        parser.error("choose exactly one operation")
    if args.final_only or args.check_final_readiness:
        curricula = load_completed_curricula()
        if args.check_final_readiness:
            print(json.dumps({"status": "FINAL_EVAL_READY", "curricula": curricula, "training_launched": False}))
        else:
            report = final_comparison(curricula)
            print(json.dumps({"status": "complete", "report": str(ARTIFACTS / "final_comparison.json"), "training_launched": False}))
    elif args.preflight_only:
        gate = preflight_gate()
        atomic_json(ARTIFACTS / "preflight/launch_preflight.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        if gate["status"] != "PASS":
            raise SystemExit(2)
    else:
        report = coordinator()
        print(json.dumps({"status": "complete", "report": str(ARTIFACTS / "final_comparison.json"), "scientific_classification": report["scientific_classification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
