#!/usr/bin/env python3
"""Z05/Z07 unified-decay full IQN curriculum runner."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import CoCapTrainer, load_config
from tools import iqn_token_matched_20260919 as matched
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file


SCHEMA = "iqn-z-unified-decay-curriculum-v1"
CONFIG_DIR = ROOT / "configs/experiments/iqn_z_unified_decay_curriculum_20260919"
FINAL_DIR = ROOT / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802"
ARMS = ("z05", "z07")
ALPHAS = {"z05": 0.5, "z07": 0.7}
STAGE_ORDER = ("stage1", "stage2", "stage3")
CONFIGS = {
    "z05": {
        "stage1": CONFIG_DIR / "z05_stage1_4p1e1obs_2m.yaml",
        "stage2": CONFIG_DIR / "z05_stage2_8p2e2obs_700k.yaml",
        "stage3": CONFIG_DIR / "z05_stage3_12p3e3obs_700k.yaml",
    },
    "z07": {
        "stage1": CONFIG_DIR / "z07_stage1_4p1e1obs_2m.yaml",
        "stage2": CONFIG_DIR / "z07_stage2_8p2e2obs_700k.yaml",
        "stage3": CONFIG_DIR / "z07_stage3_12p3e3obs_700k.yaml",
    },
}
FINAL_CONFIGS = {
    "stage1": FINAL_DIR / "stage1_4p1e1obs_scratch2m.yaml",
    "stage2": FINAL_DIR / "stage2_8p2e2obs_700k.yaml",
    "stage3": FINAL_DIR / "stage3_12p3e3obs_700k.yaml",
}
MILESTONES = {
    "stage1": tuple(range(100_000, 2_000_001, 100_000)),
    "stage2": tuple(range(100_000, 700_001, 100_000)),
    "stage3": tuple(range(100_000, 700_001, 100_000)),
}
OPERATIONAL_PATHS = {
    "device",
    "output_root",
    "run_name",
}
ALLOWED_ARM_DIFFS = {"z_state.lambda", "z_state.eta"}
FINAL_INTENTIONAL_PREFIXES = (
    "checkpointing.",
    "env.collision_semantics",
    "experiment_metadata.",
    "formal_evaluation.",
    "iqn.include_is_pursuing",
    "iqn.include_z_state",
    "iqn.pursuer_feature_dim",
    "iqn.pursuing_embed_dim",
    "iqn.pursuing_late_fusion",
    "iqn.self_feature_dim",
    "normsense_v2.",
    "perception.friend_ordering_mode",
    "perception.include_is_pursuing",
    "perception.include_z_state",
    "perception.self_feature_dim",
    "runtime_semantic_assertions.",
    "voradj.enemy_sensing_radius",
    "voradj.obstacle_sensing_radius",
    "voradj.onboard_sensing.",
    "z_state.",
)


@dataclass(frozen=True)
class StageResult:
    stage: str
    selected_checkpoint: Path
    selection_report: Path


def now_local() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolved(arm: str, stage: str) -> dict[str, Any]:
    if arm not in ARMS or stage not in STAGE_ORDER:
        raise ValueError(f"unknown arm/stage: {arm}/{stage}")
    return matched.resolved(CONFIGS[arm][stage])


def flattened(config: Mapping[str, Any]) -> dict[str, Any]:
    return matched.flatten(config)


def diff_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    return matched._diff_rows(left, right)


def write_contract_reports(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    arm_reports: dict[str, Any] = {}
    for arm in ARMS:
        per_stage = {}
        for stage in STAGE_ORDER:
            final = load_config(str(FINAL_CONFIGS[stage]))
            candidate = resolved(arm, stage)
            rows = diff_rows(flattened(final), flattened(candidate))
            unexpected = [
                row
                for row in rows
                if row["path"] not in OPERATIONAL_PATHS
                and not row["path"].startswith(FINAL_INTENTIONAL_PREFIXES)
                and row["path"] != "seed"
            ]
            report = {
                "schema": SCHEMA,
                "arm": arm,
                "stage": stage,
                "canonical_final": str(FINAL_CONFIGS[stage]),
                "candidate": str(CONFIGS[arm][stage]),
                "differences": rows,
                "unexpected_difference_count": len(unexpected),
                "unexpected_differences": unexpected,
                "canonical_hash": stable_hash(final),
                "candidate_hash": stable_hash(candidate),
            }
            matched.atomic_json(output / f"{arm.upper()}_{stage.upper()}_VS_FINAL_CONTRACT_DIFF.json", report)
            per_stage[stage] = report
        arm_reports[arm] = per_stage

    cross_stage = {}
    for stage in STAGE_ORDER:
        left = resolved("z05", stage)
        right = resolved("z07", stage)
        rows = [row for row in diff_rows(flattened(left), flattened(right)) if row["path"] not in OPERATIONAL_PATHS]
        non_alpha = [row for row in rows if row["path"] not in ALLOWED_ARM_DIFFS]
        report = {
            "schema": SCHEMA,
            "stage": stage,
            "differences": rows,
            "allowed_alpha_differences": sorted(ALLOWED_ARM_DIFFS),
            "non_alpha_difference_count": len(non_alpha),
            "non_alpha_differences": non_alpha,
        }
        matched.atomic_json(output / f"Z05_VS_Z07_{stage.upper()}_CONTRACT_DIFF.json", report)
        cross_stage[stage] = report
    matched.atomic_json(
        output / "Z05_vs_Z07_CONTRACT_DIFF.json",
        {"schema": SCHEMA, "stages": cross_stage, "non_alpha_difference_count": sum(row["non_alpha_difference_count"] for row in cross_stage.values())},
    )
    matched.atomic_json(output / "Z05_vs_FINAL_CONTRACT_DIFF.json", {"schema": SCHEMA, "stages": arm_reports["z05"]})
    matched.atomic_json(output / "Z07_vs_FINAL_CONTRACT_DIFF.json", {"schema": SCHEMA, "stages": arm_reports["z07"]})
    return {"arms": arm_reports, "cross": cross_stage}


def _close(trainer: CoCapTrainer) -> None:
    matched.close_trainer(trainer)


def _initial_models() -> tuple[CoCapIQN, CoCapIQN, str]:
    z05 = resolved("z05", "stage1")
    z07 = resolved("z07", "stage1")
    if z05["seed"] != z07["seed"]:
        raise AssertionError("Z05/Z07 model seeds differ")
    torch.manual_seed(int(z05["seed"]))
    model05 = CoCapIQN(matched.model_config(z05))
    torch.manual_seed(int(z07["seed"]))
    model07 = CoCapIQN(matched.model_config(z07))
    if list(model05.state_dict()) != list(model07.state_dict()):
        raise AssertionError("Z05/Z07 initial parameter keys differ")
    for key, value in model05.state_dict().items():
        if value.shape != model07.state_dict()[key].shape or not torch.equal(value, model07.state_dict()[key]):
            raise AssertionError(f"Z05/Z07 initial tensor differs: {key}")
    digest05, digest07 = matched.state_hash(model05), matched.state_hash(model07)
    if digest05 != digest07:
        raise AssertionError("Z05/Z07 initial state_dict SHA differs")
    return model05, model07, digest05


def _assert_production_contract(config: Mapping[str, Any], alpha: float) -> None:
    z = config["z_state"]
    if not math.isclose(float(z["lambda"]), alpha) or not math.isclose(float(z["eta"]), alpha):
        raise AssertionError("unified alpha contract drift")
    if not math.isclose(float(z["hard_zero_threshold"]), 0.10):
        raise AssertionError("hard-zero threshold drift")
    if config["train_mode"] != "voradj_mixed_coverage":
        raise AssertionError("Final mixed coverage train mode drift")
    if config["voradj"]["replay_batch_counts"] != {
        "pursuing": 64,
        "pre_capture_cover": 16,
        "post_capture_real": 32,
        "recovery_pure": 16,
    }:
        raise AssertionError("Final replay batch contract drift")
    recovery = config["voradj"]["recovery"]
    expected_recovery = {
        "capture_state_pool_capacity": 1000,
        "captured_state_ratio": 0.75,
        "map_random_ratio_within_non_capture": 0.5,
    }
    for key, value in expected_recovery.items():
        if recovery.get(key) != value:
            raise AssertionError(f"Final recovery contract drift: {key}")
    if config["perception"]["include_is_pursuing"] or not config["perception"]["include_z_state"]:
        raise AssertionError("Z is not the exclusive policy evidence state")
    if config["iqn"]["include_is_pursuing"] or not config["iqn"]["include_z_state"]:
        raise AssertionError("IQN evidence input drift")
    if config["iqn"]["pursuing_late_fusion"]:
        raise AssertionError("late fusion is enabled")
    if config["perception"]["friend_ordering_mode"] != "physical_only":
        raise AssertionError("friend ordering is not physical-only")


def preflight(output: Path, device: str = "cpu") -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reports = write_contract_reports(output / "contract_diffs")
    if any(item["unexpected_difference_count"] for arm in reports["arms"].values() for item in arm.values()):
        raise AssertionError("unexpected difference from canonical Final contract")
    if any(item["non_alpha_difference_count"] for item in reports["cross"].values()):
        raise AssertionError("Z05/Z07 non-alpha config difference")
    for arm in ARMS:
        for stage in STAGE_ORDER:
            _assert_production_contract(resolved(arm, stage), ALPHAS[arm])

    model05, model07, initial_hash = _initial_models()
    smoke_root = Path(tempfile.mkdtemp(prefix="z-v2-preflight-", dir=output))
    smoke_results = {}
    try:
        for arm in ARMS:
            cfg = copy.deepcopy(resolved(arm, "stage1"))
            cfg.update(output_root=str(smoke_root / arm), run_name="production_probe", device=device, total_timesteps=0)
            probe = CoCapTrainer(cfg)
            replay_sizes = {name: len(buffer) for name, buffer in probe.replays.items()}
            if replay_sizes != {"pursuing": 0, "pre_capture_cover": 0, "post_capture_real": 0, "recovery_pure": 0}:
                raise AssertionError(f"{arm} production replay is not fresh: {replay_sizes}")
            if probe.optimizer.state or probe.global_step != 0 or len(probe.recovery_init_pool):
                raise AssertionError(f"{arm} production scratch state is not fresh")
            if matched.state_hash(probe.model) != matched.state_hash(probe.target_model):
                raise AssertionError(f"{arm} target is not initialized from online")
            if any(np.any(env.z_state) for env in probe.envs.values()):
                raise AssertionError(f"{arm} z state is not reset to zero before rollout")
            _close(probe)

            smoke = copy.deepcopy(resolved(arm, "stage1"))
            smoke.update(output_root=str(smoke_root / arm), run_name="optimizer_smoke", device=device, total_timesteps=16, train_mode="voradj")
            smoke["iqn"].update(batch_size=8, min_replay_size=8, replay_capacity=256, train_freq=1, target_update_freq=1, checkpoint_freq=16, log_freq_steps=1)
            smoke["checkpointing"] = {"full_resume": True}
            trainer = CoCapTrainer(smoke)
            initial = {key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()}
            checkpoint = trainer.train()
            changed = any(not torch.equal(initial[key], value.detach().cpu()) for key, value in trainer.model.state_dict().items())
            if trainer.update_steps < 10 or not changed or trainer.loss_ema is None or not math.isfinite(trainer.loss_ema):
                raise AssertionError(f"{arm} real-update smoke failed")
            if trainer.target_update_count <= 0:
                raise AssertionError(f"{arm} target network did not update")
            env = trainer.envs[trainer.current_task]
            snapshot = env.z_state_dict()
            if snapshot["schema"] != "cocap-z-state-v2":
                raise AssertionError("formal smoke did not save Z-v2 state")
            resume = trainer._save_full_resume()
            resume_cfg = copy.deepcopy(smoke)
            resume_cfg["total_timesteps"] = 17
            resume_cfg["checkpointing"]["resume_path"] = str(resume)
            resumed = CoCapTrainer(resume_cfg)
            if resumed.global_step != 16 or resumed.envs[resumed.current_task].z_state_dict() != snapshot:
                raise AssertionError(f"{arm} exact resume mismatch")
            evaluation = matched.evaluate_checkpoint(
                checkpoint,
                smoke_root / arm / "formal_eval_smoke",
                device,
                episodes=1,
                seed_base=int(smoke["formal_evaluation"]["seed_base"]),
                arm=arm,
                max_steps=4,
                config_path=CONFIGS[arm]["stage1"],
            )
            if not evaluation["summary"]["z"]["update_count_per_decision_exact"]:
                raise AssertionError(f"{arm} formal evaluator saw Z update-count drift")
            smoke_results[arm] = {
                "optimizer_updates": int(trainer.update_steps),
                "loss_ema": float(trainer.loss_ema),
                "target_update_count": int(trainer.target_update_count),
                "checkpoint_exact_load": True,
                "checkpoint_sha256": sha256_file(checkpoint),
                "full_resume_exact": True,
                "full_resume_sha256": sha256_file(resume),
                "full_resume_bytes": resume.stat().st_size,
                "z_schema": snapshot["schema"],
                "formal_evaluator_smoke": True,
                "z_update_count_per_decision_exact": True,
            }
            _close(trainer)
            _close(resumed)
    finally:
        shutil.rmtree(smoke_root, ignore_errors=True)
        del model05, model07
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "schema": SCHEMA,
        "status": "pass",
        "branch": matched.git("branch", "--show-current"),
        "head": matched.git("rev-parse", "HEAD"),
        "initial_model_state_dict_sha256": initial_hash,
        "initial_models_bit_exact": True,
        "z05_z07_non_alpha_difference_count": 0,
        "hard_zero_threshold": 0.10,
        "decision_update_contract": "exactly_one_z_update_per_env_step",
        "smoke": smoke_results,
        "completed_at": now_local(),
    }
    matched.atomic_json(output / "startup_sanity.json", result)
    return result


def _mean(summary: Mapping[str, Any], *path: str) -> float:
    value: Any = summary
    for key in path:
        value = value[key]
    if isinstance(value, Mapping):
        value = value.get("mean")
    return float(value) if value is not None else math.inf


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left <= 0.0 or right <= 0.0 else 2.0 * left * right / (left + right)


def _normalized(values: list[float], *, higher_is_better: bool) -> list[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [0.0] * len(values)
    low, high = min(finite), max(finite)
    if math.isclose(low, high):
        return [1.0 if math.isfinite(value) else 0.0 for value in values]
    result = []
    for value in values:
        if not math.isfinite(value):
            result.append(0.0)
            continue
        score = (value - low) / (high - low)
        result.append(score if higher_is_better else 1.0 - score)
    return result


def select_balanced(stage_dir: Path, milestones: tuple[int, ...]) -> dict[str, Any]:
    candidates = []
    for step in milestones:
        report_path = stage_dir / "evaluations" / f"step_{step:09d}" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report["summary"]
        pure_capture = float(summary["capture"]["normal_capture_rate"])
        mixed_capture = float(summary["mixed"]["capture_rate"])
        capture_score = min(pure_capture, mixed_capture)
        coverage_score = float(summary["coverage"]["strict_ce_rate"])
        collisions = [
            float(summary[scene]["collision_rate"])
            for scene in ("coverage", "capture", "mixed")
        ]
        candidates.append(
            {
                "step": step,
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "pure_capture": pure_capture,
                "mixed_capture": mixed_capture,
                "capture_score": capture_score,
                "coverage_score": coverage_score,
                "balanced_floor": min(capture_score, coverage_score),
                "mixed_safe_complete": float(summary["mixed"]["safe_complete_rate"]),
                "mixed_post_capture_ce": float(summary["mixed"]["post_capture_ce_rate"]),
                "harmonic_capture_coverage": _harmonic(capture_score, coverage_score),
                "worst_collision_rate": max(collisions),
                "ce_rms": _mean(summary, "coverage", "ce_rms"),
                "area_cv": _mean(summary, "coverage", "area_cv"),
                "capture_seconds": _mean(summary, "capture", "capture_seconds"),
            }
        )

    fallback = all(math.isclose(row["coverage_score"], 0.0) for row in candidates)
    if fallback:
        max_capture = max(row["capture_score"] for row in candidates)
        floor = 0.5 * max_capture
        eligible = [row for row in candidates if row["capture_score"] >= floor]
        if not eligible:
            eligible = list(candidates)
        capture_norm = _normalized([row["capture_score"] for row in eligible], higher_is_better=True)
        ce_norm = _normalized([row["ce_rms"] for row in eligible], higher_is_better=False)
        cv_norm = _normalized([row["area_cv"] for row in eligible], higher_is_better=False)
        for row, capture_value, ce_value, cv_value in zip(eligible, capture_norm, ce_norm, cv_norm):
            row["fallback_capture_normalized"] = capture_value
            row["fallback_coverage_normalized"] = 0.5 * (ce_value + cv_value)
            row["fallback_maximin"] = min(row["fallback_capture_normalized"], row["fallback_coverage_normalized"])
        selected = max(
            eligible,
            key=lambda row: (
                row["fallback_maximin"],
                row["fallback_coverage_normalized"],
                row["fallback_capture_normalized"],
                row["mixed_safe_complete"],
                row["mixed_post_capture_ce"],
                -row["worst_collision_rate"],
                -row["ce_rms"],
                -row["area_cv"],
                -row["capture_seconds"],
                row["step"],
            ),
        )
        reason = "all strict coverage scores were zero; applied capture-retention filter and normalized capture/CE-RMS/area-CV maximin compromise"
    else:
        selected = max(
            candidates,
            key=lambda row: (
                row["balanced_floor"],
                row["mixed_safe_complete"],
                row["mixed_post_capture_ce"],
                row["harmonic_capture_coverage"],
                -row["worst_collision_rate"],
                -row["ce_rms"],
                -row["area_cv"],
                -row["capture_seconds"],
                row["step"],
            ),
        )
        reason = "maximized BalancedFloor with the registered tie-break order"
    report = {
        "schema": SCHEMA,
        "fallback_used": fallback,
        "selection_reason": reason,
        "capture_retention_filter": "CaptureScore >= 0.5 * max(CaptureScore)" if fallback else None,
        "candidates": candidates,
        "selected": selected,
        "completed_at": now_local(),
    }
    matched.atomic_json(stage_dir / "selection_report.json", report)
    return report


def read_last_jsonl(path: Path) -> dict[str, Any]:
    return matched.read_last_jsonl(path)


def freeze_resume(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def _stage_runtime_evidence(stage_dir: Path, step: int, config_hash: str) -> dict[str, Any]:
    training = stage_dir / "training"
    latest = read_last_jsonl(training / "metrics.jsonl")
    resume = training / "checkpoints/resume_latest.pt"
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    runtime = payload.get("runtime", {})
    evidence = {
        "schema": SCHEMA,
        "step": step,
        "config_hash": config_hash,
        "contract_hash": payload.get("contract_hash"),
        "full_resume_schema": payload.get("schema"),
        "full_resume_bytes": resume.stat().st_size,
        "global_step": int(runtime.get("global_step", -1)),
        "optimizer_updates": int(runtime.get("update_steps", -1)),
        "target_sync_count": int(runtime.get("target_update_count", -1)),
        "replay_sizes": {name: len(buffer) for name, buffer in runtime.get("replays", {}).items()},
        "recovery_pool_size": len(runtime.get("recovery_init_pool", [])),
        "latest_metrics": latest,
        "completed_at": now_local(),
    }
    matched.atomic_json(stage_dir / "runtime_evidence" / f"step_{step:09d}.json", evidence)
    return evidence


def _fresh_stage_gate(trainer: CoCapTrainer, stage: str, warm_start: Path | None) -> dict[str, Any]:
    replay_sizes = {name: len(buffer) for name, buffer in trainer.replays.items()}
    if trainer.global_step != 0 or any(replay_sizes.values()) or trainer.optimizer.state or len(trainer.recovery_init_pool):
        raise AssertionError(f"{stage} did not start with fresh runtime state")
    if any(np.any(env.z_state) for env in trainer.envs.values()):
        raise AssertionError(f"{stage} did not start with z=0")
    if matched.state_hash(trainer.model) != matched.state_hash(trainer.target_model):
        raise AssertionError(f"{stage} target is not initialized from online")
    return {
        "stage": stage,
        "global_step": 0,
        "replay_sizes": replay_sizes,
        "optimizer_state_entries": 0,
        "recovery_pool_size": 0,
        "z_all_zero": True,
        "model_target_equal": True,
        "warm_start_checkpoint": str(warm_start) if warm_start else None,
        "warm_start_checkpoint_sha256": sha256_file(warm_start) if warm_start else None,
        "completed_at": now_local(),
    }


def run_stage(
    arm: str,
    stage: str,
    output: Path,
    device: str,
    warm_start: Path | None,
    state: dict[str, Any],
) -> StageResult:
    stage_dir = output / "stages" / stage
    training_dir = stage_dir / "training"
    resume = training_dir / "checkpoints/resume_latest.pt"
    cfg = copy.deepcopy(resolved(arm, stage))
    cfg.update(output_root=str(stage_dir), run_name="training", device=device)
    cfg.setdefault("checkpointing", {})["full_resume"] = True
    if stage == "stage1":
        if (cfg.get("pretrained", {}) or {}).get("path"):
            raise AssertionError("Stage1 must be fresh scratch")
    else:
        if warm_start is None or not warm_start.is_file():
            raise FileNotFoundError(f"{stage} warm-start checkpoint is missing")
        cfg.setdefault("pretrained", {})["path"] = str(warm_start)
        cfg["pretrained"]["compatibility_mode"] = "shape_compatible"
    config_hash = stable_hash(cfg)
    milestones = MILESTONES[stage]
    current = 0
    if resume.is_file():
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        current = int(payload["runtime"]["global_step"])
    for target in milestones:
        state.update(stage=stage, target_step=target, current_step=current, phase="training", segment_start=time.time(), segment_step=current)
        if current < target:
            segment = copy.deepcopy(cfg)
            segment["total_timesteps"] = target
            if current:
                segment["checkpointing"]["resume_path"] = str(resume)
            trainer = CoCapTrainer(segment)
            if trainer.global_step != current:
                raise AssertionError("same-stage exact-resume global step mismatch")
            if current == 0:
                gate = _fresh_stage_gate(trainer, stage, warm_start)
                matched.atomic_json(stage_dir / "fresh_stage_gate.json", gate)
            trainer.train()
            if trainer.global_step != target:
                raise AssertionError("trainer did not stop at registered milestone")
            current = target
            _close(trainer)
        milestone_resume = training_dir / f"checkpoints/resume_step_{target:09d}.pt"
        freeze_resume(resume, milestone_resume)
        checkpoint = training_dir / f"checkpoints/step_{target}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state.update(phase="formal_evaluation", current_step=current)
        eval_dir = stage_dir / "evaluations" / f"step_{target:09d}"
        report_path = eval_dir / "report.json"
        if not report_path.is_file():
            matched.evaluate_checkpoint(
                checkpoint,
                eval_dir,
                device,
                episodes=int(cfg["formal_evaluation"]["episodes_per_scene"]),
                seed_base=int(cfg["formal_evaluation"]["seed_base"]),
                arm=arm,
                config_path=CONFIGS[arm][stage],
            )
        evidence = _stage_runtime_evidence(stage_dir, target, config_hash)
        matched.atomic_json(
            output / "status.json",
            {
                "schema": SCHEMA,
                "arm": arm,
                "status": "running",
                "phase": "formal_evaluation_complete",
                "stage": stage,
                "current_step": target,
                "target_step": target,
                "last_checkpoint": str(checkpoint),
                "last_full_resume": str(milestone_resume),
                "last_evaluation": str(report_path),
                "runtime_evidence": evidence,
                "updated_at": now_local(),
            },
        )
    state.update(phase="balanced_selection", current_step=current)
    selection = select_balanced(stage_dir, milestones)
    selected = Path(selection["selected"]["checkpoint"])
    return StageResult(stage=stage, selected_checkpoint=selected, selection_report=stage_dir / "selection_report.json")


def _heartbeat(output: Path, arm: str, state: Mapping[str, Any], stop: threading.Event) -> None:
    while not stop.wait(15):
        stage = str(state.get("stage", "stage1"))
        metrics = read_last_jsonl(output / "stages" / stage / "training/metrics.jsonl")
        current = int(metrics.get("global_step", state.get("current_step", 0)) or 0)
        elapsed = max(time.time() - float(state.get("segment_start", time.time())), 1e-6)
        speed = (current - int(state.get("segment_step", current))) / elapsed
        matched.atomic_json(
            output / "heartbeat.json",
            {
                "schema": SCHEMA,
                "arm": arm,
                "pid": os.getpid(),
                "status": "running",
                "phase": state.get("phase"),
                "stage": stage,
                "current_step": current,
                "target_step": state.get("target_step"),
                "steps_per_second_segment": speed if speed > 0 else None,
                "latest_metrics": metrics,
                "updated_at": now_local(),
            },
        )


def supervise_arm(arm: str, output: Path, device: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    sanity_path = ROOT / "artifacts/2026-09-19_iqn_z_unified_decay_curriculum/preflight/startup_sanity.json"
    if not sanity_path.is_file() or json.loads(sanity_path.read_text(encoding="utf-8")).get("status") != "pass":
        raise RuntimeError("shared preflight is missing or not PASS")
    state: dict[str, Any] = {"stage": "stage1", "phase": "starting", "current_step": 0, "target_step": 100_000, "segment_start": time.time(), "segment_step": 0}
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat, args=(output, arm, state, stop), daemon=True)
    thread.start()
    launch = {
        "schema": SCHEMA,
        "arm": arm,
        "alpha": ALPHAS[arm],
        "hard_zero_threshold": 0.10,
        "pid": os.getpid(),
        "branch": matched.git("branch", "--show-current"),
        "head": matched.git("rev-parse", "HEAD"),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "output": str(output),
        "started_at": now_local(),
    }
    matched.atomic_json(output / "launch.json", launch)
    try:
        results: list[StageResult] = []
        warm_start = None
        for stage in STAGE_ORDER:
            existing_selection = output / "stages" / stage / "selection_report.json"
            if existing_selection.is_file():
                selection = json.loads(existing_selection.read_text(encoding="utf-8"))
                result = StageResult(stage, Path(selection["selected"]["checkpoint"]), existing_selection)
            else:
                result = run_stage(arm, stage, output, device, warm_start, state)
            results.append(result)
            warm_start = result.selected_checkpoint
        stage_reports = []
        parent_checkpoint = None
        for result in results:
            selection = json.loads(result.selection_report.read_text(encoding="utf-8"))
            selected = selection["selected"]
            selected_step = int(selected["step"])
            evaluation_path = output / "stages" / result.stage / "evaluations" / f"step_{selected_step:09d}" / "report.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            stage_reports.append(
                {
                    "stage": result.stage,
                    "input_warm_start_checkpoint": parent_checkpoint,
                    "selected_step": selected_step,
                    "selected_checkpoint": str(result.selected_checkpoint),
                    "selected_checkpoint_sha256": sha256_file(result.selected_checkpoint),
                    "selection_report": str(result.selection_report),
                    "selection_reason": selection["selection_reason"],
                    "fallback_used": bool(selection["fallback_used"]),
                    "selected_scores": selected,
                    "formal_evaluation": str(evaluation_path),
                    "formal_summary": evaluation["summary"],
                }
            )
            parent_checkpoint = str(result.selected_checkpoint)
        final_report = {
            "schema": SCHEMA,
            "arm": arm,
            "alpha": ALPHAS[arm],
            "hard_zero_threshold": 0.10,
            "ancestry_rule": "same-line selected model weights only; fresh optimizer/replay/RNG/env/z/recovery at each promotion",
            "stages": stage_reports,
            "final_checkpoint": str(results[-1].selected_checkpoint),
            "final_checkpoint_sha256": sha256_file(results[-1].selected_checkpoint),
            "final_formal_summary": stage_reports[-1]["formal_summary"],
            "completed_at": now_local(),
        }
        final_json = output / f"{arm.upper()}_CURRICULUM_FINAL_REPORT.json"
        final_md = output / f"{arm.upper()}_CURRICULUM_FINAL_REPORT_ZH.md"
        matched.atomic_json(final_json, final_report)
        lines = [
            f"# {arm.upper()} Unified-Decay IQN 全课程最终报告",
            "",
            f"- alpha: `{ALPHAS[arm]}`",
            "- hard-zero threshold: `0.10`",
            f"- final checkpoint: `{results[-1].selected_checkpoint}`",
            f"- final SHA256: `{final_report['final_checkpoint_sha256']}`",
            "",
            "## 阶段选择与 ancestry",
            "",
        ]
        for row in stage_reports:
            summary = row["formal_summary"]
            lines.extend(
                [
                    f"### {row['stage']}",
                    "",
                    f"- selected step: `{row['selected_step']}`",
                    f"- selected checkpoint: `{row['selected_checkpoint']}`",
                    f"- input warm-start: `{row['input_warm_start_checkpoint']}`",
                    f"- selection: {row['selection_reason']}",
                    f"- fallback used: `{row['fallback_used']}`",
                    f"- pure capture: `{summary['capture']['normal_capture_rate']:.4f}`",
                    f"- mixed capture: `{summary['mixed']['capture_rate']:.4f}`",
                    f"- pure coverage strict CE: `{summary['coverage']['strict_ce_rate']:.4f}`",
                    f"- mixed post-capture CE: `{summary['mixed']['post_capture_ce_rate']:.4f}`",
                    f"- mixed safe-complete: `{summary['mixed']['safe_complete_rate']:.4f}`",
                    f"- worst collision: `{max(summary[s]['collision_rate'] for s in ('coverage', 'capture', 'mixed')):.4f}`",
                    f"- Z max lineage hop: `{summary['z']['max_lineage_hop']}`",
                    f"- Z post-capture never-release: `{summary['z']['post_capture_never_release_episodes']}`",
                    "",
                ]
            )
        final_md.write_text("\n".join(lines), encoding="utf-8")
        matched.atomic_json(output / "status.json", {**launch, "status": "complete", "phase": "complete", "final_report": str(final_json), "final_report_zh": str(final_md), "completed_at": now_local()})
        return 0
    except BaseException as exc:
        matched.atomic_json(output / "status.json", {**launch, "status": "failed_closed", "error": repr(exc), "traceback": traceback.format_exc(), "updated_at": now_local()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "select", "evaluate", "supervise"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--stage", choices=STAGE_ORDER, default="stage1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(args.output.resolve(), args.device), indent=2, ensure_ascii=False))
        return 0
    if args.arm is None:
        parser.error("--arm is required")
    if args.command == "supervise":
        return supervise_arm(args.arm, args.output.resolve(), args.device)
    if args.command == "evaluate":
        if args.checkpoint is None:
            parser.error("--checkpoint is required")
        cfg = resolved(args.arm, args.stage)
        report = matched.evaluate_checkpoint(args.checkpoint.resolve(), args.output.resolve(), args.device, args.episodes, int(cfg["formal_evaluation"]["seed_base"]), args.arm, config_path=CONFIGS[args.arm][args.stage])
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
        return 0
    report = select_balanced(args.output.resolve(), MILESTONES[args.stage])
    print(json.dumps(report["selected"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
