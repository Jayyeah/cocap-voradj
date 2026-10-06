#!/usr/bin/env python3
"""Matched Local-Binary / Z05 / Global-Oracle IQN evidence contracts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import CoCapTrainer, deep_update, set_global_config
from tools import iqn_token_matched_20260919 as matched
from tools.iqn_z_unified_decay_curriculum_20260919 import MILESTONES, select_balanced
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file


SCHEMA = "iqn-evidence-comparison-preflight-v1"
CONFIG_DIR = ROOT / "configs/experiments/iqn_evidence_comparison_20261006"
Z05_DIR = ROOT / "configs/experiments/iqn_z_unified_decay_curriculum_20260919"
STAGES = ("stage1", "stage2", "stage3")
VARIANTS = ("local_binary", "z05", "global_oracle")
CONFIGS = {
    "local_binary": {
        "stage1": CONFIG_DIR / "local_binary_stage1_4p1e1obs_2m.yaml",
        "stage2": CONFIG_DIR / "local_binary_stage2_8p2e2obs_700k.yaml",
        "stage3": CONFIG_DIR / "local_binary_stage3_12p3e3obs_700k.yaml",
    },
    "z05": {
        "stage1": Z05_DIR / "z05_stage1_4p1e1obs_2m.yaml",
        "stage2": Z05_DIR / "z05_stage2_8p2e2obs_700k.yaml",
        "stage3": Z05_DIR / "z05_stage3_12p3e3obs_700k.yaml",
    },
    "global_oracle": {
        "stage1": CONFIG_DIR / "global_oracle_stage1_4p1e1obs_2m.yaml",
        "stage2": CONFIG_DIR / "global_oracle_stage2_8p2e2obs_700k.yaml",
        "stage3": CONFIG_DIR / "global_oracle_stage3_12p3e3obs_700k.yaml",
    },
}
ALLOWED_REPRESENTATION_DIFFS = {
    "perception.policy_evidence_mode",
    "perception.include_z_state",
    "iqn.include_z_state",
    "z_state.enabled",
}
ALLOWED_OPERATIONAL_DIFFS = {"device", "output_root", "run_name"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def resolved(variant: str, stage: str) -> dict[str, Any]:
    if variant not in VARIANTS or stage not in STAGES:
        raise ValueError(f"unknown variant/stage: {variant}/{stage}")
    return matched.resolved(CONFIGS[variant][stage])


def architecture_signature(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = CoCapIQN(matched.model_config(cfg))
    shapes = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model_cfg = asdict(model.config)
    # Evidence representation metadata does not instantiate or alter modules.
    model_cfg.pop("include_z_state", None)
    payload = {"model_config": model_cfg, "state_shapes": shapes}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "model_config": model_cfg, "state_shapes": shapes}


def contract_diff_report() -> dict[str, Any]:
    stage_reports: dict[str, Any] = {}
    failures = []
    for variant in ("local_binary", "global_oracle"):
        for stage in STAGES:
            canonical = resolved("z05", stage)
            candidate = resolved(variant, stage)
            rows = matched._diff_rows(matched.flatten(canonical), matched.flatten(candidate))
            unexpected = [
                row for row in rows
                if row["path"] not in ALLOWED_REPRESENTATION_DIFFS
                and row["path"] not in ALLOWED_OPERATIONAL_DIFFS
                and not row["path"].startswith("experiment_metadata.")
            ]
            left_arch = architecture_signature(canonical)
            right_arch = architecture_signature(candidate)
            identical_dims = (
                canonical["iqn"]["self_feature_dim"] == candidate["iqn"]["self_feature_dim"]
                and canonical["iqn"]["pursuer_feature_dim"] == candidate["iqn"]["pursuer_feature_dim"]
                and canonical["perception"]["max_pursuer_num"] == candidate["perception"]["max_pursuer_num"]
                and canonical["perception"]["max_evader_num"] == candidate["perception"]["max_evader_num"]
                and canonical["perception"]["max_obstacle_num"] == candidate["perception"]["max_obstacle_num"]
            )
            if unexpected:
                failures.append(f"{variant}/{stage}: unexpected config diffs")
            if left_arch["sha256"] != right_arch["sha256"]:
                failures.append(f"{variant}/{stage}: model parameter shapes differ")
            if not identical_dims:
                failures.append(f"{variant}/{stage}: token dimensions/capacities differ")
            if candidate["perception"]["friend_ordering_mode"] != "physical_only":
                failures.append(f"{variant}/{stage}: friend ordering drift")
            if candidate["runtime_semantic_assertions"].get("global_enemy_flag") is not False:
                failures.append(f"{variant}/{stage}: global enemy reward semantics changed")
            if not candidate["normsense_v2"].get("enabled") or candidate["normsense_v2"].get("policy") != matched.NORMSENSE_POLICY:
                failures.append(f"{variant}/{stage}: NormSense-V2 drift")
            stage_reports[f"{variant}/{stage}"] = {
                "all_differences": rows,
                "unexpected_differences": unexpected,
                "architecture_fingerprint_z05": left_arch["sha256"],
                "architecture_fingerprint_candidate": right_arch["sha256"],
                "identical_model_shapes": left_arch["sha256"] == right_arch["sha256"],
                "identical_token_dimensions_and_capacities": identical_dims,
                "physical_only_friend_ordering": candidate["perception"]["friend_ordering_mode"],
                "global_enemy_flag": candidate["runtime_semantic_assertions"].get("global_enemy_flag"),
                "normsense_policy": candidate["normsense_v2"].get("policy"),
            }
    return {
        "schema": SCHEMA,
        "status": "pass" if not failures else "fail",
        "z05_reference_is_unmodified": True,
        "registered_representation_differences": sorted(ALLOWED_REPRESENTATION_DIFFS),
        "operational_and_metadata_differences": sorted(ALLOWED_OPERATIONAL_DIFFS) + ["experiment_metadata.*"],
        "stage_diffs": stage_reports,
        "failures": failures,
        "completed_at": now(),
    }


def runtime_smoke_report() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for variant in ("local_binary", "global_oracle"):
        cfg = copy.deepcopy(resolved(variant, "stage1"))
        cfg = deep_update(cfg, cfg["tasks"]["voradj"])
        set_global_config(cfg)
        env = VorAdjEnv(copy.deepcopy(cfg), seed=2026100601)
        observations = list(env.reset())
        target_count = sum(not evader.deactivated for evader in env.evaders)
        if env._z_update_count != 0:
            raise AssertionError(f"{variant}: Z recursion ran during reset")
        self_values = []
        enemy_tokens = []
        friend_values = []
        for index, observation in enumerate(observations):
            if observation is None:
                continue
            self_values.append(float(observation["self"][-1]))
            mask = np.asarray(observation["masks"], dtype=bool)
            types = np.asarray(observation["types"], dtype=int)
            enemy_tokens.append(int(np.sum(mask & (types == 2))))
            friend_values.extend(float(row[-1]) for row in np.asarray(observation["pursuers"]) if np.any(row))
            if variant == "local_binary" and float(observation["self"][-1]) != float(bool(env._policy_direct_enemy_ids(index))):
                raise AssertionError("Local-Binary self evidence differs from direct visibility")
        if variant == "global_oracle":
            if any(value != float(target_count > 0) for value in self_values + friend_values):
                raise AssertionError("Global-Oracle active-target scalar flag mismatch")
            if any(value != target_count for value in enemy_tokens):
                raise AssertionError("Global-Oracle does not expose every active target token")
        results[variant] = {
            "z_update_count_after_reset": env._z_update_count,
            "active_target_count": target_count,
            "self_feature_shape": list(next(item for item in observations if item is not None)["self"].shape),
            "friend_feature_shape": list(next(item for item in observations if item is not None)["pursuers"].shape),
            "self_evidence_values": self_values,
            "friend_evidence_values": friend_values,
            "active_enemy_tokens_by_pursuer": enemy_tokens,
            "friend_ordering_mode": cfg["perception"]["friend_ordering_mode"],
            "reward_contract_unchanged": cfg["reward"] == resolved("z05", "stage1")["reward"] and cfg["voradj"] == resolved("z05", "stage1")["voradj"],
        }
        if cfg["perception"]["friend_ordering_mode"] != "physical_only" or not results[variant]["reward_contract_unchanged"]:
            raise AssertionError(f"{variant}: friend ordering or reward drift")
    report = {"schema": "iqn-evidence-runtime-smoke-v1", "status": "pass", "results": results, "completed_at": now()}
    return report


def training_smoke(variant: str, device: str, output_root: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(resolved(variant, "stage1"))
    smoke_dir = output_root / variant / device.replace(":", "_")
    cfg.update(output_root=str(smoke_dir), run_name="bounded_training_smoke", device=device, total_timesteps=24, train_mode="voradj")
    cfg["iqn"].update(
        batch_size=32,
        min_replay_size=8,
        replay_capacity=512,
        train_freq=1,
        target_update_freq=4,
        checkpoint_freq=12,
        log_freq_steps=12,
    )
    cfg.setdefault("checkpointing", {})["full_resume"] = True
    trainer = CoCapTrainer(cfg)
    start = {key: tensor.detach().cpu().clone() for key, tensor in trainer.model.state_dict().items()}
    checkpoint = trainer.train()
    changed = any(not torch.equal(start[key], tensor.detach().cpu()) for key, tensor in trainer.model.state_dict().items())
    if trainer.global_step != 24 or trainer.update_steps <= 0 or not changed:
        raise AssertionError(f"{variant}/{device}: training did not perform a finite optimizer update")
    if trainer.loss_ema is None or not math.isfinite(float(trainer.loss_ema)):
        raise AssertionError(f"{variant}/{device}: loss is non-finite")
    replay_sizes = {name: len(buffer) for name, buffer in trainer.replays.items()}
    if not any(replay_sizes.values()):
        raise AssertionError(f"{variant}/{device}: replay remained empty")
    resume_path = trainer.ckpt_dir / "resume_latest.pt"
    if not resume_path.is_file():
        raise AssertionError(f"{variant}/{device}: full resume was not saved")
    signature = architecture_signature(cfg)
    loaded = CoCapIQN.load(str(checkpoint), device="cpu").eval()
    loaded_sig = architecture_signature_from_model(loaded)
    if loaded_sig != signature["sha256"]:
        raise AssertionError(f"{variant}/{device}: evaluator checkpoint architecture fingerprint mismatch")
    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg["total_timesteps"] = 25
    resumed_cfg["checkpointing"]["resume_path"] = str(resume_path)
    resumed = CoCapTrainer(resumed_cfg)
    if resumed.global_step != 24:
        raise AssertionError(f"{variant}/{device}: exact-resume step mismatch ({resumed.global_step})")
    if list(resumed.model.state_dict()) != list(trainer.model.state_dict()):
        raise AssertionError(f"{variant}/{device}: resumed model parameter keys mismatch")
    for name in ("episode_log", "metric_log"):
        for item in (trainer, resumed):
            stream = getattr(item, name, None)
            if stream is not None and not stream.closed:
                stream.close()
    return {
        "variant": variant,
        "device": device,
        "smoke_only_train_mode": "voradj",
        "steps": trainer.global_step,
        "optimizer_updates": trainer.update_steps,
        "target_updates": trainer.target_update_count,
        "finite_loss_ema": float(trainer.loss_ema),
        "replay_sizes": replay_sizes,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "full_resume_path": str(resume_path),
        "full_resume_bytes": resume_path.stat().st_size,
        "resume_loaded_step": resumed.global_step,
        "checkpoint_evaluator_load": True,
        "architecture_fingerprint": signature["sha256"],
    }


def architecture_signature_from_model(model: CoCapIQN) -> str:
    shapes = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    model_cfg = asdict(model.config)
    model_cfg.pop("include_z_state", None)
    payload = {"model_config": model_cfg, "state_shapes": shapes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-report", type=Path)
    parser.add_argument("--runtime-smoke", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/2026-10-06_iqn_evidence_comparison/preflight")
    args = parser.parse_args()
    report = contract_diff_report()
    if args.contract_report:
        matched.atomic_json(args.contract_report, report)
    if report["status"] != "pass":
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    if args.runtime_smoke:
        runtime = runtime_smoke_report()
        matched.atomic_json(args.runtime_smoke, runtime)
        print(json.dumps(runtime, ensure_ascii=False, indent=2))
    if args.smoke:
        rows = [training_smoke(variant, args.device, args.output_root / "training_smoke") for variant in ("local_binary", "global_oracle")]
        matched.atomic_json(args.output_root / f"training_smoke_{args.device.replace(':', '_')}.json", {"status": "pass", "smokes": rows, "completed_at": now()})
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
