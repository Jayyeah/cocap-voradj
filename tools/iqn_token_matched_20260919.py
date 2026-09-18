#!/usr/bin/env python3
"""Matched IQN ROLE-token/Z-token preflight, evaluator, and supervisor."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import yaml

from cocap_voradj.envs.density_sensing import POLICY as NORMSENSE_POLICY
from cocap_voradj.envs.density_sensing import enable_v2, runtime_metadata
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.training.trainer import CoCapTrainer, deep_update, load_config, set_global_config
from cocap_voradj.training.runtime_semantics import assert_runtime, runtime_facts
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.evaluate_vxy_stage2_cross_retention_20260903 import ring_count
from tools.run_forward_final_bridge_20260908 import atomic_json, run_episode as standard_episode

SCHEMA = "iqn-role-z-matched-v1"
CONFIG_DIR = ROOT / "configs/experiments/iqn_token_scratch_20260918"
ROLE_CONFIG = CONFIG_DIR / "role_token.yaml"
Z_CONFIG = CONFIG_DIR / "z_token.yaml"
ROLE_WORKTREE = Path("/home/yjq/rl/CoCap1/iqn-role-token-scratch-20260918")
Z_WORKTREE = Path("/home/yjq/rl/CoCap1/iqn-z-token-scratch-20260918")
DEFAULT_OUTPUTS = {
    "role": ROOT / "artifacts/2026-09-18_iqn_role_token_scratch",
    "z": ROOT / "artifacts/2026-09-18_iqn_z_token_scratch",
}
MILESTONES = tuple(range(25_000, 200_001, 25_000))
ALLOWED_TOKEN_DIFFS = {
    "perception.include_is_pursuing",
    "perception.include_z_state",
    "iqn.include_is_pursuing",
    "iqn.include_z_state",
    "z_state.enabled",
}
SOURCE_IDS = {"zero": 0, "direct": 1, "self_decay": 2, "neighbor": 3}


def now_local() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def arm_config(arm: str) -> Path:
    if arm == "role":
        return ROLE_CONFIG
    if arm == "z":
        return Z_CONFIG
    raise ValueError(f"unknown arm: {arm}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def raw_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"raw config root must be a mapping: {path}")
    return payload


def resolved(path: Path, require_v2: bool = True) -> dict[str, Any]:
    cfg = load_config(str(path))
    spec = cfg.get("normsense_v2", {}) or {}
    enabled = bool(spec.get("enabled"))
    if require_v2 and (not enabled or spec.get("policy") != NORMSENSE_POLICY):
        raise ValueError("NormSense-V2 must be explicitly enabled")
    if enabled:
        cfg = enable_v2(cfg)
    return cfg


def model_config(cfg: Mapping[str, Any]) -> CoCapNetConfig:
    iqn = cfg["iqn"]
    perception = cfg["perception"]
    return CoCapNetConfig(
        hidden_dim=int(iqn["hidden_dim"]),
        num_heads=int(iqn["num_heads"]),
        num_layers=int(iqn["num_layers"]),
        action_size=int(iqn["action_size"]),
        self_feature_dim=int(iqn["self_feature_dim"]),
        max_pursuers=int(perception["max_pursuer_num"]),
        max_evaders=int(perception["max_evader_num"]),
        max_obstacles=int(perception["max_obstacle_num"]),
        num_quantiles=int(iqn["num_quantiles"]),
        num_cosine_features=int(iqn["num_cosine_features"]),
        architecture=str(iqn["architecture"]),
        pursuing_embed_dim=int(iqn["pursuing_embed_dim"]),
        pursuer_feature_dim=int(iqn["pursuer_feature_dim"]),
        include_is_pursuing=bool(iqn["include_is_pursuing"]),
        include_z_state=bool(iqn["include_z_state"]),
        pursuing_late_fusion=bool(iqn["pursuing_late_fusion"]),
    )


def state_hash(model: CoCapIQN) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_friend_z_mapping(
    env: VorAdjEnv,
    observations: list[dict[str, np.ndarray] | None],
) -> int:
    """Prove every populated friend-token tail equals that physical friend's z_j."""

    world_frame = str(env.per_cfg.get("observation_frame", "robot")).strip().lower() in {
        "world",
        "world_frame",
    }
    distance_scale = env._distance_scale()
    verified = 0
    for observer_index, observation in enumerate(observations):
        if observation is None:
            continue
        observer = env.pursuers[observer_index]
        for row in np.asarray(observation["pursuers"], dtype=float):
            if np.allclose(row, 0.0, atol=1e-8, rtol=0.0):
                continue
            matches: list[int] = []
            for friend_index, friend in enumerate(env.pursuers):
                if friend_index == observer_index or friend.deactivated:
                    continue
                if world_frame:
                    pos_r = env._position(friend) - env._position(observer)
                    vel_r = np.asarray(friend.velocity, dtype=float)
                else:
                    pos_r = env._robot_frame(observer, env._position(friend), False)
                    vel_r = env._robot_frame(observer, friend.velocity, True)
                distance = float(np.linalg.norm(pos_r))
                expected = np.asarray(
                    [
                        pos_r[0] / distance_scale,
                        pos_r[1] / distance_scale,
                        vel_r[0],
                        vel_r[1],
                        distance / distance_scale,
                        np.arctan2(pos_r[1], pos_r[0]),
                    ],
                    dtype=float,
                )
                if np.allclose(row[:6], expected, atol=1e-6, rtol=0.0):
                    matches.append(friend_index)
            if len(matches) != 1:
                raise AssertionError(
                    f"friend physical row does not identify exactly one agent: "
                    f"observer={observer_index} matches={matches}"
                )
            friend_index = matches[0]
            if not np.isclose(
                row[-1],
                float(env.z_state[friend_index]),
                atol=1e-6,
                rtol=0.0,
            ):
                raise AssertionError(
                    f"friend token tail is not z_j: observer={observer_index} "
                    f"friend={friend_index} token={row[-1]} z={env.z_state[friend_index]}"
                )
            verified += 1
    if verified <= 0:
        raise AssertionError("friend z_j mapping sanity observed no populated friend tokens")
    return verified


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value):
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value[key], name))
        return out
    return {prefix: value}


def _diff_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    left_flat, right_flat = flatten(left), flatten(right)
    return [
        {"path": key, "role": left_flat.get(key), "z": right_flat.get(key)}
        for key in sorted(set(left_flat) | set(right_flat))
        if left_flat.get(key) != right_flat.get(key)
    ]


def _projection(cfg: Mapping[str, Any], paths: tuple[str, ...]) -> dict[str, Any]:
    flat = flatten(cfg)
    return {key: flat.get(key) for key in paths}


def config_diff(role_path: Path, output: Path) -> dict[str, Any]:
    role = resolved(role_path)
    z = resolved(Z_CONFIG)
    ignored_exact = {"output_root", "run_name", "checkpointing.full_resume_path"}
    ignored_prefixes = ("experiment_metadata.", "formal_evaluation.")
    differences = _diff_rows(role, z)
    training_differences = []
    ignored_differences = []
    for row in differences:
        key = row["path"]
        if key in ignored_exact or key.startswith(ignored_prefixes):
            ignored_differences.append(row)
        else:
            training_differences.append(row)
    unexpected = [row for row in training_differences if row["path"] not in ALLOWED_TOKEN_DIFFS]
    raw_differences = _diff_rows(raw_config(role_path), raw_config(Z_CONFIG))
    raw_allowed = ALLOWED_TOKEN_DIFFS | {
        "experiment_metadata.comparison_arm",
        "experiment_metadata.evidence_token",
    }
    raw_unexpected = [row for row in raw_differences if row["path"] not in raw_allowed]
    role_model = dataclasses.asdict(model_config(role))
    z_model = dataclasses.asdict(model_config(z))
    model_differences = _diff_rows(role_model, z_model)
    model_unexpected = [
        row for row in model_differences
        if row["path"] not in {"include_is_pursuing", "include_z_state"}
    ]
    replay_paths = (
        "iqn.batch_size", "iqn.replay_capacity", "iqn.min_replay_size",
        "iqn.train_freq", "iqn.target_update_freq", "iqn.gamma",
        "voradj.replay_batch_counts.pursuing",
        "voradj.replay_batch_counts.pre_capture_cover",
        "voradj.replay_batch_counts.post_capture_real",
        "voradj.replay_batch_counts.recovery_pure",
    )
    trainer_paths = (
        "seed", "total_timesteps", "train_mode", "iqn.learning_rate",
        "iqn.epsilon_start", "iqn.epsilon_final", "iqn.epsilon_decay_steps",
        "iqn.num_quantiles", "iqn.num_cosine_features", "iqn.checkpoint_freq",
        "env.width", "env.height", "env.num_pursuers", "env.num_evaders",
        "env.collision_semantics", "action.mode", "action.decision_dt",
    )
    evaluator_paths = tuple(f"formal_evaluation.{key}" for key in (
        "episodes_per_scene", "seed_base", "scenes", "deterministic_quantiles",
        "checkpoint_interval",
    ))
    role_contract = copy.deepcopy(role)
    z_contract = copy.deepcopy(z)
    for cfg in (role_contract, z_contract):
        for section, key in (
            ("perception", "include_is_pursuing"), ("perception", "include_z_state"),
            ("iqn", "include_is_pursuing"), ("iqn", "include_z_state"),
            ("z_state", "enabled"),
        ):
            cfg.get(section, {}).pop(key, None)
        cfg.pop("experiment_metadata", None)
        cfg.pop("output_root", None)
        cfg.pop("run_name", None)
    report = {
        "schema": SCHEMA,
        "status": "pass" if not (unexpected or raw_unexpected or model_unexpected) else "fail",
        "role_config": str(role_path),
        "z_config": str(Z_CONFIG),
        "raw_config": {
            "differences": raw_differences,
            "unexpected_differences": raw_unexpected,
        },
        "resolved_config": {
            "role_sha256": stable_hash(role),
            "z_sha256": stable_hash(z),
            "matched_contract_sha256": stable_hash(role_contract),
            "z_matched_contract_sha256": stable_hash(z_contract),
        },
        "differences": training_differences,
        "ignored_non_training_differences": ignored_differences,
        "allowed_paths": sorted(ALLOWED_TOKEN_DIFFS),
        "unexpected_differences": unexpected,
        "non_token_diff_count": len(unexpected) + len(raw_unexpected) + len(model_unexpected),
        "model_config": {
            "role": role_model,
            "z": z_model,
            "differences": model_differences,
            "unexpected_differences": model_unexpected,
        },
        "trainer_config": {
            "role": _projection(role, trainer_paths),
            "z": _projection(z, trainer_paths),
        },
        "replay_config": {
            "role": _projection(role, replay_paths),
            "z": _projection(z, replay_paths),
        },
        "evaluator_config": {
            "role": _projection(role, evaluator_paths),
            "z": _projection(z, evaluator_paths),
        },
        "generated_at": now_local(),
    }
    if report["resolved_config"]["matched_contract_sha256"] != report["resolved_config"]["z_matched_contract_sha256"]:
        report["status"] = "fail"
        report["non_token_diff_count"] += 1
    atomic_json(output, report)
    if report["status"] != "pass":
        raise RuntimeError(f"ROLE-vs-Z config drift: {report}")
    return report


def scene_config(root_cfg: Mapping[str, Any], scene: str) -> dict[str, Any]:
    task = "voradj_coverage" if scene == "coverage" else "voradj"
    cfg = deep_update(copy.deepcopy(dict(root_cfg)), root_cfg["tasks"][task])
    if scene == "capture":
        cfg.setdefault("voradj", {})["capture_episode_ends_on_capture"] = True
        cfg["voradj"]["capture_episode_success_on_capture"] = True
    return cfg


def make_env(root_cfg: Mapping[str, Any], scene: str, seed: int) -> tuple[VorAdjEnv, list[Any]]:
    cfg = scene_config(root_cfg, scene)
    set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=int(seed))
    observations = list(env.reset())
    meta = runtime_metadata(env)
    if meta["policy"] != NORMSENSE_POLICY:
        raise AssertionError("NormSense-V2 runtime policy drift")
    assert_runtime(env)
    return env, observations


def numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": float(np.mean(finite)) if finite else None,
        "p50": float(np.percentile(finite, 50)) if finite else None,
        "p90": float(np.percentile(finite, 90)) if finite else None,
        "max": float(np.max(finite)) if finite else None,
    }


class ZDiagnostics:
    def __init__(self, scene: str):
        self.scene = scene
        self.direct_checks = 0
        self.direct_violations = 0
        self.support: list[float] = []
        self.coverage: list[float] = []
        self.pure_coverage: list[float] = []
        self.neighbor: list[float] = []
        self.hops: list[int] = []
        self.saturation_09 = 0
        self.saturation_05 = 0
        self.steps = 0
        self.capture_seen = False
        self.post_capture_max_z: list[float] = []
        self.release_step: int | None = None

    def transition(self, env, local, global_state, q, greedy, active, state, phase, step, outcome):
        del local, global_state, q, greedy, state
        values = []
        for index in active:
            meta = outcome.infos[index]["replay_metadata"]
            z = float(meta["z"])
            values.append(z)
            if bool(meta["z_direct_visible"]):
                self.direct_checks += 1
                self.direct_violations += int(not np.isclose(z, 1.0))
            role = str(meta.get("reward_role", ""))
            if bool(meta.get("support_candidate", False)):
                self.support.append(z)
            elif role == "coverage":
                self.coverage.append(z)
            if self.scene == "coverage":
                self.pure_coverage.append(z)
            if str(meta["z_dominant_source"]) == "neighbor":
                self.neighbor.append(z)
            self.hops.append(int(meta["z_lineage_hops"]))
        if values:
            self.steps += 1
            self.saturation_09 += int(all(value >= 0.9 for value in values))
            self.saturation_05 += int(all(value >= 0.5 for value in values))
        captured_now = bool(env.evaders and all(ev.deactivated and not ev.collision for ev in env.evaders))
        self.capture_seen = self.capture_seen or captured_now
        if self.capture_seen and self.scene == "mixed":
            current = float(np.max(env.z_state)) if len(env.z_state) else 0.0
            self.post_capture_max_z.append(current)
            if self.release_step is None and current < 0.1:
                self.release_step = len(self.post_capture_max_z) - 1

    def finish(self, decision_seconds: float) -> dict[str, Any]:
        return {
            "direct_checks": self.direct_checks,
            "direct_violations": self.direct_violations,
            "direct_visible_z_one_rate": 1.0 - self.direct_violations / max(self.direct_checks, 1),
            "support_z": numeric_summary(self.support),
            "coverage_z": numeric_summary(self.coverage),
            "pure_coverage_z": numeric_summary(self.pure_coverage),
            "pure_coverage_exact_zero": bool(not self.pure_coverage or max(abs(v) for v in self.pure_coverage) == 0.0),
            "neighbor_dominant_count": len(self.neighbor),
            "neighbor_dominant_z": numeric_summary(self.neighbor),
            "max_lineage_hop": max(self.hops, default=-1),
            "whole_swarm_z_ge_0_9_fraction": self.saturation_09 / max(self.steps, 1),
            "whole_swarm_z_ge_0_5_fraction": self.saturation_05 / max(self.steps, 1),
            "post_capture_max_z": self.post_capture_max_z,
            "release_steps_max_z_lt_0_1": self.release_step,
            "release_seconds_max_z_lt_0_1": None if self.release_step is None else self.release_step * decision_seconds,
            "post_capture_never_released": bool(self.capture_seen and self.scene == "mixed" and self.release_step is None),
        }


class RoleDiagnostics:
    def transition(self, env, local, global_state, q, greedy, active, state, phase, step, outcome):
        del env, local, global_state, q, greedy, active, state, phase, step, outcome

    def finish(self, decision_seconds: float) -> dict[str, Any]:
        del decision_seconds
        return {}


def evaluate_checkpoint(
    checkpoint: Path,
    output: Path,
    device: str,
    episodes: int,
    seed_base: int,
    arm: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cfg = resolved(arm_config(arm))
    model = CoCapIQN.load(str(checkpoint), device=device).eval()
    expected_role = arm == "role"
    if (
        model.config.include_is_pursuing is not expected_role
        or model.config.include_z_state is expected_role
        or model.config.pursuing_late_fusion
        or model.pursuing_embed is not None
    ):
        raise ValueError(f"checkpoint is not the registered {arm}-token architecture")
    records = []
    total = episodes * 3
    started = time.monotonic()
    support_values: list[float] = []
    coverage_values: list[float] = []
    pure_coverage_values: list[float] = []
    neighbor_values: list[float] = []
    (output / "episodes.jsonl").write_text("", encoding="utf-8")
    for scene_index, scene in enumerate(("coverage", "capture", "mixed")):
        for index in range(episodes):
            seed = int(seed_base + scene_index * 100_000 + index)
            holder: dict[str, Any] = {}
            diag = ZDiagnostics(scene) if arm == "z" else RoleDiagnostics()

            def factory(requested_scene: str, requested_seed: int):
                env, obs = make_env(cfg, requested_scene, requested_seed)
                holder["env"] = env
                return env, obs

            row = standard_episode(
                model,
                scene,
                seed,
                device,
                max_steps=max_steps,
                on_transition=diag.transition,
                env_factory=factory,
            )
            env = holder["env"]
            record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
            row["area_cv"] = float(record["coverage_strict_area_cv"])
            if arm == "z":
                support_values.extend(diag.support)
                coverage_values.extend(diag.coverage)
                pure_coverage_values.extend(diag.pure_coverage)
                neighbor_values.extend(diag.neighbor)
            row["ce_max"] = float(record["coverage_ce_center_max"])
            row["z_diagnostics"] = diag.finish(float(env.pursuers[0].dt * env.pursuers[0].N))
            events = row["mission_events"]["summary"]
            ring2 = events.get("capture_region_2_to_3", {})
            ring3 = events.get("capture_region_2_to_geometry", {})
            row["ring2_seen"] = bool(ring2.get("opportunities", 0))
            row["ring3_reached"] = bool(ring2.get("completed", 0) or ring3.get("completed", 0))
            if scene == "capture":
                row["safe_complete"] = bool(row["captured"] and not row["collision"])
                row["mission_seconds"] = row["capture_seconds"] if row["safe_complete"] else None
            records.append(row)
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            elapsed = time.monotonic() - started
            atomic_json(
                output / "progress.json",
                {
                    "status": "running",
                    "completed": len(records),
                    "total": total,
                    "scene": scene,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / max(len(records), 1) * (total - len(records)),
                },
            )

    def rows(scene: str) -> list[dict[str, Any]]:
        return [row for row in records if row["scene"] == scene]

    def rate(items, key):
        return sum(bool(row[key]) for row in items) / max(len(items), 1)

    coverage, capture, mixed = rows("coverage"), rows("capture"), rows("mixed")
    action_hist = np.sum(
        [np.asarray(row["policy_diagnostics"]["action_histogram"], dtype=np.int64) for row in records],
        axis=0,
    ).tolist()
    all_diag = [row["z_diagnostics"] for row in records] if arm == "z" else []
    release = [row["release_seconds_max_z_lt_0_1"] for row in all_diag if row["release_seconds_max_z_lt_0_1"] is not None]
    summary = {
        "coverage": {
            "episodes": len(coverage),
            "strict_ce_rate": rate(coverage, "ce_success"),
            "ce_rms": numeric_summary([row["ce_rms"] for row in coverage]),
            "ce_max": numeric_summary([row["ce_max"] for row in coverage]),
            "area_cv": numeric_summary([row["area_cv"] for row in coverage]),
            "collision_rate": rate(coverage, "collision"),
            "time_to_ce_seconds": numeric_summary([row["mission_seconds"] for row in coverage]),
        },
        "capture": {
            "episodes": len(capture),
            "capture_rate": rate(capture, "captured"),
            "normal_capture_rate": rate(capture, "normal_capture"),
            "stationary_capture_rate": rate(capture, "stationary_capture"),
            "ring2_seen_rate": rate(capture, "ring2_seen"),
            "ring3_reached_rate": rate(capture, "ring3_reached"),
            "collision_rate": rate(capture, "collision"),
            "capture_seconds": numeric_summary([row["capture_seconds"] for row in capture]),
        },
        "mixed": {
            "episodes": len(mixed),
            "capture_rate": rate(mixed, "captured"),
            "collision_rate": rate(mixed, "collision"),
            "post_capture_ce_rate": rate(mixed, "ce_success"),
            "safe_complete_rate": rate(mixed, "safe_complete"),
            "capture_seconds": numeric_summary([row["capture_seconds"] for row in mixed]),
            "recovery_seconds": numeric_summary([row["recovery_seconds"] for row in mixed]),
            "mission_seconds": numeric_summary([row["mission_seconds"] for row in mixed]),
        },
        "actions": {
            "histogram": action_hist,
            "distribution": [count / max(sum(action_hist), 1) for count in action_hist],
        },
    }
    if arm == "z":
        summary["z"] = {
            "direct_checks": sum(row["direct_checks"] for row in all_diag),
            "direct_violations": sum(row["direct_violations"] for row in all_diag),
            "direct_visible_z_one_rate": 1.0 - sum(row["direct_violations"] for row in all_diag) / max(sum(row["direct_checks"] for row in all_diag), 1),
            "support_z": numeric_summary(support_values),
            "coverage_z": numeric_summary(coverage_values),
            "pure_coverage_z": numeric_summary(pure_coverage_values),
            "neighbor_dominant_z": numeric_summary(neighbor_values),
            "pure_coverage_exact_zero": all(row["pure_coverage_exact_zero"] for row in all_diag if row["pure_coverage_z"]["n"]),
            "neighbor_dominant_count": sum(row["neighbor_dominant_count"] for row in all_diag),
            "max_lineage_hop": max((row["max_lineage_hop"] for row in all_diag), default=-1),
            "whole_swarm_z_ge_0_9_fraction": float(np.mean([row["whole_swarm_z_ge_0_9_fraction"] for row in all_diag])),
            "whole_swarm_z_ge_0_5_fraction": float(np.mean([row["whole_swarm_z_ge_0_5_fraction"] for row in all_diag])),
            "post_capture_release_seconds": numeric_summary(release),
            "post_capture_never_release_episodes": sum(row["post_capture_never_released"] for row in all_diag),
        }
    report = {
        "schema": SCHEMA,
        "arm": arm,
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "episodes_per_scene": episodes,
        "seed_base": seed_base,
        "max_steps_smoke_only": max_steps,
        "summary": summary,
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": now_local(),
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "progress.json", {"status": "complete", "completed": total, "total": total})
    return report


def close_trainer(trainer: CoCapTrainer) -> None:
    for name in ("episode_log", "metric_log"):
        stream = getattr(trainer, name, None)
        if stream is not None and not stream.closed:
            stream.close()


def nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def contract_preflight(output: Path, device: str, arm: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    diff = config_diff(ROLE_CONFIG, output / "ROLE_vs_Z_resolved_config_diff.json")
    zcfg, rolecfg = resolved(Z_CONFIG), resolved(ROLE_CONFIG)
    selected = copy.deepcopy(resolved(arm_config(arm)))
    for cfg in (zcfg, rolecfg):
        if (cfg.get("pretrained", {}) or {}).get("path"):
            raise AssertionError("scratch line cannot load pretrained weights")
        if (cfg.get("multitask_training", {}) or {}).get("initialization"):
            raise AssertionError("scratch line cannot load specialist weights")
    if zcfg["seed"] != rolecfg["seed"]:
        raise AssertionError("ROLE/Z seeds differ")

    torch.manual_seed(int(zcfg["seed"]))
    z_model = CoCapIQN(model_config(zcfg))
    torch.manual_seed(int(rolecfg["seed"]))
    role_model = CoCapIQN(model_config(rolecfg))
    initial_z_hash, initial_role_hash = state_hash(z_model), state_hash(role_model)
    if list(z_model.state_dict()) != list(role_model.state_dict()):
        raise AssertionError("matched scratch parameter keys differ")
    for key, value in z_model.state_dict().items():
        if value.shape != role_model.state_dict()[key].shape or not torch.equal(value, role_model.state_dict()[key]):
            raise AssertionError(f"matched scratch initial tensor differs: {key}")
    if initial_z_hash != initial_role_hash:
        raise AssertionError("matched scratch initial weights differ")
    for model in (z_model, role_model):
        if model.pursuing_embed is not None or any("pursuing_embed" in key for key in model.state_dict()):
            raise AssertionError("late fusion survived in matched model")

    z_env, z_obs = make_env(zcfg, "mixed", int(zcfg["seed"]))
    role_env, role_obs = make_env(rolecfg, "mixed", int(rolecfg["seed"]))
    z_meta, role_meta = runtime_metadata(z_env), runtime_metadata(role_env)
    z_facts, role_facts = runtime_facts(z_env), runtime_facts(role_env)
    if z_meta != role_meta or z_facts != role_facts:
        raise AssertionError("ROLE/Z runtime sensing or environment facts differ")
    physical_rows = 0
    role_token_checks = 0
    for index, (left, right) in enumerate(zip(role_obs, z_obs)):
        if left is None or right is None:
            if left is not right:
                raise AssertionError("ROLE/Z active observation masks differ")
            continue
        if left["self"].shape != (9,) or left["pursuers"].shape != (8, 7):
            raise AssertionError("ROLE observation shape drift")
        if right["self"].shape != (9,) or right["pursuers"].shape != (8, 7):
            raise AssertionError("Z observation shape drift")
        np.testing.assert_array_equal(left["self"][:-1], right["self"][:-1])
        np.testing.assert_array_equal(left["pursuers"][:, :6], right["pursuers"][:, :6])
        np.testing.assert_array_equal(left["evaders"], right["evaders"])
        np.testing.assert_array_equal(left["obstacles"], right["obstacles"])
        if float(left["self"][-1]) not in (0.0, 1.0):
            raise AssertionError("ROLE self evidence is not binary")
        if not np.isclose(right["self"][-1], z_env.z_state[index]):
            raise AssertionError("Z self evidence is not z_i")
        populated = ~np.all(np.isclose(left["pursuers"], 0.0), axis=1)
        if not np.all(np.isin(left["pursuers"][populated, -1], (0.0, 1.0))):
            raise AssertionError("ROLE friend evidence is not binary")
        role_token_checks += int(populated.sum()) + 1
        physical_rows += int(populated.sum())

    roles = {index: bool(index % 2) for index in range(len(role_env.pursuers))}
    original_effective = role_env._effective_is_pursuing
    role_env._effective_is_pursuing = lambda index, raw: roles[index]
    role_a = role_env.get_policy_observations(True, False)
    roles = {index: not value for index, value in roles.items()}
    role_b = role_env.get_policy_observations(True, False)
    role_env._effective_is_pursuing = original_effective
    for left, right in zip(role_a, role_b):
        if left is not None and right is not None:
            np.testing.assert_array_equal(left["pursuers"][:, :6], right["pursuers"][:, :6])

    z_env.z_state[:] = np.linspace(0.1, 0.4, len(z_env.pursuers), dtype=np.float32)
    friend_z_mapping_checks = assert_friend_z_mapping(z_env, z_env.get_policy_observations(False, True))
    z_env._reset_z_state()
    source = {0}
    original_has_enemy = z_env._has_enemy_neighbor
    z_env._has_enemy_neighbor = lambda data, key: bool(key[0] == "pursuer" and key[1] in source)
    adjacency = {
        ("pursuer", 0): {("pursuer", 1)},
        ("pursuer", 1): {("pursuer", 0), ("pursuer", 2)},
        ("pursuer", 2): {("pursuer", 1), ("pursuer", 3)},
        ("pursuer", 3): {("pursuer", 2)},
    }
    z_env._advance_z_state({"adjacency": adjacency})
    first = z_env.z_state.copy()
    source.clear()
    z_env._advance_z_state({"adjacency": adjacency})
    second = z_env.z_state.copy()
    z_env._has_enemy_neighbor = original_has_enemy
    np.testing.assert_allclose(first, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(second, [0.95, 0.85, 0.0, 0.0])

    bad = copy.deepcopy(selected)
    bad["runtime_semantic_assertions"]["collision_semantics"] = "legacy_end_step"
    try:
        make_env(bad, "mixed", int(selected["seed"]))
    except ValueError as exc:
        fail_closed_trigger = "runtime assertion collision_semantics" in str(exc)
    else:
        fail_closed_trigger = False
    if not fail_closed_trigger:
        raise AssertionError("runtime fail-closed assertion did not trigger")

    smoke_dir = Path(tempfile.mkdtemp(prefix=f"{arm}-sanity-", dir=output))
    production_probe_cfg = copy.deepcopy(selected)
    production_probe_cfg.update(output_root=str(smoke_dir), run_name="production_contract_probe", device=device, total_timesteps=1)
    production_probe = CoCapTrainer(production_probe_cfg)
    production_initial_replay_sizes = {name: len(buffer) for name, buffer in production_probe.replays.items()}
    expected_replay_classes = set(selected["voradj"]["replay_batch_counts"])
    if set(production_initial_replay_sizes) != expected_replay_classes or any(production_initial_replay_sizes.values()):
        raise AssertionError(f"production scratch replay is not fresh: {production_initial_replay_sizes}")
    if production_probe.optimizer.state:
        raise AssertionError("production optimizer is not fresh")
    close_trainer(production_probe)

    smoke_cfg = copy.deepcopy(selected)
    smoke_cfg.update(output_root=str(smoke_dir), run_name="training", device=device, total_timesteps=16, train_mode="voradj")
    smoke_cfg["iqn"].update(batch_size=8, min_replay_size=8, replay_capacity=256, train_freq=1, target_update_freq=1, checkpoint_freq=16, log_freq_steps=1)
    smoke_cfg["checkpointing"] = {"full_resume": True}
    trainer = CoCapTrainer(smoke_cfg)
    smoke_initial_replay_sizes = {name: len(buffer) for name, buffer in trainer.replays.items()}
    if any(smoke_initial_replay_sizes.values()) or trainer.optimizer.state:
        raise AssertionError("optimizer smoke did not start fresh")
    initial_parameters = {key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()}
    checkpoint = trainer.train()
    changed = any(not torch.equal(initial_parameters[key], value.detach().cpu()) for key, value in trainer.model.state_dict().items())
    if trainer.update_steps < 10 or not changed or trainer.loss_ema is None or not math.isfinite(trainer.loss_ema):
        raise AssertionError("real IQN optimizer update sanity failed")
    if sum(count > 0 for count in trainer.action_histogram) < 2:
        raise AssertionError("action distribution is fixed to one action")
    if trainer.target_update_count <= 0:
        raise AssertionError("target network never updated")
    for key, value in trainer.model.state_dict().items():
        if not torch.equal(value.detach().cpu(), trainer.target_model.state_dict()[key].detach().cpu()):
            raise AssertionError("target network is not synchronized after target update")
    loaded = CoCapIQN.load(str(checkpoint), device="cpu")
    if state_hash(loaded) != state_hash(trainer.model):
        raise AssertionError("ordinary checkpoint load is not exact")

    source_env = trainer.envs[trainer.current_task]
    saved_z = None
    if arm == "z":
        state = source_env.z_state_dict()
        state["values"] = np.linspace(0.11, 0.44, len(source_env.pursuers)).tolist()
        source_env.load_z_state_dict(state)
        saved_z = source_env.z_state_dict()
    resume = trainer._save_full_resume()
    saved_model_hash = state_hash(trainer.model)
    saved_target_hash = state_hash(trainer.target_model)
    saved_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    saved_replays = {name: len(buffer) for name, buffer in trainer.replays.items()}
    saved_step = int(trainer.global_step)
    saved_epsilon = float(trainer.epsilon())
    resume_cfg = copy.deepcopy(smoke_cfg)
    resume_cfg["total_timesteps"] = 17
    resume_cfg["checkpointing"]["resume_path"] = str(resume)
    resumed = CoCapTrainer(resume_cfg)
    if state_hash(resumed.model) != saved_model_hash or state_hash(resumed.target_model) != saved_target_hash:
        raise AssertionError("full resume model/target mismatch")
    if not nested_equal(resumed.optimizer.state_dict(), saved_optimizer):
        raise AssertionError("full resume optimizer mismatch")
    if {name: len(buffer) for name, buffer in resumed.replays.items()} != saved_replays:
        raise AssertionError("full resume replay mismatch")
    if resumed.global_step != saved_step or not np.isclose(resumed.epsilon(), saved_epsilon):
        raise AssertionError("full resume global step/epsilon mismatch")
    if arm == "z" and resumed.envs[resumed.current_task].z_state_dict() != saved_z:
        raise AssertionError("full resume did not restore z bit-exact")

    evaluator = evaluate_checkpoint(checkpoint, smoke_dir / "formal_eval_smoke", device, episodes=1, seed_base=int(selected["formal_evaluation"]["seed_base"]), arm=arm, max_steps=4)
    metrics = [json.loads(line) for line in (smoke_dir / "training/metrics.jsonl").read_text().splitlines() if line.strip()]
    latest = metrics[-1]
    observation_contract = {
        "self_shape": [9],
        "friend_shape": [8, 7],
        "physical_rows_matched": physical_rows,
        "role_token_checks": role_token_checks,
        "friend_z_mapping_checks": friend_z_mapping_checks,
        "normsense_runtime": role_meta,
        "runtime_facts": role_facts,
    }
    result = {
        "schema": SCHEMA,
        "arm": arm,
        "status": "pass",
        "head": git("rev-parse", "HEAD"),
        "config_diff": diff,
        "contract_hashes": {
            "resolved_config": stable_hash(selected),
            "runtime_observation": stable_hash(observation_contract),
            "matched_non_token_contract": diff["resolved_config"]["matched_contract_sha256"],
        },
        "initialization": {
            "scratch": True,
            "seed": int(selected["seed"]),
            "z_initial_state_hash": initial_z_hash,
            "role_initial_state_hash": initial_role_hash,
            "bit_exact_equal": True,
            "parameter_keys_equal": True,
            "parameter_shapes_equal": True,
            "teacher_transfer": False,
            "fresh_optimizer": True,
            "fresh_replay": True,
            "production_initial_replay_sizes": production_initial_replay_sizes,
            "optimizer_smoke_initial_replay_sizes": smoke_initial_replay_sizes,
        },
        "observation_contract": observation_contract,
        "architecture": {
            "is_pursuing_policy_input": arm == "role",
            "z_policy_input": arm == "z",
            "late_fusion": False,
            "special_evidence_branch": False,
            "friend_ordering": "physical_only",
        },
        "z": {
            "lambda": 0.95,
            "eta": 0.85,
            "synchronous_previous_step": True,
            "direct_current_decision": True,
            "reset_zero": True,
            "checkpoint_resume_bit_exact": arm != "z" or saved_z is not None,
            "hard_floor": False,
        },
        "optimizer": {
            "real_environment_rollout": True,
            "update_steps": int(trainer.update_steps),
            "parameters_changed": changed,
            "loss": float(trainer.loss_ema),
            "loss_finite": True,
            "target_update_count": int(trainer.target_update_count),
            "target_last_step": trainer.last_target_update_step,
            "action_histogram": list(trainer.action_histogram),
            "action_distribution_nonfixed": True,
            "latest_metrics": latest,
        },
        "checkpoint_resume": {
            "checkpoint_save": True,
            "checkpoint_exact_load": True,
            "full_resume": str(resume),
            "optimizer_exact": True,
            "replay_exact": True,
            "global_step_exact": True,
            "epsilon_exact": True,
            "z_state_exact": arm != "z" or saved_z is not None,
        },
        "formal_evaluator_smoke": {"status": evaluator["status"], "episodes_per_scene": 1, "steps_per_episode": 4},
        "runtime_fail_closed_trigger": fail_closed_trigger,
        "gpu": {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
            "memory_allocated": torch.cuda.memory_allocated(torch.device(device)) if device.startswith("cuda") else 0,
            "memory_reserved": torch.cuda.memory_reserved(torch.device(device)) if device.startswith("cuda") else 0,
        },
        "completed_at": now_local(),
    }
    atomic_json(output / "startup_sanity.json", result)
    close_trainer(trainer)
    close_trainer(resumed)
    del trainer, resumed, z_model, role_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(smoke_dir)
    return result


def read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def eta_payload(current_step: int, steps_per_second: float | None) -> dict[str, Any]:
    result = {}
    now = time.time()
    for target in (25_000, 50_000, 100_000, 200_000):
        remaining = max(target - current_step, 0)
        seconds = None if not steps_per_second or steps_per_second <= 0 else remaining / steps_per_second
        result[str(target)] = {
            "remaining_seconds": seconds,
            "estimated_local_time": None if seconds is None else time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now + seconds)),
        }
    return result


def freeze_resume(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def update_trend(output: Path, step: int, report: Mapping[str, Any], training: Mapping[str, Any]) -> None:
    path = output / "trend_ledger.json"
    payload = json.loads(path.read_text()) if path.is_file() else {"schema": SCHEMA, "milestones": []}
    payload["milestones"] = [row for row in payload["milestones"] if int(row["step"]) != step]
    payload["milestones"].append(
        {
            "step": step,
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "formal_summary": report["summary"],
            "training_metrics": dict(training),
            "completed_at": now_local(),
        }
    )
    payload["milestones"].sort(key=lambda row: int(row["step"]))
    atomic_json(path, payload)


def supervisor(output: Path, device: str, arm: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    try:
        contract_preflight(output / "preflight", device, arm)
    except BaseException as exc:
        blocked_status = {
            "schema": SCHEMA,
            "arm": arm,
            "status": "blocked_prelaunch",
            "phase": "prelaunch_contract_gate",
            "role_config": str(ROLE_CONFIG),
            "z_config": str(Z_CONFIG),
            "error": repr(exc),
            "updated_at": now_local(),
        }
        atomic_json(status_path, blocked_status)
        raise
    sanity_path = output / "preflight/startup_sanity.json"
    sanity = json.loads(sanity_path.read_text())
    if sanity.get("status") != "pass":
        raise RuntimeError("startup sanity is not PASS")
    cfg = resolved(arm_config(arm))
    cfg.update(output_root=str(output), run_name="training", device=device)
    training_dir = output / "training"
    resume = training_dir / "checkpoints/resume_latest.pt"
    state: dict[str, Any] = {
        "phase": "starting",
        "segment_start_time": time.time(),
        "segment_start_step": 0,
        "current_target": MILESTONES[0],
    }
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(15):
            latest = read_last_jsonl(training_dir / "metrics.jsonl")
            step = int(latest.get("global_step", state.get("segment_start_step", 0)))
            elapsed = max(time.time() - float(state["segment_start_time"]), 1e-6)
            speed = (step - int(state["segment_start_step"])) / elapsed
            atomic_json(
                output / "heartbeat.json",
                {
                    "schema": SCHEMA,
                    "status": "running",
                    "pid": os.getpid(),
                    "phase": state["phase"],
                    "current_step": step,
                    "current_target": state["current_target"],
                    "steps_per_second_segment": speed if speed > 0 else None,
                    "seconds_per_1k": 1000.0 / speed if speed > 0 else None,
                    "eta": eta_payload(step, speed),
                    "latest_metrics": latest,
                    "updated_at": now_local(),
                },
            )

    thread = threading.Thread(target=heartbeat, name=f"iqn-{arm}-heartbeat", daemon=True)
    thread.start()
    contract_hashes = sanity["contract_hashes"]
    launch = {
        "schema": SCHEMA,
        "arm": arm,
        "status": "running",
        "pid": os.getpid(),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "worktree": str(ROOT),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config": str(arm_config(arm)),
        "role_config": str(ROLE_CONFIG),
        "z_config": str(Z_CONFIG),
        "resolved_config_hash": contract_hashes["resolved_config"],
        "runtime_contract_hash": contract_hashes["runtime_observation"],
        "matched_non_token_contract_hash": contract_hashes["matched_non_token_contract"],
        "seed": int(cfg["seed"]),
        "milestones": list(MILESTONES),
        "stop_target": 200_000,
        "command": " ".join(sys.argv),
        "resume_command": f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} python3 tools/iqn_token_matched_20260919.py supervise --arm {arm} --output {output} --device {device}",
        "started_at": now_local(),
    }
    atomic_json(output / "launch.json", launch)
    try:
        current = 0
        if resume.is_file():
            payload = torch.load(resume, map_location="cpu", weights_only=False)
            current = int(payload["runtime"]["global_step"])
        if current > 200_000:
            raise RuntimeError("resume step exceeds registered 200k stop")
        for target in MILESTONES:
            state["current_target"] = target
            if current < target:
                state.update(phase="training", segment_start_time=time.time(), segment_start_step=current)
                segment_cfg = copy.deepcopy(cfg)
                segment_cfg["total_timesteps"] = target
                segment_cfg.setdefault("checkpointing", {})["full_resume"] = True
                if current:
                    segment_cfg["checkpointing"]["resume_path"] = str(resume)
                trainer = CoCapTrainer(segment_cfg)
                if trainer.global_step != current:
                    raise AssertionError("resume global step mismatch")
                if current == 0:
                    replay_sizes = {name: len(buffer) for name, buffer in trainer.replays.items()}
                    if any(replay_sizes.values()) or trainer.optimizer.state:
                        raise AssertionError("formal scratch gate found non-empty optimizer/replay")
                    if not np.isclose(trainer.epsilon(), trainer.epsilon_start):
                        raise AssertionError("formal scratch gate epsilon is not initial")
                    if state_hash(trainer.model) != state_hash(trainer.target_model):
                        raise AssertionError("formal scratch gate target is not initialized from online")
                    atomic_json(
                        output / "scratch_gate.json",
                        {
                            "schema": SCHEMA,
                            "arm": arm,
                            "status": "pass",
                            "global_step": trainer.global_step,
                            "replay_sizes": replay_sizes,
                            "optimizer_state_entries": len(trainer.optimizer.state),
                            "epsilon": trainer.epsilon(),
                            "model_hash": state_hash(trainer.model),
                            "target_hash": state_hash(trainer.target_model),
                            "completed_at": now_local(),
                        },
                    )
                trainer.train()
                if trainer.global_step != target:
                    raise AssertionError("trainer did not stop exactly at milestone")
                current = target
                close_trainer(trainer)
                del trainer
            milestone_resume = training_dir / f"checkpoints/resume_step_{target:09d}.pt"
            freeze_resume(resume, milestone_resume)
            checkpoint = training_dir / f"checkpoints/step_{target}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            eval_dir = output / "evaluations" / f"step_{target:09d}"
            report_path = eval_dir / "report.json"
            state.update(phase="formal_evaluation", segment_start_time=time.time(), segment_start_step=current)
            if report_path.is_file():
                report = json.loads(report_path.read_text())
            else:
                report = evaluate_checkpoint(
                    checkpoint,
                    eval_dir,
                    device,
                    episodes=int(cfg["formal_evaluation"]["episodes_per_scene"]),
                    seed_base=int(cfg["formal_evaluation"]["seed_base"]),
                    arm=arm,
                )
            latest = read_last_jsonl(training_dir / "metrics.jsonl")
            update_trend(output, target, report, latest)
            atomic_json(
                output / "status.json",
                {
                    **launch,
                    "status": "running" if target < 200_000 else "complete",
                    "current_step": target,
                    "last_full_resume": str(milestone_resume),
                    "last_checkpoint": str(checkpoint),
                    "last_evaluation": str(report_path),
                    "latest_metrics": latest,
                    "updated_at": now_local(),
                },
            )
        launch["status"] = "complete"
        launch["current_step"] = 200_000
        launch["completed_at"] = now_local()
        atomic_json(output / "status.json", launch)
        return 0
    except BaseException as exc:
        atomic_json(
            output / "status.json",
            {
                **launch,
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "updated_at": now_local(),
            },
        )
        raise
    finally:
        stop.set()
        thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "evaluate", "supervise", "status"))
    parser.add_argument("--arm", choices=("role", "z"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2026092801)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    output = (args.output or DEFAULT_OUTPUTS[args.arm]).resolve()
    if args.command == "preflight":
        print(json.dumps(contract_preflight(output / "preflight", args.device, args.arm), indent=2, ensure_ascii=False))
        return 0
    if args.command == "evaluate":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for evaluate")
        print(json.dumps(evaluate_checkpoint(args.checkpoint.resolve(), output, args.device, args.episodes, args.seed_base, args.arm, args.max_steps)["summary"], indent=2))
        return 0
    if args.command == "supervise":
        return supervisor(output, args.device, args.arm)
    for name in ("launch.json", "heartbeat.json", "status.json", "trend_ledger.json"):
        path = output / name
        if path.is_file():
            print(path.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
