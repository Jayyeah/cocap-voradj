#!/usr/bin/env python3
"""Formal IQN-Z-TOKEN scratch preflight, evaluator, and detached supervisor."""
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

from cocap_voradj.envs.density_sensing import POLICY as NORMSENSE_POLICY
from cocap_voradj.envs.density_sensing import enable_v2, runtime_metadata
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.training.trainer import CoCapTrainer, deep_update, load_config, set_global_config
from cocap_voradj.training.runtime_semantics import assert_runtime
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from tools.evaluate_vxy_stage2_cross_retention_20260903 import ring_count
from tools.run_forward_final_bridge_20260908 import atomic_json, run_episode as standard_episode

SCHEMA = "iqn-z-token-scratch-v1"
CONFIG_DIR = ROOT / "configs/experiments/iqn_token_scratch_20260918"
Z_CONFIG = CONFIG_DIR / "z_token.yaml"
ROLE_REFERENCE = CONFIG_DIR / "role_reference.yaml"
ROLE_WORKTREE = Path("/home/yjq/rl/CoCap1/iqn-role-token-scratch-20260918")
DEFAULT_OUTPUT = ROOT / "artifacts/2026-09-18_iqn_z_token_scratch"
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


def config_diff(role_path: Path, output: Path) -> dict[str, Any]:
    role = resolved(role_path, require_v2=False)
    z = resolved(Z_CONFIG)
    role_flat, z_flat = flatten(role), flatten(z)
    ignored_exact = {"output_root", "run_name", "checkpointing.full_resume_path"}
    ignored_prefixes = ("experiment_metadata.", "formal_evaluation.")
    differences = []
    ignored_differences = []
    for key in sorted(set(role_flat) | set(z_flat)):
        if role_flat.get(key) != z_flat.get(key):
            row = {"path": key, "role": role_flat.get(key), "z": z_flat.get(key)}
            if key in ignored_exact or key.startswith(ignored_prefixes):
                ignored_differences.append(row)
            else:
                differences.append(row)
    unexpected = [row for row in differences if row["path"] not in ALLOWED_TOKEN_DIFFS]
    report = {
        "schema": SCHEMA,
        "status": "pass" if not unexpected else "fail",
        "role_config": str(role_path),
        "z_config": str(Z_CONFIG),
        "differences": differences,
        "ignored_non_training_differences": ignored_differences,
        "allowed_paths": sorted(ALLOWED_TOKEN_DIFFS),
        "unexpected_differences": unexpected,
        "generated_at": now_local(),
    }
    atomic_json(output, report)
    if unexpected:
        raise RuntimeError(f"ROLE-vs-Z config drift: {unexpected}")
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


def evaluate_checkpoint(
    checkpoint: Path,
    output: Path,
    device: str,
    episodes: int,
    seed_base: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cfg = resolved(Z_CONFIG)
    model = CoCapIQN.load(str(checkpoint), device=device).eval()
    if (
        not model.config.include_z_state
        or model.config.include_is_pursuing
        or model.config.pursuing_late_fusion
        or model.pursuing_embed is not None
    ):
        raise ValueError("checkpoint is not the registered Z-token architecture")
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
            diag = ZDiagnostics(scene)

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
    all_diag = [row["z_diagnostics"] for row in records]
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
        "z": {
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
        },
    }
    report = {
        "schema": SCHEMA,
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


def contract_preflight(output: Path, device: str, role_config: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    diff = config_diff(role_config, output / "config_diff.json")
    zcfg, rolecfg = resolved(Z_CONFIG), resolved(role_config)
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
    if initial_z_hash != initial_role_hash:
        raise AssertionError("matched scratch initial weights differ")
    if z_model.pursuing_embed is not None or any("pursuing_embed" in key for key in z_model.state_dict()):
        raise AssertionError("late fusion survived in Z model")

    env, observations = make_env(zcfg, "mixed", int(zcfg["seed"]))
    for index, obs in enumerate(observations):
        if obs is None:
            continue
        if obs["self"].shape != (9,) or obs["pursuers"].shape != (8, 7):
            raise AssertionError("Z observation shape drift")
        if not np.isclose(obs["self"][-1], env.z_state[index]):
            raise AssertionError("self last column is not z_i")
    z_before = [None if obs is None else np.asarray(obs["pursuers"]).copy() for obs in observations]
    env.z_state[:] = np.linspace(0.1, 0.4, len(env.pursuers), dtype=np.float32)
    z_after = env.get_policy_observations(False, True)
    for before, after in zip(z_before, z_after):
        if before is not None and after is not None:
            np.testing.assert_array_equal(before[:, :6], after["pursuers"][:, :6])
    friend_z_mapping_checks = assert_friend_z_mapping(env, z_after)

    env._reset_z_state()
    source = {0}
    original_has_enemy = env._has_enemy_neighbor
    env._has_enemy_neighbor = lambda data, key: bool(key[0] == "pursuer" and key[1] in source)
    adjacency = {
        ("pursuer", 0): {("pursuer", 1)},
        ("pursuer", 1): {("pursuer", 0), ("pursuer", 2)},
        ("pursuer", 2): {("pursuer", 1), ("pursuer", 3)},
        ("pursuer", 3): {("pursuer", 2)},
    }
    env._advance_z_state({"adjacency": adjacency})
    first = env.z_state.copy()
    source.clear()
    env._advance_z_state({"adjacency": adjacency})
    second = env.z_state.copy()
    env._has_enemy_neighbor = original_has_enemy
    np.testing.assert_allclose(first, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(second, [0.95, 0.85, 0.0, 0.0])

    direct_env, _ = make_env(zcfg, "mixed", int(zcfg["seed"]) + 1)
    p0, target = direct_env.pursuers[0], direct_env.evaders[0]
    target.x = float(np.clip(p0.x + (5.0 if p0.x < 110 else -5.0), 5.0, 115.0))
    target.y = float(p0.y)
    direct_env._reset_z_state()
    direct_env._invalidate_voronoi_cache()
    direct_env._advance_z_state(direct_env._capture_voronoi_map())
    if not any(direct_env._z_last_direct) or any(
        direct and not np.isclose(value, 1.0)
        for direct, value in zip(direct_env._z_last_direct, direct_env.z_state)
    ):
        raise AssertionError("direct-visible z=1 invariant failed")

    coverage_env, _ = make_env(zcfg, "coverage", int(zcfg["seed"]) + 2)
    if np.any(coverage_env.z_state != 0):
        raise AssertionError("pure coverage reset z is not exact zero")

    role_env, _ = make_env(rolecfg, "mixed", int(rolecfg["seed"]))
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

    smoke_dir = Path(tempfile.mkdtemp(prefix="sanity-", dir=output))
    smoke_cfg = copy.deepcopy(zcfg)
    production_probe_cfg = copy.deepcopy(zcfg)
    production_probe_cfg.update(
        output_root=str(smoke_dir),
        run_name="production_contract_probe",
        device=device,
        total_timesteps=1,
    )
    production_probe = CoCapTrainer(production_probe_cfg)
    production_initial_replay_sizes = {
        name: len(buffer) for name, buffer in production_probe.replays.items()
    }
    expected_replay_classes = set(zcfg["voradj"]["replay_batch_counts"])
    if set(production_initial_replay_sizes) != expected_replay_classes:
        raise AssertionError("production replay class set drift")
    if any(production_initial_replay_sizes.values()):
        raise AssertionError(f"production scratch replay is not fresh: {production_initial_replay_sizes}")
    close_trainer(production_probe)
    smoke_cfg.update(
        output_root=str(smoke_dir),
        run_name="training",
        device=device,
        total_timesteps=16,
        train_mode="voradj",
    )
    smoke_cfg["iqn"].update(
        batch_size=8,
        min_replay_size=8,
        replay_capacity=256,
        train_freq=1,
        target_update_freq=1,
        checkpoint_freq=16,
        log_freq_steps=1,
    )
    smoke_cfg["checkpointing"] = {"full_resume": True}
    trainer = CoCapTrainer(smoke_cfg)
    smoke_initial_replay_sizes = {name: len(buffer) for name, buffer in trainer.replays.items()}
    if any(smoke_initial_replay_sizes.values()):
        raise AssertionError(f"optimizer smoke replay is not fresh: {smoke_initial_replay_sizes}")
    initial_parameters = {key: value.detach().cpu().clone() for key, value in trainer.model.state_dict().items()}
    checkpoint = trainer.train()
    changed = any(not torch.equal(initial_parameters[key], value.detach().cpu()) for key, value in trainer.model.state_dict().items())
    if trainer.update_steps <= 0 or not changed or trainer.loss_ema is None or not math.isfinite(trainer.loss_ema):
        raise AssertionError("real IQN optimizer update sanity failed")
    if trainer.target_update_count <= 0:
        raise AssertionError("target network never updated")
    for key, value in trainer.model.state_dict().items():
        if not torch.equal(value.detach().cpu(), trainer.target_model.state_dict()[key].detach().cpu()):
            raise AssertionError("target network is not synchronized after target update")
    source_env = trainer.envs[trainer.current_task]
    state = source_env.z_state_dict()
    state["values"] = np.linspace(0.11, 0.44, len(source_env.pursuers)).tolist()
    source_env.load_z_state_dict(state)
    resume = trainer._save_full_resume()
    saved_z = source_env.z_state_dict()
    resume_cfg = copy.deepcopy(smoke_cfg)
    resume_cfg["total_timesteps"] = 17
    resume_cfg["checkpointing"]["resume_path"] = str(resume)
    resumed = CoCapTrainer(resume_cfg)
    if resumed.envs[resumed.current_task].z_state_dict() != saved_z:
        raise AssertionError("full resume did not restore z bit-exact")
    close_trainer(resumed)

    evaluator = evaluate_checkpoint(
        checkpoint,
        smoke_dir / "formal_eval_smoke",
        device,
        episodes=1,
        seed_base=int(zcfg["formal_evaluation"]["seed_base"]),
        max_steps=4,
    )
    metrics = [
        json.loads(line)
        for line in (smoke_dir / "training/metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    latest = metrics[-1]
    gpu = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
        "memory_allocated": torch.cuda.memory_allocated(torch.device(device)) if device.startswith("cuda") else 0,
        "memory_reserved": torch.cuda.memory_reserved(torch.device(device)) if device.startswith("cuda") else 0,
    }
    result = {
        "schema": SCHEMA,
        "status": "pass",
        "head": git("rev-parse", "HEAD"),
        "config_diff": diff,
        "initialization": {
            "scratch": True,
            "seed": int(zcfg["seed"]),
            "z_initial_state_hash": initial_z_hash,
            "role_initial_state_hash": initial_role_hash,
            "bit_exact_equal": True,
            "teacher_transfer": False,
            "fresh_replay": True,
            "production_initial_replay_sizes": production_initial_replay_sizes,
            "optimizer_smoke_initial_replay_sizes": smoke_initial_replay_sizes,
        },
        "architecture": {
            "self_shape": [9],
            "friend_shape": [8, 7],
            "self_last_is_z": True,
            "friend_last_is_z": True,
            "friend_z_mapping_checks": friend_z_mapping_checks,
            "is_pursuing_policy_input": False,
            "late_fusion": False,
            "z_special_branch": False,
            "friend_ordering": "physical_only",
        },
        "z": {
            "lambda": 0.95,
            "eta": 0.85,
            "synchronous_previous_step": True,
            "direct_visible_z_one": True,
            "pure_coverage_exact_zero": True,
            "reset_zero": True,
            "checkpoint_resume_bit_exact": True,
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
            "latest_metrics": latest,
        },
        "formal_evaluator_smoke": {
            "status": evaluator["status"],
            "episodes_per_scene": 1,
            "steps_per_episode": 4,
        },
        "gpu": gpu,
        "completed_at": now_local(),
    }
    atomic_json(output / "startup_sanity.json", result)
    close_trainer(trainer)
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


def supervisor(output: Path, device: str, role_config: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    # Never let a cached startup sanity bypass comparison with the ROLE config
    # that is actually current at launch time.
    try:
        contract_preflight(output / "preflight", device, role_config)
    except BaseException as exc:
        atomic_json(
            output / "status.json",
            {
                "schema": SCHEMA,
                "status": "blocked_prelaunch",
                "phase": "prelaunch_contract_gate",
                "role_config": str(role_config),
                "z_config": str(Z_CONFIG),
                "error": repr(exc),
                "updated_at": now_local(),
            },
        )
        raise
    sanity_path = output / "preflight/startup_sanity.json"
    sanity = json.loads(sanity_path.read_text())
    if sanity.get("status") != "pass":
        raise RuntimeError("startup sanity is not PASS")
    cfg = resolved(Z_CONFIG)
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

    thread = threading.Thread(target=heartbeat, name="iqn-z-heartbeat", daemon=True)
    thread.start()
    launch = {
        "schema": SCHEMA,
        "status": "running",
        "pid": os.getpid(),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "worktree": str(ROOT),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config": str(Z_CONFIG),
        "role_config": str(role_config),
        "seed": int(cfg["seed"]),
        "milestones": list(MILESTONES),
        "stop_target": 200_000,
        "command": " ".join(sys.argv),
        "resume_command": f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} python3 tools/iqn_z_token_scratch_20260918.py supervise --output {output} --device {device}",
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


def choose_role_config(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = [
        ROLE_WORKTREE / "configs/experiments/iqn_role_token_scratch_20260918/role_token.yaml",
        ROLE_WORKTREE / "configs/experiments/iqn_token_scratch_20260918/role_token.yaml",
        ROLE_WORKTREE / "configs/experiments/iqn_token_scratch_20260918/role_reference.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return ROLE_REFERENCE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "evaluate", "supervise", "status"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--role-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2026092801)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    role_config = choose_role_config(args.role_config)
    if args.command == "preflight":
        print(json.dumps(contract_preflight(output / "preflight", args.device, role_config), indent=2, ensure_ascii=False))
        return 0
    if args.command == "evaluate":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for evaluate")
        print(json.dumps(evaluate_checkpoint(args.checkpoint.resolve(), output, args.device, args.episodes, args.seed_base, args.max_steps)["summary"], indent=2))
        return 0
    if args.command == "supervise":
        return supervisor(output, args.device, role_config)
    for name in ("launch.json", "heartbeat.json", "status.json", "trend_ledger.json"):
        path = output / name
        if path.is_file():
            print(path.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
