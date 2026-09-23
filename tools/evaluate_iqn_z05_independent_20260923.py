#!/usr/bin/env python3
"""Independent, evaluation-only IQN Z05 native/generalization evaluator.

This runner deliberately reuses the registered Z-v2 environment, midpoint-32
IQN action path, formal episode runner, Z diagnostics, and censored efficiency
metric implementation.  It adds only evaluation orchestration: runtime
Density-Normalized Sensing metadata, token occupancy auditing, and GIFs made
from ten episodes selected from the same twenty-episode formal batch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.envs.density_sensing import POLICY as NORMSENSE_POLICY
from cocap_voradj.envs.density_sensing import runtime_metadata
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.runtime_semantics import assert_runtime
from cocap_voradj.training.trainer import deep_update, set_global_config
from tools import iqn_token_matched_20260919 as formal
from tools.rollout_voradj_visual import render_gif, snapshot_env
from tools.run_forward_final_bridge_20260908 import run_episode as formal_episode


SCENES = ("coverage", "capture", "mixed")
SCHEMA = "iqn-z05-independent-evaluation-v1"
DEFAULT_EPISODES = 20
DEFAULT_GIF_INDICES = tuple(range(0, 20, 2))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def now_local() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def numeric_summary(values: list[Any]) -> dict[str, Any]:
    result = formal.numeric_summary(values)
    if "p50" in result:
        result["median"] = result["p50"]
    return result


def efficiency_summary(rows: list[Mapping[str, Any]], success_key: str, value_key: str) -> dict[str, Any]:
    result = formal.efficiency_timing_summary(rows, success_key=success_key, value_key=value_key)
    if "p50" in result:
        result["median"] = result["p50"]
    return result


def scene_config(root_cfg: Mapping[str, Any], scene: str) -> dict[str, Any]:
    task = "voradj_coverage" if scene == "coverage" else "voradj"
    cfg = deep_update(copy.deepcopy(dict(root_cfg)), root_cfg["tasks"][task])
    if scene == "capture":
        cfg.setdefault("voradj", {})["capture_episode_ends_on_capture"] = True
        cfg["voradj"]["capture_episode_success_on_capture"] = True
    return cfg


def configure_large_case(base_cfg: Mapping[str, Any], pursuers: int, evaders: int, obstacles: int, post_window: int) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg.setdefault("env", {}).update(
        num_pursuers=int(pursuers), num_evaders=int(evaders), num_obstacles=int(obstacles),
        width=120.0, height=120.0, x_boundary_left=0.0, x_boundary_right=120.0,
        y_boundary_bottom=0.0, y_boundary_top=120.0,
    )
    for task_name, task_evaders, spawn_mode in (
        ("voradj", evaders, "map_random"),
        ("voradj_coverage", 0, "inner_random_cluster"),
    ):
        task = cfg.setdefault("tasks", {}).setdefault(task_name, {})
        task.setdefault("env", {}).update(
            num_pursuers=int(pursuers), num_evaders=int(task_evaders), num_obstacles=int(obstacles),
            width=120.0, height=120.0, x_boundary_left=0.0, x_boundary_right=120.0,
            y_boundary_bottom=0.0, y_boundary_top=120.0,
            pursuer_spawn_mode=spawn_mode,
        )
    cfg.setdefault("reward", {})["post_capture_coverage_window_steps"] = int(post_window)
    return cfg


def obstacle_payload(env: VorAdjEnv) -> list[dict[str, float]]:
    return [
        {"x": float(item.x), "y": float(item.y), "r": float(item.r)}
        for item in env.obstacles
    ]


def resolver_payload(env: VorAdjEnv, metadata: Mapping[str, Any], *, mode: str, coverage_spawn_radius: float | None) -> dict[str, Any]:
    obstacles = obstacle_payload(env)
    obstacle_hash = stable_hash(obstacles)
    n = int(metadata["num_pursuers"])
    width = float(metadata["map_size"][0])
    height = float(metadata["map_size"][1])
    a_eff = float(metadata["free_space_area"])
    k = float(metadata["k"])
    raw = float(k * math.sqrt(a_eff / n))
    floor = float(metadata["radius_floor"])
    return {
        "N": n,
        "num_evaders": int(len(env.evaders)),
        "num_obstacles": int(len(env.obstacles)),
        "width": width,
        "height": height,
        "A_eff": a_eff,
        "k": k,
        "raw_R": raw,
        "resolved_R": float(metadata["resolved_onboard_radius"]),
        "R_floor": floor,
        "floor_triggered": bool(raw < floor - 1e-9),
        "map_scale": 1.0,
        "surface_distance_semantics": str(metadata["distance_semantics"]),
        "free_mask_hash": str(metadata["free_mask_sha256"]),
        "free_mask_grid_n": int(metadata["free_mask_grid_n"]),
        "free_mask_method": str(metadata["free_area_method"]),
        "robot_radius": float(max(float(p.r) for p in env.pursuers)),
        "obstacle_mask": obstacles,
        "obstacle_mask_hash": obstacle_hash,
        "policy": str(metadata["policy"]),
        "schema_version": int(metadata["schema_version"]),
        "mode": str(mode),
        "coverage_spawn_radius": None if coverage_spawn_radius is None else float(coverage_spawn_radius),
        "runtime_metadata": dict(metadata),
    }


class CapacityAudit:
    """Collect actual policy-visible occupancy and raw physical friend counts."""

    def __init__(self, env: VorAdjEnv, model: CoCapIQN) -> None:
        self.env_caps = {
            "friend_tokens": int(env.per_cfg.get("max_pursuer_num", 0)),
            "evader_tokens": int(env.per_cfg.get("max_evader_num", 0)),
            "obstacle_tokens": int(env.per_cfg.get("max_obstacle_num", 0)),
        }
        self.model_caps = {
            "friend_tokens": int(model.config.max_pursuers),
            "evader_tokens": int(model.config.max_evaders),
            "obstacle_tokens": int(model.config.max_obstacles),
        }
        self.samples: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []
        self._episode_samples: list[dict[str, Any]] = []

    def start_episode(self) -> None:
        self._episode_samples = []

    def observe(self, env: VorAdjEnv) -> None:
        observations = env.get_observations()
        data = env._capture_voronoi_map()
        active_rows: list[dict[str, Any]] = []
        friend_cap = self.env_caps["friend_tokens"]
        evader_cap = self.env_caps["evader_tokens"]
        obstacle_cap = self.env_caps["obstacle_tokens"]
        for index, obs in enumerate(observations):
            if obs is None:
                continue
            masks = np.asarray(obs["masks"], dtype=bool)
            friend_seen = int(masks[1 : 1 + friend_cap].sum())
            evader_start = 1 + friend_cap
            obstacle_start = evader_start + evader_cap
            evader_seen = int(masks[evader_start : evader_start + evader_cap].sum())
            obstacle_seen = int(masks[obstacle_start : obstacle_start + obstacle_cap].sum())
            neighbors = data.get("adjacency", {}).get(("pursuer", index), set())
            raw_friends = int(sum(1 for key in neighbors if key[0] == "pursuer" and not env.pursuers[int(key[1])].deactivated))
            active_rows.append({
                "pursuer_id": int(index),
                "friend_candidates": raw_friends,
                "friend_tokens_seen": friend_seen,
                "friend_truncated": bool(raw_friends > friend_cap),
                "evader_tokens_seen": evader_seen,
                "obstacle_tokens_seen": obstacle_seen,
                "friend_padding": int(friend_cap - friend_seen),
                "evader_padding": int(evader_cap - evader_seen),
                "obstacle_padding": int(obstacle_cap - obstacle_seen),
            })
        sample = {
            "episode_step": int(env.episode_step),
            "active_agents": int(len(active_rows)),
            "rows": active_rows,
        }
        self.samples.append(sample)
        self._episode_samples.append(sample)

    def finish_episode(self) -> dict[str, Any]:
        rows = [row for sample in self._episode_samples for row in sample["rows"]]
        episode = {
            "samples": len(self._episode_samples),
            "active_agent_rows": len(rows),
            "friend_candidate_count": numeric_summary([row["friend_candidates"] for row in rows]),
            "friend_token_occupancy": numeric_summary([row["friend_tokens_seen"] for row in rows]),
            "evader_token_occupancy": numeric_summary([row["evader_tokens_seen"] for row in rows]),
            "obstacle_token_occupancy": numeric_summary([row["obstacle_tokens_seen"] for row in rows]),
            "friend_truncation_events": sum(bool(row["friend_truncated"]) for row in rows),
            "max_friend_candidates": max((int(row["friend_candidates"]) for row in rows), default=0),
            "max_friend_tokens_seen": max((int(row["friend_tokens_seen"]) for row in rows), default=0),
        }
        self.episodes.append(episode)
        self._episode_samples = []
        return episode

    def summary(self, env: VorAdjEnv) -> dict[str, Any]:
        rows = [row for sample in self.samples for row in sample["rows"]]
        def values(key: str) -> list[float]:
            return [float(row[key]) for row in rows]
        friend_candidates = values("friend_candidates")
        friend_seen = values("friend_tokens_seen")
        evader_seen = values("evader_tokens_seen")
        obstacle_seen = values("obstacle_tokens_seen")
        truncation = [row for row in rows if row["friend_truncated"]]
        env_counts = {
            "pursuers": int(len(env.pursuers)),
            "evaders": int(len(env.evaders)),
            "obstacles": int(len(env.obstacles)),
        }
        return {
            "policy_actual_token_caps": dict(self.model_caps),
            "environment_observation_caps": dict(self.env_caps),
            "environment_counts_last_episode": env_counts,
            "friend_candidate_count": numeric_summary(friend_candidates),
            "friend_token_occupancy": numeric_summary(friend_seen),
            "evader_token_occupancy": numeric_summary(evader_seen),
            "obstacle_token_occupancy": numeric_summary(obstacle_seen),
            "friend_truncation": {
                "event_count": len(truncation),
                "sample_count": len(rows),
                "sample_rate": len(truncation) / max(len(rows), 1),
                "episode_count": sum(int(item["friend_truncation_events"] > 0) for item in self.episodes),
                "episodes": len(self.episodes),
                "max_candidates": max((int(item["max_friend_candidates"]) for item in self.episodes), default=0),
            },
            "selection_semantics": {
                "friend_ordering_mode": str(env.per_cfg.get("friend_ordering_mode")),
                "friend_selection": "physical_only sorted local adjacency then first max_pursuer_num",
                "role_or_z_affects_friend_truncation_priority": False,
                "padding_is_masked": True,
                "central_diagnostic_only_padding": "not_applicable_for_IQN_policy",
            },
            "capacity_contract": {
                "model_and_environment_friend_cap_match": self.model_caps["friend_tokens"] == self.env_caps["friend_tokens"],
                "model_and_environment_evader_cap_match": self.model_caps["evader_tokens"] == self.env_caps["evader_tokens"],
                "model_and_environment_obstacle_cap_match": self.model_caps["obstacle_tokens"] == self.env_caps["obstacle_tokens"],
                "architecture_modified": False,
            },
        }


def make_env_factory(root_cfg: Mapping[str, Any], scene: str, seed: int, mode: str, model: CoCapIQN, holder: dict[str, Any]):
    cfg = scene_config(root_cfg, scene)
    coverage_spawn_radius: float | None = None
    bootstrap_metadata: Mapping[str, Any] | None = None
    if mode == "large" and scene == "coverage":
        bootstrap_cfg = copy.deepcopy(cfg)
        bootstrap_cfg.setdefault("env", {})["pursuer_spawn_mode"] = "map_random"
        set_global_config(bootstrap_cfg)
        bootstrap = VorAdjEnv(copy.deepcopy(bootstrap_cfg), seed=int(seed))
        bootstrap.reset()
        bootstrap_metadata = runtime_metadata(bootstrap)
        coverage_spawn_radius = float(bootstrap_metadata["resolved_onboard_radius"])
        cfg.setdefault("env", {})["pursuer_spawn_mode"] = "inner_random_cluster"
        cfg["env"]["spawn_cluster_radius"] = coverage_spawn_radius
        holder["coverage_spawn_radius"] = coverage_spawn_radius
    set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=int(seed))
    observations = list(env.reset())
    metadata = runtime_metadata(env)
    if bootstrap_metadata is not None:
        for key in ("free_mask_sha256", "free_space_area", "resolved_onboard_radius", "num_pursuers", "map_size"):
            if metadata[key] != bootstrap_metadata[key]:
                raise AssertionError(f"resolver bootstrap drift for {key}: {bootstrap_metadata[key]} != {metadata[key]}")
    if metadata["policy"] != NORMSENSE_POLICY:
        raise AssertionError("Density-Normalized Sensing V2 runtime policy drift")
    assert_runtime(env)
    holder["env"] = env
    holder["resolver"] = resolver_payload(env, metadata, mode=mode, coverage_spawn_radius=coverage_spawn_radius)
    holder["config"] = cfg
    if model.config.max_pursuers != int(env.per_cfg.get("max_pursuer_num", 0)):
        raise RuntimeError("policy/environment friend token cap mismatch; architecture change would be required")
    return env, observations


def json_finite(payload: Any) -> Any:
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    if isinstance(payload, dict):
        return {str(key): json_finite(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [json_finite(value) for value in payload]
    return payload


def add_episode_metrics(row: dict[str, Any], env: VorAdjEnv, diag: Any, scene: str) -> dict[str, Any]:
    record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
    row["area_cv"] = float(record["coverage_strict_area_cv"])
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
        row["mission_steps"] = row["capture_steps"] if row["safe_complete"] else None
    return row


def summarize_records(records: list[dict[str, Any]], capacity: CapacityAudit) -> dict[str, Any]:
    def rows(scene: str) -> list[dict[str, Any]]:
        return [row for row in records if row["scene"] == scene]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return sum(bool(row.get(key, False)) for row in items) / max(len(items), 1)

    coverage, capture, mixed = rows("coverage"), rows("capture"), rows("mixed")
    action_hist = np.sum(
        [np.asarray(row["policy_diagnostics"]["action_histogram"], dtype=np.int64) for row in records], axis=0
    ).tolist()
    all_diag = [row["z_diagnostics"] for row in records]
    release = [row["release_seconds_max_z_lt_0_1"] for row in all_diag if row["release_seconds_max_z_lt_0_1"] is not None]
    return {
        "coverage": {
            "episodes": len(coverage),
            "strict_ce_rate": rate(coverage, "ce_success"),
            "collision_rate": rate(coverage, "collision"),
            "ce_rms": numeric_summary([row["ce_rms"] for row in coverage]),
            "ce_max": numeric_summary([row["ce_max"] for row in coverage]),
            "area_cv": numeric_summary([row["area_cv"] for row in coverage]),
            "time_to_CE_steps": efficiency_summary(coverage, "ce_success", "time_to_strict_CE_steps"),
            "time_to_CE_seconds": efficiency_summary(coverage, "ce_success", "time_to_strict_CE_seconds"),
            "censored_or_failed_episodes": sum(not bool(row["ce_success"]) for row in coverage),
        },
        "capture": {
            "episodes": len(capture),
            "capture_rate": rate(capture, "captured"),
            "normal_capture_rate": rate(capture, "normal_capture"),
            "stationary_capture_rate": rate(capture, "stationary_capture"),
            "collision_rate": rate(capture, "collision"),
            "capture_steps": efficiency_summary(capture, "captured", "capture_steps"),
            "capture_seconds": efficiency_summary(capture, "captured", "capture_seconds"),
            "censored_or_failed_episodes": sum(not bool(row["captured"]) for row in capture),
        },
        "mixed": {
            "episodes": len(mixed),
            "capture_rate": rate(mixed, "captured"),
            "collision_rate": rate(mixed, "collision"),
            "post_capture_ce_rate": rate(mixed, "ce_success"),
            "safe_complete_rate": rate(mixed, "safe_complete"),
            "capture_steps": efficiency_summary(mixed, "captured", "capture_steps"),
            "capture_seconds": efficiency_summary(mixed, "captured", "capture_seconds"),
            "recovery_steps": efficiency_summary(mixed, "ce_success", "recovery_steps"),
            "recovery_seconds": efficiency_summary(mixed, "ce_success", "recovery_seconds"),
            "mission_steps": efficiency_summary(mixed, "safe_complete", "mission_steps"),
            "mission_seconds": efficiency_summary(mixed, "safe_complete", "mission_seconds"),
            "censored_or_failed_episodes": sum(not bool(row["safe_complete"]) for row in mixed),
        },
        "actions": {
            "histogram": action_hist,
            "distribution": [count / max(sum(action_hist), 1) for count in action_hist],
        },
        "z": {
            "direct_checks": sum(row["direct_checks"] for row in all_diag),
            "direct_violations": sum(row["direct_violations"] for row in all_diag),
            "direct_visible_z_one_rate": 1.0 - sum(row["direct_violations"] for row in all_diag) / max(sum(row["direct_checks"] for row in all_diag), 1),
            "max_lineage_hop": max((row["max_lineage_hop"] for row in all_diag), default=-1),
            "post_capture_release_seconds": numeric_summary(release),
            "post_capture_never_release_episodes": sum(row["post_capture_never_released"] for row in all_diag),
            "update_count_checks": sum(row["z_update_count_checks"] for row in all_diag),
            "update_count_violations": sum(row["z_update_count_violations"] for row in all_diag),
            "update_count_per_decision_exact": all(row["z_update_count_per_decision_exact"] for row in all_diag),
        },
        "token_occupancy": capacity.summary(records[-1].get("_last_env") if records and records[-1].get("_last_env") else capacity_env_placeholder(capacity)),
    }


def capacity_env_placeholder(capacity: CapacityAudit) -> Any:
    # Only used to retain a stable summary call shape; the actual counts are
    # filled by the caller immediately after this function returns.
    return type("CapacityEnv", (), {"per_cfg": {"friend_ordering_mode": "physical_only"}, "pursuers": [], "evaders": [], "obstacles": []})()


def run_capacity_audit(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    base_cfg = formal.resolved(config_path)
    model = CoCapIQN.load(str(Path(args.checkpoint).resolve()), device=args.device).eval()
    root_cfg = configure_large_case(base_cfg, args.pursuers, args.evaders, args.obstacles, args.post_window)
    output = Path(args.output).resolve()
    samples: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(SCENES):
        holder: dict[str, Any] = {}
        env, _ = make_env_factory(root_cfg, scene, int(args.seed_base + scene_index * 100_000), "large", model, holder)
        audit = CapacityAudit(env, model)
        audit.start_episode()
        audit.observe(env)
        audit.finish_episode()
        samples.append({"scene": scene, "resolver": holder["resolver"], "capacity": audit.summary(env)})
    payload = {
        "schema": SCHEMA,
        "status": "pass",
        "mode": "prelaunch_capacity_audit",
        "config": str(config_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "policy_architecture": "unchanged_checkpoint_architecture",
        "central_diagnostic_only_padding": "not_applicable_for_IQN_policy",
        "samples": samples,
        "completed_at": now_local(),
    }
    atomic_json(output, json_finite(payload))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    selection: dict[str, Any] | None = None
    if args.selection_report:
        report_path = Path(args.selection_report).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selection = dict(report["selected"])
        checkpoint = Path(str(selection["checkpoint"])).resolve()
        if args.expected_step is not None and int(selection["step"]) != int(args.expected_step):
            raise AssertionError(f"selection step mismatch: {selection['step']} != {args.expected_step}")
        expected_sha = str(selection["checkpoint_sha256"])
        actual_sha = sha256_file(checkpoint)
        if actual_sha != expected_sha:
            raise AssertionError(f"selected checkpoint sha mismatch: {actual_sha} != {expected_sha}")
        return checkpoint, selection
    if not args.checkpoint:
        raise ValueError("--checkpoint or --selection-report is required")
    checkpoint = Path(args.checkpoint).resolve()
    return checkpoint, selection


def run_evaluation(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, selection = resolve_checkpoint(args)
    config_path = Path(args.config).resolve()
    base_cfg = formal.resolved(config_path)
    if args.mode == "large":
        if args.pursuers is None or args.evaders is None or args.obstacles is None or args.post_window is None:
            raise ValueError("large mode requires --pursuers --evaders --obstacles --post-window")
        root_cfg = configure_large_case(base_cfg, args.pursuers, args.evaders, args.obstacles, args.post_window)
    else:
        root_cfg = base_cfg
    model = CoCapIQN.load(str(checkpoint), device=args.device).eval()
    if model.config.include_z_state is not True or model.config.include_is_pursuing is not False:
        raise RuntimeError("checkpoint is not the exact Z-state policy contract")
    if model.config.pursuing_late_fusion:
        raise RuntimeError("checkpoint retains forbidden pursuing late-fusion architecture")
    records: list[dict[str, Any]] = []
    gifs: list[dict[str, Any]] = []
    capacity = CapacityAudit.__new__(CapacityAudit)
    capacity.samples = []
    capacity.episodes = []
    capacity._episode_samples = []
    capacity.env_caps = {}
    capacity.model_caps = capacity_model_caps(model)
    run_started = time.monotonic()
    gif_indices = tuple(int(x) for x in args.gif_indices.split(",") if x.strip())
    total = len(SCENES) * int(args.episodes)
    for scene_index, scene in enumerate(SCENES):
        for episode_index in range(int(args.episodes)):
            seed = int(args.seed_base + scene_index * 100_000 + episode_index)
            holder: dict[str, Any] = {}
            frames: list[dict[str, Any]] = []
            audit: CapacityAudit | None = None

            def factory(requested_scene: str, requested_seed: int):
                nonlocal audit
                env, observations = make_env_factory(root_cfg, requested_scene, requested_seed, args.mode, model, holder)
                audit = CapacityAudit(env, model)
                if not capacity.env_caps:
                    capacity.env_caps = dict(audit.env_caps)
                audit.start_episode()
                return env, observations

            def on_snapshot(env: VorAdjEnv, step: int) -> None:
                assert audit is not None
                audit.observe(env)
                if episode_index not in gif_indices:
                    return
                if scene == "coverage":
                    phase = "coverage"
                elif scene == "mixed" and env.evaders and all(ev.deactivated and not ev.collision for ev in env.evaders):
                    phase = "coverage"
                else:
                    phase = "capture"
                frames.append(snapshot_env(env, scene, phase, int(env.post_capture_step if phase == "coverage" else 0), int(step), int(step)))

            def on_progress(step: int) -> None:
                atomic_json(output / "progress.json", {
                    "status": "running",
                    "completed_episodes": len(records),
                    "total_episodes": total,
                    "scene": scene,
                    "episode_index": episode_index,
                    "episode_step": int(step),
                    "elapsed_seconds": time.monotonic() - run_started,
                })

            diag = formal.ZDiagnostics(scene)
            row = formal_episode(
                model,
                scene,
                seed,
                args.device,
                on_transition=diag.transition,
                on_progress=on_progress,
                on_snapshot=on_snapshot,
                env_factory=factory,
                diagnostic_central_schema=not (args.mode == "large" and int(args.pursuers) > 12),
            )
            assert audit is not None
            env = holder["env"]
            add_episode_metrics(row, env, diag, scene)
            row["resolver"] = holder["resolver"]
            row["token_occupancy_episode"] = audit.finish_episode()
            row["evaluation_mode"] = args.mode
            row["config_source"] = str(config_path)
            row["checkpoint_sha256"] = sha256_file(checkpoint)
            row = json_finite(row)
            records.append(row)
            capacity.samples.extend(audit.samples)
            capacity.episodes.extend(audit.episodes)
            if episode_index in gif_indices:
                gif_dir = output / "gifs" / scene
                source_dir = output / "gif_sources" / scene
                gif_path = gif_dir / f"rollout_{scene}_seed_{seed}.gif"
                render_record = render_gif(frames, gif_path, int(args.max_gif_frames), 70, 100, scene, False, False, False, scene == "coverage")
                source_path = source_dir / f"episode_{scene}_seed_{seed}.json"
                atomic_json(source_path, {"scene": scene, "seed": seed, "episode_index": episode_index, "source": "same_formal_20_rollout_batch", "frames": frames})
                gifs.append({"scene": scene, "episode_index": episode_index, "seed": seed, "gif": str(gif_path), "source_frames": str(source_path), **render_record})
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            atomic_json(output / "progress.json", {"status": "running", "completed_episodes": len(records), "total_episodes": total, "last_scene": scene, "last_episode_index": episode_index, "elapsed_seconds": time.monotonic() - run_started})

    # Build the formal summary directly from the same records.  The existing
    # formal efficiency function is used above; this keeps failed/censored
    # episodes out of mean/median/p90 timing distributions.
    coverage, capture, mixed = ([row for row in records if row["scene"] == scene] for scene in SCENES)
    def rate(items: list[dict[str, Any]], key: str) -> float:
        return sum(bool(row.get(key, False)) for row in items) / max(len(items), 1)
    all_diag = [row["z_diagnostics"] for row in records]
    release = [row["release_seconds_max_z_lt_0_1"] for row in all_diag if row["release_seconds_max_z_lt_0_1"] is not None]
    summary = {
        "coverage": {
            "episodes": len(coverage), "strict_ce_rate": rate(coverage, "ce_success"), "collision_rate": rate(coverage, "collision"),
            "ce_rms": numeric_summary([row["ce_rms"] for row in coverage]), "ce_max": numeric_summary([row["ce_max"] for row in coverage]),
            "area_cv": numeric_summary([row["area_cv"] for row in coverage]),
            "time_to_CE_steps": efficiency_summary(coverage, "ce_success", "time_to_strict_CE_steps"),
            "time_to_CE_seconds": efficiency_summary(coverage, "ce_success", "time_to_strict_CE_seconds"),
            "censored_or_failed_episodes": sum(not bool(row["ce_success"]) for row in coverage),
        },
        "capture": {
            "episodes": len(capture), "capture_rate": rate(capture, "captured"), "normal_capture_rate": rate(capture, "normal_capture"),
            "stationary_capture_rate": rate(capture, "stationary_capture"), "collision_rate": rate(capture, "collision"),
            "capture_steps": efficiency_summary(capture, "captured", "capture_steps"), "capture_seconds": efficiency_summary(capture, "captured", "capture_seconds"),
            "censored_or_failed_episodes": sum(not bool(row["captured"]) for row in capture),
        },
        "mixed": {
            "episodes": len(mixed), "capture_rate": rate(mixed, "captured"), "collision_rate": rate(mixed, "collision"),
            "post_capture_ce_rate": rate(mixed, "ce_success"), "safe_complete_rate": rate(mixed, "safe_complete"),
            "capture_steps": efficiency_summary(mixed, "captured", "capture_steps"), "capture_seconds": efficiency_summary(mixed, "captured", "capture_seconds"),
            "recovery_steps": efficiency_summary(mixed, "ce_success", "recovery_steps"), "recovery_seconds": efficiency_summary(mixed, "ce_success", "recovery_seconds"),
            "mission_steps": efficiency_summary(mixed, "safe_complete", "mission_steps"), "mission_seconds": efficiency_summary(mixed, "safe_complete", "mission_seconds"),
            "censored_or_failed_episodes": sum(not bool(row["safe_complete"]) for row in mixed),
        },
        "z": {
            "direct_checks": sum(row["direct_checks"] for row in all_diag), "direct_violations": sum(row["direct_violations"] for row in all_diag),
            "direct_visible_z_one_rate": 1.0 - sum(row["direct_violations"] for row in all_diag) / max(sum(row["direct_checks"] for row in all_diag), 1),
            "max_lineage_hop": max((row["max_lineage_hop"] for row in all_diag), default=-1), "post_capture_release_seconds": numeric_summary(release),
            "post_capture_never_release_episodes": sum(row["post_capture_never_released"] for row in all_diag),
            "update_count_checks": sum(row["z_update_count_checks"] for row in all_diag), "update_count_violations": sum(row["z_update_count_violations"] for row in all_diag),
            "update_count_per_decision_exact": all(row["z_update_count_per_decision_exact"] for row in all_diag),
        },
        "actions": {
            "histogram": np.sum([np.asarray(row["policy_diagnostics"]["action_histogram"], dtype=np.int64) for row in records], axis=0).tolist(),
        },
        "token_occupancy": {
            "policy_actual_token_caps": dict(capacity.model_caps),
            "environment_observation_caps": dict(capacity.env_caps),
            "samples": len(capacity.samples), "episodes": len(capacity.episodes),
            "friend_candidate_count": numeric_summary([row["friend_candidates"] for sample in capacity.samples for row in sample["rows"]]),
            "friend_token_occupancy": numeric_summary([row["friend_tokens_seen"] for sample in capacity.samples for row in sample["rows"]]),
            "evader_token_occupancy": numeric_summary([row["evader_tokens_seen"] for sample in capacity.samples for row in sample["rows"]]),
            "obstacle_token_occupancy": numeric_summary([row["obstacle_tokens_seen"] for sample in capacity.samples for row in sample["rows"]]),
            "friend_truncation": {
                "event_count": sum(bool(row["friend_truncated"]) for sample in capacity.samples for row in sample["rows"]),
                "sample_count": sum(len(sample["rows"]) for sample in capacity.samples),
                "episode_count": sum(int(item["friend_truncation_events"] > 0) for item in capacity.episodes),
                "episodes": len(capacity.episodes),
                "max_candidates": max((int(item["max_friend_candidates"]) for item in capacity.episodes), default=0),
            },
            "selection_semantics": {"friend_ordering_mode": "physical_only", "friend_selection": "physical_only sorted local adjacency then first max_pursuer_num", "role_or_z_affects_friend_truncation_priority": False, "padding_is_masked": True, "central_diagnostic_only_padding": "not_applicable_for_IQN_policy"},
            "architecture_modified": False,
        },
    }
    resolver_rows = [row["resolver"] for row in records]
    manifest = {
        "schema": SCHEMA, "status": "complete", "run_id": output.name, "mode": args.mode,
        "branch": formal.git("branch", "--show-current"), "head": formal.git("rev-parse", "HEAD"),
        "config_source": str(config_path), "resolved_config_hash": stable_hash(root_cfg),
        "selection_report": args.selection_report, "selected": selection,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "N": args.pursuers, "num_evaders": args.evaders, "num_obstacles": args.obstacles,
        "episodes_per_scene": int(args.episodes), "scenes": list(SCENES), "seed_base": int(args.seed_base),
        "gif_indices_from_same_20": list(gif_indices), "gif_count": len(gifs), "gifs": gifs,
        "formal_evaluator": {"episode_runner": "tools.run_forward_final_bridge_20260908.run_episode", "z_diagnostics": "tools.iqn_token_matched_20260919.ZDiagnostics", "efficiency_metric": "tools.iqn_token_matched_20260919.efficiency_timing_summary", "policy_quantiles": "fixed_midpoint_32_greedy", "diagnostic_central_schema": "skipped_for_large_swarm_over_12_agents; tensor is diagnostic-only and not a policy input" if args.mode == "large" and int(args.pursuers) > 12 else "enabled"},
        "resolver_contract": {"policy": NORMSENSE_POLICY, "formula": "R=k*sqrt(A_eff/N)", "k": 0.8715, "R_floor": 20.0, "surface_distance_semantics": "surface_clearance", "large_coverage_spawn_rule": "resolved_R" if args.mode == "large" else "historical_spawn_preserved"},
        "resolver_distinct_values": sorted({json.dumps({key: item[key] for key in ("N", "num_obstacles", "A_eff", "raw_R", "resolved_R", "R_floor", "floor_triggered", "map_scale", "free_mask_hash", "obstacle_mask_hash")}, sort_keys=True) for item in resolver_rows}),
        "summary": json_finite(summary), "completed_at": now_local(), "elapsed_seconds": time.monotonic() - run_started,
    }
    atomic_json(output / "report.json", json_finite({**manifest, "records": records}))
    atomic_json(output / "compact_summary.json", json_finite(manifest))
    atomic_json(output / "progress.json", {"status": "complete", "completed_episodes": len(records), "total_episodes": total, "elapsed_seconds": time.monotonic() - run_started})
    print(json.dumps(json_finite(manifest), indent=2, ensure_ascii=False))
    return 0


def capacity_model_caps(model: CoCapIQN) -> dict[str, int]:
    return {"friend_tokens": int(model.config.max_pursuers), "evader_tokens": int(model.config.max_evaders), "obstacle_tokens": int(model.config.max_obstacles)}


def recover_saved_batch(args: argparse.Namespace) -> int:
    """Recover only the report after a post-rollout aggregation failure."""
    output = Path(args.output).resolve()
    records = [json.loads(line) for line in (output / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != int(args.episodes) * len(SCENES):
        raise AssertionError(f"saved formal batch is incomplete: {len(records)} records")
    checkpoint, selection = resolve_checkpoint(args)
    config_path = Path(args.config).resolve()
    root_cfg = formal.resolved(config_path)
    grouped = {scene: [row for row in records if row["scene"] == scene] for scene in SCENES}
    def rate(items: list[dict[str, Any]], key: str) -> float:
        return sum(bool(row.get(key, False)) for row in items) / max(len(items), 1)
    all_diag = [row["z_diagnostics"] for row in records]
    release = [row["release_seconds_max_z_lt_0_1"] for row in all_diag if row["release_seconds_max_z_lt_0_1"] is not None]
    summary: dict[str, Any] = {
        "coverage": {
            "episodes": len(grouped["coverage"]), "strict_ce_rate": rate(grouped["coverage"], "ce_success"), "collision_rate": rate(grouped["coverage"], "collision"),
            "ce_rms": numeric_summary([row["ce_rms"] for row in grouped["coverage"]]), "ce_max": numeric_summary([row["ce_max"] for row in grouped["coverage"]]),
            "area_cv": numeric_summary([row["area_cv"] for row in grouped["coverage"]]),
            "time_to_CE_steps": efficiency_summary(grouped["coverage"], "ce_success", "time_to_strict_CE_steps"), "time_to_CE_seconds": efficiency_summary(grouped["coverage"], "ce_success", "time_to_strict_CE_seconds"),
            "censored_or_failed_episodes": sum(not bool(row["ce_success"]) for row in grouped["coverage"]),
        },
        "capture": {
            "episodes": len(grouped["capture"]), "capture_rate": rate(grouped["capture"], "captured"), "normal_capture_rate": rate(grouped["capture"], "normal_capture"), "stationary_capture_rate": rate(grouped["capture"], "stationary_capture"), "collision_rate": rate(grouped["capture"], "collision"),
            "capture_steps": efficiency_summary(grouped["capture"], "captured", "capture_steps"), "capture_seconds": efficiency_summary(grouped["capture"], "captured", "capture_seconds"), "censored_or_failed_episodes": sum(not bool(row["captured"]) for row in grouped["capture"]),
        },
        "mixed": {
            "episodes": len(grouped["mixed"]), "capture_rate": rate(grouped["mixed"], "captured"), "collision_rate": rate(grouped["mixed"], "collision"), "post_capture_ce_rate": rate(grouped["mixed"], "ce_success"), "safe_complete_rate": rate(grouped["mixed"], "safe_complete"),
            "capture_steps": efficiency_summary(grouped["mixed"], "captured", "capture_steps"), "capture_seconds": efficiency_summary(grouped["mixed"], "captured", "capture_seconds"), "recovery_steps": efficiency_summary(grouped["mixed"], "ce_success", "recovery_steps"), "recovery_seconds": efficiency_summary(grouped["mixed"], "ce_success", "recovery_seconds"), "mission_steps": efficiency_summary(grouped["mixed"], "safe_complete", "mission_steps"), "mission_seconds": efficiency_summary(grouped["mixed"], "safe_complete", "mission_seconds"),
            "censored_or_failed_episodes": sum(not bool(row["safe_complete"]) for row in grouped["mixed"]),
        },
        "z": {
            "direct_checks": sum(row["direct_checks"] for row in all_diag), "direct_violations": sum(row["direct_violations"] for row in all_diag), "direct_visible_z_one_rate": 1.0 - sum(row["direct_violations"] for row in all_diag) / max(sum(row["direct_checks"] for row in all_diag), 1), "max_lineage_hop": max((row["max_lineage_hop"] for row in all_diag), default=-1), "post_capture_release_seconds": numeric_summary(release), "post_capture_never_release_episodes": sum(row["post_capture_never_released"] for row in all_diag), "update_count_checks": sum(row["z_update_count_checks"] for row in all_diag), "update_count_violations": sum(row["z_update_count_violations"] for row in all_diag), "update_count_per_decision_exact": all(row["z_update_count_per_decision_exact"] for row in all_diag),
        },
    }
    occupancy = [row["token_occupancy_episode"] for row in records]
    def recovered_means(key: str) -> list[float]:
        values = []
        for item in occupancy:
            value = item.get(key, {})
            if isinstance(value, Mapping) and value.get("mean") is not None:
                values.append(float(value["mean"]))
        return values
    summary["token_occupancy"] = {
        "aggregation": "recovered_from_persisted_episode_level_occupancy_summaries",
        "friend_candidate_count": numeric_summary(recovered_means("friend_candidate_count")),
        "friend_token_occupancy": numeric_summary(recovered_means("friend_token_occupancy")),
        "evader_token_occupancy": numeric_summary(recovered_means("evader_token_occupancy")),
        "obstacle_token_occupancy": numeric_summary(recovered_means("obstacle_token_occupancy")),
        "friend_truncation": {"event_count": sum(int(item["friend_truncation_events"]) for item in occupancy), "episode_count": sum(int(item["friend_truncation_events"] > 0) for item in occupancy), "episodes": len(occupancy), "max_candidates": max((int(item["max_friend_candidates"]) for item in occupancy), default=0)},
        "policy_actual_token_caps": {"friend_tokens": 8, "evader_tokens": 8, "obstacle_tokens": 5},
        "environment_observation_caps": {"friend_tokens": 8, "evader_tokens": 8, "obstacle_tokens": 5},
        "selection_semantics": {"friend_ordering_mode": "physical_only", "friend_selection": "physical_only sorted local adjacency then first max_pursuer_num", "role_or_z_affects_friend_truncation_priority": False, "padding_is_masked": True, "central_diagnostic_only_padding": "not_applicable_for_IQN_policy"},
        "architecture_modified": False,
    }
    resolver_rows = [row["resolver"] for row in records]
    gif_paths = sorted(output.glob("gifs/*/*.gif"))
    source_paths = sorted(output.glob("gif_sources/*/*.json"))
    gifs = [{"gif": str(path), "source_frames": str(source_paths[i]) if i < len(source_paths) else None} for i, path in enumerate(gif_paths)]
    manifest = {
        "schema": SCHEMA, "status": "complete", "recovered_from_saved_formal_batch": True, "run_id": output.name, "mode": args.mode,
        "branch": formal.git("branch", "--show-current"), "head": formal.git("rev-parse", "HEAD"), "config_source": str(config_path), "resolved_config_hash": stable_hash(root_cfg), "selection_report": args.selection_report, "selected": selection, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "episodes_per_scene": int(args.episodes), "scenes": list(SCENES), "gif_indices_from_same_20": list(range(0, 20, 2)), "gif_count": len(gifs), "gifs": gifs,
        "formal_evaluator": {"episode_runner": "tools.run_forward_final_bridge_20260908.run_episode", "z_diagnostics": "tools.iqn_token_matched_20260919.ZDiagnostics", "efficiency_metric": "tools.iqn_token_matched_20260919.efficiency_timing_summary", "policy_quantiles": "fixed_midpoint_32_greedy"},
        "resolver_contract": {"policy": NORMSENSE_POLICY, "formula": "R=k*sqrt(A_eff/N)", "k": 0.8715, "R_floor": 20.0, "surface_distance_semantics": "surface_clearance", "large_coverage_spawn_rule": "historical_spawn_preserved"},
        "resolver_distinct_values": sorted({json.dumps({key: item[key] for key in ("N", "num_obstacles", "A_eff", "raw_R", "resolved_R", "R_floor", "floor_triggered", "map_scale", "free_mask_hash", "obstacle_mask_hash")}, sort_keys=True) for item in resolver_rows}),
        "summary": json_finite(summary), "completed_at": now_local(),
    }
    atomic_json(output / "report.json", json_finite({**manifest, "records": records}))
    atomic_json(output / "compact_summary.json", json_finite(manifest))
    atomic_json(output / "progress.json", {"status": "complete", "completed_episodes": len(records), "total_episodes": len(records), "recovered": True})
    print(json.dumps(json_finite(manifest), indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("capacity-audit")
    audit.add_argument("--config", required=True)
    audit.add_argument("--checkpoint", required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--pursuers", type=int, required=True)
    audit.add_argument("--evaders", type=int, required=True)
    audit.add_argument("--obstacles", type=int, required=True)
    audit.add_argument("--post-window", type=int, required=True)
    audit.add_argument("--seed-base", type=int, default=2026092301)
    recover = sub.add_parser("recover")
    recover.add_argument("--mode", choices=("native", "downward", "large"), required=True)
    recover.add_argument("--config", required=True)
    recover.add_argument("--selection-report")
    recover.add_argument("--expected-step", type=int)
    recover.add_argument("--checkpoint")
    recover.add_argument("--output", type=Path, required=True)
    recover.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=("native", "downward", "large"), required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--selection-report")
    run.add_argument("--expected-step", type=int)
    run.add_argument("--checkpoint")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    run.add_argument("--seed-base", type=int, default=2026092301)
    run.add_argument("--pursuers", type=int)
    run.add_argument("--evaders", type=int)
    run.add_argument("--obstacles", type=int)
    run.add_argument("--post-window", type=int)
    run.add_argument("--gif-indices", default=",".join(str(x) for x in DEFAULT_GIF_INDICES))
    run.add_argument("--max-gif-frames", type=int, default=300)
    args = parser.parse_args()
    try:
        if args.command == "capacity-audit":
            return run_capacity_audit(args)
        if args.command == "recover":
            return recover_saved_batch(args)
        return run_evaluation(args)
    except Exception as exc:
        output = Path(getattr(args, "output", ROOT / "artifacts/iqn_z05_independent")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "run_status.json", {"schema": SCHEMA, "status": "blocked", "error": repr(exc), "traceback": traceback.format_exc(), "completed_at": now_local()})
        print(json.dumps({"status": "blocked", "error": repr(exc), "output": str(output)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
