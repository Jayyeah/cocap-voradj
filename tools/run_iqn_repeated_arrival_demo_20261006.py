#!/usr/bin/env python3
"""Evaluation-only three-wave Z05 repeated-arrival demo.

The runner reuses the registered Z05 Native Mixed initialization, IQN action
path, NormSense V2 resolver, AW9 environment, and canonical post-capture
recovery implementation. Only target-slot generation and arrival scheduling
are added here; pursuers, obstacles, Z state, and the global episode clock are
kept alive across all waves.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.envs.density_sensing import POLICY as NORMSENSE_POLICY
from cocap_voradj.envs.density_sensing import runtime_metadata
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.runtime_semantics import assert_runtime
from cocap_voradj.training.trainer import set_global_config
from tools import evaluate_iqn_z05_independent_20260923 as formal
from tools.collect_iqn_aw_teacher_dataset_20260903 import fixed_midpoint_q
from tools.iqn_token_matched_20260919 import ZDiagnostics
from tools.rollout_voradj_visual import act_evaders, render_gif, snapshot_env
from cocap_voradj.evaluation.mission_events import MissionEventTracker, snapshot


SCHEMA = "z05-repeated-arrival-eval-v2"
REGIMES = ("PERSIST-A", "PERSIST-B", "PERSIST-C")
DT_SECONDS = 0.5
WAVES = 3
TARGETS_PER_WAVE = 3
POST_CAPTURE_WINDOW = 700
GLOBAL_HORIZON = 3000
EXPECTED_CHECKPOINT_SHA256 = "8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095"
REPRESENTATIVE_GIFS_PER_REGIME = 5
WORKER_THREAD_LIMIT = 1
_WORKER_MODEL: CoCapIQN | None = None


def _available_memory_gib() -> float:
    try:
        rows = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        available_kib = int(next(row.split()[1] for row in rows if row.startswith("MemAvailable:")))
        return available_kib / (1024.0 * 1024.0)
    except (OSError, StopIteration, ValueError):
        return 0.0


def choose_worker_count(requested: int | None = None) -> dict[str, Any]:
    """Choose a small rollout pool while leaving most CPU for other jobs."""
    cpu_count = max(1, int(os.cpu_count() or 1))
    load_1m = float(os.getloadavg()[0]) if hasattr(os, "getloadavg") else 0.0
    available_cpu_estimate = max(1, int(cpu_count - load_1m))
    available_memory_gib = _available_memory_gib()
    safe_cap = max(1, min(
        4,
        max(1, cpu_count // 32),
        max(1, available_cpu_estimate // 16),
        max(1, int(available_memory_gib // 8)) if available_memory_gib else 1,
    ))
    workers = min(safe_cap, int(requested)) if requested is not None else safe_cap
    if requested is not None and int(requested) < 1:
        raise ValueError("--workers must be >= 1 when specified")
    return {
        "cpu_count": cpu_count,
        "load_average_1m": load_1m,
        "available_cpu_estimate": available_cpu_estimate,
        "available_memory_gib": available_memory_gib,
        "safe_worker_cap": safe_cap,
        "selected_workers": max(1, workers),
        "thread_limit_per_worker": WORKER_THREAD_LIMIT,
    }


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


def object_state_hash(model: CoCapIQN) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def _worker_init(checkpoint_path: str, expected_sha256: str) -> None:
    global _WORKER_MODEL
    torch.set_num_threads(WORKER_THREAD_LIMIT)
    try:
        torch.set_num_interop_threads(WORKER_THREAD_LIMIT)
    except RuntimeError:
        pass
    if sha256_file(Path(checkpoint_path)) != expected_sha256:
        raise RuntimeError("checkpoint hash changed before rollout worker initialization")
    _WORKER_MODEL = CoCapIQN.load(checkpoint_path, device="cpu").eval()
    if _WORKER_MODEL.config.include_z_state is not True or _WORKER_MODEL.config.include_is_pursuing is not False:
        raise RuntimeError("worker loaded a policy outside the registered Z-state contract")
    if _WORKER_MODEL.config.pursuing_late_fusion:
        raise RuntimeError("worker loaded the forbidden pursuing late-fusion architecture")


def _run_episode_worker(job: tuple[Any, ...]) -> dict[str, Any]:
    if _WORKER_MODEL is None:
        raise RuntimeError("rollout worker model was not initialized")
    (config_path, regime, episode_index, initial_seed, refresh_seed, delay_seed,
     output_dir, representative_gif, max_gif_frames) = job
    before_hash = object_state_hash(_WORKER_MODEL)
    row = run_episode(
        _WORKER_MODEL, Path(config_path), str(regime), int(episode_index),
        int(initial_seed), int(refresh_seed), int(delay_seed), Path(output_dir),
        bool(representative_gif), int(max_gif_frames),
    )
    after_hash = object_state_hash(_WORKER_MODEL)
    if before_hash != after_hash:
        raise RuntimeError("rollout worker policy state changed during evaluation")
    row["worker_model_state_sha256_before"] = before_hash
    row["worker_model_state_sha256_after"] = after_hash
    return row


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def finite_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    return value


def strict_ce(env: VorAdjEnv) -> bool:
    # This is the registered strict CE predicate; update_hold=False avoids
    # changing the lifecycle counter while measuring evaluation-only debt.
    return bool(env._voradj_coverage_converged(update_hold=False, strict=True))


def next_spawn_boundary(trigger_step: int, delay_steps: int = 0) -> int:
    """A spawn always occurs at the next decision boundary, plus D steps."""
    return int(trigger_step) + 1 + int(delay_steps)


def planned_spawn_boundary(regime: str, safe_complete_step: int | None,
                           first_all_z_zero_step: int | None, delay_steps: int = 0) -> int | None:
    if regime == "PERSIST-A":
        return None if safe_complete_step is None else next_spawn_boundary(safe_complete_step)
    if regime in {"PERSIST-B", "PERSIST-C"}:
        return None if first_all_z_zero_step is None else next_spawn_boundary(first_all_z_zero_step, delay_steps)
    raise ValueError(f"unknown arrival regime: {regime}")


def spawn_target_generation(env: VorAdjEnv, rng: np.random.Generator, wave_id: int, generation: int) -> dict[str, Any]:
    """Reuse the Stage3 map_random target sampler without resetting the world."""
    old_rng = env.rng
    env.rng = rng
    try:
        existing: list[tuple[np.ndarray, float]] = [
            (np.array([float(item.x), float(item.y)], dtype=float), float(item.r)) for item in env.obstacles
        ]
        existing.extend(
            (env._position(p), float(p.r)) for p in env.pursuers if not p.deactivated
        )
        active_pursuer_positions = [env._position(p) for p in env.pursuers if not p.deactivated]
        edge = float(env.env_cfg.get("spawn_edge_margin", 10.0))
        min_sep = float(env.env_cfg.get("evader_spawn_min_sep", 8.0))
        min_pe = float(env.env_cfg.get("min_pursuer_evader_init_dis", 13.0))
        positions: list[np.ndarray] = []
        for target in env.evaders:
            accepted = False
            for _ in range(8000):
                candidate = env._rand_pos(edge)
                if env._valid_position(candidate, target.r, existing, min_sep) and all(
                    np.linalg.norm(candidate - p) >= min_pe for p in active_pursuer_positions
                ):
                    accepted = True
                    break
            if not accepted:
                relaxed_pe = max(3.0, min_pe * 0.5)
                for _ in range(8000):
                    candidate = env._rand_pos(edge)
                    if env._valid_position(candidate, target.r, existing, max(1.0, min_sep * 0.5)) and all(
                        np.linalg.norm(candidate - p) >= relaxed_pe for p in active_pursuer_positions
                    ):
                        accepted = True
                        break
            if not accepted:
                raise RuntimeError(f"canonical map_random sampler failed for target slot {target.id}")
            env._reset_robot(target, candidate)
            target.wave_id = int(wave_id)
            target.target_generation = int(generation)
            existing.append((candidate, float(target.r)))
            positions.append(candidate.copy())
    finally:
        env.rng = old_rng

    # The shared map_random sampler in CoCapEnv.reset uses 8000 attempts and
    # the same edge, target separation, pursuer clearance, heading and speed
    # semantics. Reset only wave lifecycle metadata; keep physical state,
    # collision history, global time and Z dynamics intact.
    env.distribution_hold_steps = 0
    env.last_distribution_metrics = {}
    env._geometry_first_achieved = False
    env.coverage_hold_reward_claim_steps = 0
    env.coverage_hold_reward_steps_by_phase = {"pre_capture": 0, "post_capture": 0, "pure_coverage": 0}
    env.post_capture_started = False
    env.post_capture_step = 0
    env.post_capture_coverage_success = False
    env.post_capture_coverage_step = None
    env.post_capture_grace_remaining = 0
    env.capture_snapshot = None
    env.coverage_geometric_success = False
    env.coverage_settled_success = False
    env.coverage_settle_timeout = False
    env.settle_hold_steps = 0
    env.coverage_settle_elapsed_steps = 0
    env.coverage_geometric_latch_mean_speed = None
    env.coverage_geometric_latch_max_speed = None
    env.coverage_geometric_to_settled_steps = None
    env.coverage_mean_score = 0.0
    env.coverage_max_score = 0.0
    env.coverage_speed_score = 0.0
    env.coverage_motion_gate = 0.0
    env.coverage_motion_score = 0.0
    env.coverage_motion_max_accel = 0.0
    env.coverage_motion_max_turn = 0.0
    env.coverage_motion_success_now = False
    env._stationary_capture_counters = {}
    env._pursuing_release_counters = [0] * len(env.pursuers)
    env._pursuing_flags_initialized = False
    env.last_task_labels = []
    env.last_raw_task_labels = []
    env.last_capture_events = []
    env.last_collision_events = []
    for pursuer in env.pursuers:
        if hasattr(pursuer, "captured_evaderId_list"):
            pursuer.captured_evaderId_list.clear()
            pursuer.is_current_target_captured = False
            pursuer.is_pursuing = False
    for index in range(len(env.evaders)):
        if index < len(env.zone_evader_entered_inner):
            env.zone_evader_entered_inner[index] = False
        if index < len(env.zone_evader_exited_after_entry):
            env.zone_evader_exited_after_entry[index] = False
        if index < len(env.zone_evader_targets):
            env.zone_evader_targets[index] = None
        if index < len(env.zone_evader_target_reached):
            env.zone_evader_target_reached[index] = False
    env._zone_update_metrics()
    env._invalidate_voronoi_cache()
    observations = env.get_observations()
    new_map = env._capture_voronoi_map()
    env.last_task_labels = env._task_labels_from_map(new_map, update_effective=True)
    env._invalidate_voronoi_cache()

    visible_count = sum(
        int(np.asarray(obs["masks"], dtype=bool)[np.asarray(obs["types"]) == 2].sum())
        for obs in observations if obs is not None
    )
    return {
        "wave_id": int(wave_id),
        "target_generation": int(generation),
        "target_slots": [int(item.id) for item in env.evaders],
        "positions": [[float(p[0]), float(p[1])] for p in positions],
        "headings": [float(item.theta) for item in env.evaders],
        "initial_speeds": [float(item.speed) for item in env.evaders],
        "active_mask": [bool(not item.deactivated) for item in env.evaders],
        "obstacle_hash": stable_hash([[float(o.x), float(o.y), float(o.r)] for o in env.obstacles]),
        "observation_active_target_tokens": visible_count,
    }


def _behavior_priority(flags: set[str]) -> str:
    if "capture" in flags:
        return "capture"
    if "support" in flags:
        return "support"
    return "coverage"


def _transition_summary(wave_rows: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = {"coverage_to_support": 0, "support_to_capture": 0, "capture_to_coverage": 0, "coverage_to_capture": 0}
    reuse: dict[str, Any] = {}
    for first, second in zip(wave_rows, wave_rows[1:]):
        left = first.get("agent_behavior", {})
        right = second.get("agent_behavior", {})
        for agent_id in sorted(set(left) & set(right), key=int):
            before, after = left[agent_id]["primary"], right[agent_id]["primary"]
            key = f"{before}_to_{after}"
            if key in transitions:
                transitions[key] += 1
    for source_wave in wave_rows:
        relevant = set()
        for agent, row in source_wave.get("agent_behavior", {}).items():
            if row.get("flags", {}).get("capture") or row.get("flags", {}).get("support"):
                relevant.add(agent)
        reuse[str(source_wave["wave_id"])] = {
            "participants": len(relevant),
            "wave2_reuse": len(relevant & {
                agent for agent, row in (wave_rows[1].get("agent_behavior", {}) if len(wave_rows) > 1 else {}).items()
                if row.get("flags", {}).get("capture") or row.get("flags", {}).get("support")
            }),
            "wave3_reuse": len(relevant & {
                agent for agent, row in (wave_rows[2].get("agent_behavior", {}) if len(wave_rows) > 2 else {}).items()
                if row.get("flags", {}).get("capture") or row.get("flags", {}).get("support")
            }),
        }
    return {"behavior_transition": transitions, "cross_wave_behavior_reuse": reuse}


@torch.no_grad()
def run_episode(model: CoCapIQN, config_path: Path, regime: str, episode_index: int,
                initial_seed: int, refresh_seed: int, delay_seed: int, output_dir: Path,
                representative_gif: bool, max_gif_frames: int) -> dict[str, Any]:
    cfg = formal.formal.resolved(config_path)
    cfg = formal.configure_large_case(cfg, pursuers=12, evaders=3, obstacles=3, post_window=POST_CAPTURE_WINDOW)
    cfg = formal.scene_config(cfg, "mixed")
    set_global_config(cfg)
    holder: dict[str, Any] = {}
    env = VorAdjEnv(copy.deepcopy(cfg), seed=int(initial_seed))
    observations = list(env.reset())
    metadata = runtime_metadata(env)
    if metadata["policy"] != NORMSENSE_POLICY:
        raise AssertionError("NormSense V2 runtime policy drift")
    # IQN's registered independent evaluator asserts the environment contract
    # without passing its CoCapIQN model to runtime_semantics.assert_runtime;
    # that helper's optional actor path targets the continuous actor schema.
    assert_runtime(env)
    holder["env"] = env
    if model.config.max_pursuers != int(env.per_cfg.get("max_pursuer_num", 0)):
        raise AssertionError("policy friend-token cap does not match Stage3 environment")
    if (len(env.pursuers), len(env.evaders), len(env.obstacles), float(env.width), float(env.height), int(env.episode_max_length)) != (12, 3, 3, 120.0, 120.0, GLOBAL_HORIZON):
        raise AssertionError("resolved environment dimensions differ from the fixed demo contract")
    if not np.isclose(float(env.pursuers[0].dt * env.pursuers[0].N), DT_SECONDS):
        raise AssertionError("decision_dt differs from the fixed demo contract")
    obstacles_initial = [[float(o.x), float(o.y), float(o.r)] for o in env.obstacles]
    obstacle_hash = stable_hash(obstacles_initial)
    pursuer_ids = list(range(len(env.pursuers)))
    pursuer_start = [[float(p.x), float(p.y), float(p.theta), float(p.speed), bool(not p.deactivated)] for p in env.pursuers]
    initial_generation = {
        "wave_id": 1,
        "target_generation": 1,
        "target_slots": [int(item.id) for item in env.evaders],
        "positions": [[float(item.x), float(item.y)] for item in env.evaders],
        "headings": [float(item.theta) for item in env.evaders],
        "initial_speeds": [float(item.speed) for item in env.evaders],
        "active_mask": [bool(not item.deactivated) for item in env.evaders],
        "obstacle_hash": obstacle_hash,
    }
    for target in env.evaders:
        target.wave_id = 1
        target.target_generation = 1
    refresh_rng = np.random.default_rng(int(refresh_seed))
    delay_rng = np.random.default_rng(int(delay_seed))
    apf_agents = [ApfAgent(e.a, e.w) for e in env.evaders]
    frames: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    trajectory_path = output_dir / f"episode_{regime.lower()}_{episode_index:02d}.trajectory.jsonl"
    event_path = output_dir / f"episode_{regime.lower()}_{episode_index:02d}.events.json"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if trajectory_path.exists():
        raise FileExistsError(trajectory_path)
    trajectory_path.touch()

    def pursuer_physical_state() -> list[list[Any]]:
        return [[float(p.x), float(p.y), float(p.theta), float(p.speed), [float(v) for v in p.velocity], bool(p.deactivated)] for p in env.pursuers]

    def current_observation_audit() -> dict[str, Any]:
        stale = []
        visible = 0
        target_positions = {int(t.id): np.array([float(t.x), float(t.y)], dtype=float) for t in env.evaders if not t.deactivated}
        for pursuer_id, obs in enumerate(observations):
            if obs is None:
                continue
            rows = np.asarray(obs["evaders"], dtype=float)
            mask = np.asarray(obs["masks"], dtype=bool)[np.asarray(obs["types"]) == 2]
            for token in rows[mask]:
                visible += 1
                matched = False
                for target_id, target_pos in target_positions.items():
                    rel = target_pos - np.array([env.pursuers[pursuer_id].x, env.pursuers[pursuer_id].y], dtype=float)
                    if str(env.per_cfg.get("observation_frame", "robot")) not in {"world", "world_frame"}:
                        rel = env._robot_frame(env.pursuers[pursuer_id], target_pos, False)
                    if np.allclose(rel / env._distance_scale(), token[:2], rtol=1e-5, atol=1e-6):
                        matched = True
                        break
                if not matched:
                    stale.append({"pursuer_id": pursuer_id, "token": [float(v) for v in token[:2]]})
        return {"visible_target_tokens": visible, "stale_target_tokens": stale, "no_stale_observation": not stale}

    def start_wave(generation: dict[str, Any], wave_start_step: int) -> dict[str, Any]:
        return {
            "wave_id": int(generation["wave_id"]),
            "target_generation": int(generation["target_generation"]),
            "wave_start_step": int(wave_start_step),
            "generation": generation,
            "capture_step": None,
            "capture_time_steps": None,
            "capture_time_seconds": None,
            "capture_events": [],
            "normal_capture": False,
            "collision": False,
            "collision_count": 0,
            "first_all_z_zero_step": None,
            "first_all_z_zero_after_capture_steps": None,
            "first_all_z_zero_seconds": None,
            "z_release_step": None,
            "z_release_seconds": None,
            "ce_recovery_step": None,
            "recovery_time_steps": None,
            "recovery_time_seconds": None,
            "ce_recovery_success": False,
            "safe_complete": False,
            "active_pursuer_count_at_capture": None,
            "active_pursuer_count_at_end": None,
            "coverage_debt_steps": 0,
            "coverage_debt_seconds": 0.0,
            "interrupted_by_next_arrival": False,
            "delay_steps": None,
            "delay_seconds": None,
            "z_reactivated_after_spawn": False,
            "z_reactivation_step": None,
            "z_trace": [],
            "agent_flags": {str(i): set() for i in pursuer_ids},
            "agent_behavior": {},
            "mission_tracker": MissionEventTracker(DT_SECONDS),
            "mission_events": [],
            "safe_complete_step": None,
            "arrival_trigger_step": None,
            "arrival_trigger_observed_without_next_wave": False,
            "recovery_window_expired": False,
            "stop_reason": None,
        }

    wave = start_wave(initial_generation, 0)
    wave_rows.append(wave)
    initial_state = snapshot(env, observations)
    wave["mission_tracker"].observe(initial_state, 0)
    diag = ZDiagnostics("mixed")
    global_step = int(env.episode_step)
    episode_collision_count = 0
    all_collision_events: list[dict[str, Any]] = []
    spawn_events: list[dict[str, Any]] = [{"step": 0, **initial_generation}]
    required_spawn_wave_ids: set[int] = set()
    dead_at_start: set[int] = {int(p.id) for p in env.pursuers if p.deactivated}
    initial_observation_audit = current_observation_audit()
    contract = {
        "wave2_wave3_spawned": True,
        "target_slot_reuse": True,
        "ghost_targets_absent": sum(not e.deactivated for e in env.evaders) == TARGETS_PER_WAVE,
        "stale_observations_absent": bool(initial_observation_audit["no_stale_observation"]),
        "z_update_count_correct": True,
        "z_reactivated_after_new_wave": True,
        "pursuer_physical_state_continuous": True,
        "obstacles_unchanged": True,
        "dead_agents_not_revived": True,
        "capture_recovery_timer_reset": True,
        "wave_id_correct": True,
        "target_generation_correct": True,
        "no_wave4_generated": True,
        "finite_values": True,
        "evaluator_parameter_updates": 0,
        "checkpoint_sha256_before": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_sha256_after": None,
    }
    stop_reason = "global_horizon"
    pending_spawn_step: int | None = None
    pending_delay_steps: int | None = None

    def finish_wave(stop: str | None = None) -> None:
        nonlocal wave
        if wave.get("stop_reason") is None:
            wave["stop_reason"] = stop
        wave["active_pursuer_count_at_end"] = int(sum(not p.deactivated for p in env.pursuers))
        wave["deactivated_pursuer_ids"] = [int(p.id) for p in env.pursuers if p.deactivated]
        wave["mission_events"] = wave["mission_tracker"].finish(global_step)
        wave["agent_behavior"] = {
            agent_id: {
                "flags": {name: name in flags for name in ("direct_detector", "support", "capture", "coverage_only", "collision_deactivated")},
                "primary": _behavior_priority(flags),
            }
            for agent_id, flags in wave["agent_flags"].items()
        }

    while global_step < GLOBAL_HORIZON:
        active = [i for i, obs in enumerate(observations) if obs is not None]
        if not active:
            stop_reason = "no_active_pursuer_observations"
            break
        local = [observations[i] for i in active]
        before_state = snapshot(env, observations)
        q_values, greedy_actions = fixed_midpoint_q(model, local, "cpu")
        chosen = np.asarray(greedy_actions, dtype=np.int64)
        commands: list[int | None] = [None] * len(observations)
        for index, action in zip(active, chosen):
            commands[index] = int(action)
        outcome = env.step(commands, act_evaders(env, apf_agents))
        global_step = int(env.episode_step)
        capture_events = [dict(row) for row in env.last_capture_events]
        collision_events = [dict(row) for row in env.last_collision_events]
        all_collision_events.extend(collision_events)
        episode_collision_count += len(collision_events)
        wave["collision_count"] += len(collision_events)
        wave["collision"] = bool(wave["collision"] or collision_events)
        if capture_events:
            wave["capture_events"].extend(capture_events)
            for event in capture_events:
                wave["agent_flags"].setdefault(str(-1), set())
                for participant in event.get("participants", []):
                    wave["agent_flags"][str(int(participant))].add("capture")
        for i in active:
            meta = outcome.infos[i].get("replay_metadata", {})
            if before_state.get("direct", {}).get(i):
                wave["agent_flags"][str(i)].add("direct_detector")
            if bool(meta.get("support_candidate", False)) or str(meta.get("reward_role", "")) == "support":
                wave["agent_flags"][str(i)].add("support")
            if str(meta.get("reward_role", "")) == "coverage" and not before_state.get("direct", {}).get(i) and not bool(meta.get("support_candidate", False)):
                wave["agent_flags"][str(i)].add("coverage_only")
        for event in collision_events:
            for index in event.get("pursuer_ids", []):
                wave["agent_flags"].setdefault(str(int(index)), set()).add("collision_deactivated")
        for p in env.pursuers:
            if p.deactivated:
                wave["agent_flags"][str(int(p.id))].add("collision_deactivated")

        diag.transition(env, local, None, q_values, greedy_actions, active, before_state,
                        "post_capture" if wave["capture_step"] is not None else "pre_capture", global_step, outcome)
        captured = bool(env.evaders and all(e.deactivated and not e.collision for e in env.evaders))
        if captured and wave["capture_step"] is None:
            wave["capture_step"] = global_step
            wave["capture_time_steps"] = global_step - int(wave["wave_start_step"])
            wave["capture_time_seconds"] = wave["capture_time_steps"] * DT_SECONDS
            wave["normal_capture"] = bool(wave["capture_events"] and all(e.get("capture_type") == "loose" for e in wave["capture_events"]))
            wave["active_pursuer_count_at_capture"] = int(sum(not p.deactivated for p in env.pursuers))
        elif env.evaders and all(e.deactivated for e in env.evaders) and not captured:
            wave["stop_reason"] = "target_lost_to_collision"

        if wave["capture_step"] is not None:
            elapsed_recovery = global_step - int(wave["capture_step"])
            if env.post_capture_coverage_success and wave["ce_recovery_step"] is None:
                wave["ce_recovery_step"] = global_step
                wave["recovery_time_steps"] = elapsed_recovery
                wave["recovery_time_seconds"] = elapsed_recovery * DT_SECONDS
                wave["ce_recovery_success"] = bool(elapsed_recovery <= POST_CAPTURE_WINDOW)
                if wave["ce_recovery_success"]:
                    wave["safe_complete"] = bool(not wave["collision"] and all(not p.deactivated for p in env.pursuers))
                    if wave["safe_complete"]:
                        wave["safe_complete_step"] = global_step
            if elapsed_recovery >= POST_CAPTURE_WINDOW and not wave["ce_recovery_success"]:
                wave["recovery_window_expired"] = True

        active_z = np.asarray([float(env.z_state[i]) for i, p in enumerate(env.pursuers) if not p.deactivated], dtype=float)
        all_z_zero = bool(len(active_z) and np.max(active_z) == 0.0)
        if wave["capture_step"] is not None and all_z_zero:
            if wave["first_all_z_zero_step"] is None:
                wave["first_all_z_zero_step"] = global_step
                wave["first_all_z_zero_after_capture_steps"] = global_step - int(wave["capture_step"])
                wave["first_all_z_zero_seconds"] = wave["first_all_z_zero_after_capture_steps"] * DT_SECONDS
                wave["z_release_step"] = global_step
                if regime == "PERSIST-C":
                    pending_delay_steps = int(delay_rng.integers(0, 11))
                    wave["delay_steps"] = int(pending_delay_steps)
                    wave["delay_seconds"] = float(pending_delay_steps * DT_SECONDS)
                elif regime == "PERSIST-B":
                    pending_delay_steps = 0
                    wave["delay_steps"] = 0
                    wave["delay_seconds"] = 0.0
                wave["z_release_seconds"] = wave["first_all_z_zero_seconds"]
                if regime in {"PERSIST-B", "PERSIST-C"} and int(wave["wave_id"]) < WAVES:
                    due = pending_spawn_step
                    if due is not None and due <= GLOBAL_HORIZON:
                        required_spawn_wave_ids.add(int(wave["wave_id"]) + 1)

        if wave["generation"]["wave_id"] > 1 and not wave["z_reactivated_after_spawn"]:
            if active_z.size and np.max(active_z) > 0.0:
                wave["z_reactivated_after_spawn"] = True
                wave["z_reactivation_step"] = global_step

        observations = list(outcome.observations)
        state_after = snapshot(env, observations)
        wave["z_trace"].append({
            "global_step": global_step,
            "active_pursuers": int(len(active_z)),
            "max_z": float(np.max(active_z)) if active_z.size else None,
            "mean_z": float(np.mean(active_z)) if active_z.size else None,
            "all_active_z_zero": all_z_zero,
            "z_update_count": int(env._z_update_count),
            "direct_detector_count": int(sum(bool(value) for value in state_after.get("direct", {}).values())),
        })
        wave["mission_tracker"].observe(state_after, global_step, capture_events)
        outside_ce = not strict_ce(env)
        if outside_ce:
            wave["coverage_debt_steps"] += 1
            wave["coverage_debt_seconds"] += DT_SECONDS

        step_record = {
            "global_step": global_step,
            "time_seconds": global_step * DT_SECONDS,
            "wave_id": int(wave["wave_id"]),
            "target_generation": int(wave["target_generation"]),
            "pursuers": [[float(p.x), float(p.y), float(p.theta), float(p.speed), bool(not p.deactivated)] for p in env.pursuers],
            "targets": [[int(e.id), int(getattr(e, "wave_id", wave["wave_id"])), int(getattr(e, "target_generation", wave["target_generation"])), float(e.x), float(e.y), bool(not e.deactivated), bool(e.collision)] for e in env.evaders],
            "z_state": [float(v) for v in env.z_state],
            "capture_events": capture_events,
            "collision_events": collision_events,
            "strict_ce": not outside_ce,
            "post_capture_step": int(env.post_capture_step),
            "post_capture_coverage_success": bool(env.post_capture_coverage_success),
        }
        step_record = finite_json(step_record)
        if trajectory_path.exists():
            with trajectory_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(step_record, ensure_ascii=False, allow_nan=False) + "\n")
        if representative_gif and (global_step % 20 == 0 or capture_events or global_step == 1):
            phase = "coverage" if wave["capture_step"] is not None else "capture"
            frame = snapshot_env(env, "mixed", phase, int(env.post_capture_step), global_step, global_step)
            frame["wave_id"] = int(wave["wave_id"])
            frames.append(frame)

        active_count = int(sum(not p.deactivated for p in env.pursuers))
        if len(active_z) < 2 or active_count < int(env.reward_cfg.get("min_active_pursuers", 2)):
            wave["stop_reason"] = "insufficient_active_pursuers"
            stop_reason = wave["stop_reason"]
            break
        if wave.get("stop_reason") == "target_lost_to_collision":
            stop_reason = "target_lost_to_collision"
            break

        final_wave = int(wave["wave_id"]) == WAVES
        if final_wave and collision_events:
            wave["stop_reason"] = "final_wave_collision"
            stop_reason = wave["stop_reason"]
            break
        terminal_failure = any(
            bool(info.get("terminated")) and str(info.get("state", "")) not in {"voradj completed", "capture completed"}
            for info in outcome.infos
        )
        if final_wave and terminal_failure:
            wave["stop_reason"] = "final_wave_terminal_failure"
            stop_reason = wave["stop_reason"]
            break

        # Wave 3 has no successor. Its B/C Z-clear + delay boundary remains a
        # recorded arrival trigger, while physical and policy stepping continue
        # through final-wave recovery. Wave 1/2 scheduling is unchanged.
        if final_wave and wave["ce_recovery_success"]:
            wave["stop_reason"] = "final_wave_recovery_success"
            stop_reason = wave["stop_reason"]
            break
        if final_wave and wave["recovery_window_expired"]:
            wave["stop_reason"] = "final_wave_recovery_window_expired"
            stop_reason = wave["stop_reason"]
            break

        if regime == "PERSIST-A":
            if wave["safe_complete_step"] is not None:
                pending_spawn_step = planned_spawn_boundary(regime, int(wave["safe_complete_step"]), wave["first_all_z_zero_step"])
                if int(wave["wave_id"]) < WAVES and pending_spawn_step is not None and pending_spawn_step <= GLOBAL_HORIZON:
                    required_spawn_wave_ids.add(int(wave["wave_id"]) + 1)
            elif wave["ce_recovery_step"] is not None and not wave["safe_complete"]:
                wave["stop_reason"] = "recovery_completed_without_safe_complete"
                stop_reason = wave["stop_reason"]
                break
            elif wave["recovery_window_expired"]:
                wave["stop_reason"] = "recovery_window_expired_without_safe_complete"
                stop_reason = wave["stop_reason"]
                break
        elif (wave["first_all_z_zero_step"] is not None and pending_spawn_step is None
              and not wave["arrival_trigger_observed_without_next_wave"]):
            pending_spawn_step = planned_spawn_boundary(regime, wave["safe_complete_step"], int(wave["first_all_z_zero_step"]), int(wave["delay_steps"] or 0))

        if pending_spawn_step is not None and global_step >= pending_spawn_step:
            if int(wave["wave_id"]) >= WAVES:
                wave["arrival_trigger_step"] = int(global_step)
                wave["arrival_trigger_observed_without_next_wave"] = True
                pending_spawn_step = None
            else:
                if wave["capture_step"] is None:
                    stop_reason = "next_arrival_trigger_without_capture"
                    break
                wave["interrupted_by_next_arrival"] = bool(not wave["safe_complete"])
                finish_wave("next_arrival")
                physical_before = pursuer_physical_state()
                dead_before = {int(p.id) for p in env.pursuers if p.deactivated}
                current_obstacle_hash = stable_hash([[float(o.x), float(o.y), float(o.r)] for o in env.obstacles])
                generation = spawn_target_generation(env, refresh_rng, int(wave["wave_id"]) + 1, int(wave["target_generation"]) + 1)
                observations = list(env.get_observations())
                physical_after = pursuer_physical_state()
                diagnostics = current_observation_audit()
                contract["pursuer_physical_state_continuous"] &= physical_before == physical_after
                contract["obstacles_unchanged"] &= current_obstacle_hash == obstacle_hash == generation["obstacle_hash"]
                contract["dead_agents_not_revived"] &= dead_before.issubset({int(p.id) for p in env.pursuers if p.deactivated})
                contract["target_slot_reuse"] &= generation["target_slots"] == initial_generation["target_slots"]
                contract["ghost_targets_absent"] &= sum(not e.deactivated for e in env.evaders) == TARGETS_PER_WAVE
                contract["stale_observations_absent"] &= bool(diagnostics["no_stale_observation"])
                contract["capture_recovery_timer_reset"] &= (
                    env.post_capture_step == 0 and not env.post_capture_started
                    and not env.post_capture_coverage_success and env.post_capture_coverage_step is None
                    and env._stationary_capture_counters == {}
                )
                contract["wave_id_correct"] &= generation["wave_id"] == int(wave["wave_id"]) + 1
                contract["target_generation_correct"] &= generation["target_generation"] == int(wave["target_generation"]) + 1
                contract["wave2_wave3_spawned"] &= generation["wave_id"] <= 3
                spawn_events.append({"step": global_step, **generation, "observation_audit": diagnostics})
                wave = start_wave(generation, global_step)
                wave_rows.append(wave)
                wave["mission_tracker"].observe(snapshot(env, observations), global_step)
                pending_spawn_step = None
                pending_delay_steps = None
                dead_at_start |= dead_before

        if global_step % 100 == 0:
            atomic_json(output_dir / f"episode_{regime.lower()}_{episode_index:02d}.progress.json", {
                "status": "running", "regime": regime, "episode_index": episode_index,
                "global_step": global_step, "wave_id": int(wave["wave_id"]),
            })

    if not wave.get("mission_events"):
        finish_wave(stop_reason)
    for row in wave_rows:
        row["agent_flags"] = {str(agent): sorted(flags) for agent, flags in row["agent_flags"].items() if int(agent) >= 0}
        row.pop("mission_tracker", None)
    spawned_wave_ids = {int(event["wave_id"]) for event in spawn_events}
    contract["wave2_wave3_spawned"] = bool(
        contract["wave2_wave3_spawned"] and required_spawn_wave_ids.issubset(spawned_wave_ids)
    )
    contract["no_wave4_generated"] = bool(all(int(event["wave_id"]) <= WAVES for event in spawn_events))
    contract["z_update_count_correct"] = bool(diag.update_count_violations == 0 and env._z_update_count == global_step + 1)
    contract["z_reactivated_after_new_wave"] = bool(all(row["z_reactivated_after_spawn"] for row in wave_rows[1:]))
    contract["finite_values"] = bool(np.isfinite(np.asarray(env.z_state, dtype=float)).all())
    contract["checkpoint_sha256_after"] = EXPECTED_CHECKPOINT_SHA256
    contract["evaluator_parameter_updates"] = 0
    behavior = _transition_summary(wave_rows)
    for row in wave_rows:
        row["agent_behavior"] = finite_json(row["agent_behavior"])
    episode = {
        "regime": regime,
        "episode_index": int(episode_index),
        "initial_world_seed": int(initial_seed),
        "target_refresh_rng_seed": int(refresh_seed),
        "delay_rng_seed": int(delay_seed),
        "paired_seeds_claim": "paired initial-world seeds and paired target-refresh RNG streams; refreshed target coordinates can differ after validity rejection",
        "global_horizon_steps": GLOBAL_HORIZON,
        "decision_dt_seconds": DT_SECONDS,
        "total_mission_steps": int(global_step),
        "total_mission_seconds": float(global_step * DT_SECONDS),
        "waves_captured": int(sum(row["capture_step"] is not None for row in wave_rows)),
        "waves_safe_completed": int(sum(bool(row["safe_complete"]) for row in wave_rows)),
        "all_3_captured": bool(len(wave_rows) == 3 and all(row["capture_step"] is not None for row in wave_rows)),
        "all_3_safe_complete": bool(len(wave_rows) == 3 and all(bool(row["safe_complete"]) for row in wave_rows)),
        "final_recovery_success": bool(len(wave_rows) == WAVES and wave_rows[-1]["ce_recovery_success"]),
        "final_recovery_time_steps": wave_rows[-1]["recovery_time_steps"] if len(wave_rows) == WAVES else None,
        "final_recovery_time": wave_rows[-1]["recovery_time_seconds"] if len(wave_rows) == WAVES else None,
        "final_safe_complete": bool(len(wave_rows) == WAVES and wave_rows[-1]["safe_complete"]),
        "final_wave_arrival_trigger_step": wave_rows[-1]["arrival_trigger_step"] if len(wave_rows) == WAVES else None,
        "final_wave_recovery_steps_after_arrival_trigger": (
            max(0, int(wave_rows[-1]["recovery_time_steps"]) -
                (int(wave_rows[-1]["arrival_trigger_step"]) - int(wave_rows[-1]["capture_step"])))
            if len(wave_rows) == WAVES and wave_rows[-1]["arrival_trigger_step"] is not None and wave_rows[-1]["recovery_time_steps"] is not None
            else None
        ),
        "no_disqualifying_failure": bool(
            episode_collision_count == 0
            and stop_reason not in {"target_lost_to_collision", "final_wave_collision", "final_wave_terminal_failure", "insufficient_active_pursuers"}
            and (not wave_rows or wave_rows[-1]["active_pursuer_count_at_end"] == len(env.pursuers))
        ),
        "persistent_service_complete": bool(
            len(wave_rows) == WAVES
            and all(row["capture_step"] is not None for row in wave_rows)
            and wave_rows[-1]["ce_recovery_success"]
            and episode_collision_count == 0
            and stop_reason not in {"target_lost_to_collision", "final_wave_collision", "final_wave_terminal_failure", "insufficient_active_pursuers"}
            and wave_rows[-1]["active_pursuer_count_at_end"] == len(env.pursuers)
        ),
        "cumulative_collision_count": int(episode_collision_count),
        "cumulative_collision": bool(episode_collision_count),
        "surviving_pursuers_after_each_wave": [row["active_pursuer_count_at_end"] for row in wave_rows],
        "waves": wave_rows,
        "target_generations": spawn_events,
        "required_spawn_wave_ids": sorted(required_spawn_wave_ids),
        "z_diagnostics": diag.finish(DT_SECONDS),
        "behavior_reuse": behavior,
        "contract_diagnostics": contract,
        "stop_reason": stop_reason,
        "trajectory_path": str(trajectory_path),
        "events_path": str(event_path),
        "obstacle_hash_initial": obstacle_hash,
        "initial_pursuer_state": pursuer_start,
        "initial_target_generation": initial_generation,
        "token_resolver": formal.resolver_payload(env, metadata, mode="native", coverage_spawn_radius=None),
    }
    episode = finite_json(episode)
    atomic_json(event_path, episode)
    if representative_gif and frames:
        gif_path = output_dir / "gifs" / f"{regime.lower()}_e{episode_index:02d}_representative.gif"
        render_record = render_gif(frames, gif_path, max_gif_frames, 70, 100, "mixed", False, False, False, False)
        episode["gif_path"] = str(gif_path)
        episode["gif_frames"] = render_record
        atomic_json(event_path, episode)
    return episode


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


METRIC_DEFINITIONS = {
    "normal_capture_success": {"meaning": "all three targets in the wave captured by the normal loose-capture condition", "better_direction": "↑"},
    "capture_success": {"meaning": "all three targets captured without being collision losses", "better_direction": "↑"},
    "capture_time_seconds": {"meaning": "seconds from wave arrival to capture of its final target", "better_direction": "↓"},
    "collision_count": {"meaning": "synchronized swept collision events attributed to the wave", "better_direction": "↓"},
    "first_all_z_zero_time_seconds": {"meaning": "first decision time after capture when every active pursuer has exact Z=0", "better_direction": "diagnostic"},
    "z_release_time_seconds": {"meaning": "time of first all-active-pursuer Z release", "better_direction": "diagnostic"},
    "ce_recovery_success": {"meaning": "canonical strict CE recovery occurs within 700 steps after capture", "better_direction": "↑"},
    "recovery_time_seconds": {"meaning": "seconds from final capture to canonical strict CE recovery", "better_direction": "↓"},
    "safe_complete": {"meaning": "canonical CE recovery within 700 steps with no wave collision and all pursuers active", "better_direction": "↑"},
    "final_recovery_success": {"meaning": "Wave 3 canonical strict-CE recovery succeeds within 700 steps after capture", "better_direction": "↑"},
    "final_recovery_time": {"meaning": "seconds from Wave 3 final capture to canonical strict-CE recovery; null if unsuccessful", "better_direction": "↓"},
    "final_safe_complete": {"meaning": "Wave 3 recovery succeeds within 700 steps with no Wave 3 collision and all pursuers active", "better_direction": "↑"},
    "persistent_service_complete": {"meaning": "all three waves captured AND Wave 3 recovery succeeds AND no cumulative collision, terminal failure, or active-pursuer loss", "better_direction": "↑"},
    "all_3_safe_complete": {"meaning": "all three waves individually safe-complete; directly interpretable for PERSIST-A only because B/C may interrupt Wave 1/2 recovery", "better_direction": "↑ for PERSIST-A; diagnostic for PERSIST-B/C"},
    "active_pursuer_count": {"meaning": "number of pursuers still active at wave end", "better_direction": "↑"},
    "coverage_debt_seconds": {"meaning": "time outside strict CE during the wave", "better_direction": "diagnostic"},
    "waves_captured": {"meaning": "number of the three waves captured", "better_direction": "↑"},
    "waves_safe_completed": {"meaning": "number of waves canonically safe-completed", "better_direction": "↑"},
    "total_mission_seconds": {"meaning": "persistent episode elapsed time until stop or three-wave completion", "better_direction": "↓"},
    "cumulative_collision_count": {"meaning": "all collision events across the persistent episode", "better_direction": "↓"},
    "delay_steps": {"meaning": "sampled PERSIST-C normal decision steps after first all-Z-zero; PERSIST-B uses zero", "better_direction": "diagnostic"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=600000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-regime", type=int, default=3)
    parser.add_argument("--initial-seed-base", type=int, default=2026100601)
    parser.add_argument("--refresh-seed-base", type=int, default=2026800601)
    parser.add_argument("--delay-seed-base", type=int, default=2026900601)
    parser.add_argument("--workers", type=int, default=None, help="rollout process count; defaults to the resource-aware cap")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--max-gif-frames", type=int, default=300)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, selected = formal.resolve_checkpoint(argparse.Namespace(
        selection_report=str(args.selection_report), expected_step=args.expected_step,
        checkpoint=None,
    ))
    actual_sha = sha256_file(checkpoint)
    if actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint hash conflicts with designated Z05 policy: {actual_sha}")
    config_path = args.config.resolve()
    torch.set_num_threads(WORKER_THREAD_LIMIT)
    try:
        torch.set_num_interop_threads(WORKER_THREAD_LIMIT)
    except RuntimeError:
        pass
    model = CoCapIQN.load(str(checkpoint), device="cpu").eval()
    if model.config.include_z_state is not True or model.config.include_is_pursuing is not False:
        raise SystemExit("selected policy is not the registered Z-state contract")
    if model.config.pursuing_late_fusion:
        raise SystemExit("selected policy uses a forbidden pursuing late-fusion architecture")
    model_hash_before = object_state_hash(model)
    resources = choose_worker_count(args.workers)
    worker_count = int(resources["selected_workers"])
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    jobs = []
    for regime in REGIMES:
        for episode_index in range(args.episodes_per_regime):
            jobs.append((
                str(config_path), regime, episode_index,
                int(args.initial_seed_base + episode_index),
                int(args.refresh_seed_base + episode_index),
                int(args.delay_seed_base + episode_index),
                str(output), episode_index < REPRESENTATIVE_GIFS_PER_REGIME,
                args.max_gif_frames,
            ))
    total_episodes = len(jobs)
    atomic_json(output / "run_progress.json", {
        "status": "running", "completed_episodes": 0, "total_episodes": total_episodes,
        "workers": worker_count, "resource_snapshot": resources,
        "initial_seed_base": int(args.initial_seed_base),
        "refresh_seed_base": int(args.refresh_seed_base),
        "delay_seed_base": int(args.delay_seed_base),
    })
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(str(checkpoint), EXPECTED_CHECKPOINT_SHA256),
    ) as pool:
        future_jobs = {pool.submit(_run_episode_worker, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_jobs):
            row = future.result()
            records.append(row)
            completed += 1
            atomic_json(output / "run_progress.json", {
                "status": "running", "completed_episodes": completed,
                "total_episodes": total_episodes, "workers": worker_count,
                "last_regime": row["regime"], "last_episode_index": row["episode_index"],
                "elapsed_seconds": time.monotonic() - started,
            })
    records.sort(key=lambda row: (REGIMES.index(row["regime"]), int(row["episode_index"])))
    model_hash_after = object_state_hash(model)
    final_sha = sha256_file(checkpoint)
    if model_hash_before != model_hash_after or final_sha != actual_sha:
        raise SystemExit("policy state or checkpoint bytes changed during evaluation")
    worker_hashes = {
        value for row in records for value in (
            row["worker_model_state_sha256_before"], row["worker_model_state_sha256_after"]
        )
    }
    if worker_hashes != {model_hash_before}:
        raise SystemExit("one or more rollout workers changed or loaded a different policy state")
    for row in records:
        for wave_row in row["waves"]:
            wave_row["contract_diagnostics"]=row["contract_diagnostics"]
    by_regime = {}
    for regime in REGIMES:
        group = [row for row in records if row["regime"] == regime]
        waves = [wave for row in group for wave in row["waves"]]
        by_wave = {}
        for wave_id in range(1, WAVES + 1):
            wave_group = [wave for wave in waves if int(wave["wave_id"]) == wave_id]
            wave_capture_times = [float(w["capture_time_seconds"]) for w in wave_group if w["capture_time_seconds"] is not None]
            wave_recovery_times = [float(w["recovery_time_seconds"]) for w in wave_group if w["recovery_time_seconds"] is not None]
            by_wave[str(wave_id)] = {
                "episodes_observed": len(wave_group),
                "capture_success": sum(w["capture_step"] is not None for w in wave_group),
                "capture_censored_or_failed": sum(w["capture_step"] is None for w in wave_group),
                "capture_time_seconds": summarize(wave_capture_times),
                "ce_recovery_success": sum(bool(w["ce_recovery_success"]) for w in wave_group),
                "recovery_censored_or_interrupted": sum(not bool(w["ce_recovery_success"]) for w in wave_group),
                "recovery_time_seconds": summarize(wave_recovery_times),
                "collision_count": sum(int(w["collision_count"]) for w in wave_group),
                "safe_complete": sum(bool(w["safe_complete"]) for w in wave_group),
                "active_pursuer_count": summarize([float(w["active_pursuer_count_at_end"]) for w in wave_group if w["active_pursuer_count_at_end"] is not None]),
                "coverage_debt_seconds": summarize([float(w["coverage_debt_seconds"]) for w in wave_group]),
                "interrupted_by_next_arrival": sum(bool(w["interrupted_by_next_arrival"]) for w in wave_group),
            }
        stop_reasons: dict[str, int] = {}
        for row in group:
            stop_reasons[str(row["stop_reason"])] = stop_reasons.get(str(row["stop_reason"]), 0) + 1
        failure_taxonomy = {
            "stop_reason_counts": stop_reasons,
            "episodes_missing_wave1_capture": sum(not row["waves"] or row["waves"][0]["capture_step"] is None for row in group),
            "episodes_missing_all_three_captures": sum(not bool(row["all_3_captured"]) for row in group),
            "episodes_final_recovery_failure_after_wave3_capture": sum(bool(row["all_3_captured"] and not row["final_recovery_success"]) for row in group),
            "episodes_with_collision": sum(bool(row["cumulative_collision"]) for row in group),
            "episodes_insufficient_active_pursuers": sum("insufficient_active_pursuers" in stop for stop in stop_reasons),
            "episodes_terminal_failure": sum("terminal_failure" in stop for stop in stop_reasons),
        }
        by_regime[regime] = {
            "episodes": len(group),
            "waves_captured": {"n": sum(row["waves_captured"] for row in group), "mean": float(np.mean([row["waves_captured"] for row in group]))},
            "waves_safe_completed": {"n": sum(row["waves_safe_completed"] for row in group), "mean": float(np.mean([row["waves_safe_completed"] for row in group]))},
            "all_3_captured": sum(bool(row["all_3_captured"]) for row in group),
            "all_3_safe_complete": sum(bool(row["all_3_safe_complete"]) for row in group) if regime == "PERSIST-A" else None,
            "all_3_safe_complete_diagnostic": sum(bool(row["all_3_safe_complete"]) for row in group),
            "final_recovery_success": sum(bool(row["final_recovery_success"]) for row in group),
            "final_recovery_time": summarize([float(row["final_recovery_time"]) for row in group if row["final_recovery_time"] is not None]),
            "final_recovery_time_censored_or_unobserved": sum(row["final_recovery_time"] is None for row in group),
            "final_safe_complete": sum(bool(row["final_safe_complete"]) for row in group),
            "persistent_service_complete": sum(bool(row["persistent_service_complete"]) for row in group),
            "persistent_service_complete_censored_or_failed": sum(not bool(row["persistent_service_complete"]) for row in group),
            "failure_taxonomy": failure_taxonomy,
            "per_wave_degradation": by_wave,
            "total_mission_seconds": summarize([float(row["total_mission_seconds"]) for row in group]),
            "cumulative_collision_count": summarize([float(row["cumulative_collision_count"]) for row in group]),
            "per_wave": {
                "normal_capture_success": sum(bool(w["normal_capture"]) for w in waves),
                "capture_success": sum(w["capture_step"] is not None for w in waves),
                "capture_time_seconds": summarize([float(w["capture_time_seconds"]) for w in waves if w["capture_time_seconds"] is not None]),
                "collision_count": sum(int(w["collision_count"]) for w in waves),
                "first_all_z_zero_time_seconds": summarize([float(w["first_all_z_zero_seconds"]) for w in waves if w["first_all_z_zero_seconds"] is not None]),
                "z_release_time_seconds": summarize([float(w["z_release_seconds"]) for w in waves if w["z_release_seconds"] is not None]),
                "ce_recovery_success": sum(bool(w["ce_recovery_success"]) for w in waves),
                "recovery_time_seconds": summarize([float(w["recovery_time_seconds"]) for w in waves if w["recovery_time_seconds"] is not None]),
                "safe_complete": sum(bool(w["safe_complete"]) for w in waves),
                "active_pursuer_count": summarize([float(w["active_pursuer_count_at_end"]) for w in waves]),
                "coverage_debt_seconds": summarize([float(w["coverage_debt_seconds"]) for w in waves]),
            },
            "coverage_debt_cumulative_seconds": summarize([sum(float(w["coverage_debt_seconds"]) for w in row["waves"]) for row in group]),
            "behavior_reuse": [row["behavior_reuse"] for row in group],
            "contract_diagnostics_pass_count": sum(all(bool(v) for k, v in row["contract_diagnostics"].items() if isinstance(v, bool)) for row in group),
        }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "classification": ("FORMAL_CONTRACT_PASS" if args.episodes_per_regime == 20 else "DEMO_CONTRACT_PASS") if all(
            all(bool(v) for k, v in row["contract_diagnostics"].items() if isinstance(v, bool))
            for row in records
        ) else ("FORMAL_CONTRACT_BLOCKED" if args.episodes_per_regime == 20 else "DEMO_CONTRACT_BLOCKED"),
        "selected_policy": {
            "step": int(selected["step"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha,
            "model_state_sha256_before": model_hash_before,
            "model_state_sha256_after": model_hash_after,
            "source_selection_report": str(args.selection_report.resolve()),
        },
        "source_contract": {
            "branch": "evaluation/z05-repeated-arrival-demo-20261006",
            "base_branch": "evaluation/iqn-z05-independent-20260923",
            "base_head": "1a9da054ecb5652726005d00d08ab05a2188dd8e",
            "config": str(config_path),
            "pursuers": 12, "targets_per_wave": 3, "obstacles": 3,
            "map": [120, 120], "friend_token_cap": int(model.config.max_pursuers),
            "sensing": NORMSENSE_POLICY, "collision": "synchronized_swept_v1",
            "action_contract": "AW9 fixed midpoint-32 greedy", "decision_dt_seconds": DT_SECONDS,
            "global_horizon_steps": GLOBAL_HORIZON, "post_capture_window_steps": POST_CAPTURE_WINDOW,
        },
        "paired_seed_contract": {
            "initial_world_seed_base": int(args.initial_seed_base),
            "target_refresh_rng_seed_base": int(args.refresh_seed_base),
            "persist_c_delay_rng_seed_base": int(args.delay_seed_base),
            "delay_rng_independent_from_target_refresh": True,
            "episodes_per_regime": int(args.episodes_per_regime),
            "target_coordinates_identical_claimed": False,
        },
        "primary_metric_order": [
            "persistent_service_complete", "all_3_captured", "final_recovery_success",
            "final_recovery_time", "collision_count", "per_wave_capture_success_and_time",
            "agent_reuse_and_behavior_transition",
        ],
        "parallel_execution": {
            **resources,
            "representative_gifs_per_regime": min(REPRESENTATIVE_GIFS_PER_REGIME, args.episodes_per_regime),
            "workers_use_separate_processes": True,
            "policy_state_sha256_all_workers": model_hash_before,
        },
        "metrics": METRIC_DEFINITIONS,
        "regimes": by_regime,
        "episodes": records,
        "proposed_formal_command": "python tools/run_iqn_repeated_arrival_demo_20261006.py --config configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml --selection-report /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/selection_report.json --expected-step 600000 --episodes-per-regime 20 --output <FORMAL_OUTPUT_DIR>",
        "completed_elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "summary.json", finite_json(summary))
    atomic_json(output / "run_progress.json", {"status": "complete", "completed_episodes": len(records), "total_episodes": len(records), "workers": worker_count, "classification": summary["classification"]})
    print(json.dumps({"status": "complete", "classification": summary["classification"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
