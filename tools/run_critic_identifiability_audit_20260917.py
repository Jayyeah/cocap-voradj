#!/usr/bin/env python3
"""A0 -> Gate -> A1 frozen-IQN, fixed-bank critic identifiability audit.

This runner deliberately has no actor optimizer, TD target, GAE, target
network, online replay, or policy update. A1 refuses to start unless the A0
gate file says PASS.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import yaml

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.density_sensing import POLICY as SENSING_V2
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.evaluation.mission_events import snapshot
from cocap_voradj.models.continuous.central_attention_critic import (
    CentralAttentionCritic,
    CentralCriticConfig,
)
from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackboneConfig,
)
from cocap_voradj.models.critic_identifiability import LocalQ, NeighborQ, parameter_count
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.models.small_step_ac import CentralValueNetwork
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.td3_stage1_contract import aw9_grid, discrete_task_config
from cocap_voradj.training.trainer import set_global_config
from cocap_voradj.training.td3_warmstart import OBS_KEYS, file_sha256, tensor_state_sha256
from tools.collect_iqn_aw_teacher_dataset_20260903 import fixed_midpoint_q
from tools.forward_final_single_task_20260915 import Telemetry
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_continuous_ctde_training import _split_termination_flags
from tools.run_forward_final_bridge_20260908 import act_evaders, make_env
from cocap_voradj.training.small_step_ac import tensor_tree


SCHEMA = "critic-identifiability-a0-a1-v1"
MAX_AGENTS = 12
MAX_NEIGHBORS = 8
EVENTS = (
    "pure_coverage", "coverage_steady", "ordinary_pursuit", "ring2", "ring3",
    "normal_capture", "stationary_capture", "collision",
    "capture_terminal_transition", "early_recovery", "late_recovery",
    "pure_coverage_restart",
)
EVENT_TO_ID = {name: index for index, name in enumerate(EVENTS)}
PHASES = ("pure_coverage", "pre_capture", "post_capture")
SPLITS = ("train", "validation", "test")
ACTION_SCALE = np.asarray([0.4, np.pi / 6.0], dtype=np.float32)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_scene(scene: str, seed: int):
    if scene in {"pure_capture", "pure_coverage"}:
        task = "capture" if scene == "pure_capture" else "coverage"
        cfg = discrete_task_config(task)
        set_global_config(cfg)
        env = VorAdjEnv(copy.deepcopy(cfg), seed=int(seed))
        observations = env.reset()
    elif scene == "mixed":
        env, observations = make_env("mixed", int(seed), sensing_policy=SENSING_V2)
    else:
        raise ValueError(scene)
    env.total_steps = 2_000_000
    return env, observations


def padded_local(observations: list[Any]) -> dict[str, np.ndarray]:
    live = next(value for value in observations if value is not None)
    result = {}
    for key in OBS_KEYS:
        shape = np.asarray(live[key]).shape
        dtype = np.asarray(live[key]).dtype
        result[key] = np.zeros((MAX_AGENTS, *shape), dtype=dtype)
        for index, value in enumerate(observations):
            if value is not None:
                result[key][index] = value[key]
    return result


def event_flags(
    scene: str,
    state: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    collision: bool,
    terminated: np.ndarray,
    timestep: int,
    capture_step: int | None,
    steady_steps: int,
) -> tuple[np.ndarray, str, float]:
    flags = np.zeros(len(EVENTS), dtype=bool)
    participants = [len(value["participants"]) for value in state["targets"].values()]
    participants.extend(len(value.get("participants", [])) for value in events)
    ring = max(participants, default=0)
    phase = "pure_coverage" if scene == "pure_coverage" else (
        "post_capture" if capture_step is not None else "pre_capture"
    )
    flags[EVENT_TO_ID["pure_coverage"]] = scene == "pure_coverage"
    flags[EVENT_TO_ID["coverage_steady"]] = phase in {"pure_coverage", "post_capture"} and steady_steps > 0
    flags[EVENT_TO_ID["ordinary_pursuit"]] = phase == "pre_capture" and ring < 2
    flags[EVENT_TO_ID["ring2"]] = phase == "pre_capture" and ring == 2
    flags[EVENT_TO_ID["ring3"]] = phase == "pre_capture" and ring >= 3
    flags[EVENT_TO_ID["normal_capture"]] = any(e.get("capture_type") == "loose" for e in events)
    flags[EVENT_TO_ID["stationary_capture"]] = any(e.get("capture_type") == "stationary" for e in events)
    flags[EVENT_TO_ID["collision"]] = bool(collision)
    flags[EVENT_TO_ID["capture_terminal_transition"]] = bool(events) and bool(terminated.any())
    age = -1 if capture_step is None else int(timestep - capture_step)
    flags[EVENT_TO_ID["early_recovery"]] = phase == "post_capture" and 0 <= age <= 40
    flags[EVENT_TO_ID["late_recovery"]] = phase == "post_capture" and age > 40
    flags[EVENT_TO_ID["pure_coverage_restart"]] = scene == "pure_coverage" and timestep <= 20
    return flags, phase, float(ring)


def primary_event(flags: np.ndarray) -> int:
    priority = (
        "collision", "normal_capture", "stationary_capture",
        "capture_terminal_transition", "ring3", "ring2", "early_recovery",
        "late_recovery", "coverage_steady", "pure_coverage_restart",
        "pure_coverage", "ordinary_pursuit",
    )
    for name in priority:
        if flags[EVENT_TO_ID[name]]:
            return EVENT_TO_ID[name]
    return EVENT_TO_ID["ordinary_pursuit"]


@torch.no_grad()
def collect_episode(policy, scene: str, seed: int, device: str, episode_id: int,
                    policy_kind: str = "iqn_greedy"):
    env, observations = make_scene(scene, seed)
    telemetry = (
        Telemetry(env, "capture" if scene == "pure_capture" else "coverage")
        if scene != "mixed" else None
    )
    apf = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    grids = aw9_grid().astype(np.float32)
    rows: dict[str, list[Any]] = {}
    central_rows: dict[str, list[np.ndarray]] = {}
    capture_step = None
    capture_types: list[str] = []
    collision_episode = False
    terminated_seen = False
    truncated_seen = False
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    action_rng = np.random.default_rng(int(seed))

    def add(name: str, value: Any) -> None:
        rows.setdefault(name, []).append(value)

    for timestep in range(1, env.episode_max_length + 1):
        active = [i for i, value in enumerate(observations) if value is not None]
        local = [observations[i] for i in active]
        if policy_kind in {"iqn_greedy", "iqn_epsilon05"}:
            _, chosen = fixed_midpoint_q(policy, local, device)
            if policy_kind == "iqn_epsilon05":
                explore = action_rng.random(len(active)) < 0.05
                random_actions = action_rng.integers(0, len(grids), size=len(active))
                chosen = np.where(explore, random_actions, chosen)
        elif policy_kind == "strong_bc_sample_aw9":
            batch = tensor_tree({key: np.stack([value[key] for value in local])
                                 for key in OBS_KEYS}, torch.device(device))
            chosen = policy.distribution(batch).sample().cpu().numpy()
        else:
            raise ValueError(policy_kind)
        action_index = np.full(MAX_AGENTS, 4, dtype=np.int64)
        action_aw = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
        for agent, index in zip(active, chosen):
            action_index[agent] = int(index)
            action_aw[agent] = grids[int(index)]
        local_padded = padded_local(observations)
        central = build_central_global_obs(
            env, max_agents=MAX_AGENTS, max_evaders=8, max_obstacles=5,
            self_feature_dim=9,
        )
        state = snapshot(env, observations)
        neighbors = np.full((MAX_AGENTS, MAX_NEIGHBORS), -1, dtype=np.int16)
        neighbor_mask = np.zeros((MAX_AGENTS, MAX_NEIGHBORS), dtype=bool)
        for agent in active:
            ids = sorted(state["friends"].get(agent, set()))[:MAX_NEIGHBORS]
            neighbors[agent, :len(ids)] = ids
            neighbor_mask[agent, :len(ids)] = True
        commands = [None] * len(observations)
        for agent in active:
            commands[agent] = int(action_index[agent])
        outcome = env.step(commands, act_evaders(env, apf))
        if telemetry is not None:
            telemetry.observe(env, outcome, active)
        terminated, truncated = _split_termination_flags(outcome.dones, outcome.infos)
        terminated_seen = terminated_seen or bool(terminated.any())
        truncated_seen = truncated_seen or bool(truncated.any())
        captures = list(env.last_capture_events)
        capture_types.extend(str(event.get("capture_type")) for event in captures)
        collision = bool(env.last_collision_events)
        collision_episode = collision_episode or collision
        flags, phase, ring = event_flags(
            scene, state, captures, collision, terminated, timestep, capture_step,
            int(getattr(env, "distribution_hold_steps", 0)),
        )
        if captures and capture_step is None:
            capture_step = timestep
        reward = np.zeros(MAX_AGENTS, dtype=np.float32)
        term = np.zeros(MAX_AGENTS, dtype=bool)
        trunc = np.zeros(MAX_AGENTS, dtype=bool)
        active_mask = np.zeros(MAX_AGENTS, dtype=bool)
        reward[:len(outcome.rewards)] = np.asarray(outcome.rewards, dtype=np.float32)
        term[:len(terminated)] = terminated
        trunc[:len(truncated)] = truncated
        active_mask[active] = True
        for key, value in local_padded.items():
            add("local_" + key, value)
        for key, value in central.items():
            central_rows.setdefault("global_" + key, []).append(value)
        add("action_index", action_index)
        add("action_aw", action_aw)
        add("neighbor_ids", neighbors)
        add("neighbor_mask", neighbor_mask)
        add("active_mask", active_mask)
        add("reward", reward)
        add("terminated", term)
        add("truncated", trunc)
        add("event_flags", flags)
        add("primary_event", primary_event(flags))
        add("phase_id", PHASES.index(phase))
        add("ring_count", ring)
        add("episode_id", episode_id)
        add("episode_seed", seed)
        add("timestep", timestep)
        add("scene_id", ("pure_capture", "pure_coverage", "mixed").index(scene))
        observations = outcome.observations
        if all(outcome.dones):
            break

    arrays = {key: np.stack(value) for key, value in rows.items()}
    arrays.update({key: np.stack(value) for key, value in central_rows.items()})
    rewards = arrays["reward"].astype(np.float64)
    returns = np.zeros_like(rewards)
    running = np.zeros(MAX_AGENTS, dtype=np.float64)
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + 0.99 * running
        returns[index] = running
    arrays["mc_return"] = returns.astype(np.float32)
    if telemetry is not None:
        metric = telemetry.finish(env)
    else:
        episode_record = env.episode_record(task="mix")
        metric = {
            "captured": bool(episode_record.get("captured", False)),
            "ce_success": bool(env.post_capture_coverage_success),
        }
    normal = "loose" in capture_types
    stationary = "stationary" in capture_types
    captured = bool(normal or stationary or metric.get("captured", False))
    ce_success = bool(metric.get("ce_success", False))
    success = (
        normal if scene == "pure_capture"
        else ce_success if scene == "pure_coverage"
        else captured and ce_success
    ) and not collision_episode
    record = {
        "episode_id": episode_id, "seed": seed, "scene": scene, "length": len(rewards),
        "success": bool(success), "failure": not bool(success),
        "normal_capture": bool(normal), "stationary_capture": bool(stationary),
        "capture": bool(captured), "ce_success": bool(ce_success),
        "collision": bool(collision_episode), "terminated_seen": terminated_seen,
        "truncated_seen": truncated_seen,
        "event_counts": {
            name: int(arrays["event_flags"][:, EVENT_TO_ID[name]].sum()) for name in EVENTS
        },
    }
    return arrays, record


def split_assignment(scene_index: int, count: int) -> list[int]:
    train = max(1, int(math.floor(0.70 * count)))
    validation = max(1, int(math.floor(0.15 * count)))
    if train + validation >= count:
        train = count - 2
        validation = 1
    labels = [0] * train + [1] * validation + [2] * (count - train - validation)
    del scene_index
    return labels


def concatenate(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = episodes[0]
    return {key: np.concatenate([episode[key] for episode in episodes], axis=0) for key in keys}


def event_counts(bank: Mapping[str, np.ndarray], transition_mask: np.ndarray | None = None):
    mask = np.ones(len(bank["event_flags"]), dtype=bool) if transition_mask is None else transition_mask
    active = bank["active_mask"][mask].sum(1)
    return {
        name: int((bank["event_flags"][mask, EVENT_TO_ID[name]] * active).sum())
        for name in EVENTS
    }


def verify_mc(bank: Mapping[str, np.ndarray], gamma: float) -> float:
    maximum = 0.0
    for episode in np.unique(bank["episode_id"]):
        indices = np.flatnonzero(bank["episode_id"] == episode)
        running = np.zeros(MAX_AGENTS, dtype=np.float64)
        recomputed = np.zeros((len(indices), MAX_AGENTS), dtype=np.float64)
        for offset in range(len(indices) - 1, -1, -1):
            running = bank["reward"][indices[offset]].astype(np.float64) + gamma * running
            recomputed[offset] = running
        maximum = max(maximum, float(np.max(np.abs(recomputed - bank["mc_return"][indices]))))
    return maximum


def collect_a0(config: Mapping[str, Any], output: Path, device: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / config["teacher"]["checkpoint"]
    if file_sha256(checkpoint) != config["teacher"]["sha256"]:
        raise ValueError("teacher checkpoint hash mismatch")
    policy_kind = str(config["teacher"]["policy"])
    if policy_kind == "strong_bc_sample_aw9":
        model, payload = load_actor(checkpoint, device)
    elif policy_kind in {"iqn_greedy", "iqn_epsilon05"}:
        model = CoCapIQN.load(str(checkpoint), device=device).eval()
        payload = {}
    else:
        raise ValueError("unsupported frozen policy")
    model.requires_grad_(False)
    actor_before = tensor_state_sha256(model.state_dict())
    collection = config["collection"]
    plan = []
    base = int(collection["seed_base"])
    specs = (
        ("pure_capture", int(collection["pure_capture_episodes"]), 0),
        ("pure_coverage", int(collection["pure_coverage_episodes"]), 100_000),
        ("mixed", int(collection["mixed_episodes"]), 200_000),
    )
    for scene_index, (scene, count, offset) in enumerate(specs):
        seeds = [base + offset + index for index in range(count)]
        rare_key = "known_rare_" + scene + "_seeds"
        rare = [int(value) for value in collection.get(rare_key, [])]
        for index, seed in enumerate(rare):
            seeds[-1-index] = seed
        labels = split_assignment(scene_index, count)
        rare_by_split = collection.get(rare_key + "_by_split", {})
        for split_id, split_name in enumerate(SPLITS):
            positions = [index for index, label in enumerate(labels) if label == split_id]
            split_rare = [int(value) for value in rare_by_split.get(split_name, [])]
            if len(split_rare) > len(positions):
                raise ValueError(f"too many rare {scene} seeds for {split_name}")
            for position, seed in zip(reversed(positions), reversed(split_rare)):
                seeds[position] = seed
        for seed, split in zip(seeds, labels):
            plan.append((scene, seed, split))
    episodes, records = [], []
    start = time.monotonic()
    for episode_id, (scene, seed, split) in enumerate(plan):
        arrays, record = collect_episode(model, scene, seed, device, episode_id, policy_kind)
        arrays["split_id"] = np.full(len(arrays["episode_id"]), split, dtype=np.uint8)
        arrays["episode_success"] = np.full(len(arrays["episode_id"]), record["success"], dtype=bool)
        arrays["episode_collision"] = np.full(len(arrays["episode_id"]), record["collision"], dtype=bool)
        episodes.append(arrays)
        record["split"] = SPLITS[split]
        records.append(record)
        atomic_json(output / "progress.json", {
            "status": "collecting", "completed": episode_id + 1, "total": len(plan),
            "last": record, "elapsed_seconds": time.monotonic() - start,
        })
    bank = concatenate(episodes)
    bank_path = output / "trajectory_bank.npz"
    np.savez_compressed(bank_path, **bank)
    actor_after = tensor_state_sha256(model.state_dict())
    gamma = float(config["gamma"])
    mc_error = verify_mc(bank, gamma)
    episode_sets = {
        split: sorted({int(record["episode_id"]) for record in records if record["split"] == split})
        for split in SPLITS
    }
    seed_sets = {
        split: sorted({int(record["seed"]) for record in records if record["split"] == split})
        for split in SPLITS
    }
    leakage = any(
        set(episode_sets[a]) & set(episode_sets[b]) or set(seed_sets[a]) & set(seed_sets[b])
        for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]
    )
    termination_ok = bool(
        not np.any(bank["terminated"] & bank["truncated"])
        and all(record["terminated_seen"] or record["truncated_seen"] for record in records)
    )
    qualification = {}
    qn = int(collection["qualification_episodes"])
    for scene, _, _ in specs:
        subset = [record for record in records if record["scene"] == scene][:qn]
        success_rate = float(np.mean([record["success"] for record in subset]))
        collision_rate = float(np.mean([record["collision"] for record in subset]))
        qualification[scene] = {
            "episodes": len(subset), "success_rate": success_rate,
            "collision_rate": collision_rate,
            "pass": success_rate >= float(collection["qualification_min_success_rate"])
                    and collision_rate <= float(collection["qualification_max_collision_rate"]),
        }
    counts = event_counts(bank)
    split_counts = {
        split: event_counts(bank, bank["split_id"] == index)
        for index, split in enumerate(SPLITS)
    }
    gate_cfg = config["gate"]
    successes = sum(record["success"] for record in records)
    failures = sum(record["failure"] for record in records)
    checks = {
        "teacher_qualified_all_scenes": all(value["pass"] for value in qualification.values()),
        "success_episodes": successes >= int(gate_cfg["minimum_success_episodes"]),
        "failure_episodes": failures >= int(gate_cfg["minimum_failure_episodes"]),
        "normal_capture_rows": counts["normal_capture"] >= int(gate_cfg["minimum_normal_capture_rows"]),
        "ring3_rows": counts["ring3"] >= int(gate_cfg["minimum_ring3_rows"]),
        "collision_rows": counts["collision"] >= int(gate_cfg["minimum_collision_rows"]),
        "collision_rows_each_split": all(
            value["collision"] >= int(gate_cfg["minimum_collision_rows_per_split"])
            for value in split_counts.values()
        ),
        "early_recovery_rows": counts["early_recovery"] >= int(gate_cfg["minimum_early_recovery_rows"]),
        "mc_return_exact": mc_error <= 2e-5,
        "termination_truncation_semantics": termination_ok,
        "no_train_validation_test_leakage": not leakage,
        "actor_frozen_bit_exact": actor_before == actor_after,
    }
    manifest = {
        "schema": SCHEMA, "status": "complete", "contract": "NormSense-V2 4v1",
        "teacher": {
            **config["teacher"], "path": str(checkpoint), "state_sha256_before": actor_before,
            "state_sha256_after": actor_after,
            "checkpoint_payload_schema": payload.get("schema"),
        },
        "bank": str(bank_path), "bank_sha256": file_sha256(bank_path),
        "config_sha256": canonical_hash(config), "gamma": gamma,
        "transitions": int(len(bank["episode_id"])),
        "active_agent_rows": int(bank["active_mask"].sum()),
        "episodes": len(records), "success_episodes": int(successes), "failure_episodes": int(failures),
        "event_counts": counts,
        "split_event_counts": split_counts,
        "split_transitions": {
            split: int((bank["split_id"] == index).sum()) for index, split in enumerate(SPLITS)
        },
        "episode_ids": episode_sets, "episode_seeds": seed_sets,
        "episode_records": records, "qualification": qualification,
        "mc_return": {
            "formula": "G_t=sum_k gamma^k r_(t+k), complete realized trajectory",
            "learned_target_used": False, "maximum_recompute_abs_error": mc_error,
        },
        "termination": {
            "terminated_and_truncated_overlap": int((bank["terminated"] & bank["truncated"]).sum()),
            "all_episodes_end_with_terminated_or_truncated": termination_ok,
        },
        "actor_updates": 0,
    }
    atomic_json(output / "trajectory_bank_manifest.json", manifest)
    gate = {
        "schema": SCHEMA, "decision": "PASS" if all(checks.values()) else "STOP_A0_GATE_FAILED",
        "checks": checks, "qualification": qualification, "event_counts": counts,
        "success_episodes": int(successes), "failure_episodes": int(failures),
        "mc_max_abs_error": mc_error, "actor_bit_exact": actor_before == actor_after,
        "no_leakage": not leakage, "a1_allowed": bool(all(checks.values())),
    }
    atomic_json(output / "a0_gate.json", gate)
    atomic_json(output / "progress.json", {
        "status": "a0_complete", "decision": gate["decision"],
        "elapsed_seconds": time.monotonic() - start,
    })
    return gate


def qualify_a0(config: Mapping[str, Any], output: Path, device: str) -> dict[str, Any]:
    """Run the predeclared matched-contract teacher qualification only.

    This is deliberately separate from bank construction: a teacher that cannot
    solve the three scene families must fail before a large trajectory bank is
    accepted or any critic fit begins.
    """
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / config["teacher"]["checkpoint"]
    if file_sha256(checkpoint) != config["teacher"]["sha256"]:
        raise ValueError("teacher checkpoint hash mismatch")
    policy_kind = str(config["teacher"]["policy"])
    if policy_kind == "strong_bc_sample_aw9":
        model, payload = load_actor(checkpoint, device)
    elif policy_kind in {"iqn_greedy", "iqn_epsilon05"}:
        model = CoCapIQN.load(str(checkpoint), device=device).eval()
        payload = {}
    else:
        raise ValueError("unsupported frozen policy")
    model.requires_grad_(False)
    before = tensor_state_sha256(model.state_dict())
    collection = config["collection"]
    base = int(collection["seed_base"])
    qn = int(collection["qualification_episodes"])
    specs = (
        ("pure_capture", 0),
        ("pure_coverage", 100_000),
        ("mixed", 200_000),
    )
    records = []
    start = time.monotonic()
    for scene, offset in specs:
        for index in range(qn):
            seed = base + offset + index
            _, record = collect_episode(model, scene, seed, device, len(records), policy_kind)
            records.append(record)
            atomic_json(output / "qualification_progress.json", {
                "status": "qualifying", "completed": len(records),
                "total": qn * len(specs), "last": record,
                "elapsed_seconds": time.monotonic() - start,
            })
    qualification = {}
    for scene, _ in specs:
        subset = [record for record in records if record["scene"] == scene]
        success_rate = float(np.mean([record["success"] for record in subset]))
        collision_rate = float(np.mean([record["collision"] for record in subset]))
        qualification[scene] = {
            "episodes": len(subset), "success_rate": success_rate,
            "collision_rate": collision_rate,
            "pass": success_rate >= float(collection["qualification_min_success_rate"])
                    and collision_rate <= float(collection["qualification_max_collision_rate"]),
        }
    after = tensor_state_sha256(model.state_dict())
    checks = {
        "teacher_qualified_all_scenes": all(value["pass"] for value in qualification.values()),
        "actor_frozen_bit_exact": before == after,
    }
    report = {
        "schema": SCHEMA,
        "decision": "PASS" if all(checks.values()) else "STOP_TEACHER_QUALIFICATION_FAILED",
        "teacher": {
            **config["teacher"], "path": str(checkpoint),
            "checkpoint_payload_schema": payload.get("schema"),
            "state_sha256_before": before, "state_sha256_after": after,
        },
        "checks": checks, "qualification": qualification,
        "episode_records": records, "actor_updates": 0,
    }
    atomic_json(output / "teacher_qualification.json", report)
    atomic_json(output / "qualification_progress.json", {
        "status": "complete", "decision": report["decision"],
        "elapsed_seconds": time.monotonic() - start,
    })
    return report


def scan_collision_seeds(
    config: Mapping[str, Any], output: Path, device: str,
    start_seed: int, count: int, stop_collisions: int,
) -> dict[str, Any]:
    """Search deterministic seed->trajectory mappings for rare collision episodes."""
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / config["teacher"]["checkpoint"]
    if file_sha256(checkpoint) != config["teacher"]["sha256"]:
        raise ValueError("teacher checkpoint hash mismatch")
    policy_kind = str(config["teacher"]["policy"])
    if policy_kind in {"iqn_greedy", "iqn_epsilon05"}:
        model = CoCapIQN.load(str(checkpoint), device=device).eval()
    elif policy_kind == "strong_bc_sample_aw9":
        model, _ = load_actor(checkpoint, device)
    else:
        raise ValueError("unsupported frozen policy")
    model.requires_grad_(False)
    before = tensor_state_sha256(model.state_dict())
    records = []
    collisions = []
    start = time.monotonic()
    for offset in range(int(count)):
        seed = int(start_seed) + offset
        arrays, record = collect_episode(model, "mixed", seed, device, offset, policy_kind)
        del arrays
        records.append(record)
        if record["collision"]:
            collisions.append(record)
        atomic_json(output / "collision_scan_progress.json", {
            "status": "scanning", "completed": offset + 1, "planned": int(count),
            "collisions": len(collisions), "last": record,
            "elapsed_seconds": time.monotonic() - start,
        })
        if len(collisions) >= int(stop_collisions):
            break
    after = tensor_state_sha256(model.state_dict())
    report = {
        "schema": SCHEMA, "policy": config["teacher"],
        "start_seed": int(start_seed), "planned_count": int(count),
        "evaluated": len(records), "collision_episodes": collisions,
        "collision_seeds": [int(row["seed"]) for row in collisions],
        "all_records": records, "actor_updates": 0,
        "actor_frozen_bit_exact": before == after,
    }
    atomic_json(output / "collision_scan.json", report)
    atomic_json(output / "collision_scan_progress.json", {
        "status": "complete", "evaluated": len(records),
        "collisions": len(collisions), "elapsed_seconds": time.monotonic() - start,
    })
    return report


def to_tensor(value: np.ndarray, device: torch.device):
    tensor = torch.as_tensor(value, device=device)
    return tensor.float() if tensor.dtype in (torch.float64, torch.float32, torch.float16) else tensor


class BankRows:
    def __init__(self, bank: Mapping[str, np.ndarray], device: torch.device):
        self.bank = bank
        self.device = device
        self.transition, self.agent = np.nonzero(bank["active_mask"])
        self.split = bank["split_id"][self.transition]
        self.target = bank["mc_return"][self.transition, self.agent].astype(np.float32)
        self.primary = bank["primary_event"][self.transition]

    def indices(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def batch(self, rows: np.ndarray) -> dict[str, Any]:
        t = self.transition[rows]
        a = self.agent[rows]
        obs = {key: to_tensor(self.bank["local_" + key][t, a], self.device) for key in OBS_KEYS}
        action = to_tensor(self.bank["action_aw"][t, a] / ACTION_SCALE, self.device)
        neighbor_ids = self.bank["neighbor_ids"][t, a]
        neighbor_mask = self.bank["neighbor_mask"][t, a]
        safe_ids = np.maximum(neighbor_ids, 0)
        neighbor_obs = {
            key: to_tensor(self.bank["local_" + key][t[:, None], safe_ids], self.device)
            for key in OBS_KEYS
        }
        neighbor_action = to_tensor(
            self.bank["action_aw"][t[:, None], safe_ids] / ACTION_SCALE, self.device
        )
        central = {
            key[len("global_"):]: to_tensor(self.bank[key][t], self.device)
            for key in self.bank if key.startswith("global_")
        }
        joint = to_tensor(self.bank["action_aw"][t] / ACTION_SCALE, self.device)
        return {
            "obs": obs, "action": action, "neighbor_obs": neighbor_obs,
            "neighbor_action": neighbor_action,
            "neighbor_mask": to_tensor(neighbor_mask, self.device),
            "central": central, "joint": joint,
            "agent": torch.as_tensor(a, device=self.device, dtype=torch.long),
            "transition": t, "row": rows,
        }


def make_model(kind: str, config: Mapping[str, Any], device: torch.device):
    train = config["training"]
    local_cfg = LegacyVorAdjFeatureBackboneConfig(
        hidden_dim=int(train["hidden_dim"]), num_heads=int(train["num_heads"]),
        num_layers=int(train["num_layers"]), dropout=float(train["dropout"]),
    )
    if kind == "LQ":
        model = LocalQ(local_cfg)
    elif kind == "NQ":
        model = NeighborQ(local_cfg, MAX_NEIGHBORS)
    elif kind == "CQ":
        model = CentralAttentionCritic(CentralCriticConfig(
            hidden_dim=int(train["hidden_dim"]), num_heads=int(train["num_heads"]),
            num_layers=int(train["num_layers"]), max_agents=MAX_AGENTS,
            max_evaders=8, max_obstacles=5, dropout=float(train["dropout"]),
        ))
    elif kind == "V":
        model = CentralValueNetwork(
            hidden_dim=int(train["hidden_dim"]), num_heads=int(train["num_heads"]),
            num_layers=int(train["num_layers"]), max_agents=MAX_AGENTS,
            max_evaders=8, max_obstacles=5,
        )
    else:
        raise ValueError(kind)
    return model.to(device)


def forward(kind: str, model, batch: Mapping[str, Any], action_override=None):
    action = batch["action"] if action_override is None else action_override
    if kind == "LQ":
        return model(batch["obs"], action)
    if kind == "NQ":
        return model(batch["obs"], action, batch["neighbor_obs"],
                     batch["neighbor_action"], batch["neighbor_mask"])
    if kind == "CQ":
        joint = batch["joint"].clone()
        if action_override is not None:
            joint[torch.arange(len(joint), device=joint.device), batch["agent"]] = action_override
        values = model(batch["central"], joint)
    else:
        values = model(batch["central"])
    return values[torch.arange(len(values), device=values.device), batch["agent"]]


def predict(kind: str, model, rows: BankRows, indices: np.ndarray, batch_size: int,
            action_override: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            take = indices[start:start + batch_size]
            batch = rows.batch(take)
            override = None
            if action_override is not None:
                override = to_tensor(action_override[start:start + len(take)], rows.device)
            result.append(forward(kind, model, batch, override).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


def calibration(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if not len(target):
        return {"rows": 0}
    error = prediction - target
    variance = float(np.var(target))
    ev = 1.0 - float(np.var(error)) / variance if variance > 1e-12 else None
    pearson = float(np.corrcoef(target, prediction)[0, 1]) if len(target) > 1 and np.std(prediction) > 0 else None
    order_t = np.argsort(np.argsort(target))
    order_p = np.argsort(np.argsort(prediction))
    spearman = float(np.corrcoef(order_t, order_p)[0, 1]) if len(target) > 1 else None
    slope, intercept = np.polyfit(prediction, target, 1) if len(target) > 1 and np.std(prediction) > 0 else (None, None)
    bins = np.array_split(np.argsort(prediction), min(10, len(target)))
    ece = float(np.mean([abs(float(prediction[b].mean() - target[b].mean())) for b in bins]))
    return {
        "rows": int(len(target)), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))), "explained_variance": ev,
        "pearson": pearson, "spearman": spearman,
        "calibration_slope_target_on_prediction": None if slope is None else float(slope),
        "calibration_intercept": None if intercept is None else float(intercept),
        "decile_calibration_error": ece,
        "target_mean": float(target.mean()), "prediction_mean": float(prediction.mean()),
    }


def ordering(target: np.ndarray, prediction: np.ndarray, success: np.ndarray, collision: np.ndarray):
    good = np.flatnonzero(success & ~collision)
    bad = np.flatnonzero(collision)
    if not len(good) or not len(bad):
        return {"available": False, "success_rows": int(len(good)), "collision_rows": int(len(bad))}
    return {
        "available": True, "success_rows": int(len(good)), "collision_rows": int(len(bad)),
        "target_success_mean": float(target[good].mean()),
        "target_collision_mean": float(target[bad].mean()),
        "prediction_success_mean": float(prediction[good].mean()),
        "prediction_collision_mean": float(prediction[bad].mean()),
        "target_pairwise_success_gt_collision": float((target[good, None] > target[bad][None]).mean()),
        "prediction_pairwise_success_gt_collision": float((prediction[good, None] > prediction[bad][None]).mean()),
    }


def score_model(rows: BankRows, indices: np.ndarray, prediction: np.ndarray):
    bank = rows.bank
    transition = rows.transition[indices]
    target = rows.target[indices]
    result = {"overall": calibration(target, prediction), "phase_conditioned": {}}
    for name in EVENTS:
        mask = bank["event_flags"][transition, EVENT_TO_ID[name]]
        result["phase_conditioned"][name] = calibration(target[mask], prediction[mask])
    success = bank["episode_success"][transition]
    collision_episode = bank["episode_collision"][transition]
    capture_transition = bank["event_flags"][transition, EVENT_TO_ID["normal_capture"]]
    collision_transition = bank["event_flags"][transition, EVENT_TO_ID["collision"]]
    result["success_collision_ordering"] = ordering(
        target, prediction, success & capture_transition, collision_transition,
    )
    ring3 = bank["event_flags"][transition, EVENT_TO_ID["ring3"]]
    result["near_capture_ordering"] = ordering(target[ring3], prediction[ring3],
                                                success[ring3], collision_episode[ring3])
    return result


def balanced_probabilities(rows: BankRows, train_indices: np.ndarray, cap: float):
    labels = rows.primary[train_indices]
    counts = np.bincount(labels, minlength=len(EVENTS)).astype(np.float64)
    weights = np.asarray([1.0 / max(counts[label], 1.0) for label in labels])
    positive = counts[counts > 0]
    if len(positive):
        ratio = weights / max(weights.min(), 1e-12)
        weights /= np.maximum(ratio / float(cap), 1.0)
    return weights / weights.sum(), counts.astype(int).tolist()


def fit_one(kind: str, seed: int, config: Mapping[str, Any], rows: BankRows,
            output: Path, normalization: tuple[float, float], probabilities: np.ndarray):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = make_model(kind, config, rows.device)
    train_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    train_indices = rows.indices(0)
    validation_indices = rows.indices(1)
    mean, scale = normalization
    rng = np.random.default_rng(seed)
    best = None
    history = []
    for update in range(1, int(train_cfg["updates"]) + 1):
        chosen = rng.choice(train_indices, size=min(int(train_cfg["batch_size"]), len(train_indices)),
                            replace=True, p=probabilities)
        batch = rows.batch(chosen)
        target = to_tensor((rows.target[chosen] - mean) / scale, rows.device)
        model.train()
        loss = torch.mean((forward(kind, model, batch) - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"])))
        optimizer.step()
        if update % int(train_cfg["validation_every"]) == 0 or update == int(train_cfg["updates"]):
            normalized = predict(kind, model, rows, validation_indices, int(train_cfg["batch_size"]))
            predicted = normalized * scale + mean
            score = calibration(rows.target[validation_indices], predicted)
            history.append({"update": update, "loss": float(loss.detach()), "gradient_norm": gradient,
                            "validation": score})
            if best is None or score["rmse"] < best["rmse"]:
                best = {"rmse": score["rmse"], "update": update,
                        "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
    model.load_state_dict(best.pop("state"))
    path = output / "checkpoints" / f"{kind.lower()}_seed{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "kind": kind, "seed": seed, "state_dict": model.state_dict(),
                "normalization": normalization, "best": best}, path)
    return model, {
        "kind": kind, "seed": seed, "parameter_count": parameter_count(model),
        "best_validation": best, "history": history, "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
    }


def rank_stability(kind: str, models: list[Any], rows: BankRows, indices: np.ndarray,
                   batch_size: int) -> dict[str, Any]:
    if kind == "V" or len(models) < 2:
        return {"applicable": False}
    take = indices[:min(len(indices), 2048)]
    grid = aw9_grid().astype(np.float32) / ACTION_SCALE
    twins = []
    for model in models:
        action_values = []
        for action in grid:
            override = np.broadcast_to(action, (len(take), 2)).copy()
            action_values.append(predict(kind, model, rows, take, batch_size, override))
        twins.append(np.stack(action_values, axis=1))
    ranks0 = np.argsort(np.argsort(twins[0], axis=1), axis=1)
    ranks1 = np.argsort(np.argsort(twins[1], axis=1), axis=1)
    correlations = []
    for first, second in zip(ranks0, ranks1):
        correlations.append(float(np.corrcoef(first, second)[0, 1]))
    return {
        "applicable": True, "rows": len(take),
        "twin_top1_agreement": float((twins[0].argmax(1) == twins[1].argmax(1)).mean()),
        "twin_action_rank_spearman_mean": float(np.mean(correlations)),
        "twin_action_rank_spearman_p10": float(np.percentile(correlations, 10)),
    }


def classify(results: Mapping[str, Any], target_std: float):
    scores = {kind: results[kind]["test"]["overall"] for kind in ("LQ", "NQ", "CQ", "V")}
    q = {kind: scores[kind] for kind in ("LQ", "NQ", "CQ")}
    healthy = lambda value: (
        value["explained_variance"] is not None and value["explained_variance"] >= 0.50
        and value["rmse"] <= 0.75 * target_std
    )
    improvement = lambda better, worse: (
        better["rmse"] <= 0.90 * worse["rmse"]
        and (better["explained_variance"] or -99) >= (worse["explained_variance"] or -99) + 0.05
    )
    if healthy(q["LQ"]):
        decision = "LOCAL_SUFFICIENT"
    elif improvement(q["NQ"], q["LQ"]) and healthy(q["NQ"]):
        decision = "NEIGHBOR_INFORMATION_NEEDED"
    elif improvement(q["CQ"], q["LQ"]) and improvement(q["CQ"], q["NQ"]) and healthy(q["CQ"]):
        decision = "CENTRAL_INFORMATION_NEEDED"
    elif any(healthy(value) for value in q.values()) and scores["V"]["rmse"] > 1.15 * min(value["rmse"] for value in q.values()):
        decision = "VALUE_FORMULATION_WEAK"
    elif max((value["explained_variance"] or -99) for value in scores.values()) < 0.20:
        decision = "REPRESENTATION_OR_STATE_INSUFFICIENT"
    else:
        decision = "INCONCLUSIVE"
    evidence_a2 = decision in {
        "LOCAL_SUFFICIENT", "NEIGHBOR_INFORMATION_NEEDED", "CENTRAL_INFORMATION_NEEDED"
    }
    return decision, evidence_a2


def plot_scatter(path: Path, target: np.ndarray, predictions: Mapping[str, np.ndarray]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    chosen = np.linspace(0, len(target) - 1, min(len(target), 5000), dtype=int)
    low, high = np.percentile(target, [1, 99])
    for axis, kind in zip(axes.ravel(), ("LQ", "NQ", "CQ", "V")):
        axis.scatter(target[chosen], predictions[kind][chosen], s=4, alpha=0.2)
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set(title=kind, xlabel="MC return", ylabel="prediction")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def run_a1(config: Mapping[str, Any], output: Path, device_name: str, reuse_checkpoints: bool = False):
    gate = json.loads((output / "a0_gate.json").read_text())
    if gate["decision"] != "PASS" or not gate["a1_allowed"]:
        raise RuntimeError("A0 gate did not pass; A1 is forbidden")
    manifest = json.loads((output / "trajectory_bank_manifest.json").read_text())
    bank_path = Path(manifest["bank"])
    if file_sha256(bank_path) != manifest["bank_sha256"]:
        raise ValueError("trajectory bank hash mismatch")
    loaded = np.load(bank_path, allow_pickle=False)
    bank = {key: loaded[key] for key in loaded.files}
    device = torch.device(device_name)
    rows = BankRows(bank, device)
    train_indices = rows.indices(0)
    validation_indices = rows.indices(1)
    test_indices = rows.indices(2)
    mean = float(rows.target[train_indices].mean())
    scale = max(float(rows.target[train_indices].std()), 1e-6)
    probabilities, primary_counts = balanced_probabilities(
        rows, train_indices, float(config["training"]["event_balance_cap"])
    )
    models_by_kind = {}
    runs = {}
    predictions = {}
    results = {}
    previous = None
    if reuse_checkpoints:
        previous = json.loads((output / "critic_comparison.json").read_text())
    start = time.monotonic()
    for kind in ("LQ", "NQ", "CQ", "V"):
        models, run_rows = [], []
        for seed in config["training"]["seeds"]:
            if reuse_checkpoints:
                path = output / "checkpoints" / f"{kind.lower()}_seed{int(seed)}.pt"
                payload = torch.load(path, map_location=device, weights_only=False)
                if payload.get("schema") != SCHEMA or payload.get("kind") != kind or int(payload.get("seed")) != int(seed):
                    raise ValueError(f"checkpoint contract mismatch: {path}")
                if not np.allclose(np.asarray(payload["normalization"]), np.asarray((mean, scale)), rtol=0, atol=1e-6):
                    raise ValueError(f"checkpoint normalization mismatch: {path}")
                model = make_model(kind, config, device)
                model.load_state_dict(payload["state_dict"])
                candidates = [row for row in previous["training_runs"][kind] if int(row["seed"]) == int(seed)]
                if len(candidates) != 1 or candidates[0]["checkpoint_sha256"] != file_sha256(path):
                    raise ValueError(f"checkpoint provenance mismatch: {path}")
                run = candidates[0]
            else:
                model, run = fit_one(kind, int(seed), config, rows, output, (mean, scale), probabilities)
            models.append(model)
            run_rows.append(run)
            atomic_json(output / "progress.json", {
                "status": "rescoring" if reuse_checkpoints else "fitting",
                "completed_models": sum(len(value) for value in models_by_kind.values()) + len(models),
                "current": kind, "seed": seed, "elapsed_seconds": time.monotonic() - start,
            })
        models_by_kind[kind] = models
        runs[kind] = run_rows
        split_scores = {}
        for split_id, split in enumerate(SPLITS):
            indices = rows.indices(split_id)
            twin = [
                predict(kind, model, rows, indices, int(config["training"]["batch_size"])) * scale + mean
                for model in models
            ]
            ensemble = np.mean(twin, axis=0)
            split_scores[split] = score_model(rows, indices, ensemble)
            split_scores[split]["twin_prediction_rmse"] = (
                float(np.sqrt(np.mean((twin[0] - twin[1]) ** 2))) if len(twin) > 1 else None
            )
            if split == "test":
                predictions[kind] = ensemble
        split_scores["action_ranking_stability"] = rank_stability(
            kind, models, rows, test_indices, int(config["training"]["batch_size"])
        )
        results[kind] = split_scores
    classification, evidence_a2 = classify(results, float(rows.target[test_indices].std()))
    parameter_counts = {kind: runs[kind][0]["parameter_count"] for kind in runs}
    ratio = max(parameter_counts.values()) / min(parameter_counts.values())
    comparison = {
        "schema": SCHEMA, "decision": classification, "evidence_to_enter_a2": evidence_a2,
        "a2_executed": False, "a3_executed": False, "a4_executed": False,
        "target": "complete-trajectory empirical discounted MC return",
        "bootstrap_used": False, "actor_updates": 0,
        "normalization": {"train_mean": mean, "train_std": scale, "frozen": True},
        "splits": {
            split: {"rows": int(len(rows.indices(index))), "episodes": manifest["episode_ids"][split],
                    "seeds": manifest["episode_seeds"][split]}
            for index, split in enumerate(SPLITS)
        },
        "sampling": {
            "method": "inverse-primary-event-frequency capped; same schedule per seed across critics",
            "primary_event_train_counts": {name: primary_counts[index] for index, name in enumerate(EVENTS)},
            "cap": float(config["training"]["event_balance_cap"]),
        },
        "fairness": {
            "hidden_dim": config["training"]["hidden_dim"],
            "num_heads": config["training"]["num_heads"],
            "num_layers": config["training"]["num_layers"],
            "optimizer": "Adam", "updates": config["training"]["updates"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "seeds": config["training"]["seeds"],
            "parameter_counts": parameter_counts, "max_min_parameter_ratio": ratio,
            "variable_cardinality": "all set critics use masks and max_agents=12; collected 4v1 is not hard-coded in model",
            "LQ_reused_for_TD3_and_SAC": True,
        },
        "results": results, "training_runs": runs,
        "classification_rules": {
            "healthy": "test EV>=0.50 and RMSE<=0.75*test-target-std",
            "significant_improvement": "RMSE<=0.90 baseline and EV gain>=0.05",
            "value_weak": "some Q healthy and V RMSE>1.15*best-Q RMSE",
            "all_bad": "all test EV<0.20",
        },
        "limits": [
            "Action ranking is twin stability over AW9 counterfactual queries, not counterfactual ground truth.",
            "Success/collision ordering compares successful normal-capture transition rows with actual collision transition rows.",
            "MC return is a realized policy return and retains aleatoric outcome variance.",
            "A1 is representation identifiability only; no algorithm ranking or policy improvement was run.",
        ],
    }
    atomic_json(output / "critic_comparison.json", comparison)
    plot_scatter(output / "predicted_vs_mc_scatter.png", rows.target[test_indices], predictions)
    np.savez_compressed(output / "heldout_predictions.npz", target=rows.target[test_indices], **predictions)
    atomic_json(output / "progress.json", {
        "status": "complete", "decision": classification,
        "evidence_to_enter_a2": evidence_a2, "elapsed_seconds": time.monotonic() - start,
    })
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/critic_identifiability_audit_20260917/a0_a1.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/2026-09-17_critic_identifiability_audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("qualify", "scan", "collect", "fit", "rescore", "all"), default="all")
    parser.add_argument("--scan-start", type=int, default=2026096001)
    parser.add_argument("--scan-count", type=int, default=200)
    parser.add_argument("--scan-stop-collisions", type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["scope"] != "A0_GATE_A1_ONLY" or not all(config["forbidden"].values()):
        raise ValueError("scope/forbidden contract drift")
    atomic_json(args.output / "launch.json", {
        "schema": SCHEMA, "config": str(args.config), "config_sha256": canonical_hash(config),
        "device": args.device, "pid": os.getpid(), "scope": config["scope"],
        "forbidden": config["forbidden"], "started_at_unix": time.time(),
    })
    if args.stage == "qualify":
        qualification = qualify_a0(config, args.output, args.device)
        print(json.dumps(qualification, indent=2))
        return 0 if qualification["decision"] == "PASS" else 2
    if args.stage == "scan":
        report = scan_collision_seeds(
            config, args.output, args.device,
            args.scan_start, args.scan_count, args.scan_stop_collisions,
        )
        print(json.dumps({
            "evaluated": report["evaluated"],
            "collision_seeds": report["collision_seeds"],
            "actor_frozen_bit_exact": report["actor_frozen_bit_exact"],
        }, indent=2))
        return 0 if len(report["collision_seeds"]) >= args.scan_stop_collisions else 2
    if args.stage in {"collect", "all"}:
        gate = collect_a0(config, args.output, args.device)
        print(json.dumps(gate, indent=2))
        if gate["decision"] != "PASS":
            return 2
    if args.stage in {"fit", "all"}:
        comparison = run_a1(config, args.output, args.device)
        print(json.dumps({
            "decision": comparison["decision"],
            "evidence_to_enter_a2": comparison["evidence_to_enter_a2"],
            "heldout": {kind: comparison["results"][kind]["test"]["overall"] for kind in ("LQ", "NQ", "CQ", "V")},
        }, indent=2))
    if args.stage == "rescore":
        comparison = run_a1(config, args.output, args.device, reuse_checkpoints=True)
        print(json.dumps({
            "decision": comparison["decision"],
            "evidence_to_enter_a2": comparison["evidence_to_enter_a2"],
            "heldout": {kind: comparison["results"][kind]["test"]["overall"] for kind in ("LQ", "NQ", "CQ", "V")},
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
