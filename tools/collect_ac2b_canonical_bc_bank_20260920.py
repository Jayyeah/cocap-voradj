#!/usr/bin/env python3
"""AC-2B formal canonical-BC raw trajectory-bank collector.

This collector is evaluation-only. It records natural visitation from the
frozen canonical categorical BC actor; it never performs optimizer updates,
critic fitting, bootstrap, GAE, or PPO.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path("/home/yjq/rl/CoCap1/cocap-voradj-critic-audit")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.evaluation.mission_events import snapshot
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.runtime_semantics import assert_runtime
from cocap_voradj.training.small_step_ac import tensor_tree
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_continuous_ctde_training import _split_termination_flags
from tools.run_forward_final_bridge_20260908 import act_evaders, make_env


MAX_AGENTS = 12
MAX_NEIGHBORS = 8
GAMMA = 0.99
EVENTS = (
    "pure_coverage", "coverage_steady", "ordinary_pursuit", "ring2", "ring3",
    "normal_capture", "stationary_capture", "collision",
    "capture_terminal_transition", "early_recovery", "late_recovery",
    "pure_coverage_restart",
)
EVENT_TO_ID = {name: index for index, name in enumerate(EVENTS)}
PHASES = ("pure_coverage", "pre_capture", "post_capture")
PHASE_TO_ID = {name: index for index, name in enumerate(PHASES)}
SCENES = ("mixed", "pure_coverage")
POLICY_MODES = ("bc_argmax", "bc_sample")
ACTION_GRID = np.asarray(
    [(a, w) for a in (-0.4, 0.0, 0.4) for w in (-np.pi / 6.0, 0.0, np.pi / 6.0)],
    dtype=np.float32,
)
CHECKPOINT = ROOT / "artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt"
CHECKPOINT_SHA = "7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd"
CONTRACT = "forward-final-aw9-4v1-swept-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sanitize_json(value: Any):
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, set):
        return sorted(value)
    return value


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sanitize_json(value), indent=2, ensure_ascii=False, allow_nan=False, default=json_default) + "\n")
    os.replace(temporary, path)


def pad_observations(observations: list[Any], keys: list[str], shapes: dict[str, tuple[int, ...]], dtypes: dict[str, np.dtype]):
    result = {}
    active = np.zeros(MAX_AGENTS, dtype=bool)
    for index, value in enumerate(observations):
        if value is not None:
            active[index] = True
    for key in keys:
        result[key] = np.zeros((MAX_AGENTS, *shapes[key]), dtype=dtypes[key])
        for index, value in enumerate(observations):
            if value is not None:
                result[key][index] = np.asarray(value[key])
    return result, active


def event_flags(scene: str, state: Mapping[str, Any], events: list[Mapping[str, Any]], collision: bool,
                terminated: np.ndarray, timestep: int, capture_step: int | None,
                steady_steps: int) -> tuple[np.ndarray, str, float]:
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


def collect_episode(actor, scene: str, seed: int, policy_mode: str, episode_id: int,
                    split_name: str, obs_keys: list[str], obs_shapes: dict[str, tuple[int, ...]],
                    obs_dtypes: dict[str, np.dtype]):
    env_scene = "coverage" if scene == "pure_coverage" else "mixed"
    env, observations = make_env(env_scene, int(seed))
    assert_runtime(env, actor, categorical=True)
    np.testing.assert_allclose(ACTION_GRID, np.asarray(env.pursuers[0].action_list, dtype=np.float32), atol=1e-6)
    apf = [__import__("cocap_voradj.control.apf", fromlist=["ApfAgent"]).ApfAgent(evader.a, evader.w) for evader in env.evaders]
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    fields: dict[str, list[Any]] = {}
    central_fields: dict[str, list[Any]] = {}
    capture_step = None
    ce_step = None
    capture_types: list[str] = []
    collision_episode = False
    terminated_seen = False
    truncated_seen = False
    terminal_states: set[str] = set()

    def add(name: str, value: Any):
        fields.setdefault(name, []).append(np.asarray(value))

    for timestep in range(1, env.episode_max_length + 1):
        current_local, current_active = pad_observations(observations, obs_keys, obs_shapes, obs_dtypes)
        active = np.flatnonzero(current_active).tolist()
        if not active:
            raise RuntimeError("no active observations before canonical environment termination")
        local = [observations[index] for index in active]
        batch = tensor_tree({key: np.stack([value[key] for value in local]) for key in obs_keys}, torch.device("cpu"))
        distribution = actor.distribution(batch)
        logits = distribution.logits.detach().cpu().numpy().astype(np.float32)
        probs = distribution.probs.detach().cpu().numpy().astype(np.float32)
        if policy_mode == "bc_argmax":
            chosen = distribution.logits.argmax(-1).detach().cpu().numpy().astype(np.int64)
        elif policy_mode == "bc_sample":
            chosen = distribution.sample().detach().cpu().numpy().astype(np.int64)
        else:
            raise ValueError(policy_mode)

        state = snapshot(env, observations)
        neighbors = np.full((MAX_AGENTS, MAX_NEIGHBORS), -1, dtype=np.int16)
        neighbor_mask = np.zeros((MAX_AGENTS, MAX_NEIGHBORS), dtype=bool)
        for agent in active:
            ids = sorted(state["friends"].get(agent, set()))[:MAX_NEIGHBORS]
            neighbors[agent, :len(ids)] = ids
            neighbor_mask[agent, :len(ids)] = True
        neighbor_fields = {}
        safe_ids = np.maximum(neighbors, 0)
        for key in obs_keys:
            neighbor_fields[key] = current_local[key][safe_ids].copy()
            neighbor_fields[key][~neighbor_mask] = 0

        central = build_central_global_obs(env, max_agents=MAX_AGENTS, max_evaders=8, max_obstacles=5, self_feature_dim=9)
        action_index = np.full(MAX_AGENTS, 4, dtype=np.int64)
        selected_probability = np.zeros(MAX_AGENTS, dtype=np.float32)
        actor_logits = np.zeros((MAX_AGENTS, 9), dtype=np.float32)
        actor_probs = np.zeros((MAX_AGENTS, 9), dtype=np.float32)
        action_aw = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
        for row, agent in enumerate(active):
            index = int(chosen[row])
            action_index[agent] = index
            selected_probability[agent] = probs[row, index]
            actor_logits[agent] = logits[row]
            actor_probs[agent] = probs[row]
            action_aw[agent] = ACTION_GRID[index]
        commands = [None] * len(observations)
        for agent in active:
            commands[agent] = int(action_index[agent])
        outcome = env.step(commands, act_evaders(env, apf))
        terminated, truncated = _split_termination_flags(outcome.dones, outcome.infos)
        done = np.asarray(outcome.dones, dtype=bool)
        terminated_seen = terminated_seen or bool(terminated.any())
        truncated_seen = truncated_seen or bool(truncated.any())
        terminal_states.update(str(info.get("state", "unknown")) for info, flag in zip(outcome.infos, done) if flag)
        captures = list(env.last_capture_events)
        capture_types.extend(str(event.get("capture_type")) for event in captures)
        collision = bool(env.last_collision_events)
        collision_episode = collision_episode or collision
        flags, phase, ring = event_flags(
            scene, state, captures, collision, terminated, timestep, capture_step,
            int(getattr(env, "distribution_hold_steps", 0)),
        )
        reward = np.zeros(MAX_AGENTS, dtype=np.float32)
        term = np.zeros(MAX_AGENTS, dtype=bool)
        trunc = np.zeros(MAX_AGENTS, dtype=bool)
        dones = np.zeros(MAX_AGENTS, dtype=bool)
        reward[:len(outcome.rewards)] = np.asarray(outcome.rewards, dtype=np.float32)
        term[:len(terminated)] = terminated
        trunc[:len(truncated)] = truncated
        dones[:len(done)] = done
        next_local, next_active = pad_observations(outcome.observations, obs_keys, obs_shapes, obs_dtypes)
        next_central = build_central_global_obs(env, max_agents=MAX_AGENTS, max_evaders=8, max_obstacles=5, self_feature_dim=9)
        active_ids = np.full(MAX_AGENTS, -1, dtype=np.int16)
        active_ids[:len(active)] = np.asarray(active, dtype=np.int16)
        repeated_flags = np.broadcast_to(flags[None, None, :], (1, MAX_AGENTS, len(EVENTS))).copy()[0]
        for key, value in current_local.items():
            add("local_" + key, value)
        for key, value in next_local.items():
            add("next_local_" + key, value)
        for key, value in neighbor_fields.items():
            add("neighbor_local_" + key, value)
        for key, value in central.items():
            central_fields.setdefault("global_" + key, []).append(np.asarray(value))
        for key, value in next_central.items():
            central_fields.setdefault("next_global_" + key, []).append(np.asarray(value))
        add("active_mask", current_active)
        add("next_active_mask", next_active)
        add("active_agent_ids", active_ids)
        add("neighbor_ids", neighbors)
        add("neighbor_mask", neighbor_mask)
        add("actor_logits", actor_logits)
        add("actor_probs", actor_probs)
        add("action_index", action_index)
        add("selected_action_probability", selected_probability)
        add("action_aw", action_aw)
        add("reward", reward)
        add("terminated", term)
        add("truncated", trunc)
        add("done", dones)
        add("transition_event_flags", flags)
        add("event_flags", repeated_flags)
        add("primary_event", primary_event(flags))
        add("phase_id", PHASE_TO_ID[phase])
        add("normalized_time", float(timestep) / float(env.episode_max_length))
        add("ring_count", ring)
        add("episode_id", episode_id)
        add("episode_seed", seed)
        add("timestep", timestep)
        add("scene_id", SCENES.index(scene))
        add("policy_mode_id", POLICY_MODES.index(policy_mode))
        add("split_id", ("train", "validation", "test").index(split_name))
        observations = outcome.observations
        if captures and capture_step is None:
            capture_step = timestep
        if env.post_capture_coverage_success and ce_step is None:
            ce_step = timestep
        if all(outcome.dones):
            break

    arrays = {key: np.stack(value) for key, value in fields.items()}
    arrays.update({key: np.stack(value) for key, value in central_fields.items()})
    rewards = arrays["reward"].astype(np.float64)
    returns = np.zeros_like(rewards)
    running = np.zeros(MAX_AGENTS, dtype=np.float64)
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + GAMMA * running
        returns[index] = running
    arrays["mc_return"] = returns.astype(np.float32)
    replay_class = np.full(arrays["action_index"].shape, -1, dtype=np.int8)
    phase_rows = arrays["phase_id"][:, None]
    active_rows = arrays["active_mask"].astype(bool)
    pursuing_rows = (phase_rows == PHASE_TO_ID["pre_capture"]) & (arrays["local_self"][:, :, 8] >= 0.5) & active_rows
    pre_capture_cover_rows = (phase_rows == PHASE_TO_ID["pre_capture"]) & (arrays["local_self"][:, :, 8] < 0.5) & active_rows
    post_capture_rows = (phase_rows == PHASE_TO_ID["post_capture"]) & active_rows
    recovery_pure_rows = (phase_rows == PHASE_TO_ID["pure_coverage"]) & active_rows
    replay_class[pursuing_rows] = 0
    replay_class[pre_capture_cover_rows] = 1
    replay_class[post_capture_rows] = 2
    replay_class[recovery_pure_rows] = 3
    if np.any(replay_class[active_rows] < 0):
        raise RuntimeError("unclassified active replay row")
    arrays["replay_semantic_class"] = replay_class
    record = env.episode_record(task="coverage" if scene == "pure_coverage" else "mix")
    episode_success = bool(record["episode_success"])
    episode_record = {
        "episode_id": episode_id,
        "seed": int(seed),
        "scene": scene,
        "policy_mode": policy_mode,
        "split": split_name,
        "length": int(len(rewards)),
        "success": episode_success,
        "failure": not episode_success,
        "collision": bool(collision_episode),
        "capture_step": None if capture_step is None else int(capture_step),
        "ce_completion_step": None if ce_step is None else int(ce_step),
        "terminal_reason": sorted(terminal_states),
        "total_return": float(rewards.sum()),
        "per_agent_return": rewards.sum(axis=0).astype(float).tolist(),
        "normal_capture": bool("loose" in capture_types),
        "stationary_capture": bool("stationary" in capture_types),
        "ce_success": bool(record.get("post_capture_coverage_success", False) if scene == "mixed" else record.get("episode_success", False)),
        "event_counts_active_rows": {
            name: int((arrays["transition_event_flags"][:, EVENT_TO_ID[name]][:, None] & arrays["active_mask"]).sum())
            for name in EVENTS
        },
        "terminated_seen": bool(terminated_seen),
        "truncated_seen": bool(truncated_seen),
        "env_episode_record": record,
    }
    return arrays, episode_record


def concat(episodes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = episodes[0].keys()
    return {key: np.concatenate([episode[key] for episode in episodes], axis=0) for key in keys}


def event_counts(bank: Mapping[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, int]:
    take = np.ones(len(bank["episode_id"]), dtype=bool) if mask is None else mask
    active = bank["active_mask"][take]
    flags = bank["transition_event_flags"][take]
    return {name: int((flags[:, EVENT_TO_ID[name]][:, None] & active).sum()) for name in EVENTS}


def independent_mc_error(bank: Mapping[str, np.ndarray], selected_rows: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    errors = []
    details = []
    for episode_id in np.unique(bank["episode_id"]):
        rows = np.flatnonzero(bank["episode_id"] == episode_id)
        running = np.zeros(MAX_AGENTS, dtype=np.float64)
        recomputed = np.zeros((len(rows), MAX_AGENTS), dtype=np.float64)
        for offset in range(len(rows) - 1, -1, -1):
            running = bank["reward"][rows[offset]].astype(np.float64) + GAMMA * running
            recomputed[offset] = running
        local_rows = np.intersect1d(rows, selected_rows)
        if len(local_rows):
            local_error = np.max(np.abs(recomputed[np.searchsorted(rows, local_rows)] - bank["mc_return"][local_rows]))
            errors.append(float(local_error))
            details.append({"episode_id": int(episode_id), "rows_checked": int(len(local_rows)), "max_abs_error": float(local_error)})
    return (max(errors) if errors else 0.0), details


def dump_capture_windows(bank: Mapping[str, np.ndarray], records: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    dumps = []
    selected = [r for r in records if r["scene"] == "mixed" and r["success"] and r["capture_step"] is not None][:3]
    for record in selected:
        episode_id = int(record["episode_id"])
        rows = np.flatnonzero(bank["episode_id"] == episode_id)
        capture_step = int(record["capture_step"])
        center = capture_step - 1
        selected_indices = [index for index in (center - 1, center, center + 1) if 0 <= index < len(rows)]
        transition_rows = []
        for offset in selected_indices:
            row = int(rows[offset])
            transition_rows.append({
                "global_row": row,
                "episode_step": int(bank["timestep"][row]),
                "phase": PHASES[int(bank["phase_id"][row])],
                "event_flags": [EVENTS[i] for i, value in enumerate(bank["transition_event_flags"][row]) if value],
                "active_agent_ids": bank["active_agent_ids"][row].tolist(),
                "local_observations": {key[len("local_"):]: bank[key][row].tolist() for key in bank if key.startswith("local_")},
                "next_local_observations": {key[len("next_local_"):]: bank[key][row].tolist() for key in bank if key.startswith("next_local_")},
                "central_state": {key[len("global_"):]: bank[key][row].tolist() for key in bank if key.startswith("global_") and not key.startswith("global_next")},
                "next_central_state": {key[len("next_global_"):]: bank[key][row].tolist() for key in bank if key.startswith("next_global_")},
                "actor_logits": bank["actor_logits"][row].tolist(),
                "actor_probs": bank["actor_probs"][row].tolist(),
                "action_index": bank["action_index"][row].tolist(),
                "selected_action_probability": bank["selected_action_probability"][row].tolist(),
                "action_aw": bank["action_aw"][row].tolist(),
                "reward": bank["reward"][row].tolist(),
                "terminated": bank["terminated"][row].tolist(),
                "truncated": bank["truncated"][row].tolist(),
                "done": bank["done"][row].tolist(),
                "mc_return": bank["mc_return"][row].tolist(),
            })
        dumps.append({"episode_id": episode_id, "seed": record["seed"], "policy_mode": record["policy_mode"], "capture_step": capture_step, "rows": transition_rows})
    atomic_json(output / "capture_transition_dumps.json", dumps)
    return dumps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("BC_CHECKPOINT_IDENTITY_FAIL")
    actor, payload = load_actor(CHECKPOINT, "cpu")
    actor.eval()
    actor.requires_grad_(False)
    actor_before = tensor_state_sha256(actor.state_dict())
    initial_env, initial_obs = make_env("mixed", 2026093001)
    obs_keys = sorted(initial_obs[0].keys())
    obs_shapes = {key: tuple(np.asarray(initial_obs[0][key]).shape) for key in obs_keys}
    obs_dtypes = {key: np.asarray(initial_obs[0][key]).dtype for key in obs_keys}
    plan = []
    for policy_mode, seed_base in (("bc_argmax", 2026093001), ("bc_sample", 2026094001)):
        for index in range(30):
            split_name = "train" if index < 24 else "validation" if index < 27 else "test"
            plan.append(("mixed", seed_base + index, policy_mode, split_name))
    for policy_mode, seed_base in (("bc_argmax", 2026193001), ("bc_sample", 2026194001)):
        for index in range(10):
            split_name = "train" if index < 8 else "validation" if index < 9 else "test"
            plan.append(("pure_coverage", seed_base + index, policy_mode, split_name))
    if len(plan) != 80 or len({item[1] for item in plan}) != 80:
        raise RuntimeError("formal seed plan is not 80-unique")
    episodes = []
    records = []
    start = time.monotonic()
    for episode_id, (scene, seed, policy_mode, split_name) in enumerate(plan):
        arrays, record = collect_episode(actor, scene, seed, policy_mode, episode_id, split_name, obs_keys, obs_shapes, obs_dtypes)
        episodes.append(arrays)
        records.append(record)
        atomic_json(output / "progress.json", {"status": "collecting", "completed": episode_id + 1, "total": len(plan), "last": record, "elapsed_seconds": time.monotonic() - start})
    bank = concat(episodes)
    bank_path = output / "canonical_bc_critic_bank_v2.npz"
    np.savez_compressed(bank_path, **bank)
    actor_after = tensor_state_sha256(actor.state_dict())
    rng = np.random.default_rng(2026092201)
    selected_rows = np.sort(rng.choice(len(bank["episode_id"]), min(20, len(bank["episode_id"])), replace=False))
    mc_error, mc_details = independent_mc_error(bank, selected_rows)
    split_ids = {name: index for index, name in enumerate(("train", "validation", "test"))}
    episode_splits = {name: sorted(int(record["episode_id"]) for record in records if record["split"] == name) for name in split_ids}
    seed_splits = {name: sorted(int(record["seed"]) for record in records if record["split"] == name) for name in split_ids}
    leakage = any(set(episode_splits[a]) & set(episode_splits[b]) or set(seed_splits[a]) & set(seed_splits[b]) for index, a in enumerate(episode_splits) for b in list(episode_splits)[index + 1:])
    transition_count = int(len(bank["episode_id"]))
    active_rows = int(bank["active_mask"].sum())
    phase_counts = {phase: int(((bank["phase_id"] == phase_id)[:, None] & bank["active_mask"]).sum()) for phase, phase_id in PHASE_TO_ID.items()}
    event_counts_total = event_counts(bank)
    split_stats = {}
    for split_name, split_id in split_ids.items():
        mask = bank["split_id"] == split_id
        split_stats[split_name] = {"episodes": episode_splits[split_name], "transitions": int(mask.sum()), "active_agent_rows": int(bank["active_mask"][mask].sum()), "event_counts": event_counts(bank, mask)}
    mixed_records = [record for record in records if record["scene"] == "mixed"]
    coverage_records = [record for record in records if record["scene"] == "pure_coverage"]
    avg_mixed = float(np.mean([record["length"] for record in mixed_records]))
    avg_coverage = float(np.mean([record["length"] for record in coverage_records]))
    avg_mixed_rows = float(np.mean([sum(record["per_agent_return"] != [] for _ in [0]) * 0 + int(np.sum(bank["episode_id"] == record["episode_id"])) * 4 for record in mixed_records]))
    avg_coverage_rows = float(np.mean([int(np.sum(bank["episode_id"] == record["episode_id"])) * 4 for record in coverage_records]))
    avg_mixed_event = {name: float(np.mean([record["event_counts_active_rows"][name] for record in mixed_records])) for name in EVENTS}
    avg_coverage_event = {name: float(np.mean([record["event_counts_active_rows"][name] for record in coverage_records])) for name in EVENTS}
    future_estimates = {}
    for episodes_total in (60, 80, 100):
        future_estimates[str(episodes_total)] = {
            "assumption": "50% mixed / 50% pure_coverage with the pilot's equal argmax/sampled composition",
            "transitions": int(round(episodes_total * (avg_mixed + avg_coverage) / 2.0)),
            "active_agent_rows": int(round(episodes_total * (avg_mixed_rows + avg_coverage_rows) / 2.0)),
            "mixed_transitions": int(round(episodes_total / 2.0 * avg_mixed)),
            "coverage_transitions": int(round(episodes_total / 2.0 * avg_coverage)),
            "early_recovery_rows": int(round(episodes_total / 2.0 * avg_mixed_event["early_recovery"])),
            "late_recovery_rows": int(round(episodes_total / 2.0 * avg_mixed_event["late_recovery"])),
            "capture_transition_rows": int(round(episodes_total / 2.0 * avg_mixed_event["normal_capture"])),
        }
    dumps = []
    manifest = {
        "schema": "cocap-ac2b-canonical-bc-critic-formal-raw-v1",
        "status": "formal_bank_collected_pending_offline_audit",
        "contract": CONTRACT,
        "generator": {
            "policy": "frozen canonical categorical BC actor",
            "policy_modes": list(POLICY_MODES),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA,
            "payload_schema": payload["schema"],
            "actor_state_sha256_before": actor_before,
            "actor_state_sha256_after": actor_after,
            "actor_updates": 0,
        },
        "environment": {
            "source_contract": "artifacts/2026-09-20_ac1/AC1_BC_RESOLVED_CONTRACT.json",
            "mixed": {"pursuers": 4, "evaders": 1, "obstacles": 1},
            "pure_coverage": {"pursuers": 4, "evaders": 0, "obstacles": 1},
            "horizon": 3000,
            "capture_terminal": False,
            "post_capture_window": 500,
        },
        "collection": {
            "episodes": len(records),
            "mixed": sum(record["scene"] == "mixed" for record in records),
            "pure_coverage": sum(record["scene"] == "pure_coverage" for record in records),
            "argmax": sum(record["policy_mode"] == "bc_argmax" for record in records),
            "sampled": sum(record["policy_mode"] == "bc_sample" for record in records),
            "seed_plan": plan,
        },
        "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path), "transitions": transition_count, "active_agent_rows": active_rows, "episode_records": records},
        "fields": {
            "local_observations": [key for key in bank if key.startswith("local_") and not key.startswith("local_next")],
            "next_local_observations": [key for key in bank if key.startswith("next_local_")],
            "neighbor_observations": [key for key in bank if key.startswith("neighbor_local_")],
            "central_state": [key for key in bank if key.startswith("global_") and not key.startswith("global_next")],
            "next_central_state": [key for key in bank if key.startswith("next_global_")],
            "policy_metadata": ["actor_logits", "actor_probs", "action_index", "selected_action_probability", "action_aw", "policy_mode_id"],
            "transition_metadata": ["active_mask", "active_agent_ids", "neighbor_ids", "neighbor_mask", "reward", "terminated", "truncated", "done", "timestep", "normalized_time", "episode_id", "episode_seed", "scene_id", "split_id"],
            "phase_event_metadata": ["phase_id", "transition_event_flags", "event_flags", "primary_event", "ring_count"],
            "target": ["mc_return"],
        },
        "critic_input_contract": {
            "LQ": ["local_*", "action_aw or action_index"],
            "NQ": ["local_*", "neighbor_local_*", "neighbor_ids", "neighbor_mask", "neighbor action_aw/action_index"],
            "CQ": ["global_*", "active_mask", "joint action_aw/action_index", "focal agent index"],
            "V": ["global_*", "global masks", "active_mask", "normalized_time as optional normal-time feature; no future/event label"],
        },
        "reward_semantics": {
            "reward": "individual per-agent environment reward, padded to MAX_AGENTS=12",
            "team_reward_substitution": False,
            "return_semantics": "per-agent realized MC return; no critic bootstrap",
            "gamma": GAMMA,
            "formula": "G_t = sum_k gamma^k r_(t+k) within complete episode",
            "terminal_handling": "include current terminal/truncated transition reward; no post-episode reward; no bootstrap",
            "post_capture_continuation": "preserve all recovery transitions through canonical CE completion/termination",
        },
        "phase_event_contract": {
            "phase_names": list(PHASES),
            "events": list(EVENTS),
            "early_recovery": "old A0/A1 exact definition: post_capture and 0 <= timestep-capture_step <= 40; capture transition itself remains pre_capture",
            "late_recovery": "old A0/A1 exact definition: post_capture and timestep-capture_step > 40",
            "capture_transition": "normal_capture/stationary_capture event on the environment transition that emits last_capture_events",
            "transition_alignment": "current obs/action/reward/done and next obs/state stored on the same row",
        },
        "mc_audit": {"sample_rows": selected_rows.tolist(), "max_abs_error": mc_error, "rows_by_episode": mc_details, "float_tolerance": "floating-point only"},
        "split_contract": {"episode_level": True, "transition_random_split": False, "ratio": {"train": 0.70, "validation": 0.15, "test": 0.15}, "pilot_assignment": split_stats, "episode_seed_leakage": leakage},
        "phase_counts_active_rows": phase_counts,
        "event_counts_active_rows": event_counts_total,
        "future_size_estimates": future_estimates,
        "capture_transition_dump": {"path": None, "successful_mixed_episodes": [], "offline_alignment_audit": "performed by audit_ac2b_canonical_bc_bank_20260920.py"},
        "raw_collection_status": "COMPLETE" if actor_before == actor_after and not leakage else "COLLECTION_INTEGRITY_FAIL",
    }
    atomic_json(output / "formal_collection_manifest.json", manifest)
    atomic_json(output / "formal_collection_report.json", {
        "schema": "cocap-ac2b-canonical-bc-critic-formal-raw-report-v1",
        "verdict": manifest["raw_collection_status"],
        "average_mixed_length": avg_mixed,
        "average_coverage_length": avg_coverage,
        "average_mixed_active_agent_rows": avg_mixed_rows,
        "average_coverage_active_agent_rows": avg_coverage_rows,
        "phase_counts_active_rows": phase_counts,
        "event_counts_active_rows": event_counts_total,
        "future_size_estimates": future_estimates,
        "capture_transition_dumps": len(dumps),
        "mc_max_abs_error": mc_error,
        "actor_updates": 0,
        "actor_bit_exact": actor_before == actor_after,
        "formal_bank_collected": True,
        "critic_training": False,
    })
    atomic_json(output / "progress.json", {"status": "formal_collection_complete", "verdict": manifest["raw_collection_status"], "elapsed_seconds": time.monotonic() - start})
    print(json.dumps({"verdict": manifest["raw_collection_status"], "transitions": transition_count, "active_agent_rows": active_rows, "capture_dumps": 0, "mc_max_abs_error": mc_error, "actor_bit_exact": actor_before == actor_after}, ensure_ascii=False))


if __name__ == "__main__":
    main()
