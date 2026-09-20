#!/usr/bin/env python3
"""AC-4A: frozen-BC AW9 counterfactual branching pilot.

This is a small, evaluation-only audit.  It collects fresh canonical BC
states, validates an in-memory deepcopy/RNG snapshot restore, then branches
only the focal agent's first AW9 action.  Every continuation action is frozen
canonical categorical-BC argmax.  No critic, actor, bootstrap, TD, GAE, PPO,
or optimizer update is allowed here.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path("/home/yjq/rl/CoCap1/cocap-voradj-critic-audit")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.evaluation.mission_events import snapshot as mission_snapshot
from cocap_voradj.models.critic_identifiability import parameter_count
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.small_step_ac import tensor_tree
from tools.collect_ac2b_canonical_bc_bank_20260920 import (
    ACTION_GRID,
    CHECKPOINT_SHA,
    EVENTS,
    EVENT_TO_ID,
    PHASE_TO_ID,
    event_flags,
)
from tools.distill_forward_final_actor_20260908 import load_actor
from tools.run_ac3a_critic_sampler_audit_20260920 import make_config, sha256_file
from tools.run_critic_identifiability_audit_20260917 import (
    ACTION_SCALE,
    MAX_AGENTS,
    MAX_NEIGHBORS,
    forward,
    make_model,
    to_tensor,
)
from tools.run_continuous_ctde_training import _split_termination_flags
from tools.run_forward_final_bridge_20260908 import act_evaders, make_env


OUT = ROOT / "artifacts/2026-09-20_ac4a"
ACTOR_REQUESTED = ROOT / "artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt"
ACTOR_LOADED = ROOT / "artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt"
ACTOR_SHA256 = CHECKPOINT_SHA
AC3A_SUMMARY = ROOT / "artifacts/2026-09-20_ac3a/summary.json"
AC3B_SUMMARY = ROOT / "artifacts/2026-09-20_ac3b/summary.json"
BANK_SHA256 = "07e08d17ae6283d785e0d2e00307f330b13a88dcd0f8ce83f92f0d451ad7808a"
GAMMA = 0.99
ANCHOR_TARGETS = {"pursuing": 4, "pre_capture_cover": 4, "early_recovery": 4, "recovery_pure": 4}
SCENES = ("mixed", "pure_coverage")
SEED_BASE = 2026092401
MAX_MIXED_EPISODES = 40
MAX_COVERAGE_EPISODES = 20
DETERMINISM_STATES = 10
CONTINUATION_STEPS = 5


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def rng_state() -> dict[str, Any]:
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def restored_global_rng(state: Mapping[str, Any]):
    outer = rng_state()
    restore_rng(state)
    try:
        yield
    finally:
        restore_rng(outer)


def save_env_state(env: Any, observations: list[Any], capture_step: int | None = None) -> dict[str, Any]:
    """Deep-copy all env object state plus explicit process RNG state.

    VorAdjEnv owns its trajectory RNG in ``env.rng``.  The explicit process
    RNG copies cover any future runtime/helper path that uses Python, NumPy,
    or Torch global RNGs.  No serialized snapshot is written to disk.
    """
    return {
        "env": copy.deepcopy(env),
        "observations": copy.deepcopy(observations),
        "rng": rng_state(),
        "capture_step": capture_step,
    }


def restore_env_state(state: Mapping[str, Any]) -> tuple[Any, list[Any]]:
    return copy.deepcopy(state["env"]), copy.deepcopy(state["observations"])


def pad_observations(observations: list[Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    live = next(value for value in observations if value is not None)
    keys = tuple(live.keys())
    active = np.zeros(MAX_AGENTS, dtype=bool)
    shapes = {key: np.asarray(live[key]).shape for key in keys}
    dtypes = {key: np.asarray(live[key]).dtype for key in keys}
    result = {key: np.zeros((MAX_AGENTS, *shapes[key]), dtype=dtypes[key]) for key in keys}
    for index, value in enumerate(observations):
        if value is not None:
            active[index] = True
            for key in keys:
                result[key][index] = value[key]
    return result, active


def canonical_actions(actor: torch.nn.Module, observations: list[Any]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    active = np.flatnonzero([value is not None for value in observations])
    local = [observations[index] for index in active]
    batch = tensor_tree({key: np.stack([value[key] for value in local]) for key in local[0]}, torch.device("cpu"))
    with torch.no_grad():
        distribution = actor.distribution(batch)
        chosen = distribution.logits.argmax(-1).cpu().numpy().astype(np.int64)
        logits = distribution.logits.cpu().numpy().astype(np.float32)
        probs = distribution.probs.cpu().numpy().astype(np.float32)
    full = np.full(len(observations), 4, dtype=np.int64)
    full[active] = chosen
    return full, {"active": active.astype(np.int64), "logits": logits, "probs": probs}, full


def current_phase_class(
    scene: str,
    observations: list[Any],
    focal: int,
    capture_step: int | None,
    transition_step: int,
) -> tuple[str, str, int]:
    if scene == "pure_coverage":
        return "pure_coverage", "recovery_pure", -1
    if capture_step is not None:
        age = int(transition_step - capture_step)
        if 0 <= age <= 40:
            return "post_capture", "early_recovery", age
        return "post_capture", "post_capture_real", age
    self_value = np.asarray(observations[focal]["self"]).reshape(-1)
    pursuing = bool(len(self_value) > 8 and self_value[8] >= 0.5)
    return "pre_capture", "pursuing" if pursuing else "pre_capture_cover", -1


def event_signature(flags: np.ndarray, captures: list[Mapping[str, Any]], collisions: list[Mapping[str, Any]]) -> str:
    return json.dumps(json_safe({"flags": flags.astype(bool).tolist(), "captures": captures, "collisions": collisions}), sort_keys=True)


def step_once(
    env: Any,
    observations: list[Any],
    actions: np.ndarray,
    apf_agents: list[ApfAgent],
    scene: str,
    capture_step: int | None,
) -> dict[str, Any]:
    before = mission_snapshot(env, observations)
    commands = [int(actions[index]) if observations[index] is not None else None for index in range(len(observations))]
    outcome = env.step(commands, act_evaders(env, apf_agents))
    terminated, truncated = _split_termination_flags(outcome.dones, outcome.infos)
    captures = [dict(event) for event in env.last_capture_events]
    collisions = [dict(event) for event in env.last_collision_events]
    collision = bool(collisions)
    transition_step = int(env.episode_step)
    flags, phase, ring = event_flags(
        scene,
        before,
        captures,
        collision,
        terminated,
        transition_step,
        capture_step,
        int(getattr(env, "distribution_hold_steps", 0)),
    )
    next_capture_step = transition_step if captures and capture_step is None else capture_step
    return {
        "observations": outcome.observations,
        "outcome": outcome,
        "terminated": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "done": np.asarray(outcome.dones, dtype=bool),
        "captures": captures,
        "collisions": collisions,
        "flags": flags,
        "phase": phase,
        "ring": float(ring),
        "capture_step": next_capture_step,
        "event_signature": event_signature(flags, captures, collisions),
    }


def canonical_path_from_state(
    bundle: Mapping[str, Any],
    actor: torch.nn.Module,
    scene: str,
    focal: int,
    forced_action: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    with restored_global_rng(bundle["rng"]):
        env, observations = restore_env_state(bundle)
        apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
        capture_step = bundle.get("capture_step")
        rewards: list[float] = []
        trace: list[dict[str, Any]] = []
        collisions = False
        capture_types: list[str] = []
        first = True
        while True:
            if not any(value is not None for value in observations):
                break
            actions, _, _ = canonical_actions(actor, observations)
            if first and forced_action is not None:
                actions[focal] = int(forced_action)
            step = step_once(env, observations, actions, apf_agents, scene, capture_step)
            trace.append({
                "observations": copy.deepcopy(step["observations"]),
                "reward": np.asarray(step["outcome"].rewards, dtype=np.float64).copy(),
                "done": step["done"].copy(),
                "terminated": step["terminated"].copy(),
                "truncated": step["truncated"].copy(),
                "event_signature": step["event_signature"],
                "phase": step["phase"],
            })
            reward = float(step["outcome"].rewards[focal]) if focal < len(step["outcome"].rewards) else 0.0
            rewards.append(reward)
            collisions = collisions or bool(step["collisions"])
            capture_types.extend(str(event.get("capture_type")) for event in step["captures"])
            observations = step["observations"]
            capture_step = step["capture_step"]
            first = False
            if bool(step["done"].all()) or (max_steps is not None and len(trace) >= max_steps):
                break
        record = env.episode_record(task="coverage" if scene == "pure_coverage" else "mix")
        returns = float(sum((GAMMA ** index) * value for index, value in enumerate(rewards)))
        return {
            "return": returns,
            "rewards": rewards,
            "length": int(env.episode_step),
            "collision": bool(collisions or record.get("collision_event", False)),
            "captured": bool(record.get("captured", False)),
            "normal_capture": bool("loose" in capture_types),
            "stationary_capture": bool("stationary" in capture_types),
            "ce_success": bool(env.post_capture_coverage_success),
            "safe_complete": bool(record.get("episode_success", False)),
            "terminal_reason": sorted(str(value) for value in record.get("collision_type_counts", {}).keys()),
            "trace": trace,
        }


def compare_observations(first: list[Any], second: list[Any]) -> float:
    if len(first) != len(second):
        return float("inf")
    maximum = 0.0
    for left, right in zip(first, second):
        if (left is None) != (right is None):
            return float("inf")
        if left is None:
            continue
        if set(left) != set(right):
            return float("inf")
        for key in left:
            a = np.asarray(left[key], dtype=np.float64)
            b = np.asarray(right[key], dtype=np.float64)
            if a.shape != b.shape:
                return float("inf")
            maximum = max(maximum, float(np.max(np.abs(a - b), initial=0.0)))
    return maximum


def deterministic_gate(states: list[dict[str, Any]], actor: torch.nn.Module) -> dict[str, Any]:
    if len(states) < DETERMINISM_STATES:
        raise RuntimeError("SNAPSHOT_RESTORE_FAIL: fewer than 10 fresh states")
    rows = []
    maximum_obs_error = 0.0
    maximum_reward_error = 0.0
    for index, state in enumerate(states[:DETERMINISM_STATES]):
        first = canonical_path_from_state(state["bundle"], actor, state["scene"], state["focal"], max_steps=1 + CONTINUATION_STEPS)
        second = canonical_path_from_state(state["bundle"], actor, state["scene"], state["focal"], max_steps=1 + CONTINUATION_STEPS)
        if len(first["trace"]) != len(second["trace"]) or len(first["trace"]) < 1 + CONTINUATION_STEPS:
            raise RuntimeError(f"SNAPSHOT_RESTORE_FAIL: state {index} terminated before 5 continuation steps")
        obs_error = 0.0
        reward_error = 0.0
        event_match = True
        phase_match = True
        for left, right in zip(first["trace"], second["trace"]):
            obs_error = max(obs_error, compare_observations(left["observations"], right["observations"]))
            reward_error = max(reward_error, float(np.max(np.abs(left["reward"] - right["reward"]), initial=0.0)))
            event_match = event_match and left["event_signature"] == right["event_signature"]
            phase_match = phase_match and left["phase"] == right["phase"]
            if not np.array_equal(left["done"], right["done"]):
                event_match = False
            if not np.array_equal(left["truncated"], right["truncated"]):
                event_match = False
        maximum_obs_error = max(maximum_obs_error, obs_error)
        maximum_reward_error = max(maximum_reward_error, reward_error)
        if obs_error > 1e-7 or reward_error > 1e-9 or not event_match or not phase_match:
            raise RuntimeError(f"SNAPSHOT_RESTORE_FAIL: state {index} mismatch")
        rows.append({
            "state_index": index,
            "scene": state["scene"],
            "seed": state["seed"],
            "episode_step": state["episode_step"],
            "focal_agent": state["focal"],
            "steps_compared": len(first["trace"]),
            "max_observation_abs_error": obs_error,
            "max_reward_abs_error": reward_error,
            "event_phase_done_truncation_match": True,
        })
    return {
        "status": "PASS",
        "states": rows,
        "states_checked": len(rows),
        "continuation_steps_after_first": CONTINUATION_STEPS,
        "max_observation_abs_error": maximum_obs_error,
        "max_reward_abs_error": maximum_reward_error,
        "max_error": max(maximum_obs_error, maximum_reward_error),
        "snapshot_contents": ["deepcopied env object", "observations", "env-owned RNG", "python RNG", "numpy RNG", "torch RNG"],
    }


def collect_anchor_candidates(actor: torch.nn.Module) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in ANCHOR_TARGETS}
    deterministic_states: list[dict[str, Any]] = []
    episode_count = {"mixed": 0, "pure_coverage": 0}
    start = time.monotonic()
    for scene in SCENES:
        max_episodes = MAX_MIXED_EPISODES if scene == "mixed" else MAX_COVERAGE_EPISODES
        scene_targets = ("pursuing", "pre_capture_cover", "early_recovery") if scene == "mixed" else ("recovery_pure",)
        for episode_index in range(max_episodes):
            if all(len(selected[name]) >= ANCHOR_TARGETS[name] for name in scene_targets):
                break
            seed = SEED_BASE + episode_index + (100000 if scene == "pure_coverage" else 0)
            env_scene = "coverage" if scene == "pure_coverage" else "mixed"
            env, observations = make_env(env_scene, seed)
            apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
            capture_step = None
            episode_count[scene] += 1
            for _ in range(env.episode_max_length):
                if not any(value is not None for value in observations):
                    break
                actions, policy_meta, _ = canonical_actions(actor, observations)
                active = policy_meta["active"].tolist()
                state = mission_snapshot(env, observations)
                transition_step = int(env.episode_step + 1)
                if deterministic_states.__len__() < DETERMINISM_STATES:
                    deterministic_states.append({
                        "bundle": save_env_state(env, observations, capture_step),
                        "scene": scene,
                        "seed": int(seed),
                        "episode_step": int(env.episode_step),
                        "focal": int(active[0]),
                        "capture_step": capture_step,
                    })
                for focal in active:
                    phase, semantic_class, age = current_phase_class(scene, observations, int(focal), capture_step, transition_step)
                    if semantic_class not in ANCHOR_TARGETS:
                        continue
                    if len(selected[semantic_class]) >= ANCHOR_TARGETS[semantic_class]:
                        continue
                    local_full, _ = pad_observations(observations)
                    neighbor_ids = np.full((MAX_AGENTS, MAX_NEIGHBORS), -1, dtype=np.int16)
                    neighbor_mask = np.zeros((MAX_AGENTS, MAX_NEIGHBORS), dtype=bool)
                    ids = sorted(state["friends"].get(int(focal), set()))[:MAX_NEIGHBORS]
                    neighbor_ids[int(focal), :len(ids)] = ids
                    neighbor_mask[int(focal), :len(ids)] = True
                    central = build_central_global_obs(env, max_agents=MAX_AGENTS, max_evaders=8, max_obstacles=5, self_feature_dim=9)
                    joint_aw = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
                    joint_aw[:len(actions)] = ACTION_GRID[np.clip(actions, 0, 8)]
                    selected[semantic_class].append({
                        "bundle": save_env_state(env, observations, capture_step),
                        "scene": scene,
                        "seed": int(seed),
                        "episode_step": int(env.episode_step),
                        "transition_step": transition_step,
                        "focal": int(focal),
                        "semantic_class": semantic_class,
                        "phase": phase,
                        "recovery_age": int(age),
                        "capture_step": capture_step,
                        "local_full": local_full,
                        "neighbor_ids": neighbor_ids,
                        "neighbor_mask": neighbor_mask,
                        "central": central,
                        "joint_action_indices": actions.copy(),
                        "joint_aw": joint_aw,
                        "original_bc_action": int(actions[int(focal)]),
                        "event_labels_at_anchor": {
                            "phase": phase,
                            "semantic_class": semantic_class,
                            "recovery_age": int(age),
                            "ring_participant_count": int(max((len(value["participants"]) for value in state["targets"].values()), default=0)),
                        },
                    })
                step = step_once(env, observations, actions, apf_agents, scene, capture_step)
                observations = step["observations"]
                capture_step = step["capture_step"]
                if bool(step["done"].all()):
                    break
            atomic_json(OUT / "progress.json", {
                "status": "fresh_anchor_rollout",
                "scene": scene,
                "episodes_completed": episode_count,
                "anchor_counts": {name: len(selected[name]) for name in ANCHOR_TARGETS},
                "determinism_states": len(deterministic_states),
                "elapsed_seconds": time.monotonic() - start,
            })
        if not all(len(selected[name]) >= ANCHOR_TARGETS[name] for name in scene_targets):
            break
    missing = {name: ANCHOR_TARGETS[name] - len(selected[name]) for name in ANCHOR_TARGETS if len(selected[name]) < ANCHOR_TARGETS[name]}
    if missing:
        raise RuntimeError(f"ANCHOR_COMPOSITION_FAIL: {missing}")
    anchors = []
    for name in ANCHOR_TARGETS:
        anchors.extend(selected[name][:ANCHOR_TARGETS[name]])
    for anchor_id, anchor in enumerate(anchors):
        anchor["anchor_id"] = int(anchor_id)
    return anchors, deterministic_states, {
        "episodes": episode_count,
        "anchor_composition": {name: int(len(selected[name][:ANCHOR_TARGETS[name]])) for name in ANCHOR_TARGETS},
        "elapsed_seconds": time.monotonic() - start,
    }


def select_critic_checkpoint(kind: str) -> dict[str, Any]:
    if kind in {"LQ", "NQ"}:
        summary = json.loads(AC3A_SUMMARY.read_text())
        condition = summary["conditions"][f"{kind}_natural"]
    else:
        summary = json.loads(AC3B_SUMMARY.read_text())
        condition = summary["new_conditions"][f"{kind}_natural"]
    run = min(condition["runs"], key=lambda value: float(value["best_validation"]["validation_rmse"]))
    path = ROOT / str(run["checkpoint"])
    if not path.exists() or sha256_file(path) != run["checkpoint_sha256"]:
        raise RuntimeError(f"CRITIC_CHECKPOINT_IDENTITY_FAIL_{kind}: {path}")
    return {
        "kind": kind,
        "seed": int(run["seed"]),
        "path": str(path),
        "sha256": sha256_file(path),
        "validation_rmse": float(run["best_validation"]["validation_rmse"]),
        "validation_update": int(run["best_validation"]["update"]),
    }


def load_critics(device: torch.device) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    config = make_config()
    models = {}
    provenance = {}
    for kind in ("LQ", "NQ", "CQ"):
        selected = select_critic_checkpoint(kind)
        payload = torch.load(selected["path"], map_location=device, weights_only=False)
        model = make_model(kind, config, device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        selected["parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
        model.requires_grad_(False)
        selected["state_sha256_before"] = tensor_state_sha256(model.state_dict())
        models[kind] = model
        provenance[kind] = selected
    return models, provenance


def critic_batch(anchor: Mapping[str, Any], action_indices: np.ndarray, device: torch.device) -> dict[str, Any]:
    count = len(action_indices)
    focal = int(anchor["focal"])
    local_full = anchor["local_full"]
    neighbor_ids = np.asarray(anchor["neighbor_ids"])[focal]
    neighbor_mask = np.asarray(anchor["neighbor_mask"])[focal]
    safe_ids = np.maximum(neighbor_ids, 0)
    obs = {key: np.repeat(np.asarray(value)[focal][None], count, axis=0) for key, value in local_full.items()}
    neighbor_obs = {
        key: np.repeat(np.asarray(value)[safe_ids][None], count, axis=0)
        for key, value in local_full.items()
    }
    neighbor_action = np.repeat(np.asarray(anchor["joint_aw"])[safe_ids][None], count, axis=0)
    neighbor_action[:, ~neighbor_mask] = 0.0
    joint = np.repeat(np.asarray(anchor["joint_aw"])[None], count, axis=0)
    candidate_aw = ACTION_GRID[np.asarray(action_indices, dtype=np.int64)]
    joint[:, focal] = candidate_aw
    central = {key: np.repeat(np.asarray(value)[None], count, axis=0) for key, value in anchor["central"].items()}
    return {
        "obs": {key: to_tensor(value, device) for key, value in obs.items()},
        "action": to_tensor(candidate_aw / ACTION_SCALE, device),
        "neighbor_obs": {key: to_tensor(value, device) for key, value in neighbor_obs.items()},
        "neighbor_action": to_tensor(neighbor_action / ACTION_SCALE, device),
        "neighbor_mask": to_tensor(np.repeat(neighbor_mask[None], count, axis=0), device),
        "central": {key: to_tensor(value, device) for key, value in central.items()},
        "joint": to_tensor(joint / ACTION_SCALE, device),
        "agent": torch.full((count,), focal, device=device, dtype=torch.long),
    }


def critic_predictions(models: Mapping[str, torch.nn.Module], anchor: Mapping[str, Any], mean: float, scale: float, device: torch.device) -> dict[str, np.ndarray]:
    actions = np.arange(9, dtype=np.int64)
    batch = critic_batch(anchor, actions, device)
    predictions = {}
    with torch.no_grad():
        for kind, model in models.items():
            value = forward(kind, model, batch).detach().cpu().numpy().astype(np.float64)
            predictions[kind] = value * scale + mean
    return predictions


def ordinal_ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(np.float64)


def ranking_metrics(prediction: np.ndarray, empirical: np.ndarray, bc_action: int) -> dict[str, Any]:
    predicted_order = np.argsort(-prediction, kind="mergesort")
    empirical_order = np.argsort(-empirical, kind="mergesort")
    top3_overlap = len(set(predicted_order[:3].tolist()).intersection(empirical_order[:3].tolist())) / 3.0
    pred_rank = ordinal_ranks(prediction)
    empirical_rank = ordinal_ranks(empirical)
    if np.std(pred_rank) == 0 or np.std(empirical_rank) == 0:
        spearman = None
    else:
        spearman = float(np.corrcoef(pred_rank, empirical_rank)[0, 1])
    empirical_best = int(empirical_order[0])
    critic_best = int(predicted_order[0])
    alternatives = np.asarray([index for index in range(9) if index != int(bc_action)], dtype=np.int64)
    empirical_advantage = empirical[alternatives] - empirical[int(bc_action)]
    predicted_advantage = prediction[alternatives] - prediction[int(bc_action)]
    sign_accuracy = float(np.mean(np.sign(empirical_advantage) == np.sign(predicted_advantage)))
    bc_position = int(np.flatnonzero(empirical_order == int(bc_action))[0]) + 1
    return {
        "top1_agreement": bool(critic_best == empirical_best),
        "top3_overlap": float(top3_overlap),
        "spearman": spearman,
        "regret": float(empirical[empirical_best] - empirical[critic_best]),
        "bc_action_empirical_rank": bc_position,
        "sign_accuracy_vs_bc": sign_accuracy,
        "empirical_best_action": empirical_best,
        "critic_best_action": critic_best,
        "predicted_order_desc": predicted_order.astype(int).tolist(),
        "empirical_order_desc": empirical_order.astype(int).tolist(),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("top1_agreement", "top3_overlap", "spearman", "regret", "bc_action_empirical_rank", "sign_accuracy_vs_bc")
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row[key] is not None]
        result[key] = {
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values)) if values else None,
            "n": len(values),
        }
    result["top1_count"] = int(sum(bool(row["top1_agreement"]) for row in rows))
    result["count"] = len(rows)
    return result


def anchor_manifest_row(anchor: Mapping[str, Any], branch_results: list[dict[str, Any]], event_labels: Mapping[str, Any]) -> dict[str, Any]:
    focal = int(anchor["focal"])
    return {
        "anchor_id": int(anchor["anchor_id"]),
        "scene": anchor["scene"],
        "seed": int(anchor["seed"]),
        "episode_step": int(anchor["episode_step"]),
        "transition_step": int(anchor["transition_step"]),
        "focal_agent_id": focal,
        "semantic_class": anchor["semantic_class"],
        "phase": anchor["phase"],
        "recovery_age": int(anchor["recovery_age"]),
        "capture_step": anchor["capture_step"],
        "event_labels": json_safe(event_labels),
        "original_bc_action": int(anchor["original_bc_action"]),
        "local_observation": json_safe({key: np.asarray(value)[focal] for key, value in anchor["local_full"].items()}),
        "neighbor_ids": np.asarray(anchor["neighbor_ids"])[focal].astype(int).tolist(),
        "neighbor_mask": np.asarray(anchor["neighbor_mask"])[focal].astype(bool).tolist(),
        "central_state_metadata": json_safe(anchor["central"]),
        "joint_bc_action_indices": np.asarray(anchor["joint_action_indices"]).astype(int).tolist(),
        "original_immediate_reward": float(branch_results[int(anchor["original_bc_action"])] ["rewards"][0]),
        "original_return": float(branch_results[int(anchor["original_bc_action"])] ["return"]),
        "snapshot_saved_to_disk": False,
    }


def run_branches(anchors: list[dict[str, Any]], actor: torch.nn.Module, models: Mapping[str, torch.nn.Module], mean: float, scale: float, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    branch_rows = []
    returns_matrix = np.zeros((len(anchors), 9), dtype=np.float32)
    prediction_matrix = {kind: np.zeros((len(anchors), 9), dtype=np.float32) for kind in models}
    anchor_rows = []
    for index, anchor in enumerate(anchors):
        branches = [canonical_path_from_state(anchor["bundle"], actor, anchor["scene"], int(anchor["focal"]), forced_action=action) for action in range(9)]
        empirical = np.asarray([branch["return"] for branch in branches], dtype=np.float64)
        predictions = critic_predictions(models, anchor, mean, scale, device)
        returns_matrix[index] = empirical.astype(np.float32)
        for kind in models:
            prediction_matrix[kind][index] = predictions[kind].astype(np.float32)
        original = branches[int(anchor["original_bc_action"])]
        event_labels = {
            "first_step_flags": json_safe(np.asarray(original["trace"][0]["event_signature"])),
            "branch_original": {
                "collision": original["collision"],
                "captured": original["captured"],
                "ce_success": original["ce_success"],
                "safe_complete": original["safe_complete"],
            },
        }
        anchor_rows.append(anchor_manifest_row(anchor, branches, event_labels))
        for action, branch in enumerate(branches):
            branch_rows.append({
                "anchor_id": int(anchor["anchor_id"]),
                "scene": anchor["scene"],
                "semantic_class": anchor["semantic_class"],
                "focal_agent_id": int(anchor["focal"]),
                "candidate_action": int(action),
                "bc_action": int(anchor["original_bc_action"]),
                "empirical_return": float(empirical[action]),
                "collision": bool(branch["collision"]),
                "captured": bool(branch["captured"]),
                "ce_success": bool(branch["ce_success"]),
                "safe_complete": bool(branch["safe_complete"]),
                "length": int(branch["length"]),
                "terminal_reason": branch["terminal_reason"],
                "critic_predictions": {kind: float(predictions[kind][action]) for kind in models},
            })
        for kind in models:
            metrics = ranking_metrics(predictions[kind], empirical, int(anchor["original_bc_action"]))
            branch_rows[-1 - 8][f"{kind}_ranking"] = metrics
        atomic_json(OUT / "progress.json", {"status": "counterfactual_branching", "anchors_completed": index + 1, "anchors_total": len(anchors)})
    ranking = {}
    for kind in models:
        rows = []
        for anchor_index, anchor in enumerate(anchors):
            rows.append(ranking_metrics(prediction_matrix[kind][anchor_index].astype(float), returns_matrix[anchor_index].astype(float), int(anchor["original_bc_action"])))
        ranking[kind] = {
            "overall": aggregate_metrics(rows),
            "per_anchor": rows,
            "by_phase": {
                phase: aggregate_metrics([row for row, anchor in zip(rows, anchors) if anchor["semantic_class"] == phase])
                for phase in ANCHOR_TARGETS
            },
        }
    empirical_bc = []
    catastrophic = []
    for index, anchor in enumerate(anchors):
        empirical = returns_matrix[index].astype(float)
        order = np.argsort(-empirical, kind="mergesort")
        bc = int(anchor["original_bc_action"])
        empirical_bc.append({
            "anchor_id": int(anchor["anchor_id"]),
            "empirical_rank": int(np.flatnonzero(order == bc)[0]) + 1,
            "top1": bool(order[0] == bc),
            "top3": bool(bc in order[:3]),
            "regret": float(empirical[order[0]] - empirical[bc]),
        })
        for action in range(9):
            row = branch_rows[index * 9 + action]
            if row["collision"] or not row["safe_complete"]:
                catastrophic.append({"anchor_id": int(anchor["anchor_id"]), "action": action, "collision": row["collision"], "safe_complete": row["safe_complete"], "captured": row["captured"], "ce_success": row["ce_success"], "length": row["length"]})
    teacher = {
        "empirical_rank1_fraction": float(np.mean([row["top1"] for row in empirical_bc])),
        "empirical_top3_fraction": float(np.mean([row["top3"] for row in empirical_bc])),
        "mean_regret": float(np.mean([row["regret"] for row in empirical_bc])),
        "per_anchor": empirical_bc,
    }
    return anchor_rows, returns_matrix, {"ranking": ranking, "teacher_bc": teacher, "catastrophic_branches": catastrophic, "branch_rows": branch_rows, "prediction_matrix": prediction_matrix}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    before_disk = shutil.disk_usage(ROOT)
    if sha256_file(ACTOR_LOADED) != ACTOR_SHA256:
        raise RuntimeError("BC_CHECKPOINT_IDENTITY_FAIL")
    actor, actor_payload = load_actor(ACTOR_LOADED, "cpu")
    actor.eval()
    actor.requires_grad_(False)
    actor_hash_before = tensor_state_sha256(actor.state_dict())
    canonical_grid = np.asarray(actor.action_grid.cpu(), dtype=np.float32)
    if not np.array_equal(canonical_grid, ACTION_GRID):
        raise RuntimeError("ACTION_MAPPING_FAIL")
    actor_action_mapping = {"status": "MATCH", "indices": list(range(9)), "grid": canonical_grid.tolist(), "ordering": "[(a,w) for a in (-0.4,0,0.4) for w in (-pi/6,0,pi/6)]"}

    anchors, deterministic_states, collection = collect_anchor_candidates(actor)
    gate = deterministic_gate(deterministic_states, actor)
    device = torch.device(args.device)
    models, critic_provenance = load_critics(device)
    ac3a = json.loads(AC3A_SUMMARY.read_text())
    ac3b = json.loads(AC3B_SUMMARY.read_text())
    mean = float(ac3b["contract"]["normalization"]["train_mean"])
    scale = float(ac3b["contract"]["normalization"]["train_std"])
    anchor_rows, returns_matrix, branch_report = run_branches(anchors, actor, models, mean, scale, device)
    npz_path = output / "counterfactual_returns.npz"
    arrays = {
        "anchor_ids": np.asarray([row["anchor_id"] for row in anchor_rows], dtype=np.int64),
        "returns": returns_matrix,
        "action_grid": ACTION_GRID.astype(np.float32),
        "bc_actions": np.asarray([row["original_bc_action"] for row in anchor_rows], dtype=np.int64),
        "semantic_class": np.asarray([row["semantic_class"] for row in anchor_rows]),
    }
    for kind, values in branch_report["prediction_matrix"].items():
        arrays[f"{kind.lower()}_predictions"] = values
    np.savez_compressed(npz_path, **arrays)
    atomic_json(output / "anchor_manifest.json", {
        "schema": "ac4a-anchor-manifest-v1",
        "status": "complete",
        "snapshot_saved_to_disk": False,
        "anchors": anchor_rows,
    })
    actor_hash_after = tensor_state_sha256(actor.state_dict())
    critic_hash_after = {kind: tensor_state_sha256(model.state_dict()) for kind, model in models.items()}
    after_disk = shutil.disk_usage(ROOT)
    summary = {
        "schema": "ac4a-counterfactual-aw9-branching-pilot-v1",
        "status": "complete",
        "git_head_at_run": git_head(),
        "contract": {
            "bc_checkpoint_requested": str(ACTOR_REQUESTED),
            "bc_checkpoint_loaded": str(ACTOR_LOADED),
            "bc_checkpoint_sha256": ACTOR_SHA256,
            "environment_contract": "forward-final-aw9-4v1-swept-v1",
            "gamma": GAMMA,
            "return": "individual focal reward full continuation, no bootstrap/V/learned-Q tail",
            "continuation": "all agents frozen canonical BC argmax after the first forced focal action",
            "focal_action": "one unilateral AW9 action 0..8; teammate actions fixed at anchor BC argmax",
            "target_normalization": {"train_mean": mean, "train_std": scale, "source": str(AC3B_SUMMARY)},
        },
        "action_mapping": actor_action_mapping,
        "snapshot_restore": gate,
        "anchors": collection,
        "anchor_composition": {name: int(sum(row["semantic_class"] == name for row in anchor_rows)) for name in ANCHOR_TARGETS},
        "branches": {"anchors": len(anchor_rows), "actions_per_anchor": 9, "total": len(anchor_rows) * 9},
        "critic_provenance": critic_provenance,
        "ranking": branch_report["ranking"],
        "teacher_bc": branch_report["teacher_bc"],
        "catastrophic_branches": {"count": len(branch_report["catastrophic_branches"]), "rows": branch_report["catastrophic_branches"]},
        "forbidden_executed": {
            "actor_updates": 0,
            "critic_optimizer_steps": 0,
            "critic_training": False,
            "bootstrap": False,
            "GAE": False,
            "PPO": False,
            "BC_training": False,
            "reward_change": False,
            "environment_contract_change": False,
        },
        "state_hashes": {
            "actor_before": actor_hash_before,
            "actor_after": actor_hash_after,
            "actor_bit_exact": actor_hash_before == actor_hash_after,
            "critics_after": critic_hash_after,
        },
        "artifacts": {
            "anchor_manifest": str(output / "anchor_manifest.json"),
            "counterfactual_returns": str(npz_path),
            "counterfactual_returns_sha256": sha256_file(npz_path),
        },
        "storage": {
            "disk_free_bytes_before": int(before_disk.free),
            "disk_free_bytes_after": int(after_disk.free),
            "output_bytes": int(sum(path.stat().st_size for path in output.rglob("*") if path.is_file())),
        },
        "references": {
            "ac3a_summary": str(AC3A_SUMMARY),
            "ac3a_summary_sha256": sha256_file(AC3A_SUMMARY),
            "ac3b_summary": str(AC3B_SUMMARY),
            "ac3b_summary_sha256": sha256_file(AC3B_SUMMARY),
            "formal_bank_sha256_reference": BANK_SHA256,
        },
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {"status": "complete", "anchors": len(anchor_rows), "branches": len(anchor_rows) * 9})
    for model in models.values():
        del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({
        "status": "complete",
        "snapshot_restore": gate["status"],
        "anchors": len(anchor_rows),
        "branches": len(anchor_rows) * 9,
        "catastrophic_branches": len(branch_report["catastrophic_branches"]),
        "output_bytes": summary["storage"]["output_bytes"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
