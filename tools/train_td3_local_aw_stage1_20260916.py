#!/usr/bin/env python3
"""Formal isolated local-TD3 Stage-1 trainer for pure Coverage/Capture."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
    LegacyVorAdjFeatureBackboneConfig,
)
from cocap_voradj.models.continuous.local_td3 import LocalTD3Actor, LocalTD3Critic
from cocap_voradj.training.continuous.joint_replay import (
    JointReplayBuffer,
    ROLE_COVERAGE,
    ROLE_INACTIVE,
    ROLE_PURSUING,
    ROLE_SUPPORT,
    UniformJointReplaySampler,
)
from cocap_voradj.training.continuous.local_td3 import LocalTD3Config, LocalTD3Trainer
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.td3_stage1_contract import (
    A_MAX,
    W_MAX,
    check_td3_env,
    make_td3_env,
)
from cocap_voradj.training.td3_warmstart import file_sha256, tensor_state_sha256
from cocap_voradj.training.trainer import load_config, set_global_config
from tools import preflight_forward_final_scratch_20260914 as preflight
from tools.forward_final_single_task_20260915 import Telemetry, summarize


SCHEMA = "td3-local-aw-stage1-run-v1"
MAX_AGENTS = 4
OBS_KEYS = ("self", "pursuers", "evaders", "obstacles", "masks", "types")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def implementation_manifest() -> dict[str, Any]:
    paths = [
        ROOT / "src/cocap_voradj/models/continuous/local_td3.py",
        ROOT / "src/cocap_voradj/models/continuous/local_entity_token_encoder.py",
        ROOT / "src/cocap_voradj/training/continuous/local_td3.py",
        ROOT / "src/cocap_voradj/training/continuous/joint_replay.py",
        ROOT / "src/cocap_voradj/training/td3_stage1_contract.py",
        ROOT / "src/cocap_voradj/training/td3_warmstart.py",
        Path(__file__).resolve(),
    ]
    files = {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    return {"git_head": commit, "files": files, "sha256": canonical_hash(files)}


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _zero_obs_shapes(network: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    tokens = 1 + int(network["max_pursuers"]) + int(network["max_evaders"]) + int(network["max_obstacles"])
    return {
        "self": (int(network["self_feature_dim"]),),
        "pursuers": (int(network["max_pursuers"]), 7),
        "evaders": (int(network["max_evaders"]), 7),
        "obstacles": (int(network["max_obstacles"]), 5),
        "masks": (tokens,),
        "types": (tokens,),
    }


def pad_local_observations(
    observations: Iterable[Mapping[str, Any] | None], network: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Pad inactive rows without fabricating any globally visible feature."""
    observations = list(observations)
    if len(observations) != MAX_AGENTS:
        raise ValueError(f"expected {MAX_AGENTS} pursuer observation slots")
    shapes = _zero_obs_shapes(network)
    result: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
    for observation in observations:
        for key in OBS_KEYS:
            dtype = bool if key == "masks" else (np.int64 if key == "types" else np.float32)
            value = (
                np.zeros(shapes[key], dtype=dtype)
                if observation is None
                else np.asarray(observation[key], dtype=dtype)
            )
            if value.shape != shapes[key]:
                raise ValueError(f"local observation {key} shape {value.shape} != {shapes[key]}")
            if not np.all(np.isfinite(value.astype(float, copy=False))):
                raise ValueError(f"local observation {key} contains NaN/Inf")
            result[key].append(value.copy())
    return {key: np.stack(values) for key, values in result.items()}


def split_termination_flags(
    dones: Iterable[bool], infos: Iterable[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    done = np.asarray(list(dones), dtype=bool)
    infos = list(infos)
    truncated_states = {"too long episode", "pre-capture timeout"}
    truncated = np.asarray(
        [
            bool(
                terminal
                and (
                    (info["truncated"] and not info["terminated"])
                    if "terminated" in info and "truncated" in info
                    else info.get("state") in truncated_states
                )
            )
            for terminal, info in zip(done, infos)
        ],
        dtype=bool,
    )
    return done & ~truncated, truncated


def derive_roles(
    task: str, infos: Iterable[Mapping[str, Any]], active: np.ndarray
) -> np.ndarray:
    roles = np.full(MAX_AGENTS, ROLE_INACTIVE, dtype=np.uint8)
    if task == "coverage":
        roles[active] = ROLE_COVERAGE
        return roles
    for index, info in enumerate(infos):
        if not active[index]:
            continue
        metadata = info.get("replay_metadata", {}) or {}
        reward_role = str(metadata.get("reward_role", "")).lower()
        if reward_role == "support" or bool(metadata.get("support_candidate", False)):
            roles[index] = ROLE_SUPPORT
        elif reward_role == "capture" or str(metadata.get("task_label", "")) == "capture":
            roles[index] = ROLE_PURSUING
        else:
            roles[index] = ROLE_COVERAGE
    return roles


def evader_actions(env, agents) -> list[int | None]:
    if not env.evaders:
        return []
    env.configure_evader_apf_agents(agents)
    observations = env.get_evader_observations_for_apf()
    return [
        None if observation is None else int(agents[index].act(observation))
        for index, observation in enumerate(observations)
    ]


def make_networks(config: Mapping[str, Any], device: str) -> LocalTD3Trainer:
    network = LegacyVorAdjFeatureBackboneConfig(**config["network"])
    actor = LocalTD3Actor(LegacyVorAdjFeatureBackbone(network), A_MAX, W_MAX)
    critic1 = LocalTD3Critic(LegacyVorAdjFeatureBackbone(network))
    critic2 = LocalTD3Critic(LegacyVorAdjFeatureBackbone(network))
    training = config["training"]
    td3_config = LocalTD3Config(
        gamma=float(training["gamma"]),
        tau=float(training["tau"]),
        actor_lr=float(training["actor_lr"]),
        critic_lr=float(training["critic_lr"]),
        policy_noise=float(config["action"]["normalized_target_policy_std"]),
        noise_clip=float(config["action"]["normalized_target_noise_clip"]),
        policy_delay=int(training["policy_delay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        saturation_threshold=float(training["saturation_threshold"]),
    )
    return LocalTD3Trainer(actor, critic1, critic2, td3_config, device)


def load_warm_actor(trainer: LocalTD3Trainer, path: Path, task: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "td3-local-aw-iqn-warmstart-v1" or payload.get("task") != task:
        raise ValueError("warm actor schema/task mismatch")
    q1_before = tensor_state_sha256(trainer.critic1.state_dict())
    q2_before = tensor_state_sha256(trainer.critic2.state_dict())
    trainer.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    trainer.target_actor.load_state_dict(trainer.actor.state_dict(), strict=True)
    if tensor_state_sha256(trainer.critic1.state_dict()) != q1_before:
        raise AssertionError("warm start changed Q1")
    if tensor_state_sha256(trainer.critic2.state_dict()) != q2_before:
        raise AssertionError("warm start changed Q2")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "actor_state_sha256": payload["actor_state_sha256"],
        "teacher_checkpoint_sha256": payload["teacher"]["checkpoint_sha256"],
        "critic_transfer": "none",
        "bc_final": payload["behavior_cloning"]["final"],
    }


def sample_policy_actions(
    trainer: LocalTD3Trainer,
    observations: list[Mapping[str, Any] | None],
    exploration_rng: np.random.Generator,
    exploration_std: float,
) -> tuple[np.ndarray, dict[str, float]]:
    active_ids = [index for index, observation in enumerate(observations) if observation is not None]
    if not active_ids:
        raise ValueError("cannot act with no active pursuers")
    local = stack_obs([observations[index] for index in active_ids], str(trainer.device))
    with torch.no_grad():
        base = trainer.actor(local).cpu().numpy()
    normalized_noise = exploration_rng.normal(0.0, float(exploration_std), size=base.shape)
    scale = np.asarray([A_MAX, W_MAX], dtype=np.float32)
    physical_noise = normalized_noise * scale
    executed = np.clip(base + physical_noise, -scale, scale).astype(np.float32)
    padded = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
    padded[active_ids] = executed
    normalized = executed / scale
    metrics = {
        "exploration_std_normalized": float(exploration_std),
        "exploration_noise_normalized_abs_mean": float(np.abs(normalized_noise).mean()),
        "exploration_noise_physical_abs_mean": float(np.abs(physical_noise).mean()),
        "action_a_mean": float(executed[:, 0].mean()),
        "action_a_std": float(executed[:, 0].std()),
        "action_w_mean": float(executed[:, 1].mean()),
        "action_w_std": float(executed[:, 1].std()),
        "action_saturation_ratio": float((np.abs(normalized) >= 0.99).mean()),
    }
    return padded, metrics


def _mean_metrics(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
        if all(np.isfinite(float(row[key])) for row in rows)
    }


def _capture_geometry(env) -> dict[str, Any]:
    distances = [
        float(np.hypot(pursuer.x - evader.x, pursuer.y - evader.y))
        for pursuer in env.pursuers
        if not pursuer.deactivated
        for evader in env.evaders
        if not evader.deactivated
    ]
    inside = int(sum(distance <= 8.0 for distance in distances))
    return {
        "min_target_distance": min(distances, default=None),
        "inside_capture_radius": inside,
        "distances": distances,
    }


@torch.no_grad()
def evaluate(
    trainer: LocalTD3Trainer,
    config: Mapping[str, Any],
    step: int,
    output: Path,
    *,
    episodes: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    task = str(config["task"])
    evaluation = config["evaluation"]
    episodes = int(evaluation["episodes"] if episodes is None else episodes)
    max_steps = int(evaluation["max_steps"] if max_steps is None else max_steps)
    seed_base = int(evaluation[f"{task}_seed_base"])
    path = output / "evaluation" / f"step_{step:06d}.json"
    actor_hash = tensor_state_sha256(trainer.actor.state_dict())
    if path.exists():
        result = json.loads(path.read_text())
        if result["actor_state_sha256"] != actor_hash:
            raise ValueError("immutable evaluation actor hash mismatch")
        return result
    saved_rng = rng_state()
    records: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, float]] = []
    try:
        trainer._deterministic_modes()
        for episode in range(episodes):
            seed = seed_base + episode
            seed_all(seed)
            env, observations = make_td3_env(task, seed)
            env.total_steps = 2_000_000 + int(step)
            telemetry = Telemetry(env, task)
            apf_agents = [preflight.ApfAgent(evader.a, evader.w) for evader in env.evaders]
            action_rows: list[np.ndarray] = []
            q1_rows: list[float] = []
            q2_rows: list[float] = []
            reward_rows: list[float] = []
            last_geometry: list[dict[str, Any]] = []
            minimum_distance = float("inf")
            ring_run = {2: 0, 3: 0}
            ring_max = {2: 0, 3: 0}
            geometric_step = None
            settled_step = None
            for time_index in range(max_steps):
                active_ids = [index for index, observation in enumerate(observations) if observation is not None]
                batch = stack_obs([observations[index] for index in active_ids], str(trainer.device))
                actions = trainer.actor(batch)
                q1 = trainer.critic1(batch, actions)
                q2 = trainer.critic2(batch, actions)
                executed = actions.cpu().numpy()
                commands: list[Any] = [None] * MAX_AGENTS
                for agent_id, action in zip(active_ids, executed):
                    commands[agent_id] = action
                outcome = env.step(commands, evader_actions(env, apf_agents))
                telemetry.observe(env, outcome, active_ids)
                action_rows.append(executed.copy())
                q1_rows.append(float(q1.mean()))
                q2_rows.append(float(q2.mean()))
                reward_rows.append(float(np.mean(np.asarray(outcome.rewards)[active_ids])))
                geometry = _capture_geometry(env)
                if geometry["min_target_distance"] is not None:
                    minimum_distance = min(minimum_distance, float(geometry["min_target_distance"]))
                for size in (2, 3):
                    ring_run[size] = ring_run[size] + 1 if geometry["inside_capture_radius"] >= size else 0
                    ring_max[size] = max(ring_max[size], ring_run[size])
                frame = {
                    "step": time_index + 1,
                    **geometry,
                    "actions": executed.tolist(),
                    "speeds": [float(robot.speed) for robot in env.pursuers],
                    "q1_mean": q1_rows[-1],
                    "q2_mean": q2_rows[-1],
                }
                last_geometry.append(frame)
                last_geometry = last_geometry[-40:]
                if geometric_step is None and bool(getattr(env, "coverage_geometric_success", False)):
                    geometric_step = time_index + 1
                if settled_step is None and bool(getattr(env, "post_capture_coverage_success", False)):
                    settled_step = time_index + 1
                observations = list(outcome.observations)
                if all(outcome.dones):
                    break
            row = dict(seed=seed, native_done=all(outcome.dones), **telemetry.finish(env))
            flat_actions = np.concatenate(action_rows, axis=0) if action_rows else np.zeros((0, 2))
            normalized = flat_actions / np.asarray([A_MAX, W_MAX]) if len(flat_actions) else flat_actions
            switches = []
            previous = None
            for actions in action_rows:
                signs = np.sign(actions)
                if previous is not None and previous.shape == signs.shape:
                    switches.append(float(np.any(signs != previous, axis=1).mean()))
                previous = signs
            empirical = np.zeros(len(reward_rows), dtype=np.float64)
            running = 0.0
            for index in range(len(reward_rows) - 1, -1, -1):
                running = reward_rows[index] + float(config["training"]["gamma"]) * running
                empirical[index] = running
            qmin = np.minimum(q1_rows, q2_rows)
            proxy = {
                "predicted_q1_mean": float(np.mean(q1_rows)),
                "predicted_q2_mean": float(np.mean(q2_rows)),
                "predicted_qmin_mean": float(np.mean(qmin)),
                "empirical_mc_mean": float(np.mean(empirical)),
                "qmin_minus_mc_mean": float(np.mean(np.asarray(qmin) - empirical)),
                "q_gap_abs_mean": float(np.mean(np.abs(np.asarray(q1_rows) - np.asarray(q2_rows)))),
            }
            row.update(
                min_target_distance=None if not np.isfinite(minimum_distance) else minimum_distance,
                repeated_ring2_max=ring_max[2],
                repeated_ring3_max=ring_max[3],
                geometric_success=geometric_step is not None,
                settled_success=settled_step is not None,
                geometric_success_step=geometric_step,
                settled_success_step=settled_step,
                final_speed_mean=float(np.mean([robot.speed for robot in env.pursuers])),
                final_speed_max=float(np.max([robot.speed for robot in env.pursuers])),
                action_a_mean=float(flat_actions[:, 0].mean()) if len(flat_actions) else 0.0,
                action_a_std=float(flat_actions[:, 0].std()) if len(flat_actions) else 0.0,
                action_w_mean=float(flat_actions[:, 1].mean()) if len(flat_actions) else 0.0,
                action_w_std=float(flat_actions[:, 1].std()) if len(flat_actions) else 0.0,
                action_saturation_ratio=float((np.abs(normalized) >= 0.99).mean()) if len(normalized) else 0.0,
                aw_sign_switch_rate=float(np.mean(switches)) if switches else 0.0,
                q_proxy=proxy,
                final_40_step_geometry=last_geometry,
            )
            records.append(row)
            proxy_rows.append(proxy)
        success = [
            bool(row["settled_success"] if task == "coverage" else row["normal_capture"])
            for row in records
        ]
        result = {
            "schema": SCHEMA,
            "task": task,
            "step": int(step),
            "deterministic": True,
            "episodes": episodes,
            "max_steps": max_steps,
            "seed_base": seed_base,
            "actor_state_sha256": actor_hash,
            "summary": summarize(records),
            "extended_summary": {
                "geometric_success_rate": float(np.mean([row["geometric_success"] for row in records])),
                "settled_success_rate": float(np.mean([row["settled_success"] for row in records])),
                "normal_capture_rate": float(np.mean([row["normal_capture"] for row in records])),
                "stationary_capture_rate": float(np.mean([row["stationary_capture"] for row in records])),
                "min_target_distance_mean": float(np.mean([row["min_target_distance"] for row in records if row["min_target_distance"] is not None])) if any(row["min_target_distance"] is not None for row in records) else None,
                "repeated_ring2_max_mean": float(np.mean([row["repeated_ring2_max"] for row in records])),
                "repeated_ring3_max_mean": float(np.mean([row["repeated_ring3_max"] for row in records])),
                "final_speed_mean": float(np.mean([row["final_speed_mean"] for row in records])),
                "final_speed_max_mean": float(np.mean([row["final_speed_max"] for row in records])),
                "action_saturation_ratio": float(np.mean([row["action_saturation_ratio"] for row in records])),
                "aw_sign_switch_rate": float(np.mean([row["aw_sign_switch_rate"] for row in records])),
                "q_proxy_all": _mean_metrics(proxy_rows),
                "q_proxy_success": _mean_metrics([proxy for proxy, flag in zip(proxy_rows, success) if flag]),
                "q_proxy_failure": _mean_metrics([proxy for proxy, flag in zip(proxy_rows, success) if not flag]),
            },
            "records": records,
        }
        atomic_json(path, result)
        return result
    finally:
        restore_rng_state(saved_rng)


def replay_manifest(config: Mapping[str, Any], config_hash: str, implementation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "task": config["task"],
        "initialization": config["initialization"],
        "action_mode": config["action"]["mode"],
        "critic": "local_Q_i(o_i,a_i)",
        "sampler": "uniform_joint_no_phase_or_event_weighting",
        "max_agents": MAX_AGENTS,
        "config_sha256": config_hash,
        "implementation_sha256": implementation["sha256"],
    }


def save_milestone(
    *,
    output: Path,
    step: int,
    trainer: LocalTD3Trainer,
    replay: JointReplayBuffer,
    replay_contract: Mapping[str, Any],
    run_state: Mapping[str, Any],
    config: Mapping[str, Any],
    config_hash: str,
    implementation: Mapping[str, Any],
    environment_state: Any,
    evaluation_report: Mapping[str, Any],
) -> Path:
    checkpoint = output / "checkpoints" / f"step_{step:06d}.pt"
    replay_path = output / "replay" / f"step_{step:06d}.pkl"
    manifest_path = output / "manifests" / f"step_{step:06d}.json"
    for path in (checkpoint, replay_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"immutable milestone already exists: {path}")
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{replay_path.name}.", suffix=".pkl", dir=replay_path.parent)
    os.close(fd)
    try:
        replay.save(temporary, replay_contract, runtime_state=run_state)
        os.replace(temporary, replay_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    payload = {
        "schema": SCHEMA,
        "step": int(step),
        "trainer": trainer.state_dict(),
        "run_state": dict(run_state),
        "environment_pickle": pickle.dumps(environment_state, protocol=pickle.HIGHEST_PROTOCOL),
        "rng": rng_state(),
        "config": copy.deepcopy(dict(config)),
        "config_sha256": config_hash,
        "implementation": dict(implementation),
        "replay_path": str(replay_path.resolve()),
        "replay_sha256": file_sha256(replay_path),
        "replay_manifest": dict(replay_contract),
        "evaluation_path": str((output / "evaluation" / f"step_{step:06d}.json").resolve()),
        "evaluation_actor_sha256": evaluation_report["actor_state_sha256"],
    }
    atomic_torch(checkpoint, payload)
    manifest = {
        "schema": SCHEMA,
        "step": int(step),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "replay": str(replay_path.resolve()),
        "replay_sha256": payload["replay_sha256"],
        "evaluation": payload["evaluation_path"],
        "config_sha256": config_hash,
        "implementation": implementation,
        "trainer_counts": {
            "critic_updates": trainer.update_count,
            "policy_updates": trainer.policy_update_count,
            "expected_policy_updates": trainer.update_count // int(trainer.config.policy_delay),
        },
        "local_critic": True,
        "global_state_consumed": False,
        "teammate_actions_consumed": False,
        "replay_transition_count": len(replay),
        "transition_attempt_count": run_state["transition_attempt_count"],
        "transition_add_count": run_state["transition_add_count"],
    }
    if manifest["trainer_counts"]["policy_updates"] != manifest["trainer_counts"]["expected_policy_updates"]:
        raise AssertionError("delayed actor-update counter mismatch")
    if run_state["transition_attempt_count"] != run_state["transition_add_count"]:
        raise AssertionError("replay silently dropped a transition")
    atomic_json(manifest_path, manifest)
    return checkpoint


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "td3-local-aw-stage1-v1":
        raise ValueError("TD3 config schema mismatch")
    if config.get("task") not in {"coverage", "capture"}:
        raise ValueError("TD3 task must be coverage or capture")
    if config.get("initialization") not in {"scratch", "iqn_warm"}:
        raise ValueError("TD3 initialization must be scratch or iqn_warm")
    if config["critic"] != {
        "mode": "local",
        "formula": "Q_i(o_i,a_i)",
        "teammate_actions": False,
        "global_state": False,
    }:
        raise ValueError("Stage-1 critic must remain fully local")
    forbidden = config.get("forbidden", {})
    if not all(bool(forbidden.get(key)) for key in ("phase_aware_replay", "fullmix", "central_critic", "teammate_actions_in_critic", "remove_is_pursuing")):
        raise ValueError("Stage-1 forbidden-feature locks are incomplete")
    if int(config["training"]["policy_delay"]) != 2:
        raise ValueError("first formal TD3 contract requires policy_delay=2")


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    validate_config(config)
    if args.device:
        config["device"] = args.device
    device = str(config.get("device", "cuda:1"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output or config["output"])
    if not output.is_absolute():
        output = ROOT / output
    if args.smoke_steps:
        config = copy.deepcopy(config)
        config["training"]["budget"] = int(args.smoke_steps)
        config["training"]["milestones"] = [int(args.smoke_steps)]
        config["training"]["warmup_joint_transitions"] = min(8, max(1, int(args.smoke_steps) // 4))
        config["training"]["batch_size"] = min(8, max(2, int(args.smoke_steps) // 4))
        config["evaluation"]["episodes"] = 2
        config["evaluation"]["max_steps"] = 80
    config_hash = canonical_hash(config)
    implementation = implementation_manifest()
    seed = int(config["seed"])
    seed_all(seed)
    trainer = make_networks(config, device)
    warm_manifest = None
    if config["initialization"] == "iqn_warm":
        warm_path = Path(config["warm_actor"])
        if not warm_path.is_absolute():
            warm_path = ROOT / warm_path
        warm_manifest = load_warm_actor(trainer, warm_path, str(config["task"]))
    replay_contract = replay_manifest(config, config_hash, implementation)
    sampler = UniformJointReplaySampler()
    exploration_rng = np.random.default_rng(seed + 17)
    budget = int(config["training"]["budget"])
    milestones = set(int(value) for value in config["training"]["milestones"])
    batch_size = int(config["training"]["batch_size"])
    warmup = int(config["training"]["warmup_joint_transitions"])
    exploration_std = float(config["action"]["normalized_exploration_std"])
    update_rows: list[dict[str, float]] = []
    action_rows: list[dict[str, float]] = []
    episode_rows: list[dict[str, Any]] = []
    transition_attempt_count = 0
    transition_add_count = 0
    step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != SCHEMA:
            raise ValueError("resume checkpoint schema mismatch")
        if checkpoint["config_sha256"] != config_hash:
            raise ValueError("resume config hash mismatch")
        if checkpoint["implementation"]["sha256"] != implementation["sha256"]:
            raise ValueError("resume implementation hash mismatch")
        trainer.load_state_dict(checkpoint["trainer"])
        replay = JointReplayBuffer.load(checkpoint["replay_path"], replay_contract)
        state = checkpoint["run_state"]
        step = int(checkpoint["step"])
        transition_attempt_count = int(state["transition_attempt_count"])
        transition_add_count = int(state["transition_add_count"])
        update_rows = list(state["update_rows"])
        action_rows = list(state["action_rows"])
        episode_rows = list(state["episode_rows"])
        exploration_rng.bit_generator.state = state["exploration_rng_state"]
        env, observations, telemetry, apf_agents = pickle.loads(checkpoint["environment_pickle"])
        restore_rng_state(checkpoint["rng"])
        check_td3_env(env, str(config["task"]))
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty run directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        replay = JointReplayBuffer(
            capacity=int(config["training"]["replay_capacity"]), max_agents=MAX_AGENTS, seed=seed + 23
        )
        env, observations = make_td3_env(str(config["task"]), seed)
        env.total_steps = 2_000_000
        telemetry = Telemetry(env, str(config["task"]))
        apf_agents = [preflight.ApfAgent(evader.a, evader.w) for evader in env.evaders]
        launch = {
            "schema": SCHEMA,
            "config": config,
            "config_sha256": config_hash,
            "implementation": implementation,
            "warm_start": warm_manifest,
            "runtime_contract": check_td3_env(env, str(config["task"])),
            "pid": os.getpid(),
            "started_at_unix": time.time(),
            "device": device,
            "existing_mappo_untouched": True,
        }
        atomic_json(output / "launch.json", launch)

    start = time.monotonic()
    while step < budget:
        before_observations = list(observations)
        active = np.asarray([observation is not None for observation in before_observations], dtype=bool)
        before_local = pad_local_observations(before_observations, config["network"])
        actions, action_metric = sample_policy_actions(
            trainer, before_observations, exploration_rng, exploration_std
        )
        commands = [actions[index] if active[index] else None for index in range(MAX_AGENTS)]
        outcome = env.step(commands, evader_actions(env, apf_agents))
        transition_attempt_count += 1
        telemetry.observe(env, outcome, np.flatnonzero(active).tolist())
        terminated, truncated = split_termination_flags(outcome.dones, outcome.infos)
        next_observations = list(outcome.observations)
        next_local = pad_local_observations(next_observations, config["network"])
        capture_events = list(getattr(env, "last_capture_events", []))
        event_ids = []
        if capture_events:
            event_ids.append("capture")
            event_ids.extend(
                "capture_stationary" if event.get("capture_type") == "stationary" else "capture_normal"
                for event in capture_events
            )
        if getattr(env, "post_capture_coverage_success", False):
            event_ids.append("ce_success")
        if getattr(env, "last_collision_events", []):
            event_ids.append("collision")
        phase = "pure_coverage" if config["task"] == "coverage" else "pre_capture"
        replay.add(
            local_obs=before_local,
            next_local_obs=next_local,
            # Kept only to reuse the verified persistent joint-transition ring;
            # LocalTD3Trainer never reads either dummy global field.
            global_state=np.zeros(1, dtype=np.float32),
            next_global_state=np.zeros(1, dtype=np.float32),
            actions=actions,
            rewards=np.asarray(outcome.rewards, dtype=np.float32),
            active_mask=active,
            terminated=terminated,
            truncated=truncated,
            metadata={
                "regime": "uniform",
                "event_ids": sorted(set(event_ids)),
                "phase": phase,
                "scene": f"pure_{config['task']}",
                "origin": "online",
                "task_label": str(config["task"]),
                "coverage_only": config["task"] == "coverage",
                "active_target": config["task"] == "capture",
            },
            agent_role_id=derive_roles(str(config["task"]), outcome.infos, active),
        )
        transition_add_count += 1
        step += 1
        env.total_steps = 2_000_000 + step
        action_rows.append(action_metric)
        action_rows = action_rows[-1000:]
        observations = next_observations
        if len(replay) >= batch_size and step >= warmup:
            for _ in range(int(config["training"]["updates_per_environment_step"])):
                batch = replay.sample(batch_size, device=device, sampler=sampler)
                metric = trainer.update(batch)
                metric["environment_step"] = float(step)
                update_rows.append(metric)
                update_rows = update_rows[-1000:]
        if all(outcome.dones):
            episode_rows.append(dict(episode=len(episode_rows), **telemetry.finish(env)))
            set_global_config(env.config)
            observations = list(env.reset())
            env.total_steps = 2_000_000 + step
            check_td3_env(env, str(config["task"]))
            telemetry = Telemetry(env, str(config["task"]))
            apf_agents = [preflight.ApfAgent(evader.a, evader.w) for evader in env.evaders]

        if step % 1000 == 0 or step in milestones:
            diagnostics = {
                "schema": SCHEMA,
                "step": step,
                "wall_seconds": time.monotonic() - start,
                "replay_size": len(replay),
                "transition_attempt_count": transition_attempt_count,
                "transition_add_count": transition_add_count,
                "critic_updates": trainer.update_count,
                "policy_updates": trainer.policy_update_count,
                "expected_policy_updates": trainer.update_count // int(config["training"]["policy_delay"]),
                "updates_tail_mean": _mean_metrics(update_rows),
                "actions_tail_mean": _mean_metrics(action_rows),
                "episodes": summarize(episode_rows),
                "sampler": dict(replay.last_sample_stats),
            }
            if diagnostics["policy_updates"] != diagnostics["expected_policy_updates"]:
                raise AssertionError("delayed update count drift")
            if transition_attempt_count != transition_add_count:
                raise AssertionError("replay transition drop detected")
            atomic_json(output / "diagnostics" / f"step_{step:06d}.json", diagnostics)
            atomic_json(output / "status.json", diagnostics)

        if step in milestones:
            report = evaluate(trainer, config, step, output)
            state = {
                "transition_attempt_count": transition_attempt_count,
                "transition_add_count": transition_add_count,
                "update_rows": update_rows,
                "action_rows": action_rows,
                "episode_rows": episode_rows,
                "exploration_rng_state": copy.deepcopy(exploration_rng.bit_generator.state),
            }
            save_milestone(
                output=output,
                step=step,
                trainer=trainer,
                replay=replay,
                replay_contract=replay_contract,
                run_state=state,
                config=config,
                config_hash=config_hash,
                implementation=implementation,
                environment_state=(env, observations, telemetry, apf_agents),
                evaluation_report=report,
            )
            print(json.dumps({"run_id": config["run_id"], "step": step, "evaluation": report["extended_summary"]}), flush=True)

    final = {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "task": config["task"],
        "initialization": config["initialization"],
        "step": step,
        "budget": budget,
        "completed": step == budget,
        "wall_seconds": time.monotonic() - start,
        "critic_updates": trainer.update_count,
        "policy_updates": trainer.policy_update_count,
        "stage2_started": False,
        "stage3_started": False,
    }
    atomic_json(output / "final.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    result = train(config, args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
