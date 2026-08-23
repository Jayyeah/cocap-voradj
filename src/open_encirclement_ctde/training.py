"""Training, evaluation, checkpoint, and resume orchestration for L0."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from .environment import CorrectedRoundupEnv, RoundupSpec
from .maddpg import MADDPG, MADDPGConfig
from .mappo import MAPPOConfig, MAPPOLearner
from .runtime import (
    append_jsonl,
    atomic_json_dump,
    atomic_torch_save,
    git_commit,
    restore_rng_state,
    rng_state,
    seed_everything,
)


CHECKPOINT_SCHEMA = "open-encirclement-full-v1"
MODEL_SCHEMA = "open-encirclement-model-v1"


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"algorithm", "seed", "run_name", "run_dir", "num_env_steps", "n_envs", "device"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config missing {sorted(missing)}")
    if config["algorithm"] not in {"mappo", "maddpg"}:
        raise ValueError(f"unsupported algorithm {config['algorithm']}")
    if config["n_envs"] <= 0 or config["num_env_steps"] <= 0:
        raise ValueError("n_envs and num_env_steps must be positive")
    config.setdefault("checkpoint_interval", 25_000)
    config.setdefault("rolling_full_interval", 5_000)
    config.setdefault("eval_interval", 25_000)
    config.setdefault("eval_episodes", 20)
    config.setdefault("final_stochastic_episodes", 20)
    config.setdefault("final_gif_count", 5)
    config.setdefault("progress_interval", 1_000)
    config.setdefault("contract", "corrected_roundup_v1")
    if config["contract"] != "corrected_roundup_v1":
        raise ValueError("L0 only accepts corrected_roundup_v1")
    return config


def _runtime_manifest(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "open-encirclement-run-manifest-v1",
        "algorithm": config["algorithm"],
        "contract": config["contract"],
        "seed": config["seed"],
        "command": sys.argv,
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": config["device"],
        "local_commit": git_commit(repo_root),
        "upstream_commits": {
            "light_mappo": "c503d89b6f28c9687ce9e45304fe66b57322ce1e",
            "MADDPG_Multi_UAV_Roundup": "15309de231f639c62d2049b0ad5b07b8975309c9",
            "KF_AA_MARL": "c8d68cab016ce9e6b18f8435b2faa0048d59da41",
        },
        "config": config,
    }


def _make_envs(config: dict[str, Any]) -> tuple[list[CorrectedRoundupEnv], np.ndarray]:
    environments = [
        CorrectedRoundupEnv(seed=int(config["seed"]) * 10_000 + index)
        for index in range(int(config["n_envs"]))
    ]
    observations = np.stack([env.reset()[0] for env in environments])
    return environments, observations


def _initial_episode_state(n_envs: int) -> dict[str, Any]:
    return {
        "returns": np.zeros((n_envs, 3), dtype=np.float64),
        "lengths": np.zeros(n_envs, dtype=np.int64),
        "containment_steps": np.zeros(n_envs, dtype=np.int64),
        "radius_steps": np.zeros(n_envs, dtype=np.int64),
        "collision_steps": np.zeros(n_envs, dtype=np.int64),
        "max_hold": np.zeros(n_envs, dtype=np.int64),
        "current_hold": np.zeros(n_envs, dtype=np.int64),
        "episode_counts": np.zeros(n_envs, dtype=np.int64),
    }


def _episode_state_to_checkpoint(state: dict[str, Any]) -> dict[str, Any]:
    return {key: np.asarray(value).copy() for key, value in state.items()}


def _record_transition(
    episode_state: dict[str, Any],
    env_index: int,
    rewards: np.ndarray,
    info: dict[str, Any],
    ended: bool,
    episodes_path: Path,
    algorithm: str,
    env_steps: int,
) -> None:
    episode_state["returns"][env_index] += rewards
    episode_state["lengths"][env_index] += 1
    episode_state["containment_steps"][env_index] += int(info["hull_contained"])
    episode_state["radius_steps"][env_index] += int(info["all_within_capture_radius"])
    episode_state["collision_steps"][env_index] += int(info["collision"])
    if info["hull_contained"] and info["all_within_capture_radius"]:
        episode_state["current_hold"][env_index] += 1
        episode_state["max_hold"][env_index] = max(
            episode_state["max_hold"][env_index], episode_state["current_hold"][env_index]
        )
    else:
        episode_state["current_hold"][env_index] = 0
    if not ended:
        return
    length = int(episode_state["lengths"][env_index])
    episode_state["episode_counts"][env_index] += 1
    append_jsonl(
        {
            "algorithm": algorithm,
            "env_steps": env_steps,
            "worker": env_index,
            "episode": int(episode_state["episode_counts"][env_index]),
            "capture": bool(info["success"]),
            "terminated": bool(info["terminated"]),
            "truncated": bool(info["truncated"]),
            "episode_length": length,
            "returns": episode_state["returns"][env_index].tolist(),
            "containment_fraction": float(episode_state["containment_steps"][env_index] / length),
            "all_radius_fraction": float(episode_state["radius_steps"][env_index] / length),
            "collision_fraction": float(episode_state["collision_steps"][env_index] / length),
            "max_hold": int(episode_state["max_hold"][env_index]),
            "final_distances": info["distances"],
            "largest_angular_gap": info["largest_angular_gap"],
        },
        episodes_path,
    )
    for key in ("returns", "lengths", "containment_steps", "radius_steps", "collision_steps", "max_hold", "current_hold"):
        episode_state[key][env_index] = 0


def _checkpoint_payload(
    *,
    algorithm: str,
    learner: Any,
    config: dict[str, Any],
    environments: list[CorrectedRoundupEnv],
    observations: np.ndarray,
    episode_state: dict[str, Any],
    env_steps: int,
    update_count: int,
    update_budget: float,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "algorithm": algorithm,
        "contains_replay": algorithm == "maddpg",
        "contains_optimizer": True,
        "contains_rng": True,
        "contains_runtime": True,
        "config": config,
        "local_commit": git_commit(repo_root),
        "env_steps": env_steps,
        "update_count": update_count,
        "update_budget": update_budget,
        "learner": learner.full_state_dict(),
        "environments": [env.state_dict() for env in environments],
        "observations": np.asarray(observations, dtype=np.float32),
        "episode_state": _episode_state_to_checkpoint(episode_state),
        "rng": rng_state(),
    }


def _save_full(payload: dict[str, Any], path: Path) -> None:
    atomic_torch_save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    required = {"schema", "learner", "environments", "observations", "episode_state", "rng", "env_steps"}
    if loaded.get("schema") != CHECKPOINT_SCHEMA or not required.issubset(loaded):
        raise RuntimeError(f"full checkpoint verification failed: {path}")
    if loaded["algorithm"] == "maddpg" and "replay" not in loaded["learner"]:
        raise RuntimeError("MADDPG full checkpoint is missing replay")


def _save_model(algorithm: str, learner: Any, config: dict[str, Any], env_steps: int, path: Path) -> None:
    atomic_torch_save(
        {
            "schema": MODEL_SCHEMA,
            "algorithm": algorithm,
            "contains_replay": False,
            "contains_optimizer": False,
            "config": config,
            "env_steps": env_steps,
            "learner": learner.model_state_dict(),
        },
        path,
    )


def _restore_full(
    path: Path,
    learner: Any,
    environments: list[CorrectedRoundupEnv],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], int, int, float]:
    payload = torch.load(path, map_location=learner.device, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not an open encirclement full checkpoint")
    if payload["config"] != config:
        raise ValueError("resolved config mismatch during full resume")
    if len(payload["environments"]) != len(environments):
        raise ValueError("environment count mismatch during resume")
    learner.load_state_dict(payload["learner"], full=True)
    for env, state in zip(environments, payload["environments"]):
        env.load_state_dict(state)
    restore_rng_state(payload["rng"])
    return (
        np.asarray(payload["observations"], dtype=np.float32),
        {key: np.asarray(value).copy() for key, value in payload["episode_state"].items()},
        int(payload["env_steps"]),
        int(payload["update_count"]),
        float(payload.get("update_budget", 0.0)),
    )


def evaluate_current(
    learner: Any,
    algorithm: str,
    *,
    seed_base: int,
    episodes: int,
    deterministic: bool,
    env_steps: int,
    gif_dir: Path | None = None,
    gif_count: int = 0,
) -> dict[str, Any]:
    saved_rng = rng_state()
    seed_everything(seed_base)
    rows = []
    try:
        for episode in range(episodes):
            env = CorrectedRoundupEnv(seed=seed_base + episode)
            obs, _ = env.reset()
            frames = []
            containment = 0
            radius = 0
            collision = 0
            hold = 0
            max_hold = 0
            last_info = env.last_info
            for step in range(env.spec.max_steps):
                batch_obs = obs[None, ...]
                if algorithm == "mappo":
                    action = learner.policy_actions(batch_obs, deterministic=deterministic)[0]
                else:
                    action = learner.actions(batch_obs, deterministic=deterministic, env_steps=env_steps)[0]
                obs, _, terminated, truncated, last_info = env.step(action)
                containment += int(last_info["hull_contained"])
                radius += int(last_info["all_within_capture_radius"])
                collision += int(last_info["collision"])
                if last_info["hull_contained"] and last_info["all_within_capture_radius"]:
                    hold += 1
                    max_hold = max(max_hold, hold)
                else:
                    hold = 0
                if gif_dir is not None and episode < gif_count:
                    frames.append(env.render())
                if terminated or truncated:
                    break
            length = step + 1
            if frames and gif_dir is not None:
                gif_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(gif_dir / f"episode_{episode:02d}.gif", frames, duration=0.08, loop=0)
            rows.append(
                {
                    "seed": seed_base + episode,
                    "capture": bool(last_info["success"]),
                    "episode_length": length,
                    "containment_fraction": containment / length,
                    "all_radius_fraction": radius / length,
                    "collision_fraction": collision / length,
                    "max_hold": max_hold,
                    "final_distances": last_info["distances"],
                    "largest_angular_gap": last_info["largest_angular_gap"],
                }
            )
    finally:
        restore_rng_state(saved_rng)
    return {
        "algorithm": algorithm,
        "contract": "corrected_roundup_v1",
        "env_steps": env_steps,
        "deterministic": deterministic,
        "episodes": episodes,
        "distinct_capture_episodes": int(sum(row["capture"] for row in rows)),
        "capture_rate": float(np.mean([row["capture"] for row in rows])),
        "mean_episode_length": float(np.mean([row["episode_length"] for row in rows])),
        "mean_containment_fraction": float(np.mean([row["containment_fraction"] for row in rows])),
        "mean_all_radius_fraction": float(np.mean([row["all_radius_fraction"] for row in rows])),
        "mean_collision_fraction": float(np.mean([row["collision_fraction"] for row in rows])),
        "max_hold": int(max(row["max_hold"] for row in rows)),
        "rollouts": rows,
    }


def _build_learner(config: dict[str, Any], device: torch.device, repo_root: Path):
    if config["algorithm"] == "mappo":
        algorithm_config = MAPPOConfig(**config["mappo"])
        if algorithm_config.n_rollout_threads != config["n_envs"]:
            raise ValueError("MAPPO n_rollout_threads must equal n_envs")
        return MAPPOLearner(algorithm_config, device, repo_root), algorithm_config
    algorithm_config = MADDPGConfig(**config["maddpg"])
    return MADDPG(algorithm_config, device, int(config["seed"])), algorithm_config


def train(config: dict[str, Any], repo_root: Path, *, resume: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    run_dir = Path(config["run_dir"])
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_config.json"
    manifest_path = run_dir / "manifest.json"
    episodes_path = run_dir / "episodes.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    state_path = run_dir / "state.json"
    if resume is None and (episodes_path.exists() or metrics_path.exists() or (checkpoints / "rolling-full.pt").exists()):
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")

    seed_everything(int(config["seed"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    learner, algorithm_config = _build_learner(config, device, repo_root)
    environments, observations = _make_envs(config)
    episode_state = _initial_episode_state(config["n_envs"])
    env_steps = 0
    update_count = 0
    update_budget = 0.0
    if resume is not None:
        observations, episode_state, env_steps, update_count, update_budget = _restore_full(
            Path(resume), learner, environments, config
        )
    atomic_json_dump(config, resolved_path)
    atomic_json_dump(_runtime_manifest(repo_root, config), manifest_path)
    if config["algorithm"] == "mappo":
        learner.initialize(observations)
        step_quantum = algorithm_config.rollout_length * config["n_envs"]
        if (config["num_env_steps"] - env_steps) % step_quantum != 0:
            raise ValueError(f"MAPPO remaining env steps must be divisible by rollout quantum {step_quantum}")

    start_time = time.monotonic()
    start_steps = env_steps
    next_milestone = ((env_steps // config["checkpoint_interval"]) + 1) * config["checkpoint_interval"]
    next_eval = ((env_steps // config["eval_interval"]) + 1) * config["eval_interval"]
    next_rolling = max(env_steps + 1, ((env_steps // config["rolling_full_interval"]) + 1) * config["rolling_full_interval"])
    last_metrics: dict[str, float] = {}
    first_full_verified = resume is not None

    def write_progress(status: str) -> None:
        elapsed = max(time.monotonic() - start_time, 1e-9)
        throughput = (env_steps - start_steps) / elapsed
        atomic_json_dump(
            {
                "status": status,
                "algorithm": config["algorithm"],
                "run_name": config["run_name"],
                "pid": os.getpid(),
                "device": config["device"],
                "env_steps": env_steps,
                "updates": update_count,
                "throughput_env_steps_per_second": throughput,
                "replay_size": len(learner.replay) if config["algorithm"] == "maddpg" else None,
                "value_normalizer": config["algorithm"] == "mappo" and learner.trainer.value_normalizer is not None,
                "last_metrics": last_metrics,
                "rolling_full": str(checkpoints / "rolling-full.pt"),
                "rolling_full_exists": (checkpoints / "rolling-full.pt").exists(),
                "rolling_full_verified": first_full_verified,
                "final_full_exists": (checkpoints / "final-full.pt").exists(),
            },
            state_path,
        )

    write_progress("RUNNING")
    while env_steps < config["num_env_steps"]:
        if config["algorithm"] == "mappo":
            for rollout_step in range(algorithm_config.rollout_length):
                collected = learner.collect(rollout_step)
                terminal_observations = []
                raw_rewards = []
                terminated_flags = []
                truncated_flags = []
                infos = []
                for env_index, env in enumerate(environments):
                    next_obs, rewards, terminated, truncated, info = env.step(collected["actions"][env_index])
                    if not np.array_equal(np.asarray(info["executed_actions"], dtype=np.float32), collected["actions"][env_index]):
                        raise RuntimeError("MAPPO proposed/executed action mismatch")
                    terminal_observations.append(next_obs)
                    raw_rewards.append(rewards)
                    terminated_flags.append(terminated)
                    truncated_flags.append(truncated)
                    infos.append(info)
                terminal_observations_array = np.stack(terminal_observations)
                reward_array = np.asarray(raw_rewards, dtype=np.float32)[..., None]
                terminated_array = np.asarray(terminated_flags, dtype=bool)
                truncated_array = np.asarray(truncated_flags, dtype=bool)
                if np.any(truncated_array):
                    terminal_values = learner.terminal_bootstrap(terminal_observations_array[truncated_array])
                    reward_array[truncated_array] += algorithm_config.gamma * terminal_values
                next_observations = terminal_observations_array.copy()
                ended_array = terminated_array | truncated_array
                for env_index, ended in enumerate(ended_array):
                    env_steps += 1
                    _record_transition(
                        episode_state,
                        env_index,
                        raw_rewards[env_index],
                        infos[env_index],
                        bool(ended),
                        episodes_path,
                        config["algorithm"],
                        env_steps,
                    )
                    if ended:
                        next_observations[env_index] = environments[env_index].reset()[0]
                ended_agents = np.repeat(ended_array[:, None], 3, axis=1)
                learner.insert(next_observations, reward_array, ended_agents, collected)
                observations = next_observations
            last_metrics = learner.finish_rollout_and_train()
            update_count = learner.update_count
            append_jsonl({"env_steps": env_steps, "updates": update_count, **last_metrics}, metrics_path)
        else:
            action_batch = learner.actions(observations, deterministic=False, env_steps=env_steps)
            next_observations = np.empty_like(observations)
            for env_index, env in enumerate(environments):
                next_obs, rewards, terminated, truncated, info = env.step(action_batch[env_index])
                executed = np.asarray(info["executed_actions"], dtype=np.float32)
                if not np.array_equal(executed, action_batch[env_index]):
                    raise RuntimeError("MADDPG proposed/executed/replay action mismatch")
                learner.replay.add(observations[env_index], executed, rewards, next_obs, terminated, truncated)
                env_steps += 1
                ended = terminated or truncated
                _record_transition(
                    episode_state,
                    env_index,
                    rewards,
                    info,
                    ended,
                    episodes_path,
                    config["algorithm"],
                    env_steps,
                )
                next_observations[env_index] = env.reset()[0] if ended else next_obs
            observations = next_observations
            if len(learner.replay) >= max(algorithm_config.warmup_steps, algorithm_config.batch_size):
                update_budget += config["n_envs"] * algorithm_config.updates_per_transition
                while update_budget >= 1.0:
                    last_metrics = learner.train_step()
                    update_budget -= 1.0
                    update_count = learner.update_count
                    append_jsonl({"env_steps": env_steps, "updates": update_count, **last_metrics}, metrics_path)

        if env_steps >= next_rolling or not first_full_verified:
            payload = _checkpoint_payload(
                algorithm=config["algorithm"],
                learner=learner,
                config=config,
                environments=environments,
                observations=observations,
                episode_state=episode_state,
                env_steps=env_steps,
                update_count=update_count,
                update_budget=update_budget,
                repo_root=repo_root,
            )
            _save_full(payload, checkpoints / "rolling-full.pt")
            first_full_verified = True
            next_rolling = ((env_steps // config["rolling_full_interval"]) + 1) * config["rolling_full_interval"]

        while env_steps >= next_milestone:
            model_path = checkpoints / f"step_{next_milestone:09d}-model.pt"
            _save_model(config["algorithm"], learner, config, next_milestone, model_path)
            next_milestone += config["checkpoint_interval"]

        while env_steps >= next_eval:
            report = evaluate_current(
                learner,
                config["algorithm"],
                seed_base=500_000 + int(config["seed"]) * 1_000,
                episodes=config["eval_episodes"],
                deterministic=True,
                env_steps=next_eval,
            )
            atomic_json_dump(report, run_dir / f"eval_step_{next_eval:09d}.json")
            next_eval += config["eval_interval"]

        write_progress("RUNNING")
        if env_steps == start_steps or env_steps % max(config["progress_interval"], 1) < config["n_envs"]:
            print(json.dumps({"run": config["run_name"], "env_steps": env_steps, "updates": update_count, **last_metrics}), flush=True)

    _save_model(config["algorithm"], learner, config, env_steps, checkpoints / "final-model.pt")
    final_payload = _checkpoint_payload(
        algorithm=config["algorithm"],
        learner=learner,
        config=config,
        environments=environments,
        observations=observations,
        episode_state=episode_state,
        env_steps=env_steps,
        update_count=update_count,
        update_budget=update_budget,
        repo_root=repo_root,
    )
    _save_full(final_payload, checkpoints / "final-full.pt")
    rolling_path = checkpoints / "rolling-full.pt"
    if rolling_path.exists():
        rolling_bytes = rolling_path.stat().st_size
        rolling_path.unlink()
        atomic_json_dump(
            {
                "removed": str(rolling_path),
                "bytes": rolling_bytes,
                "reason": "verified final-full.pt supersedes the unique rolling bundle",
                "recoverable": False,
            },
            checkpoints / "cleanup_manifest.json",
        )
    deterministic_report = evaluate_current(
        learner,
        config["algorithm"],
        seed_base=600_000 + int(config["seed"]) * 1_000,
        episodes=config["eval_episodes"],
        deterministic=True,
        env_steps=env_steps,
        gif_dir=run_dir / "gifs",
        gif_count=config["final_gif_count"],
    )
    stochastic_report = evaluate_current(
        learner,
        config["algorithm"],
        seed_base=700_000 + int(config["seed"]) * 1_000,
        episodes=config["final_stochastic_episodes"],
        deterministic=False,
        env_steps=env_steps,
    )
    atomic_json_dump(
        {"deterministic": deterministic_report, "stochastic_or_reduced_noise": stochastic_report},
        run_dir / "final_evaluation.json",
    )
    write_progress("COMPLETED")
    return json.loads(state_path.read_text(encoding="utf-8"))
