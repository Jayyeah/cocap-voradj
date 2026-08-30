#!/usr/bin/env python3
"""Unified Golden-contract runner for MAPPO-9, MAPPO-AW, IQN-VXY9 and TD3-AW."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.dynamics.continuous_action import vxy9_body_grid
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.continuous.box_actor import BoxActorConfig, SquashedGaussianAccelerationAngularVelocityActor
from cocap_voradj.models.continuous.central_attention_critic import CentralCriticConfig, CentralTwinCritics
from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
    LegacyVorAdjFeatureBackboneConfig,
    LocalEntityTokenEncoder,
    LocalEntityTokenEncoderConfig,
)
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.models.small_step_ac import CategoricalGridActor, CentralValueNetwork, DeterministicAWActor, IQNGridPolicy
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer, UniformJointReplaySampler
from cocap_voradj.training.replay import ReplayBuffer
from cocap_voradj.training.small_step_ac import IQNTrainer, MAPPOConfig, MAPPOTrainer, TD3Config, TD3Trainer, tensor_tree
from cocap_voradj.training.trainer import set_global_config
from tools.run_continuous_ctde_training import (
    _evader_actions_for_env, _pad_actions, _pad_local_obs_tree, _pad_vector,
    _screen, _split_termination_flags,
)


SCHEMA = "small-step-ac-full-v1"
ALGORITHMS = {"mappo9", "mappo9_v2", "mappo_aw", "iqn_vxy9", "td3_aw"}
MAPPO_ALGORITHMS = {"mappo9", "mappo9_v2", "mappo_aw"}
CATEGORICAL_MAPPO_ALGORITHMS = {"mappo9", "mappo9_v2"}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def ensure_finite(metrics: Mapping[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(float(value)):
            raise FloatingPointError(f"non-finite learning metric {key}={value}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def aw_grid() -> np.ndarray:
    return np.asarray([(a, w) for a in (-0.4, 0.0, 0.4) for w in (-np.pi / 6, 0.0, np.pi / 6)], dtype=np.float32)


def vxy_grid(v_max: float) -> np.ndarray:
    return vxy9_body_grid(v_max)


def configure_environment(root_config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    config = scene_config(root_config, "capture")
    config["training"]["scene_cycle"] = ["capture"]
    if algorithm == "iqn_vxy9":
        mode = "desired_velocity_2d_body"
        config["actor"]["max_pursuers"] = int(config["perception"]["max_pursuer_num"])
        config["action"]["mode"] = mode
        config["action"]["servo_acceleration_limit"] = float(config["action"].get("a_max", 0.4))
        config["action_mode"] = mode; config["env"]["action_mode"] = mode; config["pursuer"]["action_mode"] = mode
    else:
        mode = "acceleration_angular_velocity_body"
        config["action"]["mode"] = mode
        config["action_mode"] = mode; config["env"]["action_mode"] = mode; config["pursuer"]["action_mode"] = mode
    return config


def encoder_config(config: Mapping[str, Any]) -> LocalEntityTokenEncoderConfig:
    actor = config["actor"]
    return LocalEntityTokenEncoderConfig(
        hidden_dim=int(actor["hidden_dim"]), num_heads=int(actor["num_heads"]), num_layers=int(actor["num_layers"]),
        self_feature_dim=int(actor.get("self_feature_dim", 9)), max_pursuers=int(actor.get("max_pursuers", 12)),
        max_evaders=int(actor.get("max_evaders", 8)), max_obstacles=int(actor.get("max_obstacles", 5)), dropout=float(actor.get("dropout", 0.0)),
    )


def central_config(config: Mapping[str, Any]) -> CentralCriticConfig:
    critic = config["central_critic"]
    return CentralCriticConfig(
        hidden_dim=int(critic["hidden_dim"]), num_heads=int(critic["num_heads"]), num_layers=int(critic["num_layers"]),
        self_feature_dim=int(critic.get("self_feature_dim", 9)), max_agents=int(critic["max_agents"]),
        max_evaders=int(critic.get("max_evaders", 8)), max_obstacles=int(critic.get("max_obstacles", 5)), dropout=float(critic.get("dropout", 0.0)),
    )


def make_components(config: dict[str, Any], algorithm: str, device: str):
    spec = config["small_step_ac"]
    enc_cfg = encoder_config(config); critic_cfg = central_config(config)
    if algorithm in MAPPO_ALGORITHMS:
        if algorithm in CATEGORICAL_MAPPO_ALGORITHMS:
            if algorithm == "mappo9_v2":
                iqn = config["iqn"]
                actor_encoder = LegacyVorAdjFeatureBackbone(
                    LegacyVorAdjFeatureBackboneConfig(
                        hidden_dim=int(iqn["hidden_dim"]),
                        num_heads=int(iqn["num_heads"]),
                        num_layers=int(iqn["num_layers"]),
                        self_feature_dim=int(iqn["self_feature_dim"]),
                        max_pursuers=int(config["actor"]["max_pursuers"]),
                        max_evaders=int(config["actor"].get("max_evaders", 8)),
                        max_obstacles=int(config["actor"].get("max_obstacles", 5)),
                        pursuing_embed_dim=int(iqn.get("pursuing_embed_dim", 8)),
                        dropout=float(iqn.get("dropout", 0.1)),
                    )
                )
            else:
                actor_encoder = LocalEntityTokenEncoder(enc_cfg)
            actor = CategoricalGridActor(
                actor_encoder,
                aw_grid(),
                orthogonal_policy_head=algorithm == "mappo9_v2",
                output_gain=float(spec["mappo"].get("output_gain", 0.01)),
            )
        else:
            actor = SquashedGaussianAccelerationAngularVelocityActor(
                LocalEntityTokenEncoder(enc_cfg),
                BoxActorConfig(hidden_dim=enc_cfg.hidden_dim, a_max=float(config["action"]["a_max"]),
                               w_max=float(config["action"]["w_max"]), decision_dt=float(config["dynamics"]["decision_dt"]),
                               dropout=enc_cfg.dropout, context_pooling="mean"),
            )
        value = CentralValueNetwork(hidden_dim=critic_cfg.hidden_dim, num_heads=critic_cfg.num_heads,
                                    num_layers=critic_cfg.num_layers, self_feature_dim=critic_cfg.self_feature_dim,
                                    max_agents=critic_cfg.max_agents, max_evaders=critic_cfg.max_evaders,
                                    max_obstacles=critic_cfg.max_obstacles)
        ppo = spec["mappo"]
        trainer = MAPPOTrainer(actor, value, MAPPOConfig(
            gamma=float(ppo["gamma"]), gae_lambda=float(ppo["gae_lambda"]), clip_param=float(ppo["clip_param"]),
            ppo_epochs=int(ppo["ppo_epochs"]), minibatches=int(ppo["minibatches"]), actor_lr=float(ppo["actor_lr"]),
            critic_lr=float(ppo["critic_lr"]), entropy_coef=float(ppo["entropy_coef"]), value_coef=float(ppo["value_coef"]),
            max_grad_norm=float(ppo["max_grad_norm"]), target_kl=ppo.get("target_kl"),
            value_norm=bool(ppo.get("value_norm", False)),
            value_norm_beta=float(ppo.get("value_norm_beta", 0.99999)),
            value_norm_epsilon=float(ppo.get("value_norm_epsilon", 1e-5))), device)
        return trainer, None
    if algorithm == "td3_aw":
        actor = DeterministicAWActor(LocalEntityTokenEncoder(enc_cfg), float(config["action"]["a_max"]), float(config["action"]["w_max"]))
        td3 = spec["td3"]
        trainer = TD3Trainer(actor, CentralTwinCritics(critic_cfg), TD3Config(
            gamma=float(td3["gamma"]), tau=float(td3["tau"]), actor_lr=float(td3["actor_lr"]), critic_lr=float(td3["critic_lr"]),
            policy_noise=float(td3["policy_noise"]), noise_clip=float(td3["noise_clip"]), policy_delay=int(td3["policy_delay"]),
            max_grad_norm=float(td3["max_grad_norm"])), device)
        replay = JointReplayBuffer(int(spec["replay_capacity"]), int(config["training"]["max_agents"]), int(config["seed"]))
        return trainer, replay
    iqn_cfg = config["iqn"]
    model_cfg = CoCapNetConfig(hidden_dim=int(iqn_cfg["hidden_dim"]), num_heads=int(iqn_cfg["num_heads"]), num_layers=int(iqn_cfg["num_layers"]),
                              action_size=9, self_feature_dim=int(iqn_cfg["self_feature_dim"]), max_pursuers=int(config["actor"]["max_pursuers"]),
                              max_evaders=int(config["actor"].get("max_evaders", 8)), max_obstacles=int(config["actor"].get("max_obstacles", 5)),
                              num_quantiles=int(iqn_cfg["num_quantiles"]), num_cosine_features=int(iqn_cfg["num_cosine_features"]),
                              architecture="voradj_single_head", pursuing_embed_dim=int(iqn_cfg.get("pursuing_embed_dim", 8)))
    model = CoCapIQN(model_cfg)
    trainer = IQNTrainer(model, float(iqn_cfg["learning_rate"]), float(iqn_cfg["gamma"]), float(iqn_cfg["huber_kappa"]),
                         int(iqn_cfg["training_quantile_samples"]),
                         int(iqn_cfg["target_update_freq"]), device)
    return trainer, ReplayBuffer(int(iqn_cfg["replay_capacity"]))


def policy_view(trainer: Any, algorithm: str, config: Mapping[str, Any]):
    if algorithm == "iqn_vxy9":
        actor = IQNGridPolicy(trainer.model, vxy_grid(float(config["dynamics"]["v_max"])),
                              int(config["iqn"]["action_quantile_samples"]), float(config["iqn"]["epsilon_final"])).to(trainer.device)
        return SimpleNamespace(actor=actor, device=trainer.device)
    return trainer


def padded_local(observations, config):
    return _pad_local_obs_tree(list(observations), int(config["training"]["max_agents"]), int(config["actor"]["max_pursuers"]), int(config["actor"]["self_feature_dim"]))


def batched_global(env, config):
    value = build_central_global_obs(env, max_agents=int(config["training"]["max_agents"]),
                                     max_evaders=int(config["central_critic"].get("max_evaders", 8)),
                                     max_obstacles=int(config["central_critic"].get("max_obstacles", 5)),
                                     self_feature_dim=int(config["central_critic"].get("self_feature_dim", 9)))
    return value, {key: item[None] for key, item in value.items()}


def choose_actions(trainer, algorithm, local, global_batch, config, rng, step, deterministic=False):
    active_count = int(config["env"]["num_pursuers"])
    if algorithm in MAPPO_ALGORITHMS:
        actions, logp, latent, values = trainer.act(local, global_batch, deterministic)
        return actions[:active_count], logp[:active_count], latent[:active_count], values[0]
    if algorithm == "td3_aw":
        with torch.no_grad():
            actions = trainer.actor(tensor_tree(local, trainer.device)).cpu().numpy()
        if not deterministic:
            std = float(config["small_step_ac"]["td3"]["exploration_noise"])
            scale = np.asarray([config["action"]["a_max"], config["action"]["w_max"]], dtype=np.float32)
            actions = np.clip(actions + rng.normal(0.0, std, actions.shape) * scale, -scale, scale)
        return actions[:active_count], np.zeros(active_count), actions[:active_count], np.zeros(int(config["training"]["max_agents"]))
    policy = IQNGridPolicy(trainer.model, vxy_grid(float(config["dynamics"]["v_max"])), int(config["iqn"]["action_quantile_samples"])).to(trainer.device)
    with torch.no_grad(): actions, _, indices = policy.sample(tensor_tree(local, trainer.device), deterministic=True)
    indices = indices.cpu().numpy(); actions = actions.cpu().numpy()
    if not deterministic:
        iqn = config["iqn"]; fraction = min(float(step) / max(float(iqn["epsilon_decay_steps"]), 1.0), 1.0)
        epsilon = float(iqn["epsilon_start"]) + fraction * (float(iqn["epsilon_final"]) - float(iqn["epsilon_start"]))
        for index in range(active_count):
            if rng.random() < epsilon:
                indices[index] = int(rng.integers(0, 9)); actions[index] = policy.action_grid[indices[index]].cpu().numpy()
    return actions[:active_count], np.zeros(active_count), indices[:active_count], np.zeros(int(config["training"]["max_agents"]))


def empty_rollout() -> dict[str, list[Any]]:
    return {key: [] for key in ("local_obs", "global_obs", "actions", "latent", "log_prob", "values", "next_values", "rewards", "active_mask", "terminated", "truncated", "episode_end")}


def stack_rollout(rollout):
    result = {}
    for key, values in rollout.items():
        if key in {"local_obs", "global_obs"}:
            result[key] = {name: np.stack([item[name] for item in values]) for name in values[0]}
        elif key == "truncated" and not values:
            # Backward-compatible with pre-v2 in-memory/test rollouts.  Old
            # bundles had episode_end but no separately persisted timeout bit.
            result[key] = np.zeros_like(np.stack(rollout["terminated"]), dtype=bool)
        else: result[key] = np.stack(values)
    return result


def checkpoint(path: Path, *, config, algorithm, step, trainer, replay, rollout, env, observations, rng, metrics):
    payload = {"schema": SCHEMA, "config_hash": stable_hash(config), "algorithm": algorithm, "step": int(step),
               "trainer": trainer.state_dict(), "replay": replay, "rollout": rollout, "env": env, "observations": observations,
               "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "runner": rng.bit_generator.state,
                       "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               "last_metrics": metrics}
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if loaded.get("schema") != SCHEMA or int(loaded.get("step", -1)) != int(step): raise RuntimeError("checkpoint verification failed")
    os.replace(temporary, path)


def milestone_checkpoint(path: Path, *, config, algorithm, step, trainer, metrics):
    payload = {"schema": "small-step-ac-milestone-v1", "config_hash": stable_hash(config),
               "algorithm": algorithm, "step": int(step), "contains_replay": False,
               "contains_optimizer": True, "trainer": trainer.state_dict(), "last_metrics": metrics}
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if loaded.get("schema") != "small-step-ac-milestone-v1" or int(loaded.get("step", -1)) != int(step):
        raise RuntimeError("milestone checkpoint verification failed")
    os.replace(temporary, path)


def finalize_resume_checkpoint(path: Path, retain: bool) -> None:
    if not retain:
        path.unlink()


def restore(path, trainer, config, algorithm):
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("algorithm") != algorithm or payload.get("config_hash") != stable_hash(config):
        raise ValueError("resume checkpoint contract mismatch")
    trainer.load_state_dict(payload["trainer"]); random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    if torch.cuda.is_available() and payload["rng"]["cuda"]: torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng"]["cuda"]])
    rng = np.random.default_rng(); rng.bit_generator.state = payload["rng"]["runner"]
    return int(payload["step"]), payload["replay"], payload["rollout"], payload["env"], payload["observations"], rng, payload.get("last_metrics", {})


def evaluate(trainer, algorithm, root_config, config, step, run_dir, episodes, max_steps):
    view = policy_view(trainer, algorithm, config)
    evaluation_root = copy.deepcopy(root_config)
    if algorithm == "iqn_vxy9":
        action_override = {
            "action_mode": "desired_velocity_2d_body",
            "env": {"action_mode": "desired_velocity_2d_body"},
            "pursuer": {"action_mode": "desired_velocity_2d_body"},
            "action": {"mode": "desired_velocity_2d_body", "servo_acceleration_limit": float(config["action"].get("a_max", 0.4))},
        }
        evaluation_root["action_mode"] = action_override["action_mode"]
        for key in ("env", "pursuer", "action"): evaluation_root.setdefault(key, {}).update(action_override[key])
        evaluation_root["tasks"]["capture"]["action_mode"] = action_override["action_mode"]
        for key in ("env", "pursuer", "action"):
            evaluation_root["tasks"]["capture"].setdefault(key, {}).update(action_override[key])
        evaluation_root["actor"]["max_pursuers"] = int(config["actor"]["max_pursuers"])
        evaluation_root["tasks"]["capture"].setdefault("actor", {})["max_pursuers"] = int(config["actor"]["max_pursuers"])
    rows = {}
    actor_training = bool(view.actor.training)
    saved_rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                 "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    try:
        view.actor.eval()
        for mode_index, (deterministic, label) in enumerate(((True, "deterministic"), (False, "stochastic"))):
            seed_all(int(config["small_step_ac"]["evaluation_seed"]) + mode_index * 1000000)
            rows[label] = _screen(view, evaluation_root, int(config["small_step_ac"]["evaluation_seed"]), episodes, str(trainer.device),
                                  scenes=("capture",), max_steps=max_steps, deterministic=deterministic)
    finally:
        view.actor.train(actor_training); random.setstate(saved_rng["python"]); np.random.set_state(saved_rng["numpy"])
        torch.set_rng_state(saved_rng["torch"].cpu())
        if torch.cuda.is_available() and saved_rng["cuda"]: torch.cuda.set_rng_state_all([state.cpu() for state in saved_rng["cuda"]])
    atomic_json(run_dir / "evaluations" / f"step_{step:09d}.json", {"step": step, **rows})
    return rows


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--resume")
    parser.add_argument("--total-steps", type=int); parser.add_argument("--eval-episodes", type=int); parser.add_argument("--eval-max-steps", type=int)
    parser.add_argument("--device"); parser.add_argument("--run-dir"); return parser.parse_args()


def main() -> int:
    args = parse_args(); config_path = Path(args.config).resolve(); root_config = resolve_ladder_config(config_path)
    algorithm = str(root_config["small_step_ac"]["algorithm"]); assert algorithm in ALGORITHMS
    config = configure_environment(root_config, algorithm); config["seed"] = int(root_config["seed"])
    device = args.device or str(root_config.get("device", "cuda:0")); total = int(args.total_steps or root_config["small_step_ac"]["total_env_steps"])
    interval = int(root_config["small_step_ac"]["checkpoint_interval"]); eval_episodes = int(args.eval_episodes or root_config["small_step_ac"]["eval_episodes"])
    eval_max_steps = args.eval_max_steps; run_dir = Path(args.run_dir).resolve() if args.run_dir else ROOT / str(root_config["small_step_ac"]["run_dir"])
    seed_all(int(config["seed"])); rng = np.random.default_rng(int(config["seed"])); trainer, replay = make_components(config, algorithm, device)
    rollout = empty_rollout(); set_global_config(config); env = VorAdjEnv(config, seed=int(config["seed"])); observations = env.reset(); step = 0; last_metrics = {}
    if args.resume: step, replay, rollout, env, observations, rng, last_metrics = restore(Path(args.resume), trainer, config, algorithm)
    run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_json(run_dir / "manifest.json", {"schema": SCHEMA, "algorithm": algorithm, "seed": config["seed"], "device": device,
                                            "config": str(config_path), "config_hash": stable_hash(config), "pid": os.getpid(), "started": time.time()})
    atomic_json(run_dir / "status.json", {"state": "running", "step": step, "total": total,
                                           "pid": os.getpid(), "algorithm": algorithm, "updated": time.time()})
    started = time.time(); sampler = UniformJointReplaySampler(); ppo_horizon = int(config["small_step_ac"]["mappo"]["rollout_length"])
    apf_agents = [ApfAgent(e.a, e.w) for e in env.evaders]
    batch_size = int(config["small_step_ac"]["batch_size"]); warmup = int(config["small_step_ac"]["warmup_steps"]); train_freq = int(config["small_step_ac"]["train_freq"])
    while step < total:
        local = padded_local(observations, config); global_state, global_batch = batched_global(env, config)
        active = np.asarray([not p.deactivated for p in env.pursuers], dtype=bool); active_pad = _pad_vector(active, int(config["training"]["max_agents"]), bool)
        actions, logp, latent, values = choose_actions(trainer, algorithm, local, global_batch, config, rng, step)
        outcome = env.step(actions.tolist(), _evader_actions_for_env(env, apf_agents))
        step += 1; terminated, truncated = _split_termination_flags(outcome.dones, outcome.infos)
        term_pad = _pad_vector(terminated, int(config["training"]["max_agents"]), bool); trunc_pad = _pad_vector(truncated, int(config["training"]["max_agents"]), bool)
        rewards_pad = _pad_vector(outcome.rewards, int(config["training"]["max_agents"]), np.float32)
        next_observations = list(outcome.observations); next_local = padded_local(next_observations, config); next_global, next_global_batch = batched_global(env, config)
        with torch.no_grad():
            if algorithm in MAPPO_ALGORITHMS: next_values = trainer.value(tensor_tree(next_global_batch, trainer.device))[0].cpu().numpy()
            else: next_values = np.zeros(int(config["training"]["max_agents"]), dtype=np.float32)
        if algorithm in MAPPO_ALGORITHMS:
            for key, value in (("local_obs", local), ("global_obs", global_state), ("actions", _pad_actions(actions, len(active_pad))),
                               ("latent", _pad_vector(latent, len(active_pad), np.int64) if algorithm in CATEGORICAL_MAPPO_ALGORITHMS else _pad_actions(latent, len(active_pad))),
                               ("log_prob", _pad_vector(logp, len(active_pad), np.float32)), ("values", values), ("next_values", next_values),
                               ("rewards", rewards_pad), ("active_mask", active_pad), ("terminated", term_pad),
                               ("truncated", trunc_pad), ("episode_end", term_pad | trunc_pad)):
                rollout[key].append(value)
            if len(rollout["rewards"]) >= ppo_horizon or step == total:
                last_metrics = trainer.update(
                    stack_rollout(rollout), categorical=algorithm in CATEGORICAL_MAPPO_ALGORITHMS
                ); rollout = empty_rollout()
        elif algorithm == "td3_aw":
            roles = np.where(active_pad, 1, 0).astype(np.uint8)
            replay.add(local_obs=local, next_local_obs=next_local, global_state=global_state, next_global_state=next_global,
                       actions=_pad_actions(actions, len(active_pad)), rewards=rewards_pad, active_mask=active_pad, terminated=term_pad, truncated=trunc_pad,
                       agent_role_id=roles, metadata={"regime": "active_target", "event_ids": [], "phase": "pre_capture", "scene": "capture", "origin": "online"})
            if len(replay) >= max(warmup, batch_size) and step % train_freq == 0: last_metrics = trainer.update(replay.sample(batch_size, device=device, sampler=sampler))
        else:
            for index in range(len(env.pursuers)):
                if active[index]: replay.add({key: value[index] for key, value in local.items()}, int(latent[index]), float(outcome.rewards[index]),
                                             {key: value[index] for key, value in next_local.items()}, bool(terminated[index]), {"scene": "capture", "phase": "pre_capture"})
            iqn_warmup = int(config["iqn"]["min_replay_size"])
            if len(replay) >= max(iqn_warmup, batch_size) and step % train_freq == 0: last_metrics = trainer.update(replay.sample(batch_size, device), global_step=step)
        ensure_finite(last_metrics)
        ended = bool(all(outcome.dones))
        if ended:
            record = env.episode_record(task="capture"); record.update({"env_step": step, "algorithm": algorithm, "seed": config["seed"]})
            append_jsonl(run_dir / "episodes.jsonl", record); observations = env.reset()
        else: observations = next_observations
        if step % int(config["small_step_ac"]["log_interval"]) == 0:
            throughput = step / max(time.time() - started, 1e-6)
            eta_seconds = max(total - step, 0) / max(throughput, 1e-8)
            append_jsonl(run_dir / "learning_metrics.jsonl", {"step": step, "throughput": throughput,
                                                               "replay_size": len(replay) if replay is not None else 0, **last_metrics})
            atomic_json(run_dir / "status.json", {"state": "running", "step": step, "total": total,
                                                   "pid": os.getpid(), "algorithm": algorithm,
                                                   "throughput": throughput, "eta_seconds": eta_seconds,
                                                   "last_metrics": last_metrics, "updated": time.time()})
        if step % interval == 0 or step == total:
            ckpt = run_dir / "checkpoints" / f"step_{step:09d}.pt"
            resume_ckpt = run_dir / "resume_latest.pt"
            checkpoint(resume_ckpt, config=config, algorithm=algorithm, step=step, trainer=trainer,
                       replay=replay, rollout=rollout, env=env, observations=observations, rng=rng, metrics=last_metrics)
            milestone_checkpoint(ckpt, config=config, algorithm=algorithm, step=step, trainer=trainer, metrics=last_metrics)
            eval_rows = evaluate(trainer, algorithm, root_config, config, step, run_dir, eval_episodes, eval_max_steps); set_global_config(config)
            throughput = step / max(time.time() - started, 1e-6); eta_seconds = max(total - step, 0) / max(throughput, 1e-8)
            retain_full_resume = bool(root_config["small_step_ac"].get("retain_final_full_resume", True))
            if step == total and not retain_full_resume:
                # Delete only after the terminal milestone and both formal evaluations are durable.
                # Interrupted or failed runs always retain resume_latest.pt for supervisor recovery.
                finalize_resume_checkpoint(resume_ckpt, retain=False)
            atomic_json(run_dir / "status.json", {"state": "running" if step < total else "complete", "step": step, "total": total,
                                                   "checkpoint": str(ckpt),
                                                   "resume_checkpoint": str(resume_ckpt) if resume_ckpt.is_file() else None,
                                                   "full_resume_retained": resume_ckpt.is_file(),
                                                   "throughput": throughput, "eta_seconds": eta_seconds,
                                                   "last_metrics": last_metrics, "last_evaluation": eval_rows, "updated": time.time()})
    return 0


if __name__ == "__main__": raise SystemExit(main())
