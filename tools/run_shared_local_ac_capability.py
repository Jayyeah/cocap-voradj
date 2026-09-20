#!/usr/bin/env python3
"""Run only the bounded AC capability smoke or a future formal AC line.

The formal commands are intentionally explicit and this module never changes a
global proxy or starts a run unless invoked by the caller.  Checkpoint policy:
historical model-only files are periodic; one atomic latest runtime file holds
optimizer/RNG and optionally replay when explicitly enabled.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.shared_local_ac import (
    SharedLocalACNetworkConfig,
    SharedLocalActor,
    SharedLocalQ,
    assert_runtime_aw9,
    parameter_count,
)
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.shared_local_ac import (
    REPLAY_CLASSES,
    RecoveryInitPool,
    ReplayWarmupError,
    SharedLocalACConfig,
    SharedLocalACTrainer,
    SharedLocalReplay,
    Transition,
    atomic_torch_save,
    clone_obs,
    state_hash,
)
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def env_config(config: Mapping[str, Any], scene: str) -> Dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result = deep_update(result, (config.get("tasks", {}) or {}).get(scene, {}) or {})
    result.setdefault("env", {})["num_evaders"] = 0 if scene == "voradj_coverage" else 1
    return result


def make_env(config: Mapping[str, Any], scene: str, seed: int) -> VorAdjEnv:
    cfg = env_config(config, scene)
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=int(seed))
    return env


def epsilon_value(config: Mapping[str, Any], step: int) -> float:
    policy = (config.get("ac", {}) or {}).get("behavior_policy", {}) or {}
    start = float(policy.get("epsilon_start", 0.6))
    final = float(policy.get("epsilon_final", 0.05))
    decay = max(int(policy.get("epsilon_decay_steps", 500000)), 1)
    return float(start + min(max(step, 0), decay) / decay * (final - start))


def replay_class(config: Mapping[str, Any], scene: str, metadata: Mapping[str, Any]) -> str:
    mode = str((config.get("ac", {}) or {}).get("mode", "mix"))
    if mode == "coverage":
        return "recovery_pure"
    if mode == "capture":
        return "pursuing"
    phase = str(metadata.get("phase", "pre_capture"))
    task_label = str(metadata.get("task_label", "coverage"))
    if phase == "post_capture":
        return "post_capture_real"
    if phase == "pure_coverage":
        return "recovery_pure"
    return "pursuing" if task_label == "capture" else "pre_capture_cover"


class CapabilityRunner:
    def __init__(self, config: Dict[str, Any], config_path: Path, *, smoke: bool, output_root: Optional[Path] = None, resume_path: Optional[Path] = None):
        self.config = copy.deepcopy(config)
        self.config_path = config_path.resolve()
        self.smoke = bool(smoke)
        self.mode = str((self.config.get("ac", {}) or {}).get("mode", "mix"))
        if self.mode not in {"coverage", "capture", "mix"}:
            raise ValueError("ac.mode must be coverage, capture, or mix")
        self.seed = int(self.config.get("seed", 20260920))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = str(self.config.get("device", "cuda:0"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device.startswith("cuda"):
            # The eval-mode TransformerEncoder fast path can emit sporadic NaNs
            # on this CUDA/PyTorch stack for masked local observations.  The
            # reference path has identical network semantics and remains finite.
            torch.backends.mha.set_fastpath_enabled(False)
        self.output_root = Path(output_root or self.config.get("output_root", "runs"))
        self.run_name = str(self.config.get("run_name", self.config_path.stem))
        self.run_dir = self.output_root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(self.seed + 901)
        self.replay_rng = np.random.RandomState(self.seed + 902)
        self.episode_id = 0
        self.global_step = 0
        self.episode_count = 0
        self.scene_index = 0
        self.scheduler_sequence: List[str] = []
        self.reset_sources: Counter[str] = Counter()
        self.action_counts: Counter[int] = Counter()
        self.raw_actor_action_counts: Counter[int] = Counter()
        self.epsilon_values: List[float] = []
        self.sampler_reports: List[Dict[str, Any]] = []
        self.telemetry: List[Dict[str, Any]] = []
        self.episode_reports: List[Dict[str, Any]] = []
        self.formal_curve: List[Dict[str, Any]] = []
        self._formal_telemetry_cursor = 0
        self.sampled_rows = 0
        self.sampled_behavior_probability_sum = 0.0
        self.sampled_behavior_probability_min = float("inf")
        self.sampled_behavior_probability_max = 0.0
        self.sampled_epsilon_sum = 0.0
        self.sampled_probability_count = 0
        self.recovery_pool = RecoveryInitPool(
            capacity=int(((self.config.get("voradj", {}) or {}).get("recovery", {}) or {}).get("capture_state_pool_capacity", 1000)),
            captured_ratio=float(((self.config.get("voradj", {}) or {}).get("recovery", {}) or {}).get("captured_state_ratio", 0.75)),
            map_random_ratio=float(((self.config.get("voradj", {}) or {}).get("recovery", {}) or {}).get("map_random_ratio_within_non_capture", 0.5)),
        )
        self.replay = SharedLocalReplay(int((self.config.get("ac", {}) or {}).get("replay_capacity", 1_000_000)))
        self.network_config, self.trainer_config = self._runtime_configs()
        self.formal_network_config = self.network_config
        self.formal_parameter_counts = {
            "actor": parameter_count(SharedLocalActor(self.formal_network_config)),
            "critic": parameter_count(SharedLocalQ(self.formal_network_config)),
        }
        self.trainer = SharedLocalACTrainer(self.network_config, self.trainer_config, self.device)
        self.initial_hashes = {"actor": state_hash(self.trainer.actor), "critic": state_hash(self.trainer.critic), "target_actor": state_hash(self.trainer.target_actor), "target_critic": state_hash(self.trainer.target_critic)}
        self.envs = {
            "voradj": make_env(self.config, "voradj", self.seed + 43),
            "voradj_coverage": make_env(self.config, "voradj_coverage", self.seed + 47),
        }
        self.observations: Dict[str, List[Optional[Dict[str, np.ndarray]]]] = {}
        self.apf_agents: Dict[str, List[ApfAgent]] = {}
        self._reset_current_scene()
        aw9_env = self.envs[self.scenes()[0]]
        assert_runtime_aw9(aw9_env)
        self.aw9_live = [(float(a), float(w)) for a, w in aw9_env.pursuers[0].action_list]
        self.formal_config = copy.deepcopy(self.config)
        self.resume_info: Dict[str, Any] = {"resumed": False, "replay_policy": "scratch"}
        if resume_path is not None:
            self._load_resume(Path(resume_path))
        if self.smoke:
            self._apply_smoke_overrides()

    def _load_resume(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema") != SharedLocalACTrainer.schema_version:
            raise ValueError(f"unsupported AC resume schema: {payload.get('schema')}")
        self.trainer.load_payload(payload, load_optimizer=True)
        self.global_step = int(payload.get("global_step", 0))
        self.episode_count = int(payload.get("episode", 0))
        if "python_rng" in payload:
            random.setstate(payload["python_rng"])
        if "numpy_rng" in payload:
            np.random.set_state(payload["numpy_rng"])
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"])
        if "cuda_rng" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        if "replay" in payload and payload.get("replay") is not None:
            self.replay = payload["replay"]
            self.recovery_pool = payload.get("recovery_pool") or self.recovery_pool
            self.resume_info = {"resumed": True, "replay_policy": "replay_and_recovery_restored", "replay_size": len(self.replay), "recovery_pool_size": len(self.recovery_pool)}
        else:
            self.resume_info = {"resumed": True, "replay_policy": "rewarm_without_replay", "replay_size": 0, "recovery_pool_size": len(self.recovery_pool)}

    def _runtime_configs(self) -> Tuple[SharedLocalACNetworkConfig, SharedLocalACConfig]:
        ac = self.config.get("ac", {}) or {}
        network = SharedLocalACNetworkConfig(**dict(ac.get("network", {}) or {}))
        trainer = SharedLocalACConfig(
            gamma=float(ac.get("gamma", 0.99)), tau=float(ac.get("tau", 0.005)),
            actor_lr=float(ac.get("actor_lr", 1e-4)), critic_lr=float(ac.get("critic_lr", 1e-4)),
            adam_eps=float(ac.get("adam_eps", 1e-8)), batch_size=int(ac.get("batch_size", 128)),
            min_replay_size=int(ac.get("min_replay_size", 3000)), train_freq=int(ac.get("train_freq", 4)),
            max_grad_norm=float(ac.get("max_grad_norm", 0.5)), replay_capacity=int(ac.get("replay_capacity", 1_000_000)),
            replay_fallback=str(ac.get("replay_fallback", "defer")),
        )
        return network, trainer

    def _apply_smoke_overrides(self) -> None:
        smoke = (self.formal_config.get("ac", {}) or {}).get("smoke", {}) or {}
        smoke_batch = int(smoke.get("batch_size", 32))
        if self.mode == "mix":
            smoke_batch = int(sum(int(value) for value in (self.formal_config.get("ac", {}) or {}).get("replay_batch_counts", {}).values()))
        self.trainer_config = replace(
            self.trainer_config,
            batch_size=smoke_batch,
            min_replay_size=int(smoke.get("min_replay_size", 64)),
        )
        self.network_config = SharedLocalACNetworkConfig(hidden_dim=64, num_heads=4, num_layers=1, action_size=9, self_feature_dim=9, max_pursuers=8, max_evaders=8, max_obstacles=5, pursuing_embed_dim=8)
        self.trainer = SharedLocalACTrainer(self.network_config, self.trainer_config, self.device)
        self.initial_hashes = {"actor": state_hash(self.trainer.actor), "critic": state_hash(self.trainer.critic), "target_actor": state_hash(self.trainer.target_actor), "target_critic": state_hash(self.trainer.target_critic)}
        self.smoke_steps = int(smoke.get("steps", 1200))
        for env in self.envs.values():
            env.episode_max_length = int(smoke.get("episode_max_length", 24))

    def scenes(self) -> List[str]:
        if self.mode == "coverage":
            return ["voradj_coverage"]
        if self.mode == "capture":
            return ["voradj"]
        return ["voradj", "voradj_coverage"]

    def _reset_current_scene(self) -> None:
        scene = self.scenes()[self.scene_index % len(self.scenes())]
        env = self.envs[scene]
        initial_positions = None
        initial_active = None
        source = "environment_default"
        if self.mode == "mix" and scene == "voradj_coverage":
            source, snapshot = self.recovery_pool.choose(self.rng)
            if snapshot is not None:
                initial_positions = snapshot["positions"]
                initial_active = snapshot["active_mask"]
            elif source == "ordinary_map_random":
                env.env_cfg["pursuer_spawn_mode"] = "map_random"
            else:
                env.env_cfg["pursuer_spawn_mode"] = "inner_random_cluster"
        self.observations[scene] = env.reset(initial_pursuer_positions=initial_positions, initial_pursuer_active=initial_active)
        self.apf_agents[scene] = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
        self.scheduler_sequence.append(scene)
        self.reset_sources[source] += 1
        if env.episode_step != 0 or any(p.deactivated for p in env.pursuers):
            raise AssertionError("cross-init reset did not clear episode state")

    def _evader_actions(self, scene: str) -> List[Optional[int]]:
        env = self.envs[scene]
        if not env.evaders:
            return []
        set_global_config(env.config)
        observations = env.get_evader_observations_for_apf()
        if hasattr(env, "configure_evader_apf_agents"):
            env.configure_evader_apf_agents(self.apf_agents[scene])
        actions: List[Optional[int]] = []
        for idx, obs in enumerate(observations):
            if obs is None:
                actions.append(None)
            else:
                actions.append(self.apf_agents[scene][idx].act(obs))
        return actions

    def _collect_step(self, scene: str) -> None:
        env = self.envs[scene]
        old_obs = self.observations[scene]
        active = [idx for idx, obs in enumerate(old_obs) if obs is not None]
        actions: List[Optional[int]] = [None] * len(old_obs)
        epsilon = epsilon_value(self.formal_config, self.global_step)
        self.epsilon_values.append(epsilon)
        if active:
            batch = stack_obs([old_obs[idx] for idx in active], self.device)
            with torch.no_grad():
                sampled, actor_prob, behavior_prob = self.trainer.actor.sample_behavior(batch, epsilon)
                raw_policy_actions = self.trainer.actor.deterministic_action(batch)
            for pos, idx in enumerate(active):
                action = int(sampled[pos].item())
                actions[idx] = action
                self.action_counts[action] += 1
                self.raw_actor_action_counts[int(raw_policy_actions[pos].item())] += 1
                self._pending_actions[idx] = (action, float(actor_prob[pos].item()), float(behavior_prob[pos].item()), epsilon)
        result = env.step(actions, self._evader_actions(scene))
        next_obs = result.observations
        for idx in active:
            obs = old_obs[idx]
            action, actor_prob, behavior_prob, used_epsilon = self._pending_actions[idx]
            info = result.infos[idx] if idx < len(result.infos) else {}
            metadata = dict(info.get("replay_metadata", {}) or {})
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            done = bool(result.dones[idx])
            nxt = next_obs[idx] if next_obs[idx] is not None else {key: np.zeros_like(value) for key, value in obs.items()}
            self.replay.add(Transition(
                obs=clone_obs(obs), action=action, reward=float(result.rewards[idx]), next_obs=clone_obs(nxt),
                done=done, terminated=terminated, truncated=truncated, phase=str(metadata.get("phase", "pre_capture")),
                replay_class=replay_class(self.formal_config, scene, metadata), behavior_probability=behavior_prob,
                epsilon=used_epsilon, actor_probability=actor_prob, episode_id=self.episode_id,
                agent_id=idx, metadata=metadata,
            ))
        self.observations[scene] = next_obs
        self.global_step += 1
        self._maybe_update()
        if any(result.dones):
            record = env.episode_record(task=scene)
            snapshot = env.capture_snapshot
            if self.mode == "mix" and scene == "voradj" and snapshot is not None:
                self.recovery_pool.add(snapshot)
            record["scene"] = scene
            record["recovery_pool_size"] = len(self.recovery_pool)
            self.episode_reports.append(record)
            self.episode_count += 1
            self.episode_id += 1
            self.scene_index = (self.scene_index + 1) % len(self.scenes())
            self._reset_current_scene()

    def _maybe_update(self) -> None:
        ac = self.formal_config.get("ac", {}) or {}
        if self.global_step % int(ac.get("train_freq", 4)) != 0:
            return
        batch_counts = dict(ac.get("replay_batch_counts", {}) or {})
        if self.mode == "mix":
            try:
                batch, report = self.replay.sample_stratified(batch_counts, self.replay_rng, fallback="uniform" if self.smoke else "defer")
            except ReplayWarmupError:
                return
        else:
            if len(self.replay) < self.trainer_config.min_replay_size:
                return
            batch = self.replay.sample_uniform(self.trainer_config.batch_size, self.replay_rng)
            report = {"sampler": "uniform_shared", "actual_counts": dict(Counter(item.replay_class for item in batch))}
        if len(batch) != self.trainer_config.batch_size:
            return
        stats = self.trainer.update(batch)
        stats.update({"global_step": self.global_step, "sampler": report.get("sampler", "unknown"), "sample_actual_counts": report.get("actual_counts", {})})
        self.sampler_reports.append(report)
        self.telemetry.append(stats)
        behavior_probabilities = np.asarray([item.behavior_probability for item in batch], dtype=float)
        epsilons = np.asarray([item.epsilon for item in batch], dtype=float)
        self.sampled_rows += len(batch)
        self.sampled_behavior_probability_sum += float(behavior_probabilities.sum())
        self.sampled_behavior_probability_min = min(self.sampled_behavior_probability_min, float(behavior_probabilities.min()))
        self.sampled_behavior_probability_max = max(self.sampled_behavior_probability_max, float(behavior_probabilities.max()))
        self.sampled_epsilon_sum += float(epsilons.sum())
        self.sampled_probability_count += len(batch)

    def _checkpoint_smoke_save_load(self) -> Dict[str, Any]:
        payload = self.trainer.checkpoint_payload(config=self.formal_config, global_step=self.global_step, episode=self.episode_count, include_runtime=True, include_replay=False, replay=self.replay, recovery_pool=self.recovery_pool)
        with tempfile.TemporaryDirectory(prefix="ac_capability_smoke_") as temp_dir:
            path = Path(temp_dir) / "resume_latest.pt"
            atomic_torch_save(payload, path)
            loaded = torch.load(path, map_location=self.device, weights_only=False)
            restored = SharedLocalACTrainer(self.network_config, self.trainer_config, self.device)
            restored.load_payload(loaded, load_optimizer=True)
            return {"save_load_pass": bool(state_hash(restored.actor) == state_hash(self.trainer.actor) and state_hash(restored.critic) == state_hash(self.trainer.critic)), "payload_has_replay": "replay" in loaded, "payload_has_optimizer": "actor_optimizer" in loaded, "atomic_latest_only": True}

    @torch.no_grad()
    def _deterministic_eval_episode(self, seed: int) -> Dict[str, Any]:
        scene = self.scenes()[0]
        env = make_env(self.formal_config, scene, int(seed))
        obs = env.reset()
        assert_runtime_aw9(env)
        total = 0.0
        for _ in range(min(24 if self.smoke else 3000, env.episode_max_length)):
            active = [idx for idx, item in enumerate(obs) if item is not None]
            actions: List[Optional[int]] = [None] * len(obs)
            if active:
                batch = stack_obs([obs[idx] for idx in active], self.device)
                values = self.trainer.actor.deterministic_action(batch).cpu().tolist()
                for idx, action in zip(active, values):
                    actions[idx] = int(action)
            result = env.step(actions, [None] * len(env.evaders))
            total += float(np.sum(result.rewards))
            obs = result.observations
            if any(result.dones):
                break
        record = env.episode_record(task=scene)
        events = list(getattr(env, "last_capture_events", []) or [])
        capture_types = [str(event.get("capture_type", "normal")) for event in events]
        distribution = record.get("distribution_metrics", {}) or {}
        positions = np.asarray(record.get("pursuer_positions", []), dtype=float)
        pairwise_distances: List[float] = []
        if positions.ndim == 2 and positions.shape[0] >= 2:
            for left in range(positions.shape[0]):
                for right in range(left + 1, positions.shape[0]):
                    pairwise_distances.append(float(np.linalg.norm(positions[left] - positions[right])))
        position_finite = bool(positions.size == 0 or np.isfinite(positions).all())
        return {
            "ran": True,
            "scene": scene,
            "steps": int(env.episode_step),
            "return_sum": total,
            "episode_return": total,
            "completion_time": int(env.episode_step),
            "collision": bool(record.get("collision_event", False)),
            "capture_rate": float(record.get("captured", False)),
            "normal_capture": float("normal" in capture_types),
            "stationary_capture": float("stationary" in capture_types),
            "ring_2_plus": float(any(int(event.get("participant_count", len(event.get("participants", [])))) >= 2 for event in events)),
            "ring_3_plus": float(any(int(event.get("participant_count", len(event.get("participants", [])))) >= 3 for event in events)),
            "coverage_strict_success": float(record.get("coverage_strict_success", False)),
            "coverage_ce_rms": float(record.get("coverage_ce_center_rms", distribution.get("ce_center_rms", 0.0))),
            "coverage_ce_max": float(record.get("coverage_ce_center_max", distribution.get("ce_center_max", 0.0))),
            "area_cv": float(record.get("coverage_strict_area_cv", distribution.get("area_cv", 0.0))),
            "active_pursuers": int(record.get("active_pursuers", len(positions))),
            "position_finite": position_finite,
            "position_spread_mean": float(np.mean(pairwise_distances)) if pairwise_distances else None,
            "position_spread_min": float(np.min(pairwise_distances)) if pairwise_distances else None,
            "position_spread_max": float(np.max(pairwise_distances)) if pairwise_distances else None,
            "post_capture_ce": float(record.get("post_capture_coverage", False)),
            "safe_full_completion": float(record.get("fully_capture", False) and record.get("collision_free", False)),
            "early_recovery_duration": int(record.get("post_capture_step", -1)),
            "mission_time": int(env.episode_step),
        }

    def _deterministic_eval(self) -> Dict[str, Any]:
        evaluation_config = (self.formal_config.get("ac", {}) or {}).get("evaluation", {}) or {}
        episodes = 1 if self.smoke else max(int(evaluation_config.get("episodes", 20)), 1)
        records = [self._deterministic_eval_episode(self.seed + 5000 + index) for index in range(episodes)]
        numeric_keys = (
            "steps", "return_sum", "episode_return", "completion_time", "collision", "capture_rate",
            "normal_capture", "stationary_capture", "ring_2_plus", "ring_3_plus", "coverage_strict_success",
            "coverage_ce_rms", "coverage_ce_max", "area_cv", "active_pursuers", "position_spread_mean",
            "position_spread_min", "position_spread_max", "post_capture_ce", "safe_full_completion",
            "early_recovery_duration", "mission_time",
        )
        result: Dict[str, Any] = {"ran": True, "episodes": int(episodes), "scene": self.scenes()[0]}
        for key in numeric_keys:
            values = [float(row[key]) for row in records if row.get(key) is not None and np.isfinite(float(row[key]))]
            result[key] = float(np.mean(values)) if values else None
        result["position_finite"] = bool(all(bool(row.get("position_finite", False)) for row in records))
        result["collision_rate"] = result["collision"]
        result["strict_ce_success_rate"] = result["coverage_strict_success"]
        result["episode_return_mean"] = result["episode_return"]
        result["completion_time_mean"] = result["completion_time"]
        result["episode_records"] = records
        return result

    @staticmethod
    def _telemetry_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        numeric_keys = (
            "actor_entropy", "actor_loss", "actor_grad_norm", "critic_loss",
            "critic_grad_norm", "bellman_residual_abs_mean", "q_mean", "q_std",
            "q_min", "q_max", "td_target_mean", "td_target_std",
        )
        summary: Dict[str, Any] = {"updates": int(len(rows)), "all_finite": True}
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))], dtype=float)
            if values.size:
                summary[f"{key}_mean"] = float(values.mean())
                summary[f"{key}_min"] = float(values.min())
                summary[f"{key}_max"] = float(values.max())
            else:
                summary[f"{key}_mean"] = None
                summary[f"{key}_min"] = None
                summary[f"{key}_max"] = None
        summary["all_finite"] = bool(all(bool(row.get("finite", 0.0)) for row in rows))
        return summary

    def _formal_steps(self) -> List[int]:
        ac = self.formal_config.get("ac", {}) or {}
        checkpoint = ac.get("checkpoint", {}) or {}
        steps = checkpoint.get("formal_steps")
        if steps is None:
            steps = (self.formal_config.get("training", {}) or {}).get("formal_steps")
        if steps is None:
            steps = [0, 25000, 50000, 75000, 100000]
        return sorted({int(step) for step in steps if int(step) >= 0})

    def _record_formal_point(self, step: int) -> None:
        if self.formal_curve and int(self.formal_curve[-1]["step"]) == int(step):
            return
        telemetry_rows = self.telemetry[self._formal_telemetry_cursor:]
        self._formal_telemetry_cursor = len(self.telemetry)
        behavior_histogram = {str(index): int(self.action_counts.get(index, 0)) for index in range(9)}
        raw_histogram = {str(index): int(self.raw_actor_action_counts.get(index, 0)) for index in range(9)}
        behavior_counts = np.asarray(list(behavior_histogram.values()), dtype=float)
        sampled_count = max(self.sampled_probability_count, 1)
        replay_point = {
            "total_size": int(len(self.replay)),
            "class_counts": self.replay.class_counts(),
            "sampled_rows_cumulative": int(self.sampled_rows),
            "sampled_behavior_probability_mean": float(self.sampled_behavior_probability_sum / sampled_count) if self.sampled_probability_count else None,
            "sampled_behavior_probability_min": None if self.sampled_probability_count == 0 else float(self.sampled_behavior_probability_min),
            "sampled_behavior_probability_max": None if self.sampled_probability_count == 0 else float(self.sampled_behavior_probability_max),
            "sampled_epsilon_mean": float(self.sampled_epsilon_sum / sampled_count) if self.sampled_probability_count else None,
            "last_sampler": self.sampler_reports[-1] if self.sampler_reports else None,
        }
        point = {
            "step": int(step),
            "evaluation": self._deterministic_eval(),
            "epsilon": float(epsilon_value(self.formal_config, step)),
            "behavior_action_histogram_cumulative": behavior_histogram,
            "raw_actor_policy_histogram_cumulative": raw_histogram,
            "minimum_behavior_action_count": int(behavior_counts.min()),
            "behavior_action_entropy_cumulative": float(
                -(behavior_counts[behavior_counts > 0] / max(float(behavior_counts.sum()), 1.0)
                  * np.log(behavior_counts[behavior_counts > 0] / max(float(behavior_counts.sum()), 1.0))).sum()
            ) if behavior_counts.sum() else 0.0,
            "telemetry": self._telemetry_summary(telemetry_rows),
            "replay": replay_point,
        }
        self.formal_curve.append(point)
        write_json(self.run_dir / "evaluations" / f"eval_step_{int(step):09d}.json", point)
        write_json(self.run_dir / "formal_learning_curve.json", {"points": self.formal_curve})

    def run(self) -> Dict[str, Any]:
        self._pending_actions: Dict[int, Tuple[int, float, float, float]] = {}
        total = int(self.smoke_steps if self.smoke else self.formal_config.get("total_timesteps", 100000))
        formal_steps = self._formal_steps()
        if not self.smoke and 0 in formal_steps:
            self._record_formal_point(0)
        for _ in range(total):
            scene = self.scenes()[self.scene_index % len(self.scenes())]
            self._collect_step(scene)
            if not self.smoke and self.global_step in formal_steps:
                if self.global_step > 0:
                    self._write_formal_checkpoint(self.global_step)
                self._record_formal_point(self.global_step)
                if self.global_step >= 25000 and epsilon_value(self.formal_config, self.global_step) > 0.0 and self.action_counts.get(4, 0) == 0:
                    raise RuntimeError("EXPLORATION_IMPLEMENTATION_FAIL: action 4 has no behavior samples")
        checkpoint_result = self._checkpoint_smoke_save_load() if self.smoke else {"formal_checkpoint_policy_ready": True}
        eval_result = self.formal_curve[-1]["evaluation"] if self.formal_curve else self._deterministic_eval()
        action_hist = {str(index): int(self.action_counts.get(index, 0)) for index in range(9)}
        counts = np.asarray(list(action_hist.values()), dtype=float)
        probs = counts / max(float(counts.sum()), 1.0)
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
        target_hashes = {"actor": state_hash(self.trainer.actor), "critic": state_hash(self.trainer.critic), "target_actor": state_hash(self.trainer.target_actor), "target_critic": state_hash(self.trainer.target_critic)}
        result = {
            "config_path": str(self.config_path), "run_name": self.run_name, "mode": self.mode, "smoke": self.smoke,
            "device": self.device, "steps": self.global_step, "episodes": self.episode_count,
            "formal_network_parameter_counts": self.formal_parameter_counts,
            "runtime_network_parameter_counts": {"actor": parameter_count(self.trainer.actor), "critic": parameter_count(self.trainer.critic)},
            "initial_hashes": self.initial_hashes, "final_hashes": target_hashes,
            "actor_parameters_changed": target_hashes["actor"] != self.initial_hashes["actor"], "critic_parameters_changed": target_hashes["critic"] != self.initial_hashes["critic"],
            "target_parameters_updated": target_hashes["target_actor"] != self.initial_hashes["target_actor"] or target_hashes["target_critic"] != self.initial_hashes["target_critic"],
            "aw9_live": self.aw9_live, "action_counts": action_hist, "minimum_action_count": int(counts.min()), "action_entropy": entropy,
            "raw_actor_policy_action_counts": {str(index): int(self.raw_actor_action_counts.get(index, 0)) for index in range(9)},
            "epsilon_start": float(self.epsilon_values[0]) if self.epsilon_values else None, "epsilon_end": float(self.epsilon_values[-1]) if self.epsilon_values else None,
            "replay_size": len(self.replay), "replay_class_counts": self.replay.class_counts(), "recovery_pool_size": len(self.recovery_pool),
            "scheduler_sequence": self.scheduler_sequence[: min(len(self.scheduler_sequence), 24)], "reset_source_counts": dict(self.reset_sources),
            "sampler_reports": self.sampler_reports[-20:], "telemetry_tail": self.telemetry[-20:], "episode_tail": self.episode_reports[-10:],
            "all_telemetry_finite": all(bool(item.get("finite", 0.0)) for item in self.telemetry), "checkpoint_save_load": checkpoint_result,
            "deterministic_eval": eval_result, "exploration_implementation_fail": bool(self.epsilon_values and self.epsilon_values[-1] > 0 and counts.min() == 0),
            "formal_steps": formal_steps,
            "formal_evaluations": self.formal_curve,
            "strict_ce_curve": [{"step": item["step"], "value": item["evaluation"]["coverage_strict_success"]} for item in self.formal_curve],
            "ce_rms_curve": [{"step": item["step"], "value": item["evaluation"]["coverage_ce_rms"]} for item in self.formal_curve],
            "area_cv_curve": [{"step": item["step"], "value": item["evaluation"]["area_cv"]} for item in self.formal_curve],
            "collision_curve": [{"step": item["step"], "value": item["evaluation"]["collision"]} for item in self.formal_curve],
            "actor_entropy_curve": [{"step": item["step"], "value": item["telemetry"]["actor_entropy_mean"]} for item in self.formal_curve],
            "critic_loss_curve": [{"step": item["step"], "value": item["telemetry"]["critic_loss_mean"]} for item in self.formal_curve],
            "q_scale_curve": [{"step": item["step"], "q_mean": item["telemetry"]["q_mean_mean"], "q_std": item["telemetry"]["q_std_mean"], "q_min": item["telemetry"]["q_min_min"], "q_max": item["telemetry"]["q_max_max"]} for item in self.formal_curve],
            "replay_growth": [{"step": item["step"], **item["replay"]} for item in self.formal_curve],
            "formal_launch_started": False,
            "resume_info": self.resume_info,
        }
        write_json(self.run_dir / ("smoke_report.json" if self.smoke else "formal_report.json"), result)
        return result

    def _write_formal_checkpoint(self, step: int) -> None:
        ckpt = self.run_dir / "checkpoints" / f"model_step_{int(step):09d}.pt"
        payload = self.trainer.checkpoint_payload(config=self.formal_config, global_step=step, episode=self.episode_count, include_runtime=False, include_replay=False)
        atomic_torch_save(payload, ckpt)
        resume = self.run_dir / "resume_latest" / "full_resume.pt"
        resume_payload = self.trainer.checkpoint_payload(config=self.formal_config, global_step=step, episode=self.episode_count, include_runtime=True, include_replay=bool(((self.formal_config.get("ac", {}) or {}).get("resume", {}) or {}).get("save_replay", False)), replay=self.replay, recovery_pool=self.recovery_pool)
        atomic_torch_save(resume_payload, resume)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument("--output-root")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.config).resolve()
    config = load_config(str(path))
    if args.device:
        config["device"] = args.device
    if args.run_name:
        config["run_name"] = args.run_name
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    result = CapabilityRunner(config, path, smoke=args.smoke, output_root=Path(args.output_root) if args.output_root else None, resume_path=Path(args.resume_path) if args.resume_path else None).run()
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
