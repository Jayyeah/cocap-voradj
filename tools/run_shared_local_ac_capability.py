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
    ACNumericFailure,
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
        self.actor_raw_action_counts: Counter[int] = Counter()
        self.behavior_action_counts: Counter[int] = Counter()
        self.sample_action_counts: Counter[int] = Counter()
        self.sample_class_counts: Counter[str] = Counter()
        self.replay_role_counts: Counter[str] = Counter()
        self.actor_raw_probability_mass = np.zeros(9, dtype=np.float64)
        self.behavior_probability_mass = np.zeros(9, dtype=np.float64)
        self.actor_raw_entropy_sum = 0.0
        self.behavior_entropy_sum = 0.0
        self.action_row_count = 0
        self.epsilon_values: List[float] = []
        self.sampler_reports: List[Dict[str, Any]] = []
        self.telemetry: List[Dict[str, Any]] = []
        self.telemetry_aggregate: Dict[str, Dict[str, float]] = {}
        self.episode_reports: List[Dict[str, Any]] = []
        self.evaluation_reports: List[Dict[str, Any]] = []
        self.first_nonfinite_artifact_written = False
        self.evaluation_bundles: List[Dict[str, Any]] = []
        self.transition_points: Dict[str, Optional[int]] = {
            "first_learner_update_step": None,
            "first_natural_capture_step": None,
            "first_post_capture_real_row_step": None,
            "first_batch_with_nonzero_post_samples_step": None,
            "first_post_buffer_ge_32_step": None,
            "first_strict_64_16_32_16_batch_step": None,
        }
        self.update_attempts = 0
        self.warmup_deferrals = 0
        self.substitution_batch_count = 0
        self.strict_batch_count = 0
        self.capture_snapshots_added = 0
        self.artifact_dir = self.run_dir / "telemetry"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
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
        self._append_jsonl("scheduler.jsonl", {
            "episode_index": int(self.episode_count),
            "global_step": int(self.global_step),
            "scene": scene,
            "expected_scene": self.scenes()[self.episode_count % len(self.scenes())],
            "reset_source": source,
            "pool_size": len(self.recovery_pool),
        })
        if env.episode_step != 0 or any(p.deactivated for p in env.pursuers):
            raise AssertionError("cross-init reset did not clear episode state")

    def _append_jsonl(self, name: str, payload: Mapping[str, Any]) -> None:
        with (self.artifact_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(dict(payload)), ensure_ascii=False, sort_keys=True) + "\n")

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
                actor_probs = self.trainer.actor.probabilities(batch)
                behavior_probs = (1.0 - float(epsilon)) * actor_probs + float(epsilon) / float(actor_probs.shape[-1])
                sampled = torch.multinomial(behavior_probs, num_samples=1).squeeze(-1)
                actor_entropy = -(actor_probs.clamp_min(1e-8) * actor_probs.clamp_min(1e-8).log()).sum(dim=-1)
                behavior_entropy = -(behavior_probs.clamp_min(1e-8) * behavior_probs.clamp_min(1e-8).log()).sum(dim=-1)
                actor_actions = actor_probs.argmax(dim=-1)
                self.actor_raw_probability_mass += actor_probs.sum(dim=0).cpu().numpy()
                self.behavior_probability_mass += behavior_probs.sum(dim=0).cpu().numpy()
                self.actor_raw_entropy_sum += float(actor_entropy.sum().item())
                self.behavior_entropy_sum += float(behavior_entropy.sum().item())
                self.action_row_count += int(actor_probs.shape[0])
                for action in actor_actions.cpu().tolist():
                    self.actor_raw_action_counts[int(action)] += 1
            for pos, idx in enumerate(active):
                action = int(sampled[pos].item())
                actions[idx] = action
                self.action_counts[action] += 1
                self.behavior_action_counts[action] += 1
                self._pending_actions[idx] = (
                    action,
                    float(actor_probs[pos, action].item()),
                    float(behavior_probs[pos, action].item()),
                    epsilon,
                )
        result = env.step(actions, self._evader_actions(scene))
        step_after = int(self.global_step + 1)
        if scene == "voradj" and env.capture_snapshot is not None and self.transition_points["first_natural_capture_step"] is None:
            self.transition_points["first_natural_capture_step"] = step_after
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
            transition = Transition(
                obs=clone_obs(obs), action=action, reward=float(result.rewards[idx]), next_obs=clone_obs(nxt),
                done=done, terminated=terminated, truncated=truncated, phase=str(metadata.get("phase", "pre_capture")),
                replay_class=replay_class(self.formal_config, scene, metadata), behavior_probability=behavior_prob,
                epsilon=used_epsilon, actor_probability=actor_prob, episode_id=self.episode_id,
                agent_id=idx, metadata=metadata,
            )
            self.replay.add(transition)
            if transition.replay_class == "post_capture_real" and self.transition_points["first_post_capture_real_row_step"] is None:
                self.transition_points["first_post_capture_real_row_step"] = step_after
            self.replay_role_counts["support" if bool(metadata.get("support_candidate", False)) else "pursuing"] += 1
        self.observations[scene] = next_obs
        self.global_step += 1
        if self.replay.class_counts()["post_capture_real"] >= 32 and self.transition_points["first_post_buffer_ge_32_step"] is None:
            self.transition_points["first_post_buffer_ge_32_step"] = int(self.global_step)
        self._maybe_update()
        if self.global_step % 1000 == 0 or any(
            value == self.global_step for value in self.transition_points.values() if value is not None
        ):
            self._append_jsonl("replay_sizes.jsonl", {
                "global_step": int(self.global_step),
                **self.replay.class_counts(),
                "replay_total": len(self.replay),
                "recovery_pool_size": len(self.recovery_pool),
            })
        if any(result.dones):
            record = env.episode_record(task=scene)
            snapshot = env.capture_snapshot
            if self.mode == "mix" and scene == "voradj" and snapshot is not None:
                self.recovery_pool.add(snapshot)
                self.capture_snapshots_added += 1
            record["scene"] = scene
            record["recovery_pool_size"] = len(self.recovery_pool)
            self.episode_reports.append(record)
            self._append_jsonl("episodes.jsonl", {"global_step": int(self.global_step), **record})
            self.episode_count += 1
            self.episode_id += 1
            self.scene_index = (self.scene_index + 1) % len(self.scenes())
            self._reset_current_scene()

    def _accumulate_telemetry(self, stats: Mapping[str, Any]) -> None:
        for key, raw_value in stats.items():
            if key in {"global_step", "sampler", "sample_actual_counts"}:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                raise FloatingPointError(f"non-finite AC telemetry: {key}={value}")
            bucket = self.telemetry_aggregate.setdefault(
                str(key), {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": value, "max": value}
            )
            bucket["count"] += 1.0
            bucket["sum"] += value
            bucket["sum_sq"] += value * value
            bucket["min"] = min(bucket["min"], value)
            bucket["max"] = max(bucket["max"], value)

    def _safety_check_update(self, stats: Mapping[str, Any]) -> None:
        if not bool(float(stats.get("finite", 0.0))):
            raise FloatingPointError("AC update reported finite=0")
        for key in ("q_mean", "q_std", "q_min", "q_max", "td_target_mean", "td_target_std"):
            value = float(stats.get(key, 0.0))
            if not np.isfinite(value) or abs(value) > 1.0e6:
                raise FloatingPointError(f"Q/target explosion: {key}={value}")

    def _write_first_nonfinite_artifact(self, failure: ACNumericFailure, batch: Sequence[Transition]) -> None:
        if self.first_nonfinite_artifact_written:
            return
        self.first_nonfinite_artifact_written = True
        report = dict(failure.report)
        report_dir = self.run_dir / "artifacts"
        report_dir.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for key in ("self", "pursuers", "evaders", "obstacles", "masks", "types"):
            arrays[f"obs_{key}"] = np.stack([item.obs[key] for item in batch])
            arrays[f"next_obs_{key}"] = np.stack([item.next_obs[key] for item in batch])
        arrays.update(
            {
                "action": np.asarray([item.action for item in batch], dtype=np.int64),
                "reward": np.asarray([item.reward for item in batch], dtype=np.float64),
                "terminated": np.asarray([item.terminated for item in batch], dtype=np.bool_),
                "truncated": np.asarray([item.truncated for item in batch], dtype=np.bool_),
                "active_mask": np.ones(len(batch), dtype=np.bool_),
                "behavior_probability": np.asarray([item.behavior_probability for item in batch], dtype=np.float64),
                "actor_probability": np.asarray([item.actor_probability for item in batch], dtype=np.float64),
                "epsilon": np.asarray([item.epsilon for item in batch], dtype=np.float64),
                "episode_id": np.asarray([item.episode_id for item in batch], dtype=np.int64),
                "agent_id": np.asarray([item.agent_id for item in batch], dtype=np.int64),
                "offending_rows": np.asarray(report.get("offending_rows", []), dtype=np.int64),
            }
        )
        np.savez_compressed(report_dir / "first_nonfinite_batch.npz", **arrays)
        report.update(
            {
                "task": self.mode,
                "env_step": int(self.global_step),
                "replay_composition": dict(Counter(item.replay_class for item in batch)),
                "replay_classes": [str(item.replay_class) for item in batch],
                "artifact_batch_path": str(report_dir / "first_nonfinite_batch.npz"),
                "artifact_report_path": str(report_dir / "first_nonfinite_report.json"),
            }
        )
        write_json(report_dir / "first_nonfinite_report.json", report)

    def _maybe_update(self) -> None:
        ac = self.formal_config.get("ac", {}) or {}
        if self.global_step % int(ac.get("train_freq", 4)) != 0:
            return
        self.update_attempts += 1
        requested_counts = {
            name: int((ac.get("replay_batch_counts", {}) or {}).get(name, 0))
            for name in REPLAY_CLASSES
        }
        if self.mode == "mix":
            available = self.replay.class_counts()
            post_requested = requested_counts["post_capture_real"]
            post_available = min(post_requested, available["post_capture_real"])
            substitution = post_requested - post_available
            batch_counts = dict(requested_counts)
            batch_counts["post_capture_real"] = post_available
            batch_counts["recovery_pure"] += substitution
            try:
                batch, report = self.replay.sample_stratified(batch_counts, self.replay_rng, fallback="defer")
            except ReplayWarmupError:
                self.warmup_deferrals += 1
                return
            report.update({
                "contract_requested_counts": requested_counts,
                "effective_counts": batch_counts,
                "available_before": available,
                "post_to_pure_substitution": int(substitution),
            })
        else:
            if len(self.replay) < self.trainer_config.min_replay_size:
                return
            batch = self.replay.sample_uniform(self.trainer_config.batch_size, self.replay_rng)
            report = {"sampler": "uniform_shared", "actual_counts": dict(Counter(item.replay_class for item in batch))}
        if len(batch) != self.trainer_config.batch_size:
            return
        try:
            stats = self.trainer.update(batch)
        except ACNumericFailure as failure:
            self._write_first_nonfinite_artifact(failure, batch)
            raise
        self._safety_check_update(stats)
        self._accumulate_telemetry(stats)
        for item in batch:
            self.sample_action_counts[int(item.action)] += 1
            self.sample_class_counts[str(item.replay_class)] += 1
        stats.update({"global_step": self.global_step, "sampler": report.get("sampler", "unknown"), "sample_actual_counts": report.get("actual_counts", {})})
        actual = {name: int(report.get("actual_counts", {}).get(name, 0)) for name in REPLAY_CLASSES}
        if self.transition_points["first_learner_update_step"] is None:
            self.transition_points["first_learner_update_step"] = int(self.global_step)
        if actual["post_capture_real"] > 0 and self.transition_points["first_batch_with_nonzero_post_samples_step"] is None:
            self.transition_points["first_batch_with_nonzero_post_samples_step"] = int(self.global_step)
        if actual == {"pursuing": 64, "pre_capture_cover": 16, "post_capture_real": 32, "recovery_pure": 16}:
            self.strict_batch_count += 1
            if self.transition_points["first_strict_64_16_32_16_batch_step"] is None:
                self.transition_points["first_strict_64_16_32_16_batch_step"] = int(self.global_step)
        if int(report.get("post_to_pure_substitution", 0)) > 0:
            self.substitution_batch_count += 1
        report["global_step"] = int(self.global_step)
        report["actual_counts"] = actual
        self.sampler_reports.append(report)
        if len(self.sampler_reports) > 200:
            del self.sampler_reports[:-200]
        self._append_jsonl("batch_composition.jsonl", report)
        self.telemetry.append(stats)
        if len(self.telemetry) > 1000:
            del self.telemetry[:-1000]

    def _checkpoint_smoke_save_load(self) -> Dict[str, Any]:
        payload = self.trainer.checkpoint_payload(config=self.formal_config, global_step=self.global_step, episode=self.episode_count, include_runtime=True, include_replay=False, replay=self.replay, recovery_pool=self.recovery_pool)
        with tempfile.TemporaryDirectory(prefix="ac_capability_smoke_") as temp_dir:
            path = Path(temp_dir) / "resume_latest.pt"
            atomic_torch_save(payload, path)
            loaded = torch.load(path, map_location=self.device, weights_only=False)
            restored = SharedLocalACTrainer(self.network_config, self.trainer_config, self.device)
            restored.load_payload(loaded, load_optimizer=True)
            return {"save_load_pass": bool(state_hash(restored.actor) == state_hash(self.trainer.actor) and state_hash(restored.critic) == state_hash(self.trainer.critic)), "payload_has_replay": "replay" in loaded, "payload_has_optimizer": "actor_optimizer" in loaded, "atomic_latest_only": True}

    @staticmethod
    def _value_stats(values: Sequence[Optional[float]]) -> Dict[str, Any]:
        finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=float)
        if finite.size == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None}
        return {
            "count": int(finite.size),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "median": float(np.median(finite)),
            "max": float(finite.max()),
        }

    def _telemetry_summary(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, bucket in self.telemetry_aggregate.items():
            count = max(float(bucket["count"]), 1.0)
            mean = float(bucket["sum"] / count)
            variance = max(float(bucket["sum_sq"] / count - mean * mean), 0.0)
            result[key] = {
                "count": int(bucket["count"]), "mean": mean, "std": float(np.sqrt(variance)),
                "min": float(bucket["min"]), "max": float(bucket["max"]),
            }
        return result

    def _exploration_summary(self) -> Dict[str, Any]:
        count = max(int(self.action_row_count), 1)
        raw_mass = self.actor_raw_probability_mass / float(count)
        behavior_mass = self.behavior_probability_mass / float(count)
        action_hist = {str(index): int(self.action_counts.get(index, 0)) for index in range(9)}
        raw_hist = {str(index): int(self.actor_raw_action_counts.get(index, 0)) for index in range(9)}
        sample_hist = {str(index): int(self.sample_action_counts.get(index, 0)) for index in range(9)}
        return {
            "epsilon_start": float(self.epsilon_values[0]) if self.epsilon_values else None,
            "epsilon_end": float(self.epsilon_values[-1]) if self.epsilon_values else None,
            "epsilon_at_25k": epsilon_value(self.formal_config, 25000),
            "epsilon_at_100k": epsilon_value(self.formal_config, 100000),
            "epsilon_at_500k": epsilon_value(self.formal_config, 500000),
            "behavior_action_histogram": action_hist,
            "actor_raw_argmax_histogram": raw_hist,
            "sampled_action_histogram": sample_hist,
            "actor_raw_probability_mean": {str(index): float(raw_mass[index]) for index in range(9)},
            "behavior_probability_mean": {str(index): float(behavior_mass[index]) for index in range(9)},
            "actor_raw_entropy_mean": float(self.actor_raw_entropy_sum / count),
            "behavior_entropy_mean": float(self.behavior_entropy_sum / count),
            "action_rows": int(self.action_row_count),
            "minimum_behavior_action_count": int(min(action_hist.values())) if action_hist else 0,
            "action_4_sampled": int(action_hist.get("4", 0)) > 0,
        }

    def _replay_summary(self) -> Dict[str, Any]:
        return {
            "total_rows": int(len(self.replay)),
            "class_counts": {str(key): int(value) for key, value in self.replay.class_counts().items()},
            "role_counts": {str(key): int(value) for key, value in self.replay_role_counts.items()},
            "sample_counts": {str(key): int(value) for key, value in self.sample_class_counts.items()},
            "sample_action_counts": {str(index): int(self.sample_action_counts.get(index, 0)) for index in range(9)},
            "recovery_pool_size": int(len(self.recovery_pool)),
        }

    def _eval_evader_actions(self, env: VorAdjEnv, apf_agents: List[ApfAgent]) -> List[Optional[int]]:
        if not env.evaders:
            return []
        set_global_config(env.config)
        observations = env.get_evader_observations_for_apf()
        if hasattr(env, "configure_evader_apf_agents"):
            env.configure_evader_apf_agents(apf_agents)
        return [agent.act(obs) if obs is not None else None for agent, obs in zip(apf_agents, observations)]

    @staticmethod
    def _ring_size(env: VorAdjEnv) -> int:
        sizes = [
            sum(float(np.hypot(p.x - e.x, p.y - e.y)) <= 8.0 for p in env.pursuers if not p.deactivated)
            for e in env.evaders if not e.deactivated
        ]
        sizes.extend(int(len(event.get("participants", []))) for event in (env.last_capture_events or []))
        return int(max(sizes, default=0))

    @torch.no_grad()
    def _deterministic_eval(self, step: int = 0) -> Dict[str, Any]:
        scene = self.scenes()[0]
        evaluation = (self.formal_config.get("ac", {}) or {}).get("evaluation", {}) or {}
        episodes = int(evaluation.get("episodes", 20 if not self.smoke else 1))
        seed_base = int(evaluation.get("seed_base", self.seed + 5000))
        decision_dt = float(self.formal_config.get("decision_dt", 0.5))
        max_steps = min(24 if self.smoke else 3000, int(self.envs[scene].episode_max_length))
        records: List[Dict[str, Any]] = []
        saved_python = random.getstate()
        saved_numpy = np.random.get_state()
        saved_torch = torch.get_rng_state()
        saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.trainer.actor.eval()
            for episode in range(episodes):
                env = make_env(self.formal_config, scene, seed_base + episode)
                if self.smoke:
                    env.episode_max_length = max_steps
                obs = env.reset()
                assert_runtime_aw9(env)
                apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
                total_return = 0.0
                capture_step: Optional[int] = None
                capture_types: List[str] = []
                collision_types: Counter[str] = Counter()
                ring2_steps = ring3_steps = 0
                ring2_run = ring3_run = 0
                ring2_max = ring3_max = 0
                ring2_windows = ring3_windows = 0
                eval_entropy: List[float] = []
                for _ in range(max_steps):
                    active = [idx for idx, item in enumerate(obs) if item is not None]
                    actions: List[Optional[int]] = [None] * len(obs)
                    if active:
                        batch = stack_obs([obs[idx] for idx in active], self.device)
                        probs = self.trainer.actor.probabilities(batch)
                        eval_entropy.extend((-(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)).cpu().tolist())
                        values = probs.argmax(dim=-1).cpu().tolist()
                        for idx, action in zip(active, values):
                            actions[idx] = int(action)
                    result = env.step(actions, self._eval_evader_actions(env, apf_agents))
                    total_return += float(np.sum(result.rewards))
                    for event in env.last_capture_events or []:
                        capture_types.append(str(event.get("capture_type", "normal")))
                        if capture_step is None:
                            capture_step = int(env.episode_step)
                    for event in env.last_collision_events or []:
                        collision_types[str(event.get("type", "unknown"))] += 1
                    ring_size = self._ring_size(env)
                    if ring_size >= 2:
                        ring2_steps += 1
                        ring2_run += 1
                    else:
                        if ring2_run >= 2:
                            ring2_windows += 1
                        ring2_run = 0
                    if ring_size >= 3:
                        ring3_steps += 1
                        ring3_run += 1
                    else:
                        if ring3_run >= 2:
                            ring3_windows += 1
                        ring3_run = 0
                    ring2_max = max(ring2_max, ring2_run)
                    ring3_max = max(ring3_max, ring3_run)
                    obs = result.observations
                    if any(result.dones):
                        break
                if ring2_run >= 2:
                    ring2_windows += 1
                if ring3_run >= 2:
                    ring3_windows += 1
                if not any(result.dones) and not self.smoke:
                    raise RuntimeError("formal deterministic evaluation did not reach native terminal")
                record = env.episode_record(task=scene)
                distribution = record.get("distribution_metrics", {}) or {}
                captured = bool(record.get("captured", False))
                stationary = bool(captured and "stationary" in capture_types)
                normal = bool(captured and any(value != "stationary" for value in capture_types))
                collision = bool(collision_types) or bool(record.get("collision_event", False))
                records.append({
                    "episode": int(episode), "seed": int(seed_base + episode), "length": int(env.episode_step),
                    "return": float(total_return), "captured": captured, "normal_capture": normal,
                    "stationary_capture": stationary, "capture_types": capture_types,
                    "capture_time_steps": capture_step, "capture_time_seconds": None if capture_step is None else float(capture_step * decision_dt),
                    "ring_2_plus_steps": int(ring2_steps), "ring_3_plus_steps": int(ring3_steps),
                    "ring_2_plus_max_hold": int(ring2_max), "ring_3_plus_max_hold": int(ring3_max),
                    "repeated_2_plus_ring_windows": int(ring2_windows), "repeated_3_plus_ring_windows": int(ring3_windows),
                    "collision": collision, "collision_types": dict(collision_types),
                    "eval_entropy": float(np.mean(eval_entropy)) if eval_entropy else None,
                    "coverage_strict_success": bool(record.get("coverage_strict_success", False)),
                    "coverage_ce_rms": float(record.get("coverage_ce_center_rms", distribution.get("ce_center_rms", 0.0))),
                    "coverage_ce_max": float(record.get("coverage_ce_center_max", distribution.get("ce_center_max", 0.0))),
                    "area_cv": float(record.get("coverage_strict_area_cv", distribution.get("area_cv", 0.0))),
                    "post_capture_episode": bool(record.get("post_capture_episode", False)),
                    "post_capture_coverage_success": bool(record.get("post_capture_coverage_success", False)),
                    "post_capture_strict_success": bool(record.get("post_capture_episode", False) and record.get("coverage_strict_success", False)),
                    "early_recovery_duration": int((record.get("voradj_metrics", {}) or {}).get("post_capture_coverage_step", -1)),
                })
        finally:
            random.setstate(saved_python)
            np.random.set_state(saved_numpy)
            torch.set_rng_state(saved_torch)
            if saved_cuda is not None:
                torch.cuda.set_rng_state_all(saved_cuda)

        def total(key: str) -> int:
            return int(sum(bool(row.get(key, False)) for row in records))

        collision_types = Counter()
        for row in records:
            collision_types.update(row["collision_types"])
        report = {
            "schema": "ac-capability-pure-capture-eval-v2", "step": int(step), "scene": scene,
            "episodes": int(len(records)), "seed_base": seed_base, "deterministic": True,
            "records": records,
            "total_capture": total("captured"), "normal_capture": total("normal_capture"),
            "stationary_capture": total("stationary_capture"), "capture_rate": float(total("captured") / max(len(records), 1)),
            "normal_capture_rate": float(total("normal_capture") / max(len(records), 1)),
            "stationary_capture_rate": float(total("stationary_capture") / max(len(records), 1)),
            "repeated_2_plus_ring_windows": int(sum(row["repeated_2_plus_ring_windows"] for row in records)),
            "repeated_3_plus_ring_windows": int(sum(row["repeated_3_plus_ring_windows"] for row in records)),
            "ring_2_plus_repeated_episode_rate": float(np.mean([row["ring_2_plus_max_hold"] >= 2 for row in records])) if records else 0.0,
            "ring_3_plus_repeated_episode_rate": float(np.mean([row["ring_3_plus_max_hold"] >= 2 for row in records])) if records else 0.0,
            "ring_persistence": {
                "ring_2_plus_steps": self._value_stats([row["ring_2_plus_steps"] for row in records]),
                "ring_3_plus_steps": self._value_stats([row["ring_3_plus_steps"] for row in records]),
                "ring_2_plus_max_hold": self._value_stats([row["ring_2_plus_max_hold"] for row in records]),
                "ring_3_plus_max_hold": self._value_stats([row["ring_3_plus_max_hold"] for row in records]),
            },
            "collision": {"total": total("collision"), "rate": float(total("collision") / max(len(records), 1)), "types": dict(collision_types)},
            "capture_time": {
                "steps": self._value_stats([row["capture_time_steps"] for row in records]),
                "seconds": self._value_stats([row["capture_time_seconds"] for row in records]),
            },
            "return": self._value_stats([row["return"] for row in records]),
            "episode_length": self._value_stats([row["length"] for row in records]),
            "eval_entropy": self._value_stats([row["eval_entropy"] for row in records]),
            "training_telemetry": self._telemetry_summary(),
            "exploration": self._exploration_summary(),
            "replay": self._replay_summary(),
            "training_updates": int(self.trainer.update_count),
            "real_post_capture_recovery": {
                "episodes": int(sum(bool(row.get("post_capture_episode", False)) for row in records)),
                "post_capture_ce_rms": self._value_stats([row["coverage_ce_rms"] for row in records if row.get("post_capture_episode")]),
                "strict_ce_rms": self._value_stats([row["coverage_ce_rms"] for row in records if row.get("post_capture_episode")]),
                "early_recovery_duration": self._value_stats([row["early_recovery_duration"] for row in records if row.get("post_capture_episode")]),
                "post_capture_success_rate": float(np.mean([row["post_capture_coverage_success"] for row in records if row.get("post_capture_episode")])) if any(row.get("post_capture_episode") for row in records) else 0.0,
                "post_capture_strict_success_rate": float(np.mean([row["post_capture_strict_success"] for row in records if row.get("post_capture_episode")])) if any(row.get("post_capture_episode") for row in records) else 0.0,
                "safe_full_completion_rate": float(np.mean([row["captured"] and not row["collision"] and row.get("post_capture_strict_success", False) for row in records])) if records else 0.0,
            },
        }
        self.evaluation_reports.append(report)
        return report

    @torch.no_grad()
    def _deterministic_coverage_eval(self, step: int = 0) -> Dict[str, Any]:
        scene = "voradj_coverage"
        evaluation = (self.formal_config.get("ac", {}) or {}).get("evaluation", {}) or {}
        episodes = int(evaluation.get("episodes", 20 if not self.smoke else 1))
        seed_base = int(evaluation.get("seed_base", self.seed + 6000))
        max_steps = min(24 if self.smoke else 3000, int(self.envs[scene].episode_max_length))
        records: List[Dict[str, Any]] = []
        saved_python = random.getstate()
        saved_numpy = np.random.get_state()
        saved_torch = torch.get_rng_state()
        saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.trainer.actor.eval()
            for episode in range(episodes):
                env = make_env(self.formal_config, scene, seed_base + episode)
                obs = env.reset()
                assert_runtime_aw9(env)
                total_return = 0.0
                eval_entropy: List[float] = []
                done = False
                for _ in range(max_steps):
                    active = [idx for idx, item in enumerate(obs) if item is not None]
                    actions: List[Optional[int]] = [None] * len(obs)
                    if active:
                        batch = stack_obs([obs[idx] for idx in active], self.device)
                        probs = self.trainer.actor.probabilities(batch)
                        eval_entropy.extend((-(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)).cpu().tolist())
                        values = probs.argmax(dim=-1).cpu().tolist()
                        for idx, action in zip(active, values):
                            actions[idx] = int(action)
                    result = env.step(actions, [])
                    total_return += float(np.sum(result.rewards))
                    obs = result.observations
                    if any(result.dones):
                        done = True
                        break
                if not done and not self.smoke:
                    raise RuntimeError("formal pure-coverage evaluation did not reach native terminal")
                record = env.episode_record(task=scene)
                distribution = record.get("distribution_metrics", {}) or {}
                strict_distribution = record.get("coverage_strict_distribution_metrics", {}) or {}
                records.append({
                    "episode": int(episode), "seed": int(seed_base + episode), "length": int(env.episode_step),
                    "return": float(total_return), "collision": bool(record.get("collision_event", False)),
                    "coverage_strict_success": bool(record.get("coverage_strict_success", False)),
                    "coverage_ce_rms": float(record.get("coverage_ce_center_rms", strict_distribution.get("ce_center_rms", distribution.get("ce_center_rms", 0.0)))),
                    "coverage_ce_max": float(record.get("coverage_ce_center_max", strict_distribution.get("ce_center_max", distribution.get("ce_center_max", 0.0)))),
                    "area_cv": float(record.get("coverage_strict_area_cv", distribution.get("area_cv", 0.0))),
                    "safe_full_completion": bool(record.get("coverage_strict_success", False) and record.get("collision_free", False)),
                    "eval_entropy": float(np.mean(eval_entropy)) if eval_entropy else None,
                })
        finally:
            random.setstate(saved_python)
            np.random.set_state(saved_numpy)
            torch.set_rng_state(saved_torch)
            if saved_cuda is not None:
                torch.cuda.set_rng_state_all(saved_cuda)
        collision_count = int(sum(bool(row["collision"]) for row in records))
        report = {
            "schema": "ac-capability-pure-coverage-eval-v1", "step": int(step), "scene": scene,
            "episodes": int(len(records)), "seed_base": seed_base, "deterministic": True, "records": records,
            "strict_ce": self._value_stats([row["coverage_ce_rms"] for row in records]),
            "ce_rms": self._value_stats([row["coverage_ce_rms"] for row in records]),
            "ce_max": self._value_stats([row["coverage_ce_max"] for row in records]),
            "area_cv": self._value_stats([row["area_cv"] for row in records]),
            "strict_success_rate": float(np.mean([row["coverage_strict_success"] for row in records])) if records else 0.0,
            "safe_full_completion_rate": float(np.mean([row["safe_full_completion"] for row in records])) if records else 0.0,
            "collision": {"total": collision_count, "rate": float(collision_count / max(len(records), 1))},
            "mission_time": self._value_stats([row["length"] for row in records]),
            "eval_entropy": self._value_stats([row["eval_entropy"] for row in records]),
        }
        return report

    def _evaluation_bundle(self, step: int) -> Dict[str, Any]:
        bundle = {
            "schema": "ac-capability-formal-eval-bundle-v1",
            "step": int(step),
            "capture": self._deterministic_eval(step),
            "pure_coverage": self._deterministic_coverage_eval(step),
            "replay": self._replay_summary(),
            "exploration": self._exploration_summary(),
            "training_telemetry": self._telemetry_summary(),
            "transition_points": dict(self.transition_points),
            "scheduler_prefix": self.scheduler_sequence[: min(len(self.scheduler_sequence), 48)],
            "reset_source_counts": dict(self.reset_sources),
            "cross_init": {
                "capture_snapshots_added": int(self.capture_snapshots_added),
                "pool_size": int(len(self.recovery_pool)),
                "pool_capacity": int(self.recovery_pool.data.maxlen or 0),
                "configured_capture_snapshot_ratio": float(self.recovery_pool.captured_ratio),
                "configured_non_capture_map_random_ratio": float(self.recovery_pool.map_random_ratio),
            },
        }
        self.evaluation_bundles.append(bundle)
        write_json(self.run_dir / "evaluations" / f"eval_step_{int(step):09d}.json", bundle)
        write_json(self.run_dir / "evaluations" / "formal_eval_table.json", self.evaluation_bundles)
        return bundle

    def run(self) -> Dict[str, Any]:
        self._pending_actions: Dict[int, Tuple[int, float, float, float]] = {}
        total = int(self.smoke_steps if self.smoke else self.formal_config.get("total_timesteps", 500000))
        evaluation_cfg = (self.formal_config.get("ac", {}) or {}).get("evaluation", {}) or {}
        checkpoint_cfg = (self.formal_config.get("ac", {}) or {}).get("checkpoint", {}) or {}
        evaluation_steps = set(int(value) for value in evaluation_cfg.get("formal_steps", [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000, 400000, 500000]))
        checkpoint_steps = set(int(value) for value in checkpoint_cfg.get("formal_steps", [25000, 50000, 75000, 100000, 150000, 200000, 300000, 400000, 500000]))
        if not self.smoke and 0 in evaluation_steps:
            self._evaluation_bundle(0)
        for _ in range(total):
            scene = self.scenes()[self.scene_index % len(self.scenes())]
            self._collect_step(scene)
            if not self.smoke and self.global_step in checkpoint_steps:
                self._write_formal_checkpoint(self.global_step)
            if not self.smoke and self.global_step in evaluation_steps:
                self._evaluation_bundle(self.global_step)
        checkpoint_result = self._checkpoint_smoke_save_load() if self.smoke else {"formal_checkpoint_policy_ready": True}
        if not self.smoke and self.global_step not in evaluation_steps:
            self._evaluation_bundle(self.global_step)
        if not self.smoke and self.trainer.update_count <= 0:
            raise RuntimeError("safety stop: no AC updates completed")
        eval_result = self._evaluation_bundle(self.global_step) if self.smoke else self.evaluation_bundles[-1]
        exploration = self._exploration_summary()
        action_hist = exploration["behavior_action_histogram"]
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
            "exploration": exploration, "training_telemetry": self._telemetry_summary(), "replay": self._replay_summary(),
            "evaluation_steps_completed": [int(item["step"]) for item in self.evaluation_reports],
            "evaluation_curve": [
                {
                    "step": int(item["step"]), "total_capture": int(item["total_capture"]),
                    "normal_capture": int(item["normal_capture"]), "stationary_capture": int(item["stationary_capture"]),
                    "repeated_2_plus_ring_windows": int(item["repeated_2_plus_ring_windows"]),
                    "repeated_3_plus_ring_windows": int(item["repeated_3_plus_ring_windows"]),
                    "ring_2_plus_repeated_episode_rate": float(item["ring_2_plus_repeated_episode_rate"]),
                    "ring_3_plus_repeated_episode_rate": float(item["ring_3_plus_repeated_episode_rate"]),
                    "collision_rate": float(item["collision"]["rate"]),
                    "capture_time_seconds": item["capture_time"]["seconds"], "return": item["return"],
                }
                for item in self.evaluation_reports
            ],
            "evaluation_bundles": self.evaluation_bundles,
            "transition_points": dict(self.transition_points),
            "update_attempts": self.update_attempts,
            "warmup_deferrals": self.warmup_deferrals,
            "substitution_batch_count": self.substitution_batch_count,
            "strict_batch_count": self.strict_batch_count,
            "replay_size": len(self.replay), "replay_class_counts": self.replay.class_counts(), "recovery_pool_size": len(self.recovery_pool),
            "scheduler_sequence": self.scheduler_sequence[: min(len(self.scheduler_sequence), 24)], "reset_source_counts": dict(self.reset_sources),
            "capture_snapshots_added": self.capture_snapshots_added,
            "sampler_reports": self.sampler_reports[-20:], "telemetry_tail": self.telemetry[-20:], "episode_tail": self.episode_reports[-10:],
            "all_telemetry_finite": all(bool(item.get("finite", 0.0)) for item in self.telemetry), "checkpoint_save_load": checkpoint_result,
            "deterministic_eval": eval_result, "exploration_implementation_fail": bool(self.epsilon_values and self.epsilon_values[-1] > 0 and counts.min() == 0),
            "formal_launch_started": not self.smoke,
            "resume_info": self.resume_info,
        }
        write_json(self.run_dir / ("smoke_report.json" if self.smoke else "formal_run_report.json"), result)
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
