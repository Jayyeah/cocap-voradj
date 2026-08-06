"""Single-YAML formal CTDE MASAC training runner.

The runner deliberately has no local-critic, smoke, or CLI algorithm override
path.  It resolves exactly one formal YAML, deep-merges ``tasks.<scene>``,
stores one joint transition per environment step, and updates on focal-agent
items produced by ``FocalReplaySampler``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml

from cocap_voradj.dynamics.continuous_action import ActionContractError
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.continuous.central_attention_critic import CentralCriticConfig
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.training.continuous.central_sac import CentralSACConfig, CentralSACTrainer
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.continuous.curriculum_snapshots import restore_snapshot
from cocap_voradj.training.continuous.formal_config import SCENES, resolve_formal_config, scene_config
from cocap_voradj.training.continuous.joint_replay import (
    FOCAL_BUCKETS,
    FocalReplaySampler,
    JointReplayBuffer,
)
from cocap_voradj.training.trainer import set_global_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG_PATH = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-06_ctde_contract"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _replace_dir_atomic(tmp_dir: Path, final_dir: Path) -> None:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    old_dir = final_dir.parent / f"{final_dir.name}.old_{os.getpid()}"
    if final_dir.exists():
        final_dir.rename(old_dir)
    try:
        tmp_dir.rename(final_dir)
    except Exception:
        if old_dir.exists():
            old_dir.rename(final_dir)
        raise
    if old_dir.exists():
        shutil.rmtree(old_dir)


def _save_checkpoint_bundle(
    artifact_dir: Path,
    step: int,
    root_config: Dict[str, Any],
    manifest: Dict[str, Any],
    trainer: CentralSACTrainer,
    replay: JointReplayBuffer,
    runtime_state: Dict[str, Any],
    metrics_history: List[Dict[str, Any]],
    diagnostic_eval: Dict[str, Any],
) -> Path:
    bundle_dir = artifact_dir / "checkpoints" / f"step_{int(step):09d}"
    if bundle_dir.exists():
        return bundle_dir
    tmp_dir = artifact_dir / ".tmp" / f"step_{int(step):09d}_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(tmp_dir / "trainer.pt", manifest, runtime_state=runtime_state)
    replay.save(tmp_dir / "replay.pkl", manifest, runtime_state=runtime_state)
    (tmp_dir / "runtime_state.pkl").write_bytes(pickle.dumps(runtime_state))
    (tmp_dir / "effective_config.yaml").write_text(yaml.safe_dump(root_config, sort_keys=False), encoding="utf-8")
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metrics_history),
        encoding="utf-8",
    )
    (tmp_dir / "diagnostic_eval.json").write_text(json.dumps(diagnostic_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _replace_dir_atomic(tmp_dir, bundle_dir)
    return bundle_dir


def _implementation_hash() -> str:
    paths = [
        Path(__file__),
        ROOT / "src/cocap_voradj/training/continuous/joint_replay.py",
        ROOT / "src/cocap_voradj/training/continuous/central_sac.py",
        ROOT / "src/cocap_voradj/training/continuous/central_schema.py",
        ROOT / "src/cocap_voradj/training/continuous/formal_config.py",
        ROOT / "src/cocap_voradj/models/continuous/central_attention_critic.py",
        ROOT / "src/cocap_voradj/models/continuous/radial_actor.py",
        ROOT / "src/cocap_voradj/dynamics/continuous_action.py",
        ROOT / "src/cocap_voradj/dynamics/robot.py",
        ROOT / "src/cocap_voradj/envs/base.py",
        ROOT / "src/cocap_voradj/envs/voronoi_adjacency.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _tensor_obs(obs: Mapping[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in obs.items()}


def _pad_local_obs_tree(
    observations: List[Optional[Mapping[str, np.ndarray]]],
    max_agents: int,
    actor_max_pursuers: int,
) -> Dict[str, np.ndarray]:
    """Pad per-agent local observations to fixed central/replay slots."""
    max_evaders = 8
    max_obstacles = 5
    token_count = 1 + actor_max_pursuers + max_evaders + max_obstacles
    self_dim = 9
    zero_shape = {
        "self": (self_dim,),
        "pursuers": (actor_max_pursuers, 7),
        "evaders": (max_evaders, 7),
        "obstacles": (max_obstacles, 5),
        "masks": (token_count,),
        "types": (token_count,),
    }
    types_template = np.asarray([0] + [1] * actor_max_pursuers + [2] * max_evaders + [3] * max_obstacles, dtype=np.int64)
    result: Dict[str, List[np.ndarray]] = {key: [] for key in zero_shape}
    for slot in range(max_agents):
        obs = observations[slot] if slot < len(observations) else None
        if obs is None:
            values = {
                "self": np.zeros(self_dim, dtype=np.float32),
                "pursuers": np.zeros((actor_max_pursuers, 7), dtype=np.float32),
                "evaders": np.zeros((max_evaders, 7), dtype=np.float32),
                "obstacles": np.zeros((max_obstacles, 5), dtype=np.float32),
                "masks": np.zeros(token_count, dtype=bool),
                "types": types_template.copy(),
            }
            values["masks"][0] = True
        else:
            old_p = int(np.asarray(obs["pursuers"]).shape[0])
            if old_p > actor_max_pursuers:
                raise ValueError("local observation pursuer padding exceeds actor max_pursuers")
            pursuers = np.zeros((actor_max_pursuers, 7), dtype=np.float32)
            pursuers[:old_p] = np.asarray(obs["pursuers"], dtype=np.float32)
            old_masks = np.asarray(obs["masks"], dtype=bool)
            old_types = np.asarray(obs["types"], dtype=np.int64)
            new_masks = np.concatenate(
                [
                    old_masks[: 1 + old_p],
                    np.zeros(actor_max_pursuers - old_p, dtype=bool),
                    old_masks[1 + old_p :],
                ]
            )
            new_types = np.concatenate(
                [
                    old_types[: 1 + old_p],
                    np.full(actor_max_pursuers - old_p, 1, dtype=np.int64),
                    old_types[1 + old_p :],
                ]
            )
            values = {
                "self": np.asarray(obs["self"], dtype=np.float32),
                "pursuers": pursuers,
                "evaders": np.asarray(obs["evaders"], dtype=np.float32),
                "obstacles": np.asarray(obs["obstacles"], dtype=np.float32),
                "masks": new_masks,
                "types": new_types,
            }
        for key in zero_shape:
            if values[key].shape != zero_shape[key]:
                raise ValueError(f"padded local observation {key} shape mismatch: {values[key].shape} != {zero_shape[key]}")
            result[key].append(values[key])
    return {key: np.stack(result[key], axis=0) for key in zero_shape}


def _stack_with_batch(
    observations: List[Optional[Mapping[str, np.ndarray]]],
    max_agents: int,
    actor_max_pursuers: int,
) -> Dict[str, np.ndarray]:
    tree = _pad_local_obs_tree(observations, max_agents, actor_max_pursuers)
    return {key: value[None] for key, value in tree.items()}


def _pad_vector(values: Any, max_agents: int, dtype: Any) -> np.ndarray:
    result = np.zeros(max_agents, dtype=dtype)
    array = np.asarray(values)
    result[: len(array)] = array
    return result


def _pad_actions(actions: np.ndarray, max_agents: int) -> np.ndarray:
    result = np.zeros((max_agents, 2), dtype=np.float32)
    result[: actions.shape[0]] = np.asarray(actions, dtype=np.float32)
    return result


def _split_termination_flags(
    dones: List[bool] | np.ndarray,
    infos: List[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    done_array = np.asarray(dones, dtype=bool)
    truncated_states = {"too long episode", "pre-capture timeout"}
    truncated = np.asarray(
        [
            bool(done and info.get("state") in truncated_states)
            for done, info in zip(done_array, infos)
        ],
        dtype=bool,
    )
    terminated = done_array & ~truncated
    return terminated, truncated


def _sample_actions(
    trainer: CentralSACTrainer,
    padded_obs: Mapping[str, np.ndarray],
    active_count: int,
    adapter: Any,
    deterministic: bool,
    *,
    uniform_disk: bool = False,
    rng: Optional[np.random.Generator] = None,
    a_max: float = 0.4,
) -> Tuple[np.ndarray, float]:
    if uniform_disk:
        rng = rng or np.random.default_rng(0)
        radii = float(a_max) * np.sqrt(rng.uniform(0.0, 1.0, size=active_count))
        angles = rng.uniform(0.0, 2.0 * np.pi, size=active_count)
        raw = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1).astype(np.float32)
    else:
        with torch.no_grad():
            raw, _, _ = trainer.actor.sample(_tensor_obs(padded_obs, trainer.device), deterministic=deterministic)
        raw = raw.detach().cpu().numpy()
    commands: List[np.ndarray] = []
    rejected = 0
    for action in raw[:active_count]:
        validated, diagnostics = adapter.validate_with_diagnostics(action)
        rejected += int(diagnostics.action_rejected)
        commands.append(validated)
    return np.asarray(commands, dtype=np.float32), float(rejected / max(active_count, 1))


def _make_trainer(config: Dict[str, Any], device: str) -> CentralSACTrainer:
    actor_cfg = config["actor"]
    critic_cfg = config["central_critic"]
    sac_cfg = config["masac"]
    encoder = LocalEntityTokenEncoderConfig(
        hidden_dim=int(actor_cfg["hidden_dim"]),
        num_heads=int(actor_cfg["num_heads"]),
        num_layers=int(actor_cfg["num_layers"]),
        self_feature_dim=int(actor_cfg.get("self_feature_dim", 9)),
        max_pursuers=int(actor_cfg.get("max_pursuers", 12)),
        max_evaders=int(actor_cfg.get("max_evaders", 8)),
        max_obstacles=int(actor_cfg.get("max_obstacles", 5)),
        dropout=float(actor_cfg.get("dropout", 0.0)),
    )
    actor = RadialActorConfig(
        hidden_dim=int(actor_cfg["hidden_dim"]),
        a_max=float(config["action"]["a_max"]),
        decision_dt=float(config["dynamics"]["decision_dt"]),
        log_std_min=float(actor_cfg.get("log_std_min", -5.0)),
        log_std_max=float(actor_cfg.get("log_std_max", 1.0)),
        dropout=float(actor_cfg.get("dropout", 0.0)),
    )
    critic = CentralCriticConfig(
        hidden_dim=int(critic_cfg["hidden_dim"]),
        num_heads=int(critic_cfg["num_heads"]),
        num_layers=int(critic_cfg["num_layers"]),
        self_feature_dim=int(critic_cfg.get("self_feature_dim", 9)),
        max_agents=int(critic_cfg["max_agents"]),
        max_evaders=int(critic_cfg.get("max_evaders", 8)),
        max_obstacles=int(critic_cfg.get("max_obstacles", 5)),
        dropout=float(critic_cfg.get("dropout", 0.0)),
    )
    trainer_cfg = CentralSACConfig(
        hidden_dim=int(critic_cfg["hidden_dim"]),
        gamma=float(sac_cfg["gamma"]),
        tau=float(sac_cfg["tau"]),
        actor_lr=float(sac_cfg["actor_lr"]),
        critic_lr=float(sac_cfg["critic_lr"]),
        alpha_lr=float(sac_cfg["alpha_lr"]),
        alpha_init=float(sac_cfg["alpha_init"]),
        target_entropy=float(sac_cfg["target_entropy"]),
        grad_clip_norm=float(config["training"]["grad_clip_norm"]),
    )
    trainer = CentralSACTrainer(
        encoder_config=encoder,
        actor_config=actor,
        critic_config=critic,
        config=trainer_cfg,
        device=device,
        action_mode=str(config["action"]["mode"]),
    )
    trainer.warmup_steps = int(config["training"]["warmup_joint_transitions"])
    init_cfg = (config.get("initialization", {}) or {}).get("actor_encoder", {}) or {}
    init_mode = str(init_cfg.get("mode", "none")).strip().lower()
    if init_mode == "legacy_iqn":
        checkpoint = str(init_cfg.get("checkpoint", ""))
        if not checkpoint:
            raise ValueError("initialization.actor_encoder.mode=legacy_iqn requires checkpoint")
        if not bool(init_cfg.get("strict_shape_match", True)):
            raise ValueError("actor encoder transfer requires strict_shape_match=true")
        path = Path(checkpoint)
        if not path.is_absolute():
            path = ROOT / path
        payload = torch.load(path, map_location=trainer.device, weights_only=False)
        state_dict = payload.get("state_dict", payload)
        trainer.actor.encoder.load_legacy_iqn_state_dict(state_dict, strict=True)
        freeze_steps = int(init_cfg.get("freeze_env_steps", 0))
        if freeze_steps > 0:
            for parameter in trainer.actor.encoder.parameters():
                parameter.requires_grad_(False)
            trainer.freeze_encoder_env_steps = freeze_steps
    return trainer


def _trainer_contract(trainer: CentralSACTrainer) -> Dict[str, Any]:
    actor_cfg = getattr(trainer.actor, "config", None)
    critic_cfg = getattr(trainer.critic1, "config", None)
    config = trainer.config
    return {
        "actor_hidden_dim": int(getattr(actor_cfg, "hidden_dim", -1)),
        "actor_a_max": float(getattr(actor_cfg, "a_max", 0.0)),
        "actor_decision_dt": float(getattr(actor_cfg, "decision_dt", 0.0)),
        "actor_log_std_max": float(getattr(actor_cfg, "log_std_max", 0.0)),
        "critic_max_agents": int(getattr(critic_cfg, "max_agents", -1)),
        "critic_hidden_dim": int(getattr(critic_cfg, "hidden_dim", -1)),
        "critic_num_layers": int(getattr(critic_cfg, "num_layers", -1)),
        "critic_num_heads": int(getattr(critic_cfg, "num_heads", -1)),
        "gamma": float(config.gamma),
        "tau": float(config.tau),
        "actor_lr": float(config.actor_lr),
        "critic_lr": float(config.critic_lr),
        "alpha_lr": float(config.alpha_lr),
        "alpha_init": float(config.alpha_init),
        "target_entropy": float(config.target_entropy),
        "grad_clip_norm": getattr(config, "grad_clip_norm", None),
        "warmup_steps": int(getattr(trainer, "warmup_steps", 0)),
    }


def _manifest(
    config: Dict[str, Any],
    seed: int,
    tag: str,
    trainer: CentralSACTrainer,
    scene_hashes: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "manifest_schema_version": 3,
        "config": str(FORMAL_CONFIG_PATH.relative_to(ROOT)),
        "config_hash": _stable_hash(config),
        "algorithm": str(config["algorithm"]),
        "critic_mode": str(config["critic_mode"]),
        "trainer_profile": str(config.get("trainer_profile", "")),
        "action_mode": str(config["action"]["mode"]),
        "dynamics_profile": str(config["dynamics"]["profile"]),
        "a_max": float(config["action"]["a_max"]),
        "v_max": float(config["dynamics"]["v_max"]),
        "max_agents": int(config["training"]["max_agents"]),
        "batch_size": int(config["training"]["batch_size"]),
        "grad_clip_norm": float(config["training"]["grad_clip_norm"]),
        "seed": int(seed),
        "effective_config_hashes": scene_hashes,
        "implementation_hash": _implementation_hash(),
        "trainer_contract": _trainer_contract(trainer),
        "resume_contract": {
            "rng": ["torch_cpu", "torch_cuda", "python", "numpy", "replay_numpy"],
            "scene_index": "saved",
            "exact_env_state": False,
            "continuation_mode": "seeded_episode_boundary",
        },
    }


def _derive_roles(
    infos: List[Mapping[str, Any]],
    active_mask: np.ndarray,
    phase: str,
    max_agents: int,
) -> np.ndarray:
    roles = np.zeros(max_agents, dtype=np.uint8)
    if phase in {"post_capture", "pure_coverage", "pure_recovery"}:
        roles[active_mask] = 3
        return roles
    for idx, info in enumerate(infos):
        if idx >= max_agents or not bool(active_mask[idx]):
            continue
        meta = info.get("replay_metadata", {}) or {}
        task_label = str(meta.get("task_label", ""))
        support = bool(meta.get("support_candidate", False))
        if task_label == "capture":
            roles[idx] = 1
        elif support:
            roles[idx] = 2
        else:
            roles[idx] = 3
    return roles


def _screen(
    trainer: CentralSACTrainer,
    root_config: Dict[str, Any],
    seed: int,
    episodes: int,
    device: str,
    scenes: Tuple[str, ...] = SCENES,
    max_steps: int | None = None,
) -> Dict[str, Any]:
    if int(episodes) <= 0:
        return {}
    result: Dict[str, Any] = {}
    for scene_index, scene in enumerate(scenes):
        config = scene_config(root_config, scene)
        set_global_config(config)
        records = []
        for episode in range(int(episodes)):
            env = VorAdjEnv(config, seed=seed + 10000 + scene_index * 100 + episode)
            env.reset()
            adapter = env.action_adapter
            speed_limited_count = 0
            action_count = 0
            initial_positions = np.asarray([[float(p.x), float(p.y)] for p in env.pursuers if not p.deactivated], dtype=float)
            initial_evader_positions = np.asarray(
                [[float(e.x), float(e.y)] for e in env.evaders if not e.deactivated],
                dtype=float,
            )
            initial_min_distance = (
                float(np.min([np.linalg.norm(p - e) for p in initial_positions for e in initial_evader_positions]))
                if len(initial_evader_positions) and len(initial_positions)
                else None
            )
            active_positions = np.asarray([[float(p.x), float(p.y)] for p in env.pursuers if not p.deactivated], dtype=float)
            initial_ce_energy = (
                float(np.mean(env._voradj_coverage_potentials(active_positions, initial_evader_positions)))
                if len(active_positions) and env._ce_coverage_enabled()
                else None
            )
            min_distances: List[float] = []
            ce_energies: List[float] = []
            episode_action_norms: List[float] = []
            episode_speeds: List[float] = []
            discovery_seen = False
            discovery_step: Optional[int] = None
            horizon = int(config["env"]["episode_max_length"])
            if max_steps is not None and int(max_steps) > 0:
                horizon = min(horizon, int(max_steps))
            for step_idx in range(horizon):
                observations = list(env.get_observations())
                if any(item is None for item in observations):
                    break
                padded = _pad_local_obs_tree(
                    observations,
                    int(config["training"]["max_agents"]),
                    int(config["actor"]["max_pursuers"]),
                )
                actions, _ = _sample_actions(
                    trainer,
                    padded,
                    len(env.pursuers),
                    adapter,
                    deterministic=True,
                )
                outcome = env.step(actions.tolist(), [None] * len(env.evaders))
                after_positions = np.asarray([[float(p.x), float(p.y)] for p in env.pursuers if not p.deactivated], dtype=float)
                after_evader_positions = np.asarray(
                    [[float(e.x), float(e.y)] for e in env.evaders if not e.deactivated],
                    dtype=float,
                )
                if len(after_evader_positions) and len(after_positions):
                    min_distances.append(float(np.min([np.linalg.norm(p - e) for p in after_positions for e in after_evader_positions])))
                if len(after_positions) and env._ce_coverage_enabled():
                    ce_energies.append(float(np.mean(env._voradj_coverage_potentials(after_positions, after_evader_positions))))
                episode_action_norms.extend(float(np.linalg.norm(a)) for a in actions)
                episode_speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
                if not discovery_seen and any(
                    str(info.get("replay_metadata", {}).get("task_label", "")) == "capture"
                    for info in outcome.infos
                ):
                    discovery_seen = True
                    discovery_step = int(step_idx)
                for info in outcome.infos:
                    diagnostics = info.get("action_diagnostics", {})
                    speed_limited_count += int(bool(diagnostics.get("speed_limited", False)))
                    action_count += 1
                if all(outcome.dones):
                    break
            record = env.episode_record(task=scene)
            print(
                f"[eval] scene={scene} episode={episode + 1}/{int(episodes)} "
                f"length={int(record['length'])} captured={bool(record['captured'])} "
                f"collision={bool(record['collision_event'])}",
                flush=True,
            )
            records.append(
                {
                    "length": int(record["length"]),
                    "episode_success": bool(record["episode_success"]),
                    "captured": bool(record["captured"]),
                    "collision_event": bool(record["collision_event"]),
                    "coverage_strict_success": bool(record["coverage_strict_success"]),
                    "coverage_cv015_success": bool(record["coverage_cv015_success"]),
                    "speed_limited_rate": float(speed_limited_count / max(action_count, 1)),
                    "initial_min_distance": initial_min_distance,
                    "final_min_distance": min_distances[-1] if min_distances else None,
                    "min_min_distance": float(np.min(min_distances)) if min_distances else None,
                    "min_distance_auc": float(np.trapz(min_distances)) if len(min_distances) > 1 else 0.0,
                    "distance_progress": (
                        float(initial_min_distance - min_distances[-1])
                        if initial_min_distance is not None and min_distances
                        else None
                    ),
                    "discovery_step": discovery_step,
                    "detected": bool(discovery_seen),
                    "initial_ce_energy": initial_ce_energy,
                    "final_ce_energy": ce_energies[-1] if ce_energies else None,
                    "min_ce_energy": float(np.min(ce_energies)) if ce_energies else None,
                    "ce_energy_auc": float(np.trapz(ce_energies)) if len(ce_energies) > 1 else 0.0,
                    "ce_energy_progress": (
                        float(initial_ce_energy - ce_energies[-1])
                        if initial_ce_energy is not None and ce_energies
                        else None
                    ),
                    "action_norm_mean": float(np.mean(episode_action_norms)) if episode_action_norms else 0.0,
                    "action_norm_max": float(np.max(episode_action_norms)) if episode_action_norms else 0.0,
                    "speed_mean": float(np.mean(episode_speeds)) if episode_speeds else 0.0,
                    "speed_max": float(np.max(episode_speeds)) if episode_speeds else 0.0,
                }
            )
        result[scene] = {
            "episodes": len(records),
            "success_rate": float(np.mean([bool(item["episode_success"]) for item in records])) if records else 0.0,
            "capture_rate": float(np.mean([bool(item["captured"]) for item in records])) if records else 0.0,
            "collision_rate": float(np.mean([bool(item["collision_event"]) for item in records])) if records else 0.0,
            "speed_limited_rate": float(np.mean([item["speed_limited_rate"] for item in records])) if records else 0.0,
            "records": records,
        }
    return result


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _metrics_record(
    step: int,
    update_count: int,
    window_updates: List[Dict[str, Any]],
    replay: JointReplayBuffer,
    sampling_stats: Dict[str, Any],
    action_norms: Sequence[float],
    speeds: Sequence[float],
    terminated_count: int,
    truncated_count: int,
    collision_count: int,
    scene_counts: Mapping[str, int],
    origin_counts: Mapping[str, int],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "step": int(step),
        "update_count": int(update_count),
        "replay_size": len(replay),
        "focal_index_sizes": replay.focal_index_sizes,
        "sampling_stats": dict(sampling_stats or {}),
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
        "action_norm_std": float(np.std(action_norms)) if action_norms else 0.0,
        "action_norm_p5": _percentile(action_norms, 5),
        "action_norm_p50": _percentile(action_norms, 50),
        "action_norm_p95": _percentile(action_norms, 95),
        "action_norm_max": float(np.max(action_norms)) if action_norms else 0.0,
        "near_zero_action_rate": float(np.mean([float(v) <= 0.02 for v in action_norms])) if action_norms else 0.0,
        "speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "speed_max": float(np.max(speeds)) if speeds else 0.0,
        "terminated_count": int(terminated_count),
        "truncated_count": int(truncated_count),
        "collision_count": int(collision_count),
        "scene_counts": {str(k): int(v) for k, v in scene_counts.items()},
        "origin_counts": {str(k): int(v) for k, v in origin_counts.items()},
    }
    if window_updates:
        first = window_updates[0]
        scalar_keys = [k for k, value in first.items() if isinstance(value, (int, float))]
        for key in scalar_keys:
            record[f"mean_{key}"] = float(np.mean([item[key] for item in window_updates]))
        last = window_updates[-1]
        for key in ("q1_summary", "q2_summary", "target_q_summary", "td_error_summary"):
            if key in last:
                record[key] = dict(last[key])
    return record


def _append_metrics_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _verify_resume_steps(trainer: CentralSACTrainer, replay: JointReplayBuffer) -> None:
    trainer_step = (getattr(trainer, "resume_runtime_state", {}) or {}).get("transition_count")
    replay_step = (getattr(replay, "runtime_state", {}) or {}).get("transition_count")
    if trainer_step is not None and replay_step is not None and int(trainer_step) != int(replay_step):
        raise ValueError(
            f"checkpoint/replay step mismatch: trainer={trainer_step}, replay={replay_step}; refusing unsafe resume"
        )


def _reset_pure_recovery(
    config: Dict[str, Any],
    recovery_pool: Deque[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], str, Optional[List[List[float]]], Optional[List[bool]]]:
    recovery_cfg = config.get("recovery", {}) or {}
    capture_ratio = float(recovery_cfg.get("captured_state_ratio", 0.75))
    map_random_ratio = float(recovery_cfg.get("map_random_ratio_within_non_capture", 0.5))
    initial_positions: Optional[List[List[float]]] = None
    initial_active: Optional[List[bool]] = None
    origin = "inner_cluster"
    if recovery_pool and rng.random() < capture_ratio:
        snapshot = recovery_pool[int(rng.integers(0, len(recovery_pool)))]
        initial_positions = snapshot.get("positions")
        initial_active = snapshot.get("active_mask")
        origin = "capture_snapshot"
    elif rng.random() < map_random_ratio:
        config["env"]["pursuer_spawn_mode"] = "map_random"
        config["env"]["pursuer_spawn_min_sep"] = 15.0
        origin = "map_random"
    else:
        config["env"]["pursuer_spawn_mode"] = "inner_random_cluster"
        config["env"]["pursuer_spawn_min_sep"] = 7.0
        origin = "inner_cluster"
    return config, origin, initial_positions, initial_active


def _load_snapshot_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"snapshot dataset is empty: {path}")
    return rows


def _reset_from_snapshot(
    config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    scene: str,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], str, VorAdjEnv]:
    candidates = [item for item in dataset if str(item.get("scene", "")) == scene]
    if not candidates:
        raise ValueError(f"snapshot dataset has no snapshots for scene {scene}")
    snapshot = candidates[int(rng.integers(0, len(candidates)))]
    env, _ = restore_snapshot(snapshot, config, seed=int(rng.integers(0, 2**31 - 1)), mode="geometry_reset")
    origin = f"snapshot:{snapshot.get('phase', 'unknown')}"
    return config, origin, env


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root_config = resolve_formal_config(args.config)
    _set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = _make_trainer(root_config, device)
    max_agents = int(root_config["training"]["max_agents"])
    actor_max_pursuers = int(root_config["actor"]["max_pursuers"])
    scenes: Tuple[str, ...] = tuple(item.strip() for item in args.scenes.split(",") if item.strip())
    unknown_scenes = set(scenes).difference(SCENES)
    if unknown_scenes:
        raise ValueError(f"unknown scenes: {sorted(unknown_scenes)}")
    scene_hashes = {scene: _stable_hash(scene_config(root_config, scene)) for scene in scenes}
    manifest = _manifest(root_config, args.seed, args.tag, trainer, scene_hashes)
    recovery_cfg = root_config.get("recovery", {}) or {}
    recovery_pool: Deque[Dict[str, Any]] = deque(maxlen=int(recovery_cfg.get("capture_state_pool_capacity", 1000)))
    rng = np.random.default_rng(args.seed)
    snapshot_dataset_path = args.snapshot_dataset or (root_config.get("teacher_snapshot_restore", {}) or {}).get("dataset", "")
    snapshot_dataset = _load_snapshot_dataset(snapshot_dataset_path) if snapshot_dataset_path else None
    artifact_dir = ARTIFACT_ROOT / args.tag
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.resume_checkpoint) != bool(args.resume_replay):
        raise ValueError("--resume-checkpoint and --resume-replay must be supplied together")
    resume_info = None
    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint, manifest)
        replay = JointReplayBuffer.load(args.resume_replay, manifest)
        _verify_resume_steps(trainer, replay)
        runtime_state = dict(getattr(trainer, "resume_runtime_state", {}) or {})
        runtime_state.update(getattr(replay, "runtime_state", {}) or {})
        transition_count = int(args.resume_step if args.resume_step >= 0 else runtime_state.get("transition_count", len(replay)))
        scene_index = int(runtime_state.get("next_scene_index", transition_count))
        saved_pool = runtime_state.get("recovery_pool", [])
        for snapshot in saved_pool:
            recovery_pool.append(snapshot)
        resume_info = {
            "checkpoint": str(args.resume_checkpoint),
            "replay": str(args.resume_replay),
            "step": transition_count,
            "next_scene_index": scene_index,
        }
    else:
        replay = JointReplayBuffer(
            capacity=int(root_config["replay"]["capacity_joint"]),
            max_agents=max_agents,
            seed=args.seed,
        )
        transition_count = 0
        scene_index = 0

    quotas = dict(root_config["training"]["focal_quota"])
    sampler = FocalReplaySampler(
        quotas,
        max_focal_items_per_joint_transition=int(root_config["training"]["max_focal_items_per_joint_transition"]),
        fallback_matrix=root_config["training"].get("fallback_matrix", None),
        seed=args.seed,
    )
    total_steps = int(args.total_steps if args.total_steps is not None else root_config["training"]["total_env_steps"])
    warmup = int(root_config["training"]["warmup_joint_transitions"])
    update_every = int(root_config["training"]["update_every_env_steps"])
    batch_size = int(root_config["training"]["batch_size"])
    checkpoint_interval = int(root_config["training"].get("checkpoint_interval_env_steps", 25000))
    metrics_flush_interval = int(root_config["training"].get("metrics_flush_interval_env_steps", 1000))
    diagnostic_eval_interval = int(root_config["training"].get("diagnostic_eval_interval_env_steps", 25000))
    diagnostic_rollout_cap = int((root_config.get("evaluation", {}) or {}).get("diagnostic_rollout_cap", 400))
    warmup_action_mode = str((root_config.get("warmup_action_policy", {}) or {}).get("mode", "actor_prior")).strip().lower()
    if args.warmup_action_mode:
        warmup_action_mode = args.warmup_action_mode
    if warmup_action_mode not in {"actor_prior", "uniform_disk"}:
        raise ValueError("warmup_action_policy.mode must be actor_prior or uniform_disk")

    action_norms: List[float] = []
    speeds: List[float] = []
    window_action_norms: List[float] = []
    window_speeds: List[float] = []
    speed_limited_count = 0
    action_sample_count = 0
    terminated_count = 0
    truncated_count = 0
    collision_count = 0
    window_terminated_count = 0
    window_truncated_count = 0
    window_collision_count = 0
    transition_attempt_count = 0
    updates: List[Dict[str, float]] = []
    window_updates: List[Dict[str, float]] = []
    metrics_history: List[Dict[str, Any]] = []
    sampling_stats_accum: Dict[str, Any] = {}
    scene_counts: Dict[str, int] = {}
    origin_counts: Dict[str, int] = {}
    env = None
    observations: List[Optional[Dict[str, np.ndarray]]] = []

    while transition_count < total_steps:
        scene = scenes[scene_index % len(scenes)]
        scene_index += 1
        config = scene_config(root_config, scene)
        origin = "map_random"
        initial_positions = None
        initial_active = None
        if snapshot_dataset is not None:
            config, origin, env = _reset_from_snapshot(config, snapshot_dataset, scene, rng)
            adapter = env.action_adapter
            observations = list(env.get_observations())
            while transition_count < total_steps:
                if any(item is None for item in observations):
                    break
                before_labels = list(getattr(env, "last_task_labels", []))
                before_active_target = bool(any(not e.deactivated for e in env.evaders))
                before_coverage_success = bool(
                    getattr(env, "post_capture_coverage_success", False)
                    or getattr(env, "coverage_geometric_success", False)
                )
                before_active = _pad_vector([not p.deactivated for p in env.pursuers], max_agents, bool)
                before_obs_batch = _stack_with_batch(observations, max_agents, actor_max_pursuers)
                before_obs_padded = {key: value[0] for key, value in before_obs_batch.items()}
                before_global = build_central_global_obs(env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=9)
                actions, validation_rate = _sample_actions(
                    trainer,
                    before_obs_padded,
                    len(env.pursuers),
                    adapter,
                    deterministic=False,
                    uniform_disk=warmup_action_mode == "uniform_disk" and transition_count < warmup,
                    rng=rng,
                    a_max=float(root_config["action"]["a_max"]),
                )
                action_norms.extend(float(np.linalg.norm(action)) for action in actions)
                window_action_norms.extend(float(np.linalg.norm(action)) for action in actions)
                try:
                    outcome = env.step(actions.tolist(), [None] * len(env.evaders))
                except ActionContractError as exc:
                    raise RuntimeError("formal CTDE action contract violated") from exc
                transition_attempt_count += 1
                next_observations = list(outcome.observations)
                speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
                window_speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
                for info in outcome.infos:
                    diagnostics = info.get("action_diagnostics", {})
                    speed_limited_count += int(bool(diagnostics.get("speed_limited", False)))
                    action_sample_count += 1
                terminated, truncated = _split_termination_flags(outcome.dones, outcome.infos)
                terminated_count += int(any(terminated))
                truncated_count += int(any(truncated))
                collision_count += int(any(info.get("state") == "deactivated after collision" for info in outcome.infos))
                window_terminated_count += int(any(terminated))
                window_truncated_count += int(any(truncated))
                window_collision_count += int(any(info.get("state") == "deactivated after collision" for info in outcome.infos))
                scene_counts[scene] = scene_counts.get(scene, 0) + 1
                origin_counts[origin] = origin_counts.get(origin, 0) + 1
                phase = str(outcome.infos[0].get("replay_metadata", {}).get("phase", "pre_capture"))
                roles = _derive_roles(outcome.infos, before_active, phase, max_agents)
                event_ids: List[str] = []
                after_labels = [str(info.get("replay_metadata", {}).get("next_task_label", "")) for info in outcome.infos]
                if any(
                    before not in {"", "capture", "inactive"} and after == "capture"
                    for before, after in zip(before_labels, after_labels)
                ):
                    event_ids.append("discovery")
                if getattr(env, "last_capture_events", []):
                    event_ids.append("capture")
                collision_states = {"collision", "deactivated after collision", "evader collision", "zone breach"}
                if any(str(info.get("state", "")) in collision_states for info in outcome.infos):
                    event_ids.append("collision")
                if (
                    (getattr(env, "post_capture_coverage_success", False)
                     or getattr(env, "coverage_geometric_success", False))
                    and not before_coverage_success
                ):
                    event_ids.append("ce_success")
                metadata = {
                    "phase": phase,
                    "scene": scene,
                    "origin": origin,
                    "regime": "active_target" if phase == "pre_capture" else "coverage_only",
                    "coverage_only": phase != "pre_capture",
                    "active_target": phase == "pre_capture",
                    "event_ids": event_ids,
                    "task_label": str(outcome.infos[0].get("replay_metadata", {}).get("task_label", "")),
                    "recovery_reset_source": origin,
                }
                next_obs_batch = _stack_with_batch(next_observations, max_agents, actor_max_pursuers)
                next_obs_padded = {key: value[0] for key, value in next_obs_batch.items()}
                next_global = build_central_global_obs(env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=9)
                replay.add(
                    local_obs=before_obs_padded,
                    next_local_obs=next_obs_padded,
                    global_state=before_global,
                    next_global_state=next_global,
                    actions=_pad_actions(actions, max_agents),
                    rewards=_pad_vector(outcome.rewards, max_agents, np.float32),
                    active_mask=before_active,
                    terminated=_pad_vector(terminated, max_agents, bool),
                    truncated=_pad_vector(truncated, max_agents, bool),
                    metadata=metadata,
                    agent_role_id=roles,
                )
                transition_count += 1
                observations = next_observations
                freeze_steps = int(getattr(trainer, "freeze_encoder_env_steps", 0) or 0)
                if freeze_steps > 0 and transition_count >= freeze_steps:
                    for parameter in trainer.actor.encoder.parameters():
                        parameter.requires_grad_(True)
                    trainer.freeze_encoder_env_steps = 0
                if (
                    len(replay) >= batch_size
                    and transition_count >= warmup
                    and transition_count % update_every == 0
                ):
                    batch = replay.sample(batch_size, device=device, sampler=sampler)
                    metric = trainer.update(batch)
                    updates.append(metric)
                    window_updates.append(metric)
                    sampling_stats_accum = dict(batch.get("sampling_stats", {}))
                if transition_count % metrics_flush_interval == 0:
                    record = _metrics_record(
                        transition_count,
                        len(updates),
                        window_updates,
                        replay,
                        sampling_stats_accum,
                        window_action_norms,
                        window_speeds,
                        window_terminated_count,
                        window_truncated_count,
                        window_collision_count,
                        scene_counts,
                        origin_counts,
                    )
                    metrics_history.append(record)
                    _append_metrics_jsonl(artifact_dir / "metrics.jsonl", record)
                    window_updates = []
                    window_action_norms = []
                    window_speeds = []
                    window_terminated_count = 0
                    window_truncated_count = 0
                    window_collision_count = 0
                if transition_count == 1 or transition_count % checkpoint_interval == 0:
                    diagnostic_eval = (
                        _screen(
                            trainer,
                            root_config,
                            args.seed,
                            episodes=int(args.diagnostic_eval_episodes),
                            device=device,
                            scenes=scenes,
                            max_steps=diagnostic_rollout_cap,
                        )
                        if transition_count % diagnostic_eval_interval == 0
                        else {}
                    )
                    runtime_state = {
                        "transition_count": int(transition_count),
                        "update_count": int(len(updates)),
                        "next_scene_index": int(scene_index),
                        "current_scene": str(scene),
                        "recovery_pool": list(recovery_pool),
                        "metrics_history": list(metrics_history),
                    }
                    _save_checkpoint_bundle(
                        artifact_dir,
                        transition_count,
                        root_config,
                        manifest,
                        trainer,
                        replay,
                        runtime_state,
                        metrics_history,
                        diagnostic_eval,
                    )
                if all(outcome.dones) or any(item is None for item in next_observations):
                    break
            if snapshot_dataset is not None:
                snapshot = getattr(env, "capture_snapshot", None)
                if isinstance(snapshot, dict) and scene in {"capture", "mixed_crms"}:
                    recovery_pool.append(
                        {
                            "step": int(snapshot.get("step", 0)),
                            "positions": snapshot.get("positions", []),
                            "active_mask": snapshot.get("active_mask", []),
                        }
                    )
                continue
        elif scene == "pure_ce":
            config, origin, initial_positions, initial_active = _reset_pure_recovery(
                config,
                recovery_pool,
                rng,
            )
        set_global_config(config)
        env = VorAdjEnv(config, seed=args.seed + scene_index)
        env.reset(
            initial_pursuer_positions=initial_positions,
            initial_pursuer_active=initial_active,
        )
        adapter = env.action_adapter
        observations = list(env.get_observations())
        while transition_count < total_steps:
            if any(item is None for item in observations):
                break
            before_labels = list(getattr(env, "last_task_labels", []))
            before_active_target = bool(any(not e.deactivated for e in env.evaders))
            before_coverage_success = bool(
                getattr(env, "post_capture_coverage_success", False)
                or getattr(env, "coverage_geometric_success", False)
            )
            before_active = _pad_vector([not p.deactivated for p in env.pursuers], max_agents, bool)
            before_obs_batch = _stack_with_batch(observations, max_agents, actor_max_pursuers)
            before_obs_padded = {key: value[0] for key, value in before_obs_batch.items()}
            before_global = build_central_global_obs(env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=9)
            actions, validation_rate = _sample_actions(
                trainer,
                before_obs_padded,
                len(env.pursuers),
                adapter,
                deterministic=False,
                uniform_disk=warmup_action_mode == "uniform_disk" and transition_count < warmup,
                rng=rng,
                a_max=float(root_config["action"]["a_max"]),
            )
            action_norms.extend(float(np.linalg.norm(action)) for action in actions)
            window_action_norms.extend(float(np.linalg.norm(action)) for action in actions)
            try:
                outcome = env.step(actions.tolist(), [None] * len(env.evaders))
            except ActionContractError as exc:
                raise RuntimeError("formal CTDE action contract violated") from exc
            transition_attempt_count += 1
            next_observations = list(outcome.observations)
            speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
            window_speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
            for info in outcome.infos:
                diagnostics = info.get("action_diagnostics", {})
                speed_limited_count += int(bool(diagnostics.get("speed_limited", False)))
                action_sample_count += 1
            terminated, truncated = _split_termination_flags(outcome.dones, outcome.infos)
            terminated_count += int(any(terminated))
            truncated_count += int(any(truncated))
            collision_count += int(any(info.get("state") == "deactivated after collision" for info in outcome.infos))
            window_terminated_count += int(any(terminated))
            window_truncated_count += int(any(truncated))
            window_collision_count += int(any(info.get("state") == "deactivated after collision" for info in outcome.infos))
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            phase = str(outcome.infos[0].get("replay_metadata", {}).get("phase", "pre_capture"))
            roles = _derive_roles(outcome.infos, before_active, phase, max_agents)
            event_ids: List[str] = []
            after_labels = [str(info.get("replay_metadata", {}).get("next_task_label", "")) for info in outcome.infos]
            if any(
                before not in {"", "capture", "inactive"} and after == "capture"
                for before, after in zip(before_labels, after_labels)
            ):
                event_ids.append("discovery")
            if getattr(env, "last_capture_events", []):
                event_ids.append("capture")
            collision_states = {"collision", "deactivated after collision", "evader collision", "zone breach"}
            if any(str(info.get("state", "")) in collision_states for info in outcome.infos):
                event_ids.append("collision")
            if (
                (getattr(env, "post_capture_coverage_success", False)
                 or getattr(env, "coverage_geometric_success", False))
                and not before_coverage_success
            ):
                event_ids.append("ce_success")
            metadata = {
                "phase": phase,
                "scene": scene,
                "origin": origin,
                "regime": "active_target" if phase == "pre_capture" else "coverage_only",
                "coverage_only": phase != "pre_capture",
                "active_target": phase == "pre_capture",
                "event_ids": event_ids,
                "task_label": str(outcome.infos[0].get("replay_metadata", {}).get("task_label", "")),
                "recovery_reset_source": origin,
            }
            next_obs_batch = _stack_with_batch(next_observations, max_agents, actor_max_pursuers)
            next_obs_padded = {key: value[0] for key, value in next_obs_batch.items()}
            next_global = build_central_global_obs(env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=9)
            replay.add(
                local_obs=before_obs_padded,
                next_local_obs=next_obs_padded,
                global_state=before_global,
                next_global_state=next_global,
                actions=_pad_actions(actions, max_agents),
                rewards=_pad_vector(outcome.rewards, max_agents, np.float32),
                active_mask=before_active,
                terminated=_pad_vector(terminated, max_agents, bool),
                truncated=_pad_vector(truncated, max_agents, bool),
                metadata=metadata,
                agent_role_id=roles,
            )
            transition_count += 1
            observations = next_observations
            freeze_steps = int(getattr(trainer, "freeze_encoder_env_steps", 0) or 0)
            if freeze_steps > 0 and transition_count >= freeze_steps:
                for parameter in trainer.actor.encoder.parameters():
                    parameter.requires_grad_(True)
                trainer.freeze_encoder_env_steps = 0
            if (
                len(replay) >= batch_size
                and transition_count >= warmup
                and transition_count % update_every == 0
            ):
                batch = replay.sample(batch_size, device=device, sampler=sampler)
                metric = trainer.update(batch)
                updates.append(metric)
                window_updates.append(metric)
                sampling_stats_accum = dict(batch.get("sampling_stats", {}))
            if transition_count % metrics_flush_interval == 0:
                record = _metrics_record(
                    transition_count,
                    len(updates),
                    window_updates,
                    replay,
                    sampling_stats_accum,
                    window_action_norms,
                    window_speeds,
                    window_terminated_count,
                    window_truncated_count,
                    window_collision_count,
                    scene_counts,
                    origin_counts,
                )
                metrics_history.append(record)
                _append_metrics_jsonl(artifact_dir / "metrics.jsonl", record)
                window_updates = []
                window_action_norms = []
                window_speeds = []
                window_terminated_count = 0
                window_truncated_count = 0
                window_collision_count = 0
            if transition_count == 1 or transition_count % checkpoint_interval == 0:
                diagnostic_eval = (
                    _screen(
                        trainer,
                        root_config,
                        args.seed,
                        episodes=int(args.diagnostic_eval_episodes),
                        device=device,
                        scenes=scenes,
                        max_steps=diagnostic_rollout_cap,
                    )
                    if transition_count % diagnostic_eval_interval == 0
                    else {}
                )
                runtime_state = {
                    "transition_count": int(transition_count),
                    "update_count": int(len(updates)),
                    "next_scene_index": int(scene_index),
                    "current_scene": str(scene),
                    "recovery_pool": list(recovery_pool),
                    "metrics_history": list(metrics_history),
                }
                _save_checkpoint_bundle(
                    artifact_dir,
                    transition_count,
                    root_config,
                    manifest,
                    trainer,
                    replay,
                    runtime_state,
                    metrics_history,
                    diagnostic_eval,
                )
            if all(outcome.dones) or any(item is None for item in next_observations):
                break
        if env is not None:
            snapshot = getattr(env, "capture_snapshot", None)
            if isinstance(snapshot, dict) and scene in {"capture", "mixed_crms"}:
                recovery_pool.append(
                    {
                        "step": int(snapshot.get("step", 0)),
                        "positions": snapshot.get("positions", []),
                        "active_mask": snapshot.get("active_mask", []),
                    }
                )

    effective_path = artifact_dir / "effective_config.yaml"
    effective_path.write_text(yaml.safe_dump(root_config, sort_keys=False), encoding="utf-8")
    scene_config_dir = artifact_dir / "scene_configs"
    scene_config_dir.mkdir(exist_ok=True)
    for scene in scenes:
        (scene_config_dir / f"{scene}.yaml").write_text(
            yaml.safe_dump(scene_config(root_config, scene), sort_keys=False),
            encoding="utf-8",
        )
    checkpoint = artifact_dir / f"{args.tag}_step{transition_count}.pt"
    runtime_state = {
        "transition_count": int(transition_count),
        "update_count": int(len(updates)),
        "next_scene_index": int(scene_index),
        "current_scene": str(scene),
        "recovery_pool": list(recovery_pool),
        "metrics_history": list(metrics_history),
    }
    trainer.save_checkpoint(checkpoint, manifest, runtime_state=runtime_state)
    replay_path = artifact_dir / f"{args.tag}_replay.pkl"
    replay.save(replay_path, manifest, runtime_state=runtime_state)

    screening = (
        _screen(
            trainer,
            root_config,
            args.seed,
            episodes=int(args.screen_episodes),
            device=device,
            scenes=scenes,
        )
        if int(args.screen_episodes) > 0
        else {}
    )
    final_diagnostic_eval = (
        _screen(
            trainer,
            root_config,
            args.seed,
            episodes=int(args.diagnostic_eval_episodes),
            device=device,
            scenes=scenes,
            max_steps=diagnostic_rollout_cap,
        )
        if int(args.diagnostic_eval_episodes) > 0
        else {}
    )
    final_bundle = _save_checkpoint_bundle(
        artifact_dir,
        transition_count,
        root_config,
        manifest,
        trainer,
        replay,
        runtime_state,
        metrics_history,
        final_diagnostic_eval,
    )
    report = {
        "schema_version": 1,
        "kind": "continuous_ctde_formal",
        "algorithm": root_config["algorithm"],
        "critic_mode": root_config["critic_mode"],
        "effective_config_path": str(effective_path),
        "scene_config_hashes": scene_hashes,
        "seed": args.seed,
        "requested_steps": total_steps,
        "transition_count": transition_count,
        "replay_size": len(replay),
        "focal_index_sizes": replay.focal_index_sizes,
        "recovery_pool_size": len(recovery_pool),
        "updates": len(updates),
        "update_metrics_tail": updates[-10:],
        "metrics_history_tail": metrics_history[-20:],
        "metrics_jsonl_path": str(artifact_dir / "metrics.jsonl"),
        "checkpoint_bundles": sorted(
            str(path)
            for path in (artifact_dir / "checkpoints").glob("step_*")
            if path.is_dir()
        ) if (artifact_dir / "checkpoints").exists() else [],
        "final_checkpoint_bundle": str(final_bundle),
        "diagnostic_eval_400": final_diagnostic_eval,
        "all_finite": bool(all(item["finite"] == 1.0 for item in updates)) if updates else True,
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
        "action_norm_max": float(np.max(action_norms)) if action_norms else 0.0,
        "speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "speed_max": float(np.max(speeds)) if speeds else 0.0,
        "speed_limit_rate": float(speed_limited_count / max(action_sample_count, 1)),
        "terminated_transition_count": terminated_count,
        "truncated_transition_count": truncated_count,
        "collision_transition_count": collision_count,
        "transition_attempt_count": transition_attempt_count,
        "sampling_stats": sampling_stats_accum,
        "screening": screening,
        "manifest": manifest,
        "resumed_from": resume_info,
        "replay_path": str(replay_path),
        "checkpoint": str(checkpoint),
        "diagnostic_rollout_cap": diagnostic_rollout_cap,
    }
    report_path = artifact_dir / f"{args.tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal CTDE MASAC training runner")
    parser.add_argument("--config", default=str(FORMAL_CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=2026080601)
    parser.add_argument("--device", default="")
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--snapshot-dataset", default="")
    parser.add_argument("--warmup-action-mode", choices=("actor_prior", "uniform_disk"), default="")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--screen-episodes", type=int, default=2)
    parser.add_argument("--diagnostic-eval-episodes", type=int, default=4)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-replay", default="")
    parser.add_argument("--resume-step", type=int, default=-1)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["all_finite"] and report["replay_size"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
