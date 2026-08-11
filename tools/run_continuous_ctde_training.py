"""Single-YAML formal CTDE MASAC training runner.

The runner deliberately has no local-critic, smoke, or CLI algorithm override
path.  It resolves exactly one formal YAML, deep-merges ``tasks.<scene>``,
stores one joint transition per environment step, and uses the manifest-bound
focal-item or standard all-active-agent optimizer unit.
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
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml

from cocap_voradj.dynamics.continuous_action import ActionContractError
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.models.continuous.central_attention_critic import CentralCriticConfig
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.models.continuous.box_actor import BoxActorConfig
from cocap_voradj.training.continuous.central_sac import CentralSACConfig, CentralSACTrainer
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.continuous.curriculum_snapshots import restore_snapshot
from cocap_voradj.training.continuous.formal_config import (
    AW_ACTION_MODE,
    BODY_ACTION_MODE,
    SCENES,
    resolve_formal_config,
    resolve_ladder_config,
    scene_config,
    training_mode_contract,
)
from cocap_voradj.training.continuous.joint_replay import (
    FOCAL_BUCKETS,
    FocalReplaySampler,
    JointReplayBuffer,
    UniformJointReplaySampler,
)
from cocap_voradj.training.trainer import load_config, set_global_config


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


def _link_or_copy_atomic(source: Path, destination: Path) -> None:
    """Expose a bundle file under the legacy standalone name without duplication."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.tmp_{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(source, tmp)
    except OSError:
        shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def _periodic_checkpoint_due(step: int, total_steps: int, interval: int) -> bool:
    """Return true for non-final periodic milestones; final save owns total_steps."""
    return int(step) < int(total_steps) and (
        int(step) == 1 or int(step) % int(interval) == 0
    )


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
    overwrite: bool = False,
    include_replay: bool = True,
) -> Path:
    bundle_dir = artifact_dir / "checkpoints" / f"step_{int(step):09d}"
    if bundle_dir.exists() and not overwrite:
        return bundle_dir
    tmp_dir = artifact_dir / ".tmp" / f"step_{int(step):09d}_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(tmp_dir / "trainer.pt", manifest, runtime_state=runtime_state)
    if include_replay:
        replay.save(tmp_dir / "replay.pkl", manifest, runtime_state=runtime_state)
    (tmp_dir / "runtime_state.pkl").write_bytes(pickle.dumps(runtime_state))
    (tmp_dir / "effective_config.yaml").write_text(yaml.safe_dump(root_config, sort_keys=False), encoding="utf-8")
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metrics_history),
        encoding="utf-8",
    )
    (tmp_dir / "diagnostic_eval.json").write_text(json.dumps(diagnostic_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_dir / "checkpoint_storage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "full_resume" if include_replay else "evaluation_model_only",
                "step": int(step),
                "contains_replay": bool(include_replay),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _replace_dir_atomic(tmp_dir, bundle_dir)
    return bundle_dir


def _save_rolling_resume_bundle(
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
    """Atomically keep exactly one full resume bundle beside model-only milestones."""
    final_dir = artifact_dir / "resume_latest"
    tmp_dir = artifact_dir / ".tmp" / f"resume_latest_{int(step):09d}_{os.getpid()}"
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
    (tmp_dir / "checkpoint_storage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "rolling_latest_full_resume",
                "step": int(step),
                "contains_replay": True,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _replace_dir_atomic(tmp_dir, final_dir)
    return final_dir


def _link_bundle_tree_atomic(source_dir: Path, destination_dir: Path) -> Path:
    """Expose a completed full bundle as resume_latest without duplicating large files."""
    tmp_dir = destination_dir.parent / ".tmp" / f"{destination_dir.name}_links_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        destination = tmp_dir / source.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    _replace_dir_atomic(tmp_dir, destination_dir)
    return destination_dir


def _runtime_state(
    *,
    transition_count: int,
    update_count: int,
    scene_index: int,
    current_scene: str,
    recovery_pool: Deque[Dict[str, Any]],
    metrics_history: List[Dict[str, Any]],
    runner_rng: np.random.Generator,
    focal_sampler: Optional[FocalReplaySampler],
    scene_counts: Mapping[str, int],
    origin_counts: Mapping[str, int],
    sampling_stats: Mapping[str, Any],
    all_finite: bool,
    update_metrics_tail: List[Dict[str, float]],
    training_mode: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the episode-boundary state needed for a faithful continuation."""
    state = {
        "runtime_state_schema_version": 2,
        "transition_count": int(transition_count),
        "update_count": int(update_count),
        "next_scene_index": int(scene_index),
        "current_scene": str(current_scene),
        "recovery_pool": list(recovery_pool),
        "metrics_history": list(metrics_history),
        "runner_rng_state": copy.deepcopy(runner_rng.bit_generator.state),
        "scene_counts": {str(key): int(value) for key, value in scene_counts.items()},
        "origin_counts": {str(key): int(value) for key, value in origin_counts.items()},
        "sampling_stats": copy.deepcopy(dict(sampling_stats)),
        "all_finite": bool(all_finite),
        "update_metrics_tail": copy.deepcopy(list(update_metrics_tail[-10:])),
        "training_mode": copy.deepcopy(dict(training_mode or {})),
    }
    if focal_sampler is not None:
        state["focal_sampler_state"] = focal_sampler.state_dict()
    return state



def _implementation_hash() -> str:
    paths = [
        Path(__file__),
        ROOT / "src/cocap_voradj/training/continuous/joint_replay.py",
        ROOT / "src/cocap_voradj/training/continuous/central_sac.py",
        ROOT / "src/cocap_voradj/training/continuous/central_schema.py",
        ROOT / "src/cocap_voradj/training/continuous/formal_config.py",
        ROOT / "src/cocap_voradj/models/continuous/central_attention_critic.py",
        ROOT / "src/cocap_voradj/models/continuous/radial_actor.py",
        ROOT / "src/cocap_voradj/models/continuous/box_actor.py",
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
    self_feature_dim: int = 9,
) -> Dict[str, np.ndarray]:
    """Pad per-agent local observations to fixed central/replay slots."""
    max_evaders = 8
    max_obstacles = 5
    token_count = 1 + actor_max_pursuers + max_evaders + max_obstacles
    self_dim = int(self_feature_dim)
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
    self_feature_dim: int = 9,
) -> Dict[str, np.ndarray]:
    tree = _pad_local_obs_tree(observations, max_agents, actor_max_pursuers, self_feature_dim)
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
        if hasattr(adapter, "w_max") and float(getattr(adapter, "w_max", 0.0)) > 0.0:
            raw_a = rng.uniform(-float(a_max), float(a_max), size=(active_count, 1))
            raw_w = rng.uniform(-float(adapter.w_max), float(adapter.w_max), size=(active_count, 1))
            raw = np.concatenate([raw_a, raw_w], axis=1).astype(np.float32)
        else:
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
    action_mode = str(config["action"]["mode"]).strip().lower()
    if action_mode in {AW_ACTION_MODE, "aw", "continuous_aw"}:
        actor = BoxActorConfig(
            hidden_dim=int(actor_cfg["hidden_dim"]),
            a_max=float(config["action"]["a_max"]),
            w_max=float(config["action"].get("w_max", float(np.pi / 6.0))),
            decision_dt=float(config["dynamics"]["decision_dt"]),
            log_std_min=float(actor_cfg.get("log_std_min", -5.0)),
            log_std_max=float(actor_cfg.get("log_std_max", 1.0)),
            dropout=float(actor_cfg.get("dropout", 0.0)),
        )
    else:
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
    configured_grad_clip = config["training"].get("grad_clip_norm")
    trainer_cfg = CentralSACConfig(
        hidden_dim=int(critic_cfg["hidden_dim"]),
        gamma=float(sac_cfg["gamma"]),
        tau=float(sac_cfg["tau"]),
        actor_lr=float(sac_cfg["actor_lr"]),
        critic_lr=float(sac_cfg["critic_lr"]),
        alpha_lr=float(sac_cfg["alpha_lr"]),
        alpha_init=float(sac_cfg["alpha_init"]),
        target_entropy=float(sac_cfg["target_entropy"]),
        grad_clip_norm=(
            None
            if configured_grad_clip is None
            else float(configured_grad_clip)
        ),
    )
    trainer = CentralSACTrainer(
        encoder_config=encoder,
        actor_config=actor,
        critic_config=critic,
        config=trainer_cfg,
        device=device,
        action_mode=action_mode,
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
        "actor_w_max": float(getattr(actor_cfg, "w_max", 0.0)),
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
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    source = Path(config_path).resolve() if config_path else FORMAL_CONFIG_PATH
    mode = training_mode_contract(config)
    configured_scenes = (config.get("training", {}) or {}).get("scene_cycle", SCENES)
    if isinstance(configured_scenes, str):
        configured_scenes = [
            item.strip() for item in configured_scenes.split(",") if item.strip()
        ]
    resume_rng = [
        "torch_cpu",
        "torch_cuda",
        "python",
        "numpy",
        "replay_numpy",
        "runner_numpy",
    ]
    if bool(mode["focal_training"]):
        resume_rng.append("focal_sampler_numpy")
    experiment_metadata = config.get("experiment_metadata", {}) or {}
    configured_grad_clip = config["training"].get("grad_clip_norm")
    return {
        "manifest_schema_version": 3,
        "config": str(source.relative_to(ROOT)),
        "config_hash": _stable_hash(config),
        "algorithm": str(config["algorithm"]),
        "critic_mode": str(config["critic_mode"]),
        "trainer_profile": str(config.get("trainer_profile", "")),
        "action_mode": str(config["action"]["mode"]),
        "dynamics_profile": str(config["dynamics"]["profile"]),
        "a_max": float(config["action"]["a_max"]),
        "w_max": float(config["action"].get("w_max", 0.0)),
        "v_max": float(config["dynamics"]["v_max"]),
        "max_agents": int(config["training"]["max_agents"]),
        "batch_size": int(config["training"]["batch_size"]),
        "grad_clip_norm": (
            None
            if configured_grad_clip is None
            else float(configured_grad_clip)
        ),
        "grad_clip": (
            "none" if configured_grad_clip is None else float(configured_grad_clip)
        ),
        "seed": int(seed),
        "optimizer_unit": str(mode["optimizer_unit"]),
        "replay_sampling": str(mode["replay_sampling"]),
        "focal_training": bool(mode["focal_training"]),
        "actor_q_implementation": str(
            experiment_metadata.get("actor_q_implementation", "retained_graph_v0")
        ),
        "initialization": str(
            experiment_metadata.get("training_start", "unspecified")
        ),
        "scene_cycle": [str(item) for item in configured_scenes],
        "effective_config_hashes": scene_hashes,
        "implementation_hash": _implementation_hash(),
        "trainer_contract": _trainer_contract(trainer),
        "resume_contract": {
            "rng": resume_rng,
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


def _evader_actions_for_env(env: VorAdjEnv, apf_agents: Optional[List[ApfAgent]] = None) -> List[Optional[int]]:
    """Return APF actions for moving evaders, or None (stationary) by default."""
    if not env.evaders:
        return []
    autonomous = bool(((env.config.get("evader", {}) or {}) or {}).get("autonomous", False))
    if not autonomous:
        return [None] * len(env.evaders)
    if apf_agents is None:
        apf_agents = [ApfAgent(e.a, e.w) for e in env.evaders]
    if hasattr(env, "configure_evader_apf_agents"):
        env.configure_evader_apf_agents(apf_agents)
    observations = list(env.get_evader_observations_for_apf())
    return [
        None if obs is None else int(apf_agents[idx].act(obs))
        for idx, obs in enumerate(observations)
    ]


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
            apf_agents = [ApfAgent(e.a, e.w) for e in env.evaders] if bool(((config.get("evader", {}) or {}) or {}).get("autonomous", False)) else None
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
                    int(config["actor"]["self_feature_dim"]),
                )
                actions, _ = _sample_actions(
                    trainer,
                    padded,
                    len(env.pursuers),
                    adapter,
                    deterministic=True,
                )
                outcome = env.step(actions.tolist(), _evader_actions_for_env(env, apf_agents))
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
            coverage_area_cv = float(record.get("coverage_strict_area_cv", float("inf")))
            coverage_cv_loose_threshold = float(
                record.get(
                    "coverage_cv_loose_area_cv_threshold",
                    (config.get("reward", {}) or {}).get("coverage_cv_loose_area_cv_threshold", 0.15),
                )
            )
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
                    "coverage_cv015_success": bool(coverage_area_cv <= 0.15),
                    "coverage_cv020_success": bool(coverage_area_cv <= 0.20),
                    "coverage_cv_loose_success": bool(record.get("coverage_cv_loose_success", False)),
                    "coverage_cv_loose_threshold": coverage_cv_loose_threshold,
                    "coverage_area_cv": coverage_area_cv,
                    "coverage_ce_center_rms": float(record.get("coverage_ce_center_rms", float("inf"))),
                    "coverage_ce_center_max": float(record.get("coverage_ce_center_max", float("inf"))),
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
        def mean_present(key: str) -> float | None:
            values = [
                float(item[key])
                for item in records
                if item.get(key) is not None and np.isfinite(float(item[key]))
            ]
            return float(np.mean(values)) if values else None

        result[scene] = {
            "episodes": len(records),
            "success_rate": float(np.mean([bool(item["episode_success"]) for item in records])) if records else 0.0,
            "capture_rate": float(np.mean([bool(item["captured"]) for item in records])) if records else 0.0,
            "collision_rate": float(np.mean([bool(item["collision_event"]) for item in records])) if records else 0.0,
            "coverage_strict_rate": float(np.mean([bool(item["coverage_strict_success"]) for item in records])) if records else 0.0,
            "coverage_cv015_rate": float(np.mean([bool(item["coverage_cv015_success"]) for item in records])) if records else 0.0,
            "coverage_cv020_rate": float(np.mean([bool(item["coverage_cv020_success"]) for item in records])) if records else 0.0,
            "coverage_cv_loose_rate": float(np.mean([bool(item["coverage_cv_loose_success"]) for item in records])) if records else 0.0,
            "coverage_cv_loose_threshold": (
                float(records[0]["coverage_cv_loose_threshold"]) if records else None
            ),
            "detected_rate": float(np.mean([bool(item["detected"]) for item in records])) if records else 0.0,
            "mean_discovery_step": mean_present("discovery_step"),
            "mean_episode_length": mean_present("length"),
            "mean_initial_min_distance": mean_present("initial_min_distance"),
            "mean_final_min_distance": mean_present("final_min_distance"),
            "mean_min_min_distance": mean_present("min_min_distance"),
            "mean_distance_progress": mean_present("distance_progress"),
            "mean_initial_ce_energy": mean_present("initial_ce_energy"),
            "mean_final_ce_energy": mean_present("final_ce_energy"),
            "mean_min_ce_energy": mean_present("min_ce_energy"),
            "mean_ce_energy_progress": mean_present("ce_energy_progress"),
            "mean_coverage_area_cv": mean_present("coverage_area_cv"),
            "mean_coverage_ce_center_rms": mean_present("coverage_ce_center_rms"),
            "mean_coverage_ce_center_max": mean_present("coverage_ce_center_max"),
            "mean_action_norm": mean_present("action_norm_mean"),
            "mean_speed": mean_present("speed_mean"),
            "speed_limited_rate": float(np.mean([item["speed_limited_rate"] for item in records])) if records else 0.0,
            "records": records,
        }
    return result


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _pursuit_step_geometry(env: Any, actions: Optional[np.ndarray] = None) -> Dict[str, float] | None:
    """Per-step pursuit diagnostics for capture scenes (uses env state after step)."""
    active_p = [p for p in env.pursuers if not p.deactivated]
    active_e = [e for e in env.evaders if not e.deactivated]
    if not active_p or not active_e:
        return None
    epos = np.asarray([active_e[0].x, active_e[0].y], dtype=float)
    evel = np.asarray(active_e[0].velocity, dtype=float)
    dists = []
    closing = []
    bearing = []
    turn_ok = 0
    turn_n = 0
    for p in active_p:
        ppos = np.asarray([p.x, p.y], dtype=float)
        pvel = np.asarray(p.velocity, dtype=float)
        rel = epos - ppos
        d = float(np.linalg.norm(rel))
        dists.append(d)
        if d > 1e-6:
            unit = rel / d
            closing.append(float(np.dot(pvel - evel, unit)))
            phi = float(np.arctan2(rel[1], rel[0]))
            be = abs(_wrap_angle(phi - float(p.theta)))
            bearing.append(be)
            w = None
            if actions is not None:
                idx = env.pursuers.index(p)
                if idx < len(actions):
                    w = float(actions[idx][1])
            if w is not None and be >= 0.05 and abs(w) >= 0.01:
                turn_n += 1
                if w * _wrap_angle(phi - float(p.theta)) > 0:
                    turn_ok += 1
    if not dists:
        return None
    d_sorted = sorted(dists)
    d = [float(d_sorted[i]) if i < len(d_sorted) else 99.0 for i in range(4)]
    return {
        "d1": d[0], "d2": d[1], "d3": d[2], "d4": d[3],
        "closing": float(np.mean(closing)) if closing else 0.0,
        "fraction_closing": float(np.mean([c > 0 for c in closing])) if closing else 0.0,
        "abs_bearing_error": float(np.mean(bearing)) if bearing else 0.0,
        "turn_direction_correct_rate": float(turn_ok / turn_n) if turn_n else 0.0,
        "num_within_8": int(np.sum([x < 8.0 for x in dists])),
        "num_within_10_5": int(np.sum([x < 10.5 for x in dists])),
        "num_within_12": int(np.sum([x < 12.0 for x in dists])),
        "num_within_20": int(np.sum([x < 20.0 for x in dists])),
        "num_in_ring_8_10_5": int(np.sum([8.0 <= x < 10.5 for x in dists])),
    }


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
    action_a: Sequence[float] = (),
    action_w: Sequence[float] = (),
    geometry: Sequence[Dict[str, Any]] = (),
    window_wall_time_s: float = 0.0,
    window_env_steps: int = 0,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "step": int(step),
        "update_count": int(update_count),
        "replay_size": len(replay),
        "focal_index_sizes": replay.focal_index_sizes,
        "sampling_stats": dict(sampling_stats or {}),
        "window_wall_time_s": float(window_wall_time_s),
        "window_env_steps": int(window_env_steps),
        "env_steps_per_second": float(
            window_env_steps / max(window_wall_time_s, 1e-12)
        ),
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
        "action_norm_std": float(np.std(action_norms)) if action_norms else 0.0,
        "action_norm_p5": _percentile(action_norms, 5),
        "action_norm_p50": _percentile(action_norms, 50),
        "action_norm_p95": _percentile(action_norms, 95),
        "action_norm_max": float(np.max(action_norms)) if action_norms else 0.0,
        "near_zero_action_rate": float(np.mean([float(v) <= 0.02 for v in action_norms])) if action_norms else 0.0,
        "speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "speed_p50": _percentile(speeds, 50),
        "speed_p95": _percentile(speeds, 95),
        "speed_max": float(np.max(speeds)) if speeds else 0.0,
        "fraction_speed_lt_0_2": float(np.mean([float(v) < 0.2 for v in speeds])) if speeds else 0.0,
        "fraction_speed_gt_2_5": float(np.mean([float(v) > 2.5 for v in speeds])) if speeds else 0.0,
        # ---- E0 / action-component diagnostics (2026-08-08) ----
        "a_mean": float(np.mean(action_a)) if action_a else 0.0,
        "a_abs_mean": float(np.mean(np.abs(action_a))) if action_a else 0.0,
        "a_std": float(np.std(action_a)) if action_a else 0.0,
        "a_p5": _percentile(action_a, 5),
        "a_p50": _percentile(action_a, 50),
        "a_p95": _percentile(action_a, 95),
        "fraction_abs_a_lt_0_05": float(np.mean([abs(float(v)) < 0.05 for v in action_a])) if action_a else 0.0,
        "fraction_abs_a_gt_0_35": float(np.mean([abs(float(v)) > 0.35 for v in action_a])) if action_a else 0.0,
        "fraction_a_positive": float(np.mean([float(v) > 0 for v in action_a])) if action_a else 0.0,
        "fraction_a_negative": float(np.mean([float(v) < 0 for v in action_a])) if action_a else 0.0,
        "omega_mean": float(np.mean(action_w)) if action_w else 0.0,
        "omega_abs_mean": float(np.mean(np.abs(action_w))) if action_w else 0.0,
        "omega_std": float(np.std(action_w)) if action_w else 0.0,
        "omega_p5": _percentile(action_w, 5),
        "omega_p50": _percentile(action_w, 50),
        "omega_p95": _percentile(action_w, 95),
        "fraction_abs_omega_lt_0_05": float(np.mean([abs(float(v)) < 0.05 for v in action_w])) if action_w else 0.0,
        "fraction_abs_omega_gt_0_45": float(np.mean([abs(float(v)) > 0.45 for v in action_w])) if action_w else 0.0,
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
        e0_keys = (
            "log_alpha", "log_prob_mean", "log_prob_std", "log_prob_p5", "log_prob_p50", "log_prob_p95",
            "entropy_proxy_mean", "entropy_residual_mean", "entropy_residual_p5", "entropy_residual_p50", "entropy_residual_p95",
            "log_std_a_mean", "log_std_omega_mean", "std_a_mean", "std_omega_mean",
        )
        for key in e0_keys:
            values = [item[key] for item in window_updates if isinstance(item.get(key), (int, float))]
            if values:
                record[f"p5_{key}"] = _percentile(values, 5)
                record[f"p50_{key}"] = _percentile(values, 50)
                record[f"p95_{key}"] = _percentile(values, 95)
        lp_means = [item.get("entropy_proxy_mean") for item in window_updates if isinstance(item.get("entropy_proxy_mean"), (int, float))]
        if lp_means:
            # normalized entropy = physical entropy + log(a_max) + log(w_max) (BoxActor affine scale)
            record["normalized_entropy_mean"] = float(np.mean(lp_means) + np.log(0.4) + np.log(np.pi / 6.0))
    if geometry:
        record.update({
            "d1_mean": float(np.mean([g["d1"] for g in geometry])),
            "d2_mean": float(np.mean([g["d2"] for g in geometry])),
            "d3_mean": float(np.mean([g["d3"] for g in geometry])),
            "d4_mean": float(np.mean([g["d4"] for g in geometry])),
            "d1_min": float(np.min([g["d1"] for g in geometry])),
            "closing_velocity_mean": float(np.mean([g["closing"] for g in geometry])),
            "closing_velocity_p50": _percentile([g["closing"] for g in geometry], 50),
            "closing_velocity_p95": _percentile([g["closing"] for g in geometry], 95),
            "fraction_closing": float(np.mean([g["fraction_closing"] for g in geometry])),
            "abs_bearing_error_mean": float(np.mean([g["abs_bearing_error"] for g in geometry])),
            "abs_bearing_error_p50": _percentile([g["abs_bearing_error"] for g in geometry], 50),
            "abs_bearing_error_p95": _percentile([g["abs_bearing_error"] for g in geometry], 95),
            "turn_direction_correct_rate": float(np.mean([g["turn_direction_correct_rate"] for g in geometry])),
            "max_num_within_8m": int(np.max([g["num_within_8"] for g in geometry])),
            "max_num_in_ring": int(np.max([g["num_in_ring_8_10_5"] for g in geometry])),
            "fraction_steps_any_within_8m": float(np.mean([g["num_within_8"] > 0 for g in geometry])),
            "fraction_steps_any_in_ring": float(np.mean([g["num_in_ring_8_10_5"] > 0 for g in geometry])),
            "fraction_steps_2plus_in_ring": float(np.mean([g["num_in_ring_8_10_5"] >= 2 for g in geometry])),
            "fraction_steps_3plus_in_ring": float(np.mean([g["num_in_ring_8_10_5"] >= 3 for g in geometry])),
        })
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
    run_started_at = time.perf_counter()
    loaded = load_config(str(args.config))
    action_mode = str(loaded.get("action", {}).get("mode", "")).strip().lower()
    is_ladder = str((loaded.get("experiment_metadata", {}) or {}).get("series_label", "")).startswith("positive_feedback_ladder")
    if is_ladder or action_mode in {
        AW_ACTION_MODE,
        BODY_ACTION_MODE,
        "aw",
        "continuous_aw",
    }:
        root_config = resolve_ladder_config(args.config)
    else:
        root_config = resolve_formal_config(args.config)
    training_mode = training_mode_contract(root_config)
    _set_seed(args.seed)
    if args.legacy_encoder_checkpoint:
        root_config.setdefault("initialization", {})["actor_encoder"] = {
            "mode": "legacy_iqn",
            "checkpoint": args.legacy_encoder_checkpoint,
            "strict_shape_match": True,
            "freeze_env_steps": 0,
        }
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = _make_trainer(root_config, device)
    max_agents = int(root_config["training"]["max_agents"])
    actor_max_pursuers = int(root_config["actor"]["max_pursuers"])
    scene_text = str(getattr(args, "scenes", "") or "").strip()
    configured_training_scenes = (root_config.get("training", {}) or {}).get(
        "scene_cycle", SCENES
    )
    if isinstance(configured_training_scenes, str):
        configured_training_scenes = tuple(
            item.strip() for item in configured_training_scenes.split(",") if item.strip()
        )
    scenes: Tuple[str, ...] = (
        tuple(item.strip() for item in scene_text.split(",") if item.strip())
        if scene_text
        else tuple(str(item).strip() for item in configured_training_scenes if str(item).strip())
    )
    if not scenes:
        raise ValueError("--scenes must contain at least one scene")
    unknown_scenes = set(scenes).difference(SCENES)
    if unknown_scenes:
        raise ValueError(f"unknown scenes: {sorted(unknown_scenes)}")
    configured_diagnostic_scenes = (root_config.get("evaluation", {}) or {}).get("diagnostic_scenes", scenes)
    if isinstance(configured_diagnostic_scenes, str):
        configured_diagnostic_scenes = configured_diagnostic_scenes.split(",")
    diagnostic_scene_text = str(getattr(args, "diagnostic_eval_scenes", "") or "").strip()
    diagnostic_scenes: Tuple[str, ...] = (
        tuple(item.strip() for item in diagnostic_scene_text.split(",") if item.strip())
        if diagnostic_scene_text
        else tuple(str(item).strip() for item in configured_diagnostic_scenes if str(item).strip())
    )
    if not diagnostic_scenes:
        diagnostic_scenes = scenes
    unknown_diagnostic_scenes = set(diagnostic_scenes).difference(SCENES)
    if unknown_diagnostic_scenes:
        raise ValueError(f"unknown diagnostic scenes: {sorted(unknown_diagnostic_scenes)}")
    all_scenes = tuple(dict.fromkeys((*scenes, *diagnostic_scenes)))
    scene_hashes = {scene: _stable_hash(scene_config(root_config, scene)) for scene in all_scenes}
    manifest = _manifest(root_config, args.seed, args.tag, trainer, scene_hashes, config_path=args.config)
    recovery_cfg = root_config.get("recovery", {}) or {}
    recovery_pool: Deque[Dict[str, Any]] = deque(maxlen=int(recovery_cfg.get("capture_state_pool_capacity", 1000)))
    rng = np.random.default_rng(args.seed)
    snapshot_dataset_path = args.snapshot_dataset or (root_config.get("teacher_snapshot_restore", {}) or {}).get("dataset", "")
    snapshot_dataset = _load_snapshot_dataset(snapshot_dataset_path) if snapshot_dataset_path else None
    artifact_root = Path(args.artifact_root) if args.artifact_root else ARTIFACT_ROOT
    artifact_dir = artifact_root / args.tag
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.resume_checkpoint) != bool(args.resume_replay):
        raise ValueError("--resume-checkpoint and --resume-replay must be supplied together")
    resume_info = None
    runtime_state: Dict[str, Any] = {}
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

    focal_sampler: Optional[FocalReplaySampler] = None
    if bool(training_mode["focal_training"]):
        quotas = dict(root_config["training"]["focal_quota"])
        focal_sampler = FocalReplaySampler(
            quotas,
            max_focal_items_per_joint_transition=int(
                root_config["training"]["max_focal_items_per_joint_transition"]
            ),
            fallback_matrix=root_config["training"].get("fallback_matrix", None),
            seed=args.seed,
        )
        sampler = focal_sampler
    else:
        sampler = UniformJointReplaySampler()

    if resume_info is not None:
        if "runner_rng_state" not in runtime_state:
            raise ValueError("resume checkpoint is missing runner RNG state")
        rng.bit_generator.state = copy.deepcopy(runtime_state["runner_rng_state"])
        if focal_sampler is not None:
            if "focal_sampler_state" not in runtime_state:
                raise ValueError("focal resume checkpoint is missing sampler RNG state")
            focal_sampler.load_state_dict(runtime_state["focal_sampler_state"])
        elif "focal_sampler_state" in runtime_state:
            raise ValueError("all-agent resume unexpectedly contains focal sampler state")

    total_steps = int(args.total_steps if args.total_steps is not None else root_config["training"]["total_env_steps"])
    warmup = int(root_config["training"]["warmup_joint_transitions"])
    update_every = int(root_config["training"]["update_every_env_steps"])
    batch_size = int(root_config["training"]["batch_size"])
    checkpoint_interval = int(root_config["training"].get("checkpoint_interval_env_steps", 25000))
    periodic_checkpoint_replay_mode = str(
        root_config["training"].get("periodic_checkpoint_replay_mode", "full")
    ).strip().lower()
    if periodic_checkpoint_replay_mode not in {"full", "rolling_latest"}:
        raise ValueError("training.periodic_checkpoint_replay_mode must be full or rolling_latest")
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
    window_action_a: List[float] = []
    window_action_w: List[float] = []
    window_geometry: List[Dict[str, Any]] = []
    speed_limited_count = 0
    action_sample_count = 0
    terminated_count = 0
    truncated_count = 0
    collision_count = 0
    window_terminated_count = 0
    window_truncated_count = 0
    window_collision_count = 0
    transition_attempt_count = 0
    update_count = int(runtime_state.get("update_count", 0))
    window_start_step = int(transition_count)
    window_started_at = time.perf_counter()
    updates: List[Dict[str, float]] = list(runtime_state.get("update_metrics_tail", []))
    window_updates: List[Dict[str, float]] = []
    metrics_history: List[Dict[str, Any]] = list(runtime_state.get("metrics_history", []))
    sampling_stats_accum: Dict[str, Any] = dict(runtime_state.get("sampling_stats", {}))
    scene_counts: Dict[str, int] = {
        str(key): int(value) for key, value in dict(runtime_state.get("scene_counts", {})).items()
    }
    origin_counts: Dict[str, int] = {
        str(key): int(value) for key, value in dict(runtime_state.get("origin_counts", {})).items()
    }
    all_finite_so_far = bool(runtime_state.get("all_finite", True))
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
            apf_agents = [ApfAgent(e.a, e.w) for e in env.evaders] if bool(((config.get("evader", {}) or {}) or {}).get("autonomous", False)) else None
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
                before_obs_batch = _stack_with_batch(observations, max_agents, actor_max_pursuers, int(root_config["actor"]["self_feature_dim"]))
                before_obs_padded = {key: value[0] for key, value in before_obs_batch.items()}
                before_global = build_central_global_obs(env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=int(root_config["actor"]["self_feature_dim"]))
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
                window_action_a.extend(float(action[0]) for action in actions)
                window_action_w.extend(float(action[1]) for action in actions)
                try:
                    outcome = env.step(actions.tolist(), _evader_actions_for_env(env, apf_agents))
                except ActionContractError as exc:
                    raise RuntimeError("formal CTDE action contract violated") from exc
                transition_attempt_count += 1
                next_observations = list(outcome.observations)
                speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
                window_speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
                if scene in {"capture", "mixed_crms"}:
                    _geom = _pursuit_step_geometry(env, actions)
                    if _geom is not None:
                        window_geometry.append(_geom)
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
                next_obs_batch = _stack_with_batch(next_observations, max_agents, actor_max_pursuers, int(root_config["actor"]["self_feature_dim"]))
                next_obs_padded = {key: value[0] for key, value in next_obs_batch.items()}
                next_global = build_central_global_obs(env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=int(root_config["actor"]["self_feature_dim"]))
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
                    update_count += 1
                    all_finite_so_far = all_finite_so_far and float(metric.get("finite", 1.0)) == 1.0
                    updates.append(metric)
                    del updates[:-10]
                    window_updates.append(metric)
                    sampling_stats_accum = dict(batch.get("sampling_stats", {}))
                if transition_count % metrics_flush_interval == 0:
                    record = _metrics_record(
                        transition_count,
                        update_count,
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
                        action_a=window_action_a,
                        action_w=window_action_w,
                        geometry=window_geometry,
                        window_wall_time_s=max(time.perf_counter() - window_started_at, 1e-12),
                        window_env_steps=transition_count - window_start_step,
                    )
                    metrics_history.append(record)
                    _append_metrics_jsonl(artifact_dir / "metrics.jsonl", record)
                    window_updates = []
                    window_action_norms = []
                    window_speeds = []
                    window_action_a = []
                    window_action_w = []
                    window_geometry = []
                    window_terminated_count = 0
                    window_truncated_count = 0
                    window_collision_count = 0
                    window_start_step = int(transition_count)
                    window_started_at = time.perf_counter()
                if _periodic_checkpoint_due(
                    transition_count, total_steps, checkpoint_interval
                ):
                    diagnostic_eval = (
                        _screen(
                            trainer,
                            root_config,
                            args.seed,
                            episodes=int(args.diagnostic_eval_episodes),
                            device=device,
                            scenes=diagnostic_scenes,
                            max_steps=diagnostic_rollout_cap,
                        )
                        if transition_count % diagnostic_eval_interval == 0
                        else {}
                    )
                    runtime_state = _runtime_state(
                        transition_count=transition_count,
                        update_count=update_count,
                        scene_index=scene_index,
                        current_scene=scene,
                        recovery_pool=recovery_pool,
                        metrics_history=metrics_history,
                        runner_rng=rng,
                        focal_sampler=focal_sampler,
                        training_mode=training_mode,
                        scene_counts=scene_counts,
                        origin_counts=origin_counts,
                        sampling_stats=sampling_stats_accum,
                        all_finite=all_finite_so_far,
                        update_metrics_tail=updates,
                    )
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
                        include_replay=periodic_checkpoint_replay_mode == "full",
                    )
                    if periodic_checkpoint_replay_mode == "rolling_latest":
                        _save_rolling_resume_bundle(
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
        apf_agents = [ApfAgent(e.a, e.w) for e in env.evaders] if bool(((config.get("evader", {}) or {}) or {}).get("autonomous", False)) else None
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
            before_obs_batch = _stack_with_batch(observations, max_agents, actor_max_pursuers, int(root_config["actor"]["self_feature_dim"]))
            before_obs_padded = {key: value[0] for key, value in before_obs_batch.items()}
            before_global = build_central_global_obs(env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=int(root_config["actor"]["self_feature_dim"]))
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
            window_action_a.extend(float(action[0]) for action in actions)
            window_action_w.extend(float(action[1]) for action in actions)
            try:
                outcome = env.step(actions.tolist(), _evader_actions_for_env(env, apf_agents))
            except ActionContractError as exc:
                raise RuntimeError("formal CTDE action contract violated") from exc
            transition_attempt_count += 1
            next_observations = list(outcome.observations)
            speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
            window_speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
            if scene in {"capture", "mixed_crms"}:
                _geom = _pursuit_step_geometry(env, actions)
                if _geom is not None:
                    window_geometry.append(_geom)
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
            next_obs_batch = _stack_with_batch(next_observations, max_agents, actor_max_pursuers, int(root_config["actor"]["self_feature_dim"]))
            next_obs_padded = {key: value[0] for key, value in next_obs_batch.items()}
            next_global = build_central_global_obs(env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=int(root_config["actor"]["self_feature_dim"]))
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
                update_count += 1
                all_finite_so_far = all_finite_so_far and float(metric.get("finite", 1.0)) == 1.0
                updates.append(metric)
                del updates[:-10]
                window_updates.append(metric)
                sampling_stats_accum = dict(batch.get("sampling_stats", {}))
            if transition_count % metrics_flush_interval == 0:
                record = _metrics_record(
                    transition_count,
                    update_count,
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
                    action_a=window_action_a,
                    action_w=window_action_w,
                    geometry=window_geometry,
                    window_wall_time_s=max(time.perf_counter() - window_started_at, 1e-12),
                    window_env_steps=transition_count - window_start_step,
                )
                metrics_history.append(record)
                _append_metrics_jsonl(artifact_dir / "metrics.jsonl", record)
                window_updates = []
                window_action_norms = []
                window_speeds = []
                window_action_a = []
                window_action_w = []
                window_geometry = []
                window_terminated_count = 0
                window_truncated_count = 0
                window_collision_count = 0
                window_start_step = int(transition_count)
                window_started_at = time.perf_counter()
            if _periodic_checkpoint_due(
                transition_count, total_steps, checkpoint_interval
            ):
                diagnostic_eval = (
                    _screen(
                        trainer,
                        root_config,
                        args.seed,
                        episodes=int(args.diagnostic_eval_episodes),
                        device=device,
                        scenes=diagnostic_scenes,
                        max_steps=diagnostic_rollout_cap,
                    )
                    if transition_count % diagnostic_eval_interval == 0
                    else {}
                )
                runtime_state = _runtime_state(
                    transition_count=transition_count,
                    update_count=update_count,
                    scene_index=scene_index,
                    current_scene=scene,
                    recovery_pool=recovery_pool,
                    metrics_history=metrics_history,
                    runner_rng=rng,
                    focal_sampler=focal_sampler,
                    training_mode=training_mode,
                    scene_counts=scene_counts,
                    origin_counts=origin_counts,
                    sampling_stats=sampling_stats_accum,
                    all_finite=all_finite_so_far,
                    update_metrics_tail=updates,
                )
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
                    include_replay=periodic_checkpoint_replay_mode == "full",
                )
                if periodic_checkpoint_replay_mode == "rolling_latest":
                    _save_rolling_resume_bundle(
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
    for scene in all_scenes:
        (scene_config_dir / f"{scene}.yaml").write_text(
            yaml.safe_dump(scene_config(root_config, scene), sort_keys=False),
            encoding="utf-8",
        )
    runtime_state = _runtime_state(
        transition_count=transition_count,
        update_count=update_count,
        scene_index=scene_index,
        current_scene=scene,
        recovery_pool=recovery_pool,
        metrics_history=metrics_history,
        runner_rng=rng,
        focal_sampler=focal_sampler,
        training_mode=training_mode,
        scene_counts=scene_counts,
        origin_counts=origin_counts,
        sampling_stats=sampling_stats_accum,
        all_finite=all_finite_so_far,
        update_metrics_tail=updates,
    )
    screening = (
        _screen(
            trainer,
            root_config,
            args.seed,
            episodes=int(args.screen_episodes),
            device=device,
            scenes=diagnostic_scenes,
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
            scenes=diagnostic_scenes,
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
        overwrite=True,
    )
    if periodic_checkpoint_replay_mode == "rolling_latest":
        _link_bundle_tree_atomic(final_bundle, artifact_dir / "resume_latest")
    checkpoint = artifact_dir / f"{args.tag}_step{transition_count}.pt"
    replay_path = artifact_dir / f"{args.tag}_replay.pkl"
    _link_or_copy_atomic(final_bundle / "trainer.pt", checkpoint)
    _link_or_copy_atomic(final_bundle / "replay.pkl", replay_path)
    run_wall_time_s = max(time.perf_counter() - run_started_at, 1e-12)
    report = {
        "schema_version": 1,
        "kind": "continuous_ctde_formal",
        "algorithm": root_config["algorithm"],
        "critic_mode": root_config["critic_mode"],
        "optimizer_unit": str(training_mode["optimizer_unit"]),
        "replay_sampling": str(training_mode["replay_sampling"]),
        "focal_training": bool(training_mode["focal_training"]),
        "effective_config_path": str(effective_path),
        "scene_config_hashes": scene_hashes,
        "seed": args.seed,
        "requested_steps": total_steps,
        "transition_count": transition_count,
        "replay_size": len(replay),
        "focal_index_sizes": replay.focal_index_sizes,
        "recovery_pool_size": len(recovery_pool),
        "updates": update_count,
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
        "all_finite": bool(all_finite_so_far),
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
        "action_norm_max": float(np.max(action_norms)) if action_norms else 0.0,
        "speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "speed_max": float(np.max(speeds)) if speeds else 0.0,
        "speed_limit_rate": float(speed_limited_count / max(action_sample_count, 1)),
        "terminated_transition_count": terminated_count,
        "truncated_transition_count": truncated_count,
        "collision_transition_count": collision_count,
        "transition_attempt_count": transition_attempt_count,
        "run_wall_time_s": float(run_wall_time_s),
        "env_steps_per_second": float(transition_count / run_wall_time_s),
        "sampling_stats": sampling_stats_accum,
        "screening": screening,
        "manifest": manifest,
        "resumed_from": resume_info,
        "replay_path": str(replay_path),
        "checkpoint": str(checkpoint),
        "diagnostic_rollout_cap": diagnostic_rollout_cap,
        "training_scenes": list(scenes),
        "diagnostic_scenes": list(diagnostic_scenes),
        "periodic_checkpoint_replay_mode": periodic_checkpoint_replay_mode,
    }
    report_path = artifact_dir / f"{args.tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal CTDE MASAC training runner")
    parser.add_argument("--config", default=str(FORMAL_CONFIG_PATH))
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--seed", type=int, default=2026080601)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--scenes",
        default="",
        help="Comma-separated training scenes. Empty uses training.scene_cycle from the config.",
    )
    parser.add_argument("--snapshot-dataset", default="")
    parser.add_argument("--warmup-action-mode", choices=("actor_prior", "uniform_disk"), default="")
    parser.add_argument("--legacy-encoder-checkpoint", default="")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--screen-episodes", type=int, default=2)
    parser.add_argument("--diagnostic-eval-episodes", type=int, default=4)
    parser.add_argument("--diagnostic-eval-scenes", default="")
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
