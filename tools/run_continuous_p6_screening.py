"""P6 4v1 continuous-action screening runner.

The runner is intentionally isolated from legacy IQN training.  It keeps one
joint replay contract, switches only the critic mode, applies the P1 feasible
acceleration action contract at environment boundaries, and records deterministic
screening from step zero.  It is suitable for 25k preflight and later
100k/200k screening; it does not auto-promote a factor to formal training.

Checkpoint/replay persistence is decoupled from screening: the runner saves a
checkpoint + joint-replay snapshot at step 1 and then every ``--save-interval``
steps (default 25000), while deterministic screening still runs at
``--screen-interval``.  This bounds the worst-case progress loss after a server
shutdown to one save interval, resumable as seeded episode-boundary continuation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from cocap_voradj.config import ConfigManager
from cocap_voradj.dynamics.continuous_action import (
    ActionContractError,
    AccelerationActionAdapter,
    AccelerationAngularVelocityActionAdapter,
)
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.continuous.central_attention_critic import CentralCriticConfig
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.models.continuous.box_actor import BoxActorConfig
from cocap_voradj.training.continuous.central_sac import CentralSACConfig, CentralSACTrainer
from cocap_voradj.training.continuous.central_schema import build_central_global_obs
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer, JointReplaySampler
from cocap_voradj.training.continuous.local_sac import LocalSACConfig, LocalSACTrainer
from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/continuous_marl_20260804/p5_central_critic_contract_4v1.yaml"
FORMAL_TRAINER_CONFIG_PATH = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_local_sac_4v1.yaml"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-04_continuous_marl_refactor/p6_screening"
BASE_SEED = 2026080410
DEFAULT_SAVE_INTERVAL = 25000
SCENES = ("pure_ce", "capture", "mixed_crms")
SCENE_SETS = {
    "full": SCENES,
    "pure_coverage": ("pure_ce",),
    "broadcast_capture": ("capture",),
    "broadcast_mixed": ("mixed_crms",),
}


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_hash() -> str:
    paths = [
        Path(__file__),
        ROOT / "src/cocap_voradj/training/continuous/local_sac.py",
        ROOT / "src/cocap_voradj/training/continuous/central_sac.py",
        ROOT / "src/cocap_voradj/training/continuous/joint_replay.py",
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


def _trainer_contract(trainer: Any) -> Dict[str, Any]:
    actor = trainer.actor
    encoder = getattr(actor, "encoder", None)
    actor_config = getattr(actor, "config", None)
    critic_config = getattr(getattr(trainer, "critic1", None), "config", None)
    config = trainer.config
    return {
        "encoder_hidden_dim": int(getattr(getattr(encoder, "config", None), "hidden_dim", -1)),
        "encoder_num_layers": int(getattr(getattr(encoder, "config", None), "num_layers", -1)),
        "actor_hidden_dim": int(getattr(actor_config, "hidden_dim", -1)),
        "actor_a_max": float(getattr(actor_config, "a_max", 0.0)),
        "actor_w_max": float(getattr(actor_config, "w_max", 0.0)),
        "actor_decision_dt": float(getattr(actor_config, "decision_dt", 0.0)),
        "actor_log_std_max": float(getattr(actor_config, "log_std_max", 0.0)),
        "critic_hidden_dim": int(getattr(critic_config, "hidden_dim", config.hidden_dim)),
        "critic_num_layers": int(getattr(critic_config, "num_layers", 1)),
        "critic_num_heads": int(getattr(critic_config, "num_heads", 1)),
        "gamma": float(config.gamma),
        "tau": float(config.tau),
        "actor_lr": float(config.actor_lr),
        "critic_lr": float(config.critic_lr),
        "alpha_lr": float(config.alpha_lr),
        "alpha_init": float(config.alpha_init),
        "target_entropy": float(config.target_entropy),
        "warmup_steps": int(getattr(trainer, "warmup_steps", 0)),
        "grad_clip_norm": getattr(config, "grad_clip_norm", None),
    }


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _persistence_due(transition_count: int, save_interval: int) -> bool:
    """True when the runner must persist checkpoint + joint replay snapshot."""
    return transition_count == 1 or (int(save_interval) > 0 and transition_count % int(save_interval) == 0)


def _scene_config(
    scene: str,
    a_max: float,
    *,
    action_mode: str = "axay",
    w_max: float = float(np.pi / 6.0),
    coverage_control_ablation: bool = False,
    global_evader_visibility: bool = False,
) -> Dict[str, Any]:
    config = copy.deepcopy(load_config(str(CONFIG_PATH)))
    config["env"]["num_pursuers"] = 4
    config["env"]["num_evaders"] = 0 if scene == "pure_ce" else 1
    config["env"]["episode_max_length"] = 128
    config["env"]["num_obstacles"] = 0
    config["a_max"] = float(a_max)
    config["action_mode"] = (
        "acceleration_angular_velocity_body" if str(action_mode).lower() in {"aw", "acceleration_angular_velocity_body"}
        else "acceleration_2d_body"
    )
    config["w_max"] = float(w_max)
    config.setdefault("pursuer", {})["a_max"] = float(a_max)
    config.setdefault("pursuer", {})["w_max"] = float(w_max)
    config.setdefault("perception", {})["global_evader_visibility"] = bool(global_evader_visibility)
    if coverage_control_ablation:
        reward = config.setdefault("reward", {})
        reward["coverage_ce_speed_weight"] = 0.0
        reward["coverage_ce_acceleration_weight"] = 0.0
        reward["coverage_ce_angular_velocity_weight"] = 0.0
        reward["coverage_cell_center_speed_penalty_enabled"] = False
    ConfigManager.get_instance().update_config(config)
    return config


def _stack_obs(observations: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.stack([observation[key] for observation in observations], axis=0) for key in observations[0]}


def _stack_obs_with_padding(
    observations: List[Optional[Mapping[str, np.ndarray]]],
    padding: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Stack fixed agent slots, zero-padding observations for inactive agents.

    The environment returns ``None`` for a pursuer that was permanently
    deactivated. The joint replay contract still requires one slot per
    pursuer so that the transition containing the terminal collision reward is
    retained. Padding is only used for the next observation of an inactive
    slot; its active/termination mask decides whether it contributes to SAC.
    """
    if not padding:
        raise ValueError("padding observation tree cannot be empty")
    agent_count = len(observations)
    result: Dict[str, np.ndarray] = {}
    for key, stacked_value in padding.items():
        stacked = np.asarray(stacked_value)
        if stacked.ndim < 1 or stacked.shape[0] != agent_count:
            raise ValueError(f"padding.{key} must have leading agent dimension {agent_count}")
        rows: List[np.ndarray] = []
        for index, observation in enumerate(observations):
            expected_shape = stacked[index].shape
            if observation is None:
                rows.append(np.zeros(expected_shape, dtype=stacked.dtype))
                continue
            if key not in observation:
                raise ValueError(f"next observation slot {index} is missing key {key}")
            value = np.asarray(observation[key])
            if value.shape != expected_shape:
                raise ValueError(
                    f"next observation {key}[{index}] shape mismatch: "
                    f"expected {expected_shape}, got {value.shape}"
                )
            rows.append(value)
        result[key] = np.stack(rows, axis=0)
    return result


def _split_termination_flags(
    dones: List[bool] | np.ndarray,
    infos: List[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate true task termination from the environment time limit.

    The current environment reports both causes through ``dones``. Its
    per-agent info state is the only causal distinction available at this
    runner boundary: ``too long episode`` is a time-limit truncation and all
    other done states are task/agent terminations.
    """
    done_array = np.asarray(dones, dtype=bool)
    if len(infos) != len(done_array):
        raise ValueError("dones and infos must have the same agent count")
    truncated = np.asarray(
        [bool(done and info.get("state") == "too long episode") for done, info in zip(done_array, infos)],
        dtype=bool,
    )
    terminated = done_array & ~truncated
    return terminated, truncated


def _coverage_success_latched(env: Any) -> bool:
    """Return the post-step CE/coverage success latch without inventing labels."""
    reward_terms = getattr(env, "last_reward_terms", {}) or {}
    return bool(
        float(reward_terms.get("coverage_success", 0.0)) > 0.0
        or getattr(env, "post_capture_coverage_success", False)
        or getattr(env, "coverage_geometric_success", False)
        or getattr(env, "coverage_settled_success", False)
    )


def _derive_transition_metadata(
    env: Any,
    *,
    scene: str,
    before_active_target: bool,
    before_coverage_success: bool,
    infos: List[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Derive replay regime/events from action-time and outcome env state.

    Only compact IDs enter replay.  The returned source map is kept in the
    screening report so a missing/fallback event can be audited separately.
    """
    replay_items = [dict(info.get("replay_metadata", {}) or {}) for info in infos]
    phase = next((str(item["phase"]) for item in replay_items if item.get("phase")), "")
    phase_source = "info.replay_metadata.phase" if phase else "fallback.before_active_target"
    if not phase:
        phase = "pre_capture" if before_active_target else ("pure_coverage" if not getattr(env, "evaders", []) else "post_capture")
    task_labels = [str(item.get("task_label", "")) for item in replay_items]
    next_task_labels = [str(item.get("next_task_label", "")) for item in replay_items]
    event_ids: List[str] = []
    event_sources: Dict[str, str] = {}

    def add_event(event_id: str, source: str) -> None:
        if event_id not in event_ids:
            event_ids.append(event_id)
            event_sources[event_id] = source

    if any(
        before not in {"", "capture", "inactive"} and after == "capture"
        for before, after in zip(task_labels, next_task_labels)
    ):
        add_event("discovery", "info.replay_metadata.task_transition")
    if bool(getattr(env, "last_capture_events", [])):
        add_event("capture", "env.last_capture_events")
    collision_states = {
        "collision",
        "deactivated after collision",
        "evader collision",
        "zone breach",
    }
    if any(str(info.get("state", "")) in collision_states for info in infos):
        add_event("collision", "info.state")
    elif any(bool(getattr(pursuer, "collision", False) or getattr(pursuer, "boundary_collision", False)) for pursuer in getattr(env, "pursuers", [])):
        add_event("collision", "env.pursuer_collision_flags")
    if _coverage_success_latched(env) and not before_coverage_success:
        add_event("ce_success", "env.coverage_success_latch")

    metadata = {
        "regime": "active_target" if before_active_target else "coverage_only",
        "coverage_only": not before_active_target,
        "active_target": before_active_target,
        "event_ids": event_ids,
        "scene": str(scene),
        "phase": phase,
        "task_label": next((label for label in task_labels if label), ""),
    }
    sources = {
        "regime": "env.active_evaders_before_step",
        "phase": phase_source,
        **{f"event:{event_id}": source for event_id, source in event_sources.items()},
    }
    return metadata, sources


def _tensor_obs(obs: Mapping[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in obs.items()}


def _sample_actions(
    trainer: Any,
    observations: List[Dict[str, np.ndarray]],
    adapter: Any,
    *,
    deterministic: bool,
) -> Tuple[np.ndarray, float]:
    device = trainer.device
    obs = _tensor_obs(_stack_obs(observations), device)
    with torch.no_grad():
        raw, _, _ = trainer.actor.sample(obs, deterministic=deterministic)
    commands: List[np.ndarray] = []
    rejected = 0
    for action in raw.detach().cpu().numpy():
        validated, diagnostics = adapter.validate_with_diagnostics(action)
        rejected += int(diagnostics.action_rejected)
        commands.append(validated)
    return np.asarray(commands, dtype=np.float32), float(rejected / max(len(commands), 1))


def _make_trainer(
    critic_mode: str,
    a_max: float,
    device: str,
    *,
    action_mode: str = "axay",
    w_max: float = float(np.pi / 6.0),
    actor_log_std_max: float = 1.0,
    trainer_profile: str = "smoke",
) -> Any:
    profile = str(trainer_profile).strip().lower()
    if profile == "smoke":
        hidden_dim, num_heads, num_layers = 16, 4, 1
        learning_rate = 3e-4
        warmup_steps = 0
        grad_clip_norm = None
    elif profile == "formal_p6":
        formal = load_config(str(FORMAL_TRAINER_CONFIG_PATH))
        encoder_cfg = formal.get("continuous_encoder", {}) or {}
        sac_cfg = formal.get("local_sac", {}) or {}
        hidden_dim = int(encoder_cfg.get("hidden_dim", 256))
        num_heads = int(encoder_cfg.get("num_heads", 8))
        num_layers = int(encoder_cfg.get("num_layers", 4))
        learning_rate = float(sac_cfg.get("actor_lr", 1e-4))
        warmup_steps = int(formal.get("warmup_steps", 5000))
        grad_clip_norm = float(formal.get("grad_clip_norm", 10.0))
        if hidden_dim != 256 or learning_rate != 1e-4 or warmup_steps != 5000 or grad_clip_norm != 10.0:
            raise ValueError("formal_p6 profile no longer matches the frozen 256/1e-4/5k/10 contract")
    else:
        raise ValueError("trainer_profile must be smoke or formal_p6")
    encoder = LocalEntityTokenEncoderConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_pursuers=8,
    )
    if str(action_mode).lower() in {"aw", "acceleration_angular_velocity_body"}:
        actor = BoxActorConfig(
            hidden_dim=hidden_dim,
            a_max=float(a_max),
            w_max=float(w_max),
            decision_dt=0.5,
            log_std_max=float(actor_log_std_max),
        )
    else:
        actor = RadialActorConfig(
            hidden_dim=hidden_dim,
            a_max=float(a_max),
            decision_dt=0.5,
            log_std_max=float(actor_log_std_max),
        )
    if critic_mode == "local":
        trainer = LocalSACTrainer(
            encoder_config=encoder,
            actor_config=actor,
            config=LocalSACConfig(
                hidden_dim=hidden_dim,
                actor_lr=learning_rate,
                critic_lr=learning_rate,
                alpha_lr=learning_rate,
                grad_clip_norm=grad_clip_norm,
            ),
            device=device,
            action_mode=action_mode,
        )
    elif critic_mode == "central":
        trainer = CentralSACTrainer(
            encoder_config=encoder,
            actor_config=actor,
            critic_config=CentralCriticConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                max_agents=4,
            ),
            config=CentralSACConfig(
                hidden_dim=hidden_dim,
                actor_lr=learning_rate,
                critic_lr=learning_rate,
                alpha_lr=learning_rate,
                grad_clip_norm=grad_clip_norm,
            ),
            device=device,
            action_mode=action_mode,
        )
    else:
        raise ValueError("critic_mode must be local or central")
    trainer.training_profile = profile
    trainer.warmup_steps = warmup_steps
    return trainer


def _contract(
    critic_mode: str,
    a_max: float,
    seed: int,
    *,
    action_mode: str = "axay",
    w_max: float = float(np.pi / 6.0),
    scene_set: str = "full",
    coverage_control_ablation: bool = False,
    global_evader_visibility: bool = False,
    actor_log_std_max: float = 1.0,
    trainer_profile: str = "smoke",
) -> Dict[str, Any]:
    contract = {
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "action_mode": (
            "acceleration_angular_velocity_body" if str(action_mode).lower() in {"aw", "acceleration_angular_velocity_body"}
            else "acceleration_2d_body"
        ),
        "critic_mode": "local" if critic_mode == "local" else "central_attention_focal",
        "global_schema": "world_entity_v1",
        "v_max": 3.0,
        "a_max": float(a_max),
        "w_max": float(w_max),
        "decision_dt": 0.5,
        "max_agents": 4,
        "seed": int(seed),
        "trainer_profile": str(trainer_profile),
    }
    # Keep diagnostic factors inside the resume contract. A checkpoint from a
    # different scene set/ablation must never silently cross-resume; the
    # trainer profile is also part of the contract for every run.
    if scene_set != "full" or coverage_control_ablation or global_evader_visibility or float(actor_log_std_max) != 1.0:
        contract.update(
            {
                "scene_set": str(scene_set),
                "coverage_control_ablation": bool(coverage_control_ablation),
                "global_evader_visibility": bool(global_evader_visibility),
            }
        )
        if float(actor_log_std_max) != 1.0:
            contract["actor_log_std_max"] = float(actor_log_std_max)
    return contract


def _screen(
    trainer: Any,
    a_max: float,
    seed: int,
    episodes: int = 2,
    *,
    action_mode: str = "axay",
    w_max: float = float(np.pi / 6.0),
    scenes: Tuple[str, ...] = SCENES,
    coverage_control_ablation: bool = False,
    global_evader_visibility: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for scene_index, scene in enumerate(scenes):
        config = _scene_config(
            scene,
            a_max,
            action_mode=action_mode,
            w_max=w_max,
            coverage_control_ablation=coverage_control_ablation,
            global_evader_visibility=global_evader_visibility,
        )
        records = []
        for episode in range(int(episodes)):
            env = VorAdjEnv(config, seed=seed + 10000 + scene_index * 100 + episode)
            env.reset()
            adapter = env.action_adapter
            if adapter is None:
                raise RuntimeError("P6 screening requires an initialized continuous action adapter")
            speed_limited_count = 0
            action_sample_count = 0
            for _ in range(128):
                observations = [item for item in env.get_observations() if item is not None]
                if len(observations) != len(env.pursuers):
                    break
                actions, _ = _sample_actions(trainer, observations, adapter, deterministic=True)
                outcome = env.step(actions.tolist(), [None] * len(env.evaders))
                for info in outcome.infos:
                    diagnostics = info.get("action_diagnostics", {})
                    speed_limited_count += int(bool(diagnostics.get("speed_limited", False)))
                    action_sample_count += 1
                if all(outcome.dones):
                    break
            record = env.episode_record(task=scene)
            records.append(
                {
                    "length": int(record["length"]),
                    "episode_success": bool(record["episode_success"]),
                    "captured": bool(record["captured"]),
                    "collision_event": bool(record["collision_event"]),
                    "coverage_strict_success": bool(record["coverage_strict_success"]),
                    "coverage_cv015_success": bool(record["coverage_cv015_success"]),
                    "coverage_strict_area_cv": float(record["coverage_strict_area_cv"]),
                    "episode_end_mean_speed": float(record["episode_end_mean_speed"]),
                    "episode_end_max_speed": float(record["episode_end_max_speed"]),
                    "speed_limited_rate": float(speed_limited_count / max(action_sample_count, 1)),
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.init != "scratch":
        raise ValueError("P6 transfer is intentionally disabled until a validated IQN mapping is supplied")
    _set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scenes = SCENE_SETS[str(args.scene_set)]
    trainer = _make_trainer(
        args.critic_mode,
        args.a_max,
        device,
        action_mode=args.action_mode,
        w_max=float(args.w_max),
        actor_log_std_max=float(args.actor_log_std_max),
        trainer_profile=str(args.trainer_profile),
    )
    manifest = _contract(
        args.critic_mode,
        args.a_max,
        args.seed,
        action_mode=args.action_mode,
        w_max=float(args.w_max),
        scene_set=args.scene_set,
        coverage_control_ablation=bool(args.coverage_control_ablation),
        global_evader_visibility=bool(args.global_evader_visibility),
        actor_log_std_max=float(args.actor_log_std_max),
        trainer_profile=str(args.trainer_profile),
    )
    effective_config_hashes = {}
    for scene in scenes:
        effective_config_hashes[scene] = _stable_hash(
            _scene_config(
                scene,
                args.a_max,
                action_mode=args.action_mode,
                w_max=float(args.w_max),
                coverage_control_ablation=bool(args.coverage_control_ablation),
                global_evader_visibility=bool(args.global_evader_visibility),
            )
        )
    manifest.update(
        {
            "manifest_schema_version": 2,
            "trainer_config": str(
                (FORMAL_TRAINER_CONFIG_PATH if args.trainer_profile == "formal_p6" else CONFIG_PATH).relative_to(ROOT)
            ),
            "trainer_config_hash": _stable_hash(
                load_config(str(FORMAL_TRAINER_CONFIG_PATH if args.trainer_profile == "formal_p6" else CONFIG_PATH))
            ),
            "effective_config_hashes": effective_config_hashes,
            "implementation_hash": _implementation_hash(),
            "trainer_contract": _trainer_contract(trainer),
            "resume_contract": {
                "rng": ["torch_cpu", "torch_cuda", "python", "numpy", "replay_numpy"],
                "scene_index": "saved",
                "exact_env_state": False,
                "continuation_mode": "seeded_episode_boundary",
            },
        }
    )
    if bool(args.resume_checkpoint) != bool(args.resume_replay):
        raise ValueError("--resume-checkpoint and --resume-replay must be supplied together")
    resume_info = None
    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint, manifest)
        replay = JointReplayBuffer.load(args.resume_replay, manifest)
        runtime_state = dict(getattr(trainer, "resume_runtime_state", {}) or {})
        runtime_state.update(getattr(replay, "runtime_state", {}) or {})
        transition_count = int(
            args.resume_step if args.resume_step >= 0 else runtime_state.get("transition_count", len(replay))
        )
        scene_index = int(runtime_state.get("next_scene_index", transition_count))
        resume_info = {
            "checkpoint": str(args.resume_checkpoint),
            "replay": str(args.resume_replay),
            "step": transition_count,
            "next_scene_index": scene_index,
            "continuation_mode": manifest["resume_contract"]["continuation_mode"],
            "exact_env_state": bool(manifest["resume_contract"]["exact_env_state"]),
        }
    else:
        # Training length and replay retention are independent knobs. The old
        # ``max(total_steps, replay_capacity)`` rule made a long 500k run
        # allocate the entire trajectory even when only a bounded hot replay
        # was requested, which is both wasteful and prone to host OOM.
        replay = JointReplayBuffer(
            capacity=max(int(args.replay_capacity), int(args.batch_size)),
            max_agents=4,
            seed=args.seed,
        )
        transition_count = 0
        scene_index = 0
    sampler = JointReplaySampler()
    rng = np.random.default_rng(args.seed)
    env = None
    observations: List[Optional[Dict[str, np.ndarray]]] = []
    updates: List[Dict[str, float]] = []
    screening: List[Dict[str, Any]] = []
    action_validation_rates: List[float] = []
    terminal_transition_count = 0
    terminated_transition_count = 0
    truncated_transition_count = 0
    transition_attempt_count = 0
    inactive_next_slot_transition_count = 0
    collision_transition_count = 0
    collision_reward_sum = 0.0
    reward_term_sums: Dict[str, float] = {}
    metadata_event_counts: Dict[str, int] = {}
    metadata_event_source_counts: Dict[str, int] = {}
    metadata_source_counts: Dict[str, int] = {}
    sampling_event_fallback_count = 0
    sampling_batch_count = 0
    sampling_source_slot_counts: Dict[str, int] = {}
    sampling_regime_counts: Dict[str, int] = {}
    sampling_event_count = 0
    sampling_coverage_only_count = 0
    sampling_coverage_floor_min_ratio = 1.0
    while transition_count < int(args.total_steps):
        scene = scenes[scene_index % len(scenes)]
        scene_index += 1
        config = _scene_config(
            scene,
            args.a_max,
            action_mode=args.action_mode,
            w_max=float(args.w_max),
            coverage_control_ablation=bool(args.coverage_control_ablation),
            global_evader_visibility=bool(args.global_evader_visibility),
        )
        env = VorAdjEnv(config, seed=args.seed + scene_index)
        env.reset()
        adapter = env.action_adapter
        if adapter is None:
            raise RuntimeError("P6 screening requires an initialized continuous action adapter")
        observations = list(env.get_observations())
        while transition_count < int(args.total_steps):
            if len(observations) != len(env.pursuers) or any(item is None for item in observations):
                break
            active_observations = [item for item in observations if item is not None]
            before_obs = _stack_obs(active_observations)
            before_active = np.asarray([not p.deactivated for p in env.pursuers], dtype=bool)
            before_active_target = bool(any(not evader.deactivated for evader in env.evaders))
            before_coverage_success = _coverage_success_latched(env)
            before_global = build_central_global_obs(env, max_agents=4, max_evaders=8, max_obstacles=5)
            actions, validation_rate = _sample_actions(trainer, active_observations, adapter, deterministic=False)
            action_validation_rates.append(validation_rate)
            try:
                outcome = env.step(actions.tolist(), [None] * len(env.evaders))
            except ActionContractError as exc:
                raise RuntimeError(
                    f"P6 continuous action contract mismatch: "
                    f"a_max={getattr(env.action_adapter, 'a_max', None)}, "
                    f"actions={actions.tolist()}"
                ) from exc
            next_observations = list(outcome.observations)
            transition_attempt_count += 1
            for key, value in (getattr(env, "last_reward_terms", {}) or {}).items():
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                    continue
                reward_term_sums[str(key)] = reward_term_sums.get(str(key), 0.0) + float(value)
            next_local_obs = _stack_obs_with_padding(next_observations, before_obs)
            terminated, truncated = _split_termination_flags(outcome.dones, outcome.infos)
            terminal_transition_count += int(any(outcome.dones))
            terminated_transition_count += int(any(terminated))
            truncated_transition_count += int(any(truncated))
            inactive_next_slot_transition_count += int(any(item is None for item in next_observations))
            collision_transition_count += int(
                any(info.get("state") == "deactivated after collision" for info in outcome.infos)
            )
            collision_reward_sum += float(
                sum(
                    reward
                    for reward, info in zip(outcome.rewards, outcome.infos)
                    if info.get("state") == "deactivated after collision"
                )
            )
            metadata, metadata_sources = _derive_transition_metadata(
                env,
                scene=scene,
                before_active_target=before_active_target,
                before_coverage_success=before_coverage_success,
                infos=outcome.infos,
            )
            for source in metadata_sources.values():
                metadata_source_counts[source] = metadata_source_counts.get(source, 0) + 1
            for event_id in metadata["event_ids"]:
                metadata_event_counts[event_id] = metadata_event_counts.get(event_id, 0) + 1
                source = metadata_sources.get(f"event:{event_id}", "unknown")
                metadata_event_source_counts[source] = metadata_event_source_counts.get(source, 0) + 1
            replay.add(
                local_obs=before_obs,
                next_local_obs=next_local_obs,
                global_state=before_global,
                next_global_state=build_central_global_obs(env, max_agents=4, max_evaders=8, max_obstacles=5),
                actions=actions,
                rewards=np.asarray(outcome.rewards, dtype=np.float32),
                active_mask=before_active,
                terminated=terminated,
                truncated=truncated,
                metadata=metadata,
            )
            transition_count += 1
            observations = next_observations
            if (
                len(replay) >= args.batch_size
                and transition_count >= int(getattr(trainer, "warmup_steps", 0))
                and transition_count % args.update_every == 0
            ):
                batch = replay.sample(args.batch_size, device=device, sampler=sampler)
                sample_stats = batch.get("sampling_stats", {})
                sampling_batch_count += 1
                sampling_event_fallback_count += int(sample_stats.get("event_fallback_count", 0))
                for source, count in (sample_stats.get("source_slot_counts", {}) or {}).items():
                    sampling_source_slot_counts[source] = sampling_source_slot_counts.get(source, 0) + int(count)
                for regime, count in (sample_stats.get("regime_counts", {}) or {}).items():
                    sampling_regime_counts[regime] = sampling_regime_counts.get(regime, 0) + int(count)
                sampling_event_count += int(sample_stats.get("event_count", 0))
                coverage_only_count = int(sample_stats.get("coverage_only_count", 0))
                sampling_coverage_only_count += coverage_only_count
                actual_batch = max(int(sample_stats.get("actual", 0)), 1)
                sampling_coverage_floor_min_ratio = min(
                    sampling_coverage_floor_min_ratio,
                    float(coverage_only_count / actual_batch),
                )
                updates.append(trainer.update(batch))
            if transition_count % args.screen_interval == 0 or transition_count == 1:
                screening.append({
                    "step": transition_count,
                    "critic_mode": args.critic_mode,
                    "a_max": args.a_max,
                    "action_mode": args.action_mode,
                    "w_max": float(args.w_max),
                    "metrics": _screen(
                        trainer,
                        args.a_max,
                        args.seed,
                        episodes=args.screen_episodes,
                        scenes=scenes,
                        action_mode=args.action_mode,
                        w_max=float(args.w_max),
                        coverage_control_ablation=bool(args.coverage_control_ablation),
                        global_evader_visibility=bool(args.global_evader_visibility),
                    ),
                    "action_validation_rate_mean": float(np.mean(action_validation_rates)) if action_validation_rates else 0.0,
                })
            if _persistence_due(transition_count, args.save_interval):
                checkpoint = ARTIFACT_ROOT / f"{args.tag}_step{transition_count}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                runtime_state = {
                    "transition_count": int(transition_count),
                    "next_scene_index": int(scene_index),
                    "current_scene": str(scene),
                    "current_episode_step": int(env.episode_step),
                    "continuation_mode": manifest["resume_contract"]["continuation_mode"],
                    "exact_env_state": bool(manifest["resume_contract"]["exact_env_state"]),
                }
                trainer.save_checkpoint(checkpoint, manifest, runtime_state=runtime_state)
                replay.save(
                    ARTIFACT_ROOT / f"{args.tag}_replay_step{transition_count}.pkl",
                    manifest,
                    runtime_state=runtime_state,
                )
            if all(outcome.dones) or any(item is None for item in next_observations):
                break
    replay_path = ARTIFACT_ROOT / f"{args.tag}_replay.pkl"
    report_path = ARTIFACT_ROOT / f"{args.tag}_report.json"
    replay.save(
        replay_path,
        manifest,
        runtime_state={
            "transition_count": int(transition_count),
            "next_scene_index": int(scene_index),
            "continuation_mode": manifest["resume_contract"]["continuation_mode"],
            "exact_env_state": bool(manifest["resume_contract"]["exact_env_state"]),
        },
    )
    payload = {
        "schema_version": 1,
        "kind": "continuous_p6_screening",
        "seed": args.seed,
        "critic_mode": args.critic_mode,
        "a_max": args.a_max,
        "action_mode": args.action_mode,
        "w_max": float(args.w_max),
        "scene_set": args.scene_set,
        "coverage_control_ablation": bool(args.coverage_control_ablation),
        "global_evader_visibility": bool(args.global_evader_visibility),
        "actor_log_std_max": float(args.actor_log_std_max),
        "trainer_profile": str(args.trainer_profile),
        "warmup_steps": int(getattr(trainer, "warmup_steps", 0)),
        "init": args.init,
        "device": device,
        "requested_steps": args.total_steps,
        "save_interval": int(args.save_interval),
        "replay_size": len(replay),
        "updates": len(updates),
        "update_metrics_tail": updates[-10:],
        "all_finite": bool(all(item["finite"] == 1.0 for item in updates)) if updates else True,
        "action_validation_rate_mean": float(np.mean(action_validation_rates)) if action_validation_rates else 0.0,
        "terminal_transition_count": int(terminal_transition_count),
        "terminated_transition_count": int(terminated_transition_count),
        "truncated_transition_count": int(truncated_transition_count),
        "transition_attempt_count": int(transition_attempt_count),
        "terminal_transition_drop_count": int(max(0, transition_attempt_count - len(replay))),
        "inactive_next_slot_transition_count": int(inactive_next_slot_transition_count),
        "collision_transition_count": int(collision_transition_count),
        "collision_reward_sum": float(collision_reward_sum),
        "reward_term_sums": {key: float(value) for key, value in sorted(reward_term_sums.items())},
        "reward_term_means": {
            key: float(value / max(transition_attempt_count, 1)) for key, value in sorted(reward_term_sums.items())
        },
        "metadata_event_counts": {key: int(value) for key, value in sorted(metadata_event_counts.items())},
        "metadata_event_source_counts": {
            key: int(value) for key, value in sorted(metadata_event_source_counts.items())
        },
        "metadata_source_counts": {key: int(value) for key, value in sorted(metadata_source_counts.items())},
        "sampling_batch_count": int(sampling_batch_count),
        "sampling_event_fallback_count": int(sampling_event_fallback_count),
        "sampling_event_fallback_rate": float(
            sampling_event_fallback_count / max(sampling_batch_count, 1)
        ),
        "sampling_source_slot_counts": {
            key: int(value) for key, value in sorted(sampling_source_slot_counts.items())
        },
        "sampling_regime_counts": {key: int(value) for key, value in sorted(sampling_regime_counts.items())},
        "sampling_event_count": int(sampling_event_count),
        "sampling_coverage_only_count": int(sampling_coverage_only_count),
        "sampling_coverage_only_fraction": float(
            sampling_coverage_only_count / max(sum(sampling_regime_counts.values()), 1)
        ),
        "sampling_event_fraction": float(sampling_event_count / max(sum(sampling_regime_counts.values()), 1)),
        "sampling_coverage_floor_min_ratio": float(sampling_coverage_floor_min_ratio),
        "screening": screening,
        "replay_path": str(replay_path),
        "manifest": manifest,
        "resumed_from": resume_info,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-mode", choices=("local", "central"), default="local")
    parser.add_argument("--action-mode", choices=("axay", "aw"), default="axay")
    parser.add_argument("--a-max", type=float, choices=(0.4, 0.8, 1.6), default=0.8)
    parser.add_argument("--w-max", type=float, default=float(np.pi / 6.0))
    parser.add_argument("--scene-set", choices=tuple(SCENE_SETS), default="full")
    parser.add_argument("--coverage-control-ablation", action="store_true")
    parser.add_argument("--global-evader-visibility", action="store_true")
    parser.add_argument("--actor-log-std-max", type=float, default=1.0)
    parser.add_argument("--trainer-profile", choices=("smoke", "formal_p6"), default="smoke")
    parser.add_argument("--init", choices=("scratch", "transfer"), default="scratch")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--total-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--screen-interval", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--screen-episodes", type=int, default=2)
    parser.add_argument("--replay-capacity", type=int, default=30000)
    parser.add_argument("--device", default="")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-replay", default="")
    parser.add_argument("--resume-step", type=int, default=-1)
    args = parser.parse_args()
    if args.save_interval <= 0:
        parser.error("--save-interval must be a positive integer")
    payload = run(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["all_finite"] and payload["replay_size"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
