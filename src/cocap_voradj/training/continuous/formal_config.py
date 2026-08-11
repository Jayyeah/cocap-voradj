"""Single-source formal configuration helpers for the CTDE MASAC line."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np

from cocap_voradj.training.trainer import deep_update, load_config


FORMAL_ALGORITHM = "masac_ctde"
FORMAL_ACTION_MODE = "acceleration_2d_world"
BODY_ACTION_MODE = "acceleration_2d_body"
AW_ACTION_MODE = "acceleration_angular_velocity_body"
FORMAL_DYNAMICS_PROFILE = "continuous_parity_v1"
SCENES = ("capture", "pure_ce", "mixed_crms")


def _require_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping in the formal config")
    return value

def training_mode_contract(config: Dict[str, Any], label: str = "formal config") -> Dict[str, Any]:
    """Resolve and validate the optimizer/replay unit without changing legacy defaults."""
    training = _require_mapping(config.get("training"), "training")
    focal_training = bool(training.get("focal_training", True))
    optimizer_unit = str(
        training.get(
            "optimizer_unit",
            "focal_agent_item" if focal_training else "",
        )
    ).strip()
    replay_sampling = str(
        training.get(
            "replay_sampling",
            "focal_role_balanced" if focal_training else "",
        )
    ).strip()
    if focal_training:
        if optimizer_unit != "focal_agent_item":
            raise ValueError(f"{label}: focal training requires optimizer_unit=focal_agent_item")
        if replay_sampling != "focal_role_balanced":
            raise ValueError(f"{label}: focal training requires replay_sampling=focal_role_balanced")
        quotas = _require_mapping(training.get("focal_quota"), "training.focal_quota")
        if sum(int(value) for value in quotas.values()) != int(training.get("batch_size", 0)):
            raise ValueError(f"{label}: focal quotas must sum to training.batch_size")
    else:
        if optimizer_unit != "joint_transition_all_active_agents":
            raise ValueError(
                f"{label}: all-agent training requires "
                "optimizer_unit=joint_transition_all_active_agents"
            )
        if replay_sampling != "uniform_joint":
            raise ValueError(f"{label}: all-agent training requires replay_sampling=uniform_joint")
        if training.get("focal_quota") not in (None, {}):
            raise ValueError(f"{label}: all-agent training must explicitly disable focal_quota")
    return {
        "optimizer_unit": optimizer_unit,
        "replay_sampling": replay_sampling,
        "focal_training": focal_training,
    }



def validate_formal_config(config: Dict[str, Any], path: str | Path | None = None) -> Dict[str, Any]:
    """Validate the frozen formal CTDE contract and reject legacy/local paths."""
    label = f"formal config {path}" if path else "formal config"
    if str(config.get("algorithm", "")).strip().lower() != FORMAL_ALGORITHM:
        raise ValueError(f"{label}: algorithm must be {FORMAL_ALGORITHM!r}")
    if str(config.get("critic_mode", "")).strip().lower() != "central":
        raise ValueError(f"{label}: critic_mode must be central; local critic is forbidden")
    if "local_sac" in config:
        raise ValueError(f"{label}: local_sac block is forbidden in the formal entry")
    profile = str(config.get("trainer_profile", "")).strip().lower()
    if profile in {"smoke", "local"}:
        raise ValueError(f"{label}: smoke/local trainer profiles are forbidden")

    env = _require_mapping(config.get("env"), "env")
    if "width" not in env or "height" not in env:
        raise ValueError(f"{label}: env.width and env.height are mandatory; 55 fallback is forbidden")
    if int(env.get("num_obstacles", 0)) != 1:
        raise ValueError(f"{label}: formal Stage-1 requires env.num_obstacles == 1")
    if int(env.get("num_pursuers", 0)) != 4:
        raise ValueError(f"{label}: formal Stage-1 requires env.num_pursuers == 4")

    action = _require_mapping(config.get("action"), "action")
    if str(action.get("mode", "")).strip().lower() != FORMAL_ACTION_MODE:
        raise ValueError(
            f"{label}: action.mode must be {FORMAL_ACTION_MODE!r}; "
            "velocity_2d_body and acceleration_2d_body aliases are forbidden"
        )
    if float(action.get("a_max", 0.0)) > 0.4 + 1e-9:
        raise ValueError(f"{label}: first formal run must keep a_max <= 0.4")

    dynamics = _require_mapping(config.get("dynamics"), "dynamics")
    if str(dynamics.get("profile", "")).strip().lower() != FORMAL_DYNAMICS_PROFILE:
        raise ValueError(f"{label}: dynamics.profile must be {FORMAL_DYNAMICS_PROFILE!r}")

    _require_mapping(config.get("actor"), "actor")
    _require_mapping(config.get("central_critic"), "central_critic")
    _require_mapping(config.get("masac"), "masac")
    _require_mapping(config.get("replay"), "replay")
    training = _require_mapping(config.get("training"), "training")
    if int(training.get("batch_size", 0)) != 128:
        raise ValueError(f"{label}: training.batch_size must be 128")
    if abs(float(training.get("grad_clip_norm", 0.0)) - 0.5) > 1e-9:
        raise ValueError(f"{label}: training.grad_clip_norm must be 0.5")
    if int(training.get("max_agents", 0)) != 12:
        raise ValueError(f"{label}: training.max_agents must be 12")
    training_mode_contract(config, label)
    _require_mapping(config.get("tasks"), "tasks")
    evaluation = _require_mapping(config.get("evaluation"), "evaluation")
    if int(evaluation.get("diagnostic_rollout_cap", 0)) != 400:
        raise ValueError(f"{label}: evaluation.diagnostic_rollout_cap must be 400")
    teacher_snapshot = _require_mapping(config.get("teacher_snapshot_restore"), "teacher_snapshot_restore")
    if str(teacher_snapshot.get("mode", "geometry_reset")).strip().lower() not in {"geometry_reset", "full_state"}:
        raise ValueError(f"{label}: teacher_snapshot_restore.mode must be geometry_reset or full_state")
    warmup_policy = _require_mapping(config.get("warmup_action_policy"), "warmup_action_policy")
    if str(warmup_policy.get("mode", "actor_prior")).strip().lower() not in {"actor_prior", "uniform_disk"}:
        raise ValueError(f"{label}: warmup_action_policy.mode must be actor_prior or uniform_disk")
    _require_mapping(config.get("recovery"), "recovery")
    return config


def validate_ladder_config(config: Dict[str, Any], path: str | Path | None = None) -> Dict[str, Any]:
    """Validate a positive-feedback-ladder formal config.

    The ladder may use either the frozen world-frame disk or the continuous
    ``(a,omega)`` body action contract, and diagnostic stages are allowed to
    reduce pursuer/evader/obstacle counts.  Frozen optimization and physics
    parameters remain enforced.
    """
    label = f"ladder config {path}" if path else "ladder config"
    if str(config.get("algorithm", "")).strip().lower() != FORMAL_ALGORITHM:
        raise ValueError(f"{label}: algorithm must be {FORMAL_ALGORITHM!r}")
    if str(config.get("critic_mode", "")).strip().lower() != "central":
        raise ValueError(f"{label}: critic_mode must be central")
    if "local_sac" in config:
        raise ValueError(f"{label}: local_sac block is forbidden")

    env = _require_mapping(config.get("env"), "env")
    if abs(float(env.get("width", 0.0)) - 120.0) > 1e-9 or abs(float(env.get("height", 0.0)) - 120.0) > 1e-9:
        raise ValueError(f"{label}: map must remain 120x120")

    action = _require_mapping(config.get("action"), "action")
    action_mode = str(action.get("mode", "")).strip().lower()
    if action_mode not in {FORMAL_ACTION_MODE, BODY_ACTION_MODE, AW_ACTION_MODE, "aw", "continuous_aw"}:
        raise ValueError(f"{label}: unsupported action.mode {action_mode!r}")
    if float(action.get("a_max", 0.0)) > 0.4 + 1e-9:
        raise ValueError(f"{label}: a_max must be <= 0.4")
    if action_mode in {AW_ACTION_MODE, "aw", "continuous_aw"}:
        if abs(float(action.get("w_max", 0.0)) - float(np.pi / 6.0)) > 1e-9:
            raise ValueError(f"{label}: w_max must be pi/6 for the (a,omega) contract")

    dynamics = _require_mapping(config.get("dynamics"), "dynamics")
    if str(dynamics.get("profile", "")).strip().lower() not in {"continuous_parity_v1", "continuous_aw_v1"}:
        raise ValueError(f"{label}: dynamics.profile must be continuous_parity_v1 or continuous_aw_v1")

    _require_mapping(config.get("actor"), "actor")
    _require_mapping(config.get("central_critic"), "central_critic")
    _require_mapping(config.get("masac"), "masac")
    _require_mapping(config.get("replay"), "replay")
    training = _require_mapping(config.get("training"), "training")
    if int(training.get("batch_size", 0)) != 128:
        raise ValueError(f"{label}: training.batch_size must be 128")
    grad_clip_norm = training.get("grad_clip_norm")
    if grad_clip_norm is not None and abs(float(grad_clip_norm) - 0.5) > 1e-9:
        raise ValueError(f"{label}: training.grad_clip_norm must be 0.5 or null")
    if int(training.get("max_agents", 0)) != 12:
        raise ValueError(f"{label}: training.max_agents must be 12")
    training_mode_contract(config, label)
    _require_mapping(config.get("tasks"), "tasks")
    evaluation = _require_mapping(config.get("evaluation"), "evaluation")
    if int(evaluation.get("diagnostic_rollout_cap", 0)) != 400:
        raise ValueError(f"{label}: evaluation.diagnostic_rollout_cap must be 400")
    return config


def resolve_formal_config(path: str | Path) -> Dict[str, Any]:
    resolved = load_config(str(path))
    return validate_formal_config(resolved, path)


def resolve_ladder_config(path: str | Path) -> Dict[str, Any]:
    resolved = load_config(str(path))
    return validate_ladder_config(resolved, path)


def scene_config(config: Dict[str, Any], scene: str) -> Dict[str, Any]:
    """Deep-merge one task override onto the resolved root config."""
    tasks = config.get("tasks") or {}
    if scene not in tasks:
        raise ValueError(f"formal config is missing tasks.{scene}")
    if not isinstance(tasks[scene], dict):
        raise ValueError(f"tasks.{scene} must be a mapping")
    return deep_update(config, tasks[scene])
