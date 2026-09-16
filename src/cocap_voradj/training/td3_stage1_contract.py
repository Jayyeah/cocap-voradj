"""Strict Stage-1 local-TD3 task and continuous-AW contract.

This module intentionally changes only the action API of the current
NormSense-V2 pure single-task environments.  Reward, capture, coverage,
perception, spawn, map, horizon, and collision semantics remain inherited
from the validated single-task contract.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np

from cocap_voradj.envs.density_sensing import enable_v2, runtime_metadata
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.forward_final import flatten
from cocap_voradj.training.runtime_semantics import assert_runtime
from cocap_voradj.training.trainer import set_global_config


ACTION_MODE = "acceleration_angular_velocity_body"
A_MAX = 0.4
W_MAX = float(np.pi / 6.0)
V_MAX = 3.0
DECISION_DT = 0.5
SCHEMA = "td3-local-aw-stage1-contract-v1"


def aw9_grid() -> np.ndarray:
    """The deterministic legacy IQN-index to physical (a, omega) map."""
    return np.asarray(
        [(a, w) for a in (-A_MAX, 0.0, A_MAX) for w in (-W_MAX, 0.0, W_MAX)],
        # Preserve the exact Python/NumPy float values used by the legacy
        # Robot.action_list.  A float32 bridge measurably perturbs the final
        # velocity component at the largest angular-rate action.
        dtype=np.float64,
    )


def discrete_task_config(task: str) -> Dict[str, Any]:
    """Return the latest validated NormSense-V2 pure-task reference."""
    if task not in {"coverage", "capture"}:
        raise ValueError("Stage-1 task must be 'coverage' or 'capture'")
    # Import lazily: tools owns the current validated task definition while
    # this module owns only the TD3 action-API delta.
    from tools.forward_final_single_task_20260915 import task_config

    return enable_v2(task_config(task))


def td3_task_config(task: str) -> Dict[str, Any]:
    """Convert only the discrete action API to legacy-exact continuous AW."""
    cfg = copy.deepcopy(discrete_task_config(task))
    cfg.update(
        action_mode=ACTION_MODE,
        a_max=A_MAX,
        w_max=W_MAX,
        v_max=V_MAX,
        decision_dt=DECISION_DT,
        yaw={"init": "legacy_random"},
        action={"mode": ACTION_MODE, "a_max": A_MAX, "w_max": W_MAX},
    )
    cfg.setdefault("env", {}).update(
        action_mode=ACTION_MODE,
        decision_dt=DECISION_DT,
    )
    cfg.setdefault("pursuer", {}).update(
        action_mode=ACTION_MODE,
        a_max=A_MAX,
        w_max=W_MAX,
    )
    assert_action_only_delta(discrete_task_config(task), cfg)
    return cfg


ACTION_ONLY_DELTA = {
    "action_mode": ACTION_MODE,
    "a_max": A_MAX,
    "w_max": W_MAX,
    "v_max": V_MAX,
    "decision_dt": DECISION_DT,
    "yaw.init": "legacy_random",
    "action.mode": ACTION_MODE,
    "action.a_max": A_MAX,
    "action.w_max": W_MAX,
    "env.action_mode": ACTION_MODE,
    "env.decision_dt": DECISION_DT,
    "pursuer.action_mode": ACTION_MODE,
    "pursuer.a_max": A_MAX,
    "pursuer.w_max": W_MAX,
}


def assert_action_only_delta(
    discrete: Dict[str, Any], continuous: Dict[str, Any]
) -> Tuple[str, ...]:
    """Fail if continuous TD3 silently changes anything except its action API."""
    before, after = flatten(discrete), flatten(continuous)
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    if changed != set(ACTION_ONLY_DELTA):
        raise AssertionError(
            f"TD3 contract drift: changed={sorted(changed)}, "
            f"expected={sorted(ACTION_ONLY_DELTA)}"
        )
    for key, value in ACTION_ONLY_DELTA.items():
        actual = after[key]
        if isinstance(value, float):
            if not np.isclose(float(actual), value, rtol=0.0, atol=1e-12):
                raise AssertionError(f"{key}: expected {value!r}, got {actual!r}")
        elif actual != value:
            raise AssertionError(f"{key}: expected {value!r}, got {actual!r}")
    return tuple(sorted(changed))


def check_td3_env(env: VorAdjEnv, task: str) -> Dict[str, Any]:
    """Assert the resolved runtime contract, including local NormSense V2."""
    expected_config = td3_task_config(task)
    if env.config != expected_config:
        raise AssertionError("Resolved TD3 runtime configuration drift")
    sensing = runtime_metadata(env)
    radius = float(sensing["resolved_onboard_radius"])
    facts = assert_runtime(
        env,
        expected={
            "pursuers": 4,
            "evaders": int(task == "capture"),
            "topology": "friendly_voronoi_comm_v0",
            "enemy_token_rule": "surface_radius",
            "global_enemy_flag": False,
            "support_capture_weight": 1.0,
            "support_coverage_weight": 0.0,
            "support_blend_enabled": task == "capture",
            "capture_reward_mode": "ring_importance_ms_v0",
            "capture_radius": 8.0,
            "capture_k": 3,
            "enemy_radius": radius,
            "action_mode": ACTION_MODE,
            "decision_dt": DECISION_DT,
            "physics_dt": 0.05,
            "a_longitudinal_max": A_MAX,
            "omega_max": W_MAX,
            "adapter_a_max": A_MAX,
            "adapter_w_max": W_MAX,
            "v_max": V_MAX,
            "drag": 0.4 / 3.0,
            "collision_semantics": "synchronized_swept_v1",
        },
    )
    if env.episode_max_length != 3000 or env.reward_cfg["min_active_pursuers"] != 4:
        raise AssertionError("TD3 horizon or active-agent contract drift")
    if not env._ce_coverage_enabled() or not env._ring_importance_ms_enabled():
        raise AssertionError("TD3 reward semantics drift")
    if env._pure_capture_all_capture_enabled():
        raise AssertionError("Historical capture reward override is forbidden")
    if env._support_reward_capture_component_mode() != "approach_only":
        raise AssertionError("Capture support component drift")
    if env._support_reward_capture_target_mode() != "neighbor_visible":
        raise AssertionError("Capture support target drift")
    if env._vct_ls_sensing_radius("obstacle") != radius:
        raise AssertionError("NormSense enemy/obstacle radius mismatch")
    if not np.allclose(np.asarray(env.pursuers[0].action_list), aw9_grid(), atol=1e-12):
        raise AssertionError("Legacy AW9 teacher grid drift")
    return {
        **facts,
        "schema": SCHEMA,
        "task": task,
        "capture_terminal": task == "capture",
        "ce_reward_enabled": task == "coverage",
        "local_actor_observation": True,
        "local_critic_observation": True,
        "sensing": sensing,
        "action_only_delta": list(assert_action_only_delta(discrete_task_config(task), env.config)),
    }


def make_td3_env(task: str, seed: int) -> tuple[VorAdjEnv, Any]:
    cfg = td3_task_config(task)
    set_global_config(cfg)
    env = VorAdjEnv(copy.deepcopy(cfg), seed=int(seed))
    observations = env.reset()
    check_td3_env(env, task)
    return env, observations
