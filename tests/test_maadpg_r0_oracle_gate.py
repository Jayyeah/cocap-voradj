from __future__ import annotations

import math

import numpy as np

from maadpg_reproduction.controllers import fixed_ring_pd_actions
from maadpg_reproduction.env import MAADPGPursuitEnv
from maadpg_reproduction.target_policy import StationaryTargetPolicy


def test_stationary_obstacle_free_task_has_a_safe_strict_capture_solution():
    target = np.array([1.0, 1.0])
    angles = np.arange(3) * 2.0 * math.pi / 3.0
    pursuers = target + 0.50 * np.stack(
        [np.cos(angles), np.sin(angles)], axis=1
    )
    env = MAADPGPursuitEnv(target_policy=StationaryTargetPolicy())
    env.reset(
        seed=9001,
        options={
            "pursuer_positions": pursuers,
            "target_position": target,
            "obstacles": [],
        },
    )
    terminal_info = None
    for _step in range(120):
        actions = fixed_ring_pd_actions(
            env.state.pursuer_positions,
            env.state.pursuer_velocities,
            env.state.target_position,
            acceleration_limit=env.config.pursuer_limits.amax,
        )
        _, _, terminated, truncated, terminal_info = env.step(actions)
        if terminated or truncated:
            break
    assert terminal_info is not None
    assert terminated and not truncated
    assert terminal_info["capture"].success
    assert not np.any(terminal_info["collision"]["by_agent"])
    assert terminal_info["step_count"] <= 100


def test_ring_oracle_is_explicitly_separate_from_actor_action_path():
    import maadpg_reproduction.controllers as controllers
    import maadpg_reproduction.env as environment

    assert not hasattr(environment, "fixed_ring_pd_actions")
    assert controllers.fixed_ring_pd_actions.__module__.endswith("controllers")
