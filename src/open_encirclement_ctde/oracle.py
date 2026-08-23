"""Geometric controller used only as an environment sanity oracle."""

from __future__ import annotations

import numpy as np

from .environment import CorrectedRoundupEnv


class TriangleSlotOracle:
    """Drive hunters toward three evenly spaced target-relative slots."""

    def __init__(self, radius: float = 0.22, kp: float = 0.9, kd: float = 1.4) -> None:
        self.radius = float(radius)
        self.kp = float(kp)
        self.kd = float(kd)

    def actions(self, env: CorrectedRoundupEnv) -> np.ndarray:
        target = env.positions[3]
        # The target starts near the north wall.  Fixed identities avoid the
        # discontinuous slot swapping that a per-step nearest assignment causes.
        slot_angles = np.deg2rad(np.array([150.0, 30.0, -90.0]))
        target_velocity = env.velocities[3]
        lead_center = np.clip(
            target + 3.0 * target_velocity,
            self.radius,
            env.spec.arena_length - self.radius,
        )
        slots = lead_center + self.radius * np.column_stack((np.cos(slot_angles), np.sin(slot_angles)))
        actions = np.zeros((3, 2), dtype=np.float64)
        for hunter_idx in range(3):
            desired_velocity = target_velocity + self.kp * (slots[hunter_idx] - env.positions[hunter_idx])
            speed = np.linalg.norm(desired_velocity)
            if speed > env.spec.hunter_v_max:
                desired_velocity *= env.spec.hunter_v_max / speed
            action = (desired_velocity - env.velocities[hunter_idx]) * self.kd
            actions[hunter_idx] = np.clip(action, -env.spec.hunter_a_max, env.spec.hunter_a_max)
        return actions


def run_oracle_episode(env: CorrectedRoundupEnv, controller: TriangleSlotOracle | None = None) -> dict:
    controller = controller or TriangleSlotOracle()
    _, _ = env.reset()
    max_hold = 0
    current_hold = 0
    last_info = env.last_info
    for _ in range(env.spec.max_steps):
        _, _, terminated, truncated, last_info = env.step(controller.actions(env))
        if last_info["hull_contained"] and last_info["all_within_capture_radius"]:
            current_hold += 1
            max_hold = max(max_hold, current_hold)
        else:
            current_hold = 0
        if terminated or truncated:
            break
    return {
        "success": bool(last_info["success"]),
        "episode_length": int(last_info["step"]),
        "max_hold": int(max_hold),
        "last_info": last_info,
    }
