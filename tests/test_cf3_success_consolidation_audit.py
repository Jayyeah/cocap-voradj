from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from tools.audit_cf3_success_consolidation import (
    CATEGORY_LONG_TWO_PLUS,
    CATEGORY_NORMAL_CAPTURE,
    CATEGORY_THREE_PLUS_COLLISION,
    action_rows,
    angular_geometry,
    classify_episode,
    clearance_geometry,
    max_hold,
    relative_velocity_components,
    summarize_trace,
)


def robot(index: int, x: float, y: float, *, radius: float = 1.0):
    return SimpleNamespace(
        id=index,
        x=x,
        y=y,
        r=radius,
        theta=0.0,
        speed=0.0,
        velocity=np.zeros(2, dtype=float),
        deactivated=False,
        collision=False,
    )


def test_angular_geometry_reports_wrap_gap_and_pairwise_separation() -> None:
    geometry = angular_geometry([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    assert np.allclose(geometry["adjacent_gaps_rad"], [2.0 * math.pi / 3.0] * 3)
    assert math.isclose(geometry["largest_angular_gap_rad"], 2.0 * math.pi / 3.0)
    assert math.isclose(geometry["minimum_pairwise_separation_rad"], 2.0 * math.pi / 3.0)


def test_relative_velocity_signs_are_enemy_centred_and_explicit() -> None:
    closing = relative_velocity_components(
        [2.0, 0.0], [-1.0, 2.0], [0.0, 0.0], [0.0, 0.0],
    )
    assert closing["radial_velocity_outward"] == -1.0
    assert closing["closing_speed"] == 1.0
    assert closing["tangential_velocity_ccw"] == 2.0


def test_clearance_geometry_uses_surface_for_entities_and_centre_for_boundary() -> None:
    pursuers = [robot(0, 2.0, 2.0), robot(1, 5.0, 2.0)]
    obstacles = [SimpleNamespace(x=2.0, y=6.0, r=1.5)]
    geometry = clearance_geometry(pursuers, obstacles, width=10.0, height=10.0)
    assert geometry["min_agent_agent_surface_clearance"] == 1.0
    assert geometry["min_obstacle_surface_clearance"] == 1.5
    assert geometry["min_boundary_center_clearance"] == 2.0


def test_episode_classes_have_fixed_priority_and_hold_requirement() -> None:
    reached_three = [{
        "ring_count_pre": 2,
        "ring_count_post_alive": 3,
        "collision_now": True,
    }]
    assert classify_episode(
        reached_three,
        normal_capture=True,
        stationary_capture=False,
        collision=True,
        long_two_plus_steps=2,
    ) == CATEGORY_NORMAL_CAPTURE
    assert classify_episode(
        reached_three,
        normal_capture=False,
        stationary_capture=False,
        collision=True,
        long_two_plus_steps=2,
    ) == CATEGORY_THREE_PLUS_COLLISION
    assert classify_episode(
        reached_three,
        normal_capture=False,
        stationary_capture=True,
        collision=True,
        long_two_plus_steps=2,
    ) is None
    long_two = [
        {"ring_count_pre": 0, "ring_count_post_alive": value}
        for value in [0, 2, 2, 2, 1]
    ]
    assert classify_episode(
        long_two,
        normal_capture=False,
        stationary_capture=False,
        collision=False,
        long_two_plus_steps=3,
    ) == CATEGORY_LONG_TWO_PLUS
    assert max_hold([0, 2, 2, 1, 3, 3, 3], 2) == 3


def test_action_rows_marks_saturation_and_nontrivial_omega_sign_flip() -> None:
    env = SimpleNamespace(
        pursuers=[robot(0, 0.0, 0.0)],
        action_adapter=SimpleNamespace(a_max=0.4, w_max=0.5),
    )
    previous = {0: 0.3}
    rows = action_rows(env, np.asarray([[0.4, -0.5]], dtype=float), [0], previous)
    assert rows == [{
        "pursuer_id": 0,
        "a": 0.4,
        "omega": -0.5,
        "a_fraction_of_limit": 1.0,
        "omega_fraction_of_limit": 1.0,
        "a_saturated": True,
        "omega_saturated": True,
        "omega_sign_flip": True,
    }]


def test_trace_summary_retains_geometry_clearance_and_control_failure_signals() -> None:
    frames = []
    for step, count in enumerate([2, 3, 3], start=1):
        frames.append({
            "episode_step": step,
            "ring_count_post_alive": count,
            "post": {
                "angular": {
                    "largest_angular_gap_rad": 3.0 - 0.1 * step,
                    "minimum_pairwise_separation_rad": 0.5 + 0.1 * step,
                },
                "clearance": {
                    "min_agent_agent_surface_clearance": 0.3 - 0.2 * step,
                    "min_obstacle_surface_clearance": 2.0,
                    "min_boundary_center_clearance": 4.0,
                },
                "pursuers": [{
                    "radial_velocity_outward": -0.2,
                    "tangential_velocity_ccw": 0.4,
                }],
            },
            "actions": [{
                "a": 0.4,
                "omega": (-1.0) ** step * 0.5,
                "a_saturated": True,
                "omega_saturated": True,
                "omega_sign_flip": step > 1,
            }],
            "collision_events": [{"type": "agent_agent"}] if step == 3 else [],
        })
    summary = summarize_trace(frames)
    assert summary["max_num_in_ring"] == 3
    assert summary["max_2plus_ring_hold_steps"] == 3
    assert summary["max_3plus_ring_hold_steps"] == 2
    assert summary["agent_agent_surface_clearance"]["min"] < 0.0
    assert summary["a_saturation_rate"] == 1.0
    assert math.isclose(summary["omega_sign_flip_rate"], 2.0 / 3.0)
    assert summary["collision_event_count"] == 1
