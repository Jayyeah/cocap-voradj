import numpy as np

from maadpg_reproduction.collision_audit import (
    audit_swept_collisions,
    moving_points_minimum_distance,
    point_segment_distance,
)


def test_swept_obstacle_detects_a_tunnel_missed_by_endpoints():
    assert point_segment_distance([0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]) == 0.0
    audit = audit_swept_collisions(
        previous_pursuers=[[-1.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        current_pursuers=[[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        previous_target=[5.0, 5.0],
        current_target=[5.0, 5.0],
        obstacles=[([0.0, 0.0], 0.1)],
        obstacle_clearance=0.02,
        teammate_separation=0.04,
        target_hard_radius=0.02,
    )
    assert audit.obstacle_by_agent.tolist() == [True, False, False]


def test_synchronous_swept_teammate_and_target_crossings_are_reported():
    assert moving_points_minimum_distance([-1, 0], [1, 0], [1, 0], [-1, 0]) == 0.0
    audit = audit_swept_collisions(
        previous_pursuers=[[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
        current_pursuers=[[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]],
        previous_target=[0.0, -1.0],
        current_target=[0.0, 1.0],
        obstacles=[],
        obstacle_clearance=0.02,
        teammate_separation=0.04,
        target_hard_radius=0.02,
    )
    assert audit.teammate_by_agent.tolist() == [True, True, False]
    assert audit.target_by_agent.tolist() == [True, True, False]
    assert audit.any_by_agent.tolist() == [True, True, False]
