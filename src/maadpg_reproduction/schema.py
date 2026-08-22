"""Stable observation/action schema identity stored in every run/checkpoint."""

from __future__ import annotations

import hashlib
import json


OBSERVATION_ACTION_SCHEMA = {
    "spec_version": "maadpg-paper-v1",
    "agent_slots": ["p0", "p1", "p2"],
    "pursuer_observation": {
        "shape": [3, 26],
        "fields": [
            "self_position_div_L[2]",
            "self_velocity_div_vmax[2]",
            "other_absolute_positions_div_L_fixed_slot_order[4]",
            "lidar_ccw_from_heading_div_range[16]",
            "target_relative_distance_div_4km[1]",
            "target_world_bearing_rad[1]",
        ],
    },
    "joint_action": {
        "shape": [3, 2],
        "frame": "world",
        "meaning": "normalized_acceleration_ax_ay",
        "canonical_projection": "component_box_then_l2_unit_disk",
        "replay_value": "exact_environment_executed_action",
    },
    "joint_critic_input": {
        "shape": [84],
        "formula": "3*26 observations + 3*2 actions",
    },
}


def observation_action_schema_sha256() -> str:
    payload = json.dumps(
        OBSERVATION_ACTION_SCHEMA,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
