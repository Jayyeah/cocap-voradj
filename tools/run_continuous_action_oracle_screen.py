"""Deterministic scripted screen for the continuous pursuer action contract.

This is a P2 oracle only: it does not construct an RL policy or start training.
It compares explicit ax/ay candidates under the same v_max, decision period,
seeds and normalized acceleration sequences, and writes a JSON artifact.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from cocap_voradj.config import ConfigManager
from cocap_voradj.dynamics.continuous_action import AccelerationActionAdapter
from cocap_voradj.envs.base import Obstacle
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/continuous_marl_20260804/p1_acceleration_contract_4v1.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/2026-08-04_continuous_marl_refactor/p2_a_max_screen.json"
SEED = 2026080401
V_MAX = 3.0
DECISION_DT = 0.5
CANDIDATES = (0.4, 0.8, 1.6)
# Keep every pair farther apart than the project reset contract
# (pursuer_spawn_min_sep=15 m), so reset does not silently replace scripted
# positions with random near-boundary positions.
POSITIONS = [(20.0, 20.0), (20.0, 50.0), (50.0, 20.0), (50.0, 50.0)]


def _sequence(*chunks: tuple[int, Sequence[float]]) -> List[np.ndarray]:
    commands: List[np.ndarray] = []
    for count, command in chunks:
        commands.extend([np.asarray(command, dtype=float)] * int(count))
    return commands


SCENARIOS: Dict[str, List[np.ndarray]] = {
    # Explicit acceleration semantics: zero acceleration holds velocity, so
    # every scripted stop uses a reverse-acceleration braking segment.
    "ce_step_response": _sequence((4, (1.0, 0.0)), (4, (-1.0, 0.0)), (12, (0.0, 0.0))),
    "turn_90": _sequence((3, (1.0, 0.0)), (3, (0.0, 1.0)), (3, (0.0, -1.0)), (3, (-1.0, 0.0)), (6, (0.0, 0.0))),
    "turn_180": _sequence((3, (1.0, 0.0)), (6, (-1.0, 0.0)), (3, (1.0, 0.0)), (6, (0.0, 0.0))),
    "ring_radial_tangential": _sequence(
        (4, (0.8, 0.0)),
        (4, (0.0, 0.8)),
        (4, (-0.8, 0.0)),
        (4, (0.0, -0.8)),
        (6, (0.0, 0.0)),
    ),
    "boundary_brake": _sequence((4, (-1.0, 0.0)), (4, (1.0, 0.0)), (12, (0.0, 0.0))),
    "obstacle_brake": _sequence((4, (1.0, 0.0)), (4, (-1.0, 0.0)), (12, (0.0, 0.0))),
}


def _build_config(a_max: float, num_obstacles: int = 0) -> Dict[str, Any]:
    config = copy.deepcopy(load_config(str(CONFIG_PATH)))
    config["a_max"] = float(a_max)
    config["decision_dt"] = DECISION_DT
    config["v_max"] = V_MAX
    config["env"]["num_pursuers"] = 4
    config["env"]["num_evaders"] = 0
    config["env"]["num_obstacles"] = int(num_obstacles)
    config["env"]["episode_max_length"] = 100
    ConfigManager.get_instance().update_config(config)
    return config


def _reset_env(env: VorAdjEnv, scenario: str) -> None:
    positions = list(POSITIONS)
    if scenario == "boundary_brake":
        positions[0] = (20.0, 20.0)
    elif scenario == "obstacle_brake":
        positions[0] = (20.0, 20.0)
    env.reset(initial_pursuer_positions=positions)
    if scenario == "obstacle_brake":
        env.obstacles = [Obstacle(32.0, 20.0, 2.0)]
        env._refresh_collisions()


def _command_for_agents(
    adapter: AccelerationActionAdapter,
    desired: Sequence[Sequence[float] | None],
    a_max: float,
) -> tuple[List[Sequence[float] | None], List[Dict[str, Any]]]:
    final: List[Sequence[float] | None] = []
    diagnostics: List[Dict[str, Any]] = []
    for idx, raw in enumerate(desired):
        if raw is None:
            final.append(None)
            diagnostics.append({})
            continue
        action, diag = adapter.validate_with_diagnostics(np.asarray(raw, dtype=float) * float(a_max))
        final.append(action.tolist())
        diagnostics.append(diag.as_dict())
    return final, diagnostics


def _run_scenario(a_max: float, scenario: str) -> Dict[str, Any]:
    config = _build_config(a_max, num_obstacles=0)
    env = VorAdjEnv(config, seed=SEED)
    _reset_env(env, scenario)
    if scenario == "obstacle_brake":
        env._refresh_collisions()
    adapter = AccelerationActionAdapter(a_max, DECISION_DT)
    previous_acceleration = [0.0 for _ in env.pursuers]
    path_length = np.zeros(len(env.pursuers), dtype=float)
    max_speed = 0.0
    max_actual_acceleration = 0.0
    max_jerk = 0.0
    rejection_count = 0
    validation_delta_sum = 0.0
    speed_limited_count = 0
    min_clearance = math.inf
    nan_detected = False
    command_steps = SCENARIOS[scenario]
    for desired_common in command_steps:
        desired = [desired_common.tolist()] * len(env.pursuers)
        before = np.asarray([[p.x, p.y] for p in env.pursuers], dtype=float)
        actions, _ = _command_for_agents(adapter, desired, a_max)
        result = env.step(actions, [])
        after = np.asarray([[p.x, p.y] for p in env.pursuers], dtype=float)
        path_length += np.linalg.norm(after - before, axis=1)
        for idx, info in enumerate(result.infos):
            diagnostics = info.get("action_diagnostics", {})
            actual_acceleration = float(diagnostics.get("actual_acceleration", 0.0))
            jerk = float(abs(actual_acceleration - previous_acceleration[idx]) / DECISION_DT)
            previous_acceleration[idx] = actual_acceleration
            rejection_count += int(diagnostics.get("action_rejected", False))
            validation_delta_sum += float(diagnostics.get("validation_delta", 0.0))
            speed_limited_count += int(bool(diagnostics.get("speed_limited", False)))
            max_actual_acceleration = max(max_actual_acceleration, actual_acceleration)
            max_jerk = max(max_jerk, jerk)
            max_speed = max(max_speed, float(diagnostics.get("speed_after", 0.0)))
            nan_detected |= any(not np.isfinite(float(value)) for value in diagnostics.values() if isinstance(value, (float, int)))
        for idx in range(len(env.pursuers)):
            min_clearance = min(min_clearance, float(env._minimum_clearance(idx)))
    record = env.episode_record()
    collision = bool(record["collision_event"])
    expected_no_collision = scenario in {"ce_step_response", "turn_90", "turn_180", "ring_radial_tangential", "boundary_brake", "obstacle_brake"}
    return {
        "a_max": float(a_max),
        "scenario": scenario,
        "seed": SEED,
        "steps": len(command_steps),
        "v_max": V_MAX,
        "decision_dt": DECISION_DT,
        "max_speed": float(max_speed),
        "max_actual_acceleration": float(max_actual_acceleration),
        "max_jerk": float(max_jerk),
        "action_rejection_count": int(rejection_count),
        "action_validation_failure_rate": float(rejection_count / max(len(command_steps) * len(env.pursuers), 1)),
        "validation_delta_mean": float(validation_delta_sum / max(rejection_count, 1)),
        "speed_limited_rate": float(speed_limited_count / max(len(command_steps) * len(env.pursuers), 1)),
        "path_length_mean": float(np.mean(path_length)),
        "path_length_max": float(np.max(path_length)),
        "min_clearance": float(min_clearance),
        "collision_event": collision,
        "boundary_collision_event": bool(record["boundary_collision_event"]),
        "nan_detected": bool(nan_detected),
        "contract_ok": bool(
            max_actual_acceleration <= a_max + 1e-8
            and max_speed <= V_MAX + 1e-8
            and not nan_detected
        ),
        "no_collision_ok": bool((not expected_no_collision) or not collision),
        "episode_record": record,
    }


def run_screen(a_max_values: Iterable[float]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for a_max in a_max_values:
        for scenario in SCENARIOS:
            rows.append(_run_scenario(float(a_max), scenario))
    summary: List[Dict[str, Any]] = []
    for a_max in a_max_values:
        candidate = [row for row in rows if row["a_max"] == float(a_max)]
        summary.append(
            {
                "a_max": float(a_max),
                "contract_ok": bool(all(row["contract_ok"] for row in candidate)),
                "no_collision_ok": bool(all(row["no_collision_ok"] for row in candidate)),
                "nan_free": bool(all(not row["nan_detected"] for row in candidate)),
                "max_speed": float(max(row["max_speed"] for row in candidate)),
                "max_actual_acceleration": float(max(row["max_actual_acceleration"] for row in candidate)),
                "max_jerk": float(max(row["max_jerk"] for row in candidate)),
                "action_validation_failure_rate_mean": float(np.mean([row["action_validation_failure_rate"] for row in candidate])),
                "min_clearance": float(min(row["min_clearance"] for row in candidate)),
                "path_length_mean": float(np.mean([row["path_length_mean"] for row in candidate])),
                "candidate_gate": bool(all(row["contract_ok"] and row["no_collision_ok"] for row in candidate)),
            }
        )
    return {
        "schema_version": 1,
        "kind": "continuous_action_p2_oracle_screen",
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "seed": SEED,
        "v_max": V_MAX,
        "decision_dt": DECISION_DT,
        "a_max_candidates": [float(value) for value in a_max_values],
        "scenarios": list(SCENARIOS),
        "summary": summary,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = run_screen(CANDIDATES)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"artifact={output}")
    return 0 if all(row["candidate_gate"] for row in payload["summary"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
