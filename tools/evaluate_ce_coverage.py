from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import safe_json_dumps, set_global_config


def _load_eval_config(run_dir: Path) -> Dict[str, Any]:
    with (run_dir / "effective_config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config.setdefault("env", {})["num_evaders"] = 0
    config.setdefault("zone_demo", {})["enabled"] = False
    config.setdefault("tasks", {}).setdefault("voradj", {}).setdefault("env", {})["num_evaders"] = 0
    if config.get("reward", {}).get("coverage_objective_version") != "centroid_energy_v0":
        raise ValueError(f"Not a CE-Coverage run: {run_dir}")
    return config


def _active_actions(
    model: CoCapIQN,
    observations: List[Optional[Dict[str, np.ndarray]]],
    device: str,
) -> List[Optional[int]]:
    active = [index for index, observation in enumerate(observations) if observation is not None]
    actions: List[Optional[int]] = [None] * len(observations)
    if active:
        batch = stack_obs([observations[index] for index in active], device)
        selected = model.act(
            batch,
            mode="voradj",
            epsilon=0.0,
            deterministic_quantiles=True,
        ).detach().cpu().tolist()
        for index, action in zip(active, selected):
            actions[index] = int(action)
    return actions


def evaluate_episode(
    model: CoCapIQN,
    config: Dict[str, Any],
    seed: int,
    device: str,
    max_steps: int,
    post_success_steps: int,
) -> Dict[str, Any]:
    set_global_config(config)
    env = VorAdjEnv(config, seed=seed)
    observations = env.reset()
    reward_scale = float(config["reward"].get("coverage_ce_reward_scale", 10.0))
    center_cost_sum = 0.0
    control_cost_sum = 0.0
    objective_speed_sq_sum = 0.0
    objective_acceleration_sq_sum = 0.0
    objective_angular_velocity_sq_sum = 0.0
    objective_active_agent_steps = 0
    success_step: Optional[int] = None
    post_success_speeds: List[float] = []
    post_success_actions: List[int] = []
    post_success_center_rms: List[float] = []
    post_success_center_max: List[float] = []
    last_center_rms = float("inf")
    last_center_max = float("inf")
    min_center_rms = float("inf")
    min_center_max = float("inf")
    last_area_cv = float("inf")
    rms_threshold = float(config["reward"].get("coverage_ce_success_rms_threshold", 0.05))
    max_threshold = float(config["reward"].get("coverage_ce_success_max_threshold", 0.10))

    for _ in range(max_steps + post_success_steps):
        measuring_primary_episode = success_step is None
        actions = _active_actions(model, observations, device)
        active_before = [
            index for index, pursuer in enumerate(env.pursuers)
            if not pursuer.deactivated and actions[index] is not None
        ]
        result = env.step(actions, [])
        observations = result.observations
        metrics = env.last_distribution_metrics
        if measuring_primary_episode:
            costs = metrics.get("ce_center_costs", [])
            center_cost_sum += float(np.sum(costs)) if costs else 0.0
            control_reward = sum(
                float(info.get("replay_metadata", {}).get("reward_ce_control", 0.0))
                for info in result.infos
            )
            control_cost_sum += -control_reward / max(reward_scale, 1e-9)
            center_rms = float(metrics.get("ce_center_rms", float("inf")))
            center_max = float(metrics.get("ce_center_max", float("inf")))
            if int(metrics.get("participant_count", 0)) >= 2 and np.isfinite(center_rms):
                last_center_rms = center_rms
                last_center_max = center_max
                min_center_rms = min(min_center_rms, center_rms)
                min_center_max = min(min_center_max, center_max)
                last_area_cv = float(metrics.get("area_cv", float("inf")))
            for index in active_before:
                pursuer = env.pursuers[index]
                acceleration, angular_velocity = pursuer.action_list[int(actions[index])]
                max_acceleration = max(abs(float(value)) for value in pursuer.a)
                max_angular_velocity = max(abs(float(value)) for value in pursuer.w)
                objective_speed_sq_sum += (
                    float(pursuer.speed) / max(abs(float(pursuer.max_speed)), 1e-9)
                ) ** 2
                objective_acceleration_sq_sum += (
                    float(acceleration) / max(max_acceleration, 1e-9)
                ) ** 2
                objective_angular_velocity_sq_sum += (
                    float(angular_velocity) / max(max_angular_velocity, 1e-9)
                ) ** 2
                objective_active_agent_steps += 1

        if success_step is None and env.post_capture_coverage_success:
            success_step = int(env.episode_step)
        elif success_step is not None:
            active_speeds = [float(pursuer.speed) for pursuer in env.pursuers if not pursuer.deactivated]
            post_success_speeds.extend(active_speeds)
            post_success_actions.extend(int(action) for action in actions if action is not None)
            post_success_center_rms.append(
                float(metrics.get("ce_center_rms", float("inf")))
            )
            post_success_center_max.append(
                float(metrics.get("ce_center_max", float("inf")))
            )
            if env.episode_step - success_step >= post_success_steps:
                break

        if all(result.dones) and success_step is None:
            break
        if success_step is None and env.episode_step >= max_steps:
            break

    record = env.episode_record(task="ce_pure_random")
    idle_action = 4
    post_success_geometry_ok = [
        rms <= rms_threshold and maximum <= max_threshold
        for rms, maximum in zip(post_success_center_rms, post_success_center_max)
    ]
    return {
        "seed": int(seed),
        "success": bool(success_step is not None),
        "success_step": -1 if success_step is None else int(success_step),
        "collision": bool(record["collision_event"]),
        "active_pursuers": int(record["active_pursuers"]),
        "center_rms": last_center_rms,
        "center_max": last_center_max,
        "min_center_rms": min_center_rms,
        "min_center_max": min_center_max,
        "area_cv_diagnostic": last_area_cv,
        "episode_end_mean_speed": float(record["episode_end_mean_speed"]),
        "episode_end_max_speed": float(record["episode_end_max_speed"]),
        "center_cost_sum": float(center_cost_sum),
        "control_cost_sum": float(control_cost_sum),
        "objective_speed_sq_sum": float(objective_speed_sq_sum),
        "objective_acceleration_sq_sum": float(objective_acceleration_sq_sum),
        "objective_angular_velocity_sq_sum": float(objective_angular_velocity_sq_sum),
        "objective_active_agent_steps": int(objective_active_agent_steps),
        "post_success_observation_steps": int(post_success_steps if success_step is not None else 0),
        "post_success_mean_speed": float(np.mean(post_success_speeds)) if post_success_speeds else float("inf"),
        "post_success_max_speed": float(max(post_success_speeds)) if post_success_speeds else float("inf"),
        "post_success_idle_action_ratio": (
            float(np.mean(np.asarray(post_success_actions) == idle_action))
            if post_success_actions
            else 0.0
        ),
        "post_success_geometry_retained_ratio": (
            float(np.mean(post_success_geometry_ok)) if post_success_geometry_ok else 0.0
        ),
        "post_success_final_geometry_retained": (
            bool(post_success_geometry_ok[-1]) if post_success_geometry_ok else False
        ),
        "post_success_max_center_rms": (
            float(max(post_success_center_rms)) if post_success_center_rms else float("inf")
        ),
        "post_success_max_center_max": (
            float(max(post_success_center_max)) if post_success_center_max else float("inf")
        ),
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    successes = [record for record in records if record["success"]]

    def mean(key: str, values: List[Dict[str, Any]] = records) -> float:
        finite = [float(record[key]) for record in values if np.isfinite(float(record[key]))]
        return float(np.mean(finite)) if finite else float("inf")

    return {
        "episodes": len(records),
        "success_rate": float(np.mean([record["success"] for record in records])) if records else 0.0,
        "collision_rate": float(np.mean([record["collision"] for record in records])) if records else 0.0,
        "avg_success_step": mean("success_step", successes),
        "avg_center_rms": mean("center_rms"),
        "avg_center_max": mean("center_max"),
        "avg_min_center_rms": mean("min_center_rms"),
        "avg_min_center_max": mean("min_center_max"),
        "avg_center_cost_sum": mean("center_cost_sum"),
        "avg_control_cost_sum": mean("control_cost_sum"),
        "avg_objective_speed_sq_sum": mean("objective_speed_sq_sum"),
        "avg_objective_acceleration_sq_sum": mean("objective_acceleration_sq_sum"),
        "avg_objective_angular_velocity_sq_sum": mean("objective_angular_velocity_sq_sum"),
        "avg_objective_active_agent_steps": mean("objective_active_agent_steps"),
        "avg_episode_end_mean_speed": mean("episode_end_mean_speed"),
        "avg_episode_end_max_speed": mean("episode_end_max_speed"),
        "avg_post_success_mean_speed": mean("post_success_mean_speed", successes),
        "avg_post_success_max_speed": mean("post_success_max_speed", successes),
        "avg_post_success_idle_action_ratio": mean("post_success_idle_action_ratio", successes),
        "avg_post_success_geometry_retained_ratio": mean(
            "post_success_geometry_retained_ratio", successes
        ),
        "post_success_final_geometry_retained_rate": (
            float(np.mean([
                record["post_success_final_geometry_retained"] for record in successes
            ]))
            if successes
            else 0.0
        ),
        "avg_post_success_max_center_rms": mean(
            "post_success_max_center_rms", successes
        ),
        "avg_post_success_max_center_max": mean(
            "post_success_max_center_max", successes
        ),
    }


def evaluate_checkpoint(
    run_dir: Path,
    checkpoint: Path,
    episodes: int,
    seed: int,
    device: str,
    max_steps: int,
    post_success_steps: int,
) -> Dict[str, Any]:
    config = _load_eval_config(run_dir)
    model = CoCapIQN.load(str(checkpoint), device=device)
    model.eval()
    records = [
        evaluate_episode(
            model,
            config,
            seed=seed + episode,
            device=device,
            max_steps=max_steps,
            post_success_steps=post_success_steps,
        )
        for episode in range(episodes)
    ]
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "seed_start": int(seed),
        "max_steps": int(max_steps),
        "post_success_steps": int(post_success_steps),
        "policy_quantile_mode": "fixed_midpoint_32",
        "summary": summarize(records),
        "episodes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one CE-Coverage checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026072900)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--post-success-steps", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = evaluate_checkpoint(
        Path(args.run_dir),
        Path(args.checkpoint),
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        post_success_steps=args.post_success_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(safe_json_dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(safe_json_dumps(payload["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
