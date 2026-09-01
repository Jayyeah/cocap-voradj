from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocap_voradj.models.iqn import CoCapIQN
from tools.rollout_voradj_visual import load_config, render_gif, rollout_scenario, write_outputs


SCENARIOS = ["capture", "coverage", "ab", "ba", "mix"]


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def coverage_rollout_config(base: Dict[str, Any], episode_index: int) -> tuple[Dict[str, Any], str]:
    cfg = copy.deepcopy(base)
    task_override = ((base.get("tasks", {}) or {}).get("voradj_coverage", {}) or {})
    cfg = deep_update(cfg, task_override)
    cfg.setdefault("env", {})["num_evaders"] = 0
    if int(episode_index) % 2 == 0:
        cfg["env"]["pursuer_spawn_mode"] = "map_random"
        return cfg, "map_random"
    cfg["env"]["pursuer_spawn_mode"] = "inner_random_cluster"
    return cfg, "inner_random_cluster"


def rotation_stats(frames: List[Dict[str, Any]]) -> Dict[str, float]:
    sign_consistency: List[float] = []
    abs_angle_delta: List[float] = []
    centroid_moves: List[float] = []
    radius_cvs: List[float] = []
    prev = None
    for frame in frames:
        current: Dict[int, np.ndarray] = {}
        for item in frame.get("pursuers", []):
            if bool(item.get("active", False)):
                current[int(item["id"])] = np.asarray([float(item["x"]), float(item["y"])], dtype=float)
        if len(current) < 2:
            prev = None
            continue
        arr = np.stack(list(current.values()))
        centroid = arr.mean(axis=0)
        radii = np.linalg.norm(arr - centroid, axis=1)
        radius_cvs.append(float(radii.std() / max(radii.mean(), 1e-6)))
        if prev is not None:
            prev_centroid, prev_points = prev
            deltas = []
            for idx, point in current.items():
                if idx not in prev_points:
                    continue
                old_vec = prev_points[idx] - prev_centroid
                new_vec = point - centroid
                old_angle = math.atan2(float(old_vec[1]), float(old_vec[0]))
                new_angle = math.atan2(float(new_vec[1]), float(new_vec[0]))
                delta = (new_angle - old_angle + math.pi) % (2.0 * math.pi) - math.pi
                if abs(delta) > 1e-3:
                    deltas.append(delta)
            if len(deltas) >= 2:
                sign_sum = sum(1 if delta > 0 else -1 for delta in deltas)
                sign_consistency.append(abs(sign_sum) / len(deltas))
                abs_angle_delta.append(float(np.mean(np.abs(deltas))))
                centroid_moves.append(float(np.linalg.norm(centroid - prev_centroid)))
        prev = (centroid, current)
    return {
        "rotation_sign_consistency": float(np.mean(sign_consistency)) if sign_consistency else 0.0,
        "rotation_high_consistency_frac": float(np.mean([value >= 0.75 for value in sign_consistency])) if sign_consistency else 0.0,
        "mean_abs_angle_delta": float(np.mean(abs_angle_delta)) if abs_angle_delta else 0.0,
        "mean_centroid_move": float(np.mean(centroid_moves)) if centroid_moves else 0.0,
        "mean_radius_cv": float(np.mean(radius_cvs)) if radius_cvs else 0.0,
    }


def trajectory_stats(frames: List[Dict[str, Any]]) -> Dict[str, float]:
    if not frames:
        return {
            "initial_active_pursuers": 0.0,
            "final_active_pursuers": 0.0,
            "min_active_pursuers": 0.0,
            "mean_active_pursuers": 0.0,
            "initial_radius": 0.0,
            "final_radius": 0.0,
            "radius_growth": 0.0,
            "radius_ratio": 0.0,
            "mean_path_length": 0.0,
            "mean_start_end_displacement": 0.0,
            "final_step_mean_displacement": 0.0,
            "final_step_max_displacement": 0.0,
            "stopped_near_start": 0.0,
            "expanded_then_stopped": 0.0,
        }

    def positions(frame: Dict[str, Any]) -> Dict[int, np.ndarray]:
        return {
            int(item["id"]): np.asarray([float(item["x"]), float(item["y"])], dtype=float)
            for item in frame.get("pursuers", [])
            if bool(item.get("active", False))
        }

    def mean_radius(points: Dict[int, np.ndarray]) -> float:
        if len(points) < 2:
            return 0.0
        arr = np.stack(list(points.values()))
        centroid = arr.mean(axis=0)
        return float(np.mean(np.linalg.norm(arr - centroid, axis=1)))

    first = positions(frames[0])
    last = positions(frames[-1])
    active_counts = [len(positions(frame)) for frame in frames]
    path_lengths: Dict[int, float] = {}
    previous: Dict[int, np.ndarray] = {}
    for frame in frames:
        current = positions(frame)
        for idx, point in current.items():
            if idx in previous:
                path_lengths[idx] = path_lengths.get(idx, 0.0) + float(np.linalg.norm(point - previous[idx]))
        previous = current

    shared = sorted(set(first).intersection(last))
    start_end_displacements = [float(np.linalg.norm(last[idx] - first[idx])) for idx in shared]
    if len(frames) >= 2:
        penultimate = positions(frames[-2])
        final_step = [
            float(np.linalg.norm(last[idx] - penultimate[idx]))
            for idx in sorted(set(last).intersection(penultimate))
        ]
    else:
        final_step = []
    initial_radius = mean_radius(first)
    final_radius = mean_radius(last)
    radius_growth = final_radius - initial_radius
    mean_path_length = float(np.mean(list(path_lengths.values()))) if path_lengths else 0.0
    mean_start_end_displacement = float(np.mean(start_end_displacements)) if start_end_displacements else 0.0
    final_step_mean_displacement = float(np.mean(final_step)) if final_step else 0.0
    final_step_max_displacement = float(np.max(final_step)) if final_step else 0.0
    stopped = final_step_mean_displacement <= 0.05
    return {
        "initial_active_pursuers": float(len(first)),
        "final_active_pursuers": float(len(last)),
        "min_active_pursuers": float(min(active_counts)) if active_counts else 0.0,
        "mean_active_pursuers": float(np.mean(active_counts)) if active_counts else 0.0,
        "initial_radius": initial_radius,
        "final_radius": final_radius,
        "radius_growth": radius_growth,
        "radius_ratio": float(final_radius / max(initial_radius, 1e-6)),
        "mean_path_length": mean_path_length,
        "mean_start_end_displacement": mean_start_end_displacement,
        "final_step_mean_displacement": final_step_mean_displacement,
        "final_step_max_displacement": final_step_max_displacement,
        "stopped_near_start": float(bool(stopped and mean_path_length <= 5.0 and mean_start_end_displacement <= 2.0)),
        "expanded_then_stopped": float(bool(stopped and radius_growth >= 5.0 and mean_path_length >= 10.0)),
    }


def coverage_cv_stats(frames: List[Dict[str, Any]], threshold: float = 0.15) -> Dict[str, float]:
    values: List[float] = []
    for frame in frames:
        if str(frame.get("phase", "")) != "coverage":
            continue
        try:
            value = float(frame.get("voronoi_cv", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    final_cv = float(values[-1]) if values else float("nan")
    best_cv = float(min(values)) if values else float("nan")
    return {
        "coverage_cv015_threshold": float(threshold),
        "final_voronoi_cv": final_cv,
        "best_voronoi_cv": best_cv,
        "coverage_cv015_success": float(bool(math.isfinite(final_cv) and final_cv <= threshold)),
        "coverage_cv015_best_success": float(bool(math.isfinite(best_cv) and best_cv <= threshold)),
    }


def settle_summary_for_scenario(scenario: str, phase_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if scenario in {"ab", "ba"}:
        return next((item for item in phase_summaries if str(item.get("phase")) == "coverage"), {})
    return next((item for item in reversed(phase_summaries) if "coverage_settled_success" in item), {})


def coverage_quality_stats(frames: List[Dict[str, Any]]) -> Dict[str, float]:
    coverage = [frame for frame in frames if str(frame.get("phase", "")) == "coverage"]

    def finite_values(key: str) -> List[float]:
        values = []
        for frame in coverage:
            try:
                value = float(frame.get(key, float("nan")))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    ce_rms = finite_values("ce_center_rms")
    ce_max = finite_values("ce_center_max")
    speed_rms = finite_values("active_speed_rms")
    speed_mean = finite_values("active_speed_mean")
    speed_max = finite_values("active_speed_max")
    return {
        "final_ce_center_rms": ce_rms[-1] if ce_rms else float("nan"),
        "best_ce_center_rms": min(ce_rms) if ce_rms else float("nan"),
        "final_ce_center_max": ce_max[-1] if ce_max else float("nan"),
        "best_ce_center_max": min(ce_max) if ce_max else float("nan"),
        "avg_active_speed_rms": float(np.mean(speed_rms)) if speed_rms else 0.0,
        "final_active_speed_rms": speed_rms[-1] if speed_rms else 0.0,
        "avg_active_speed_mean": float(np.mean(speed_mean)) if speed_mean else 0.0,
        "final_active_speed_mean": speed_mean[-1] if speed_mean else 0.0,
        "max_active_speed": max(speed_max) if speed_max else 0.0,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = max(len(records), 1)

    def rate(key: str) -> float:
        return sum(bool(record.get(key, False)) for record in records) / total

    def mean_value(key: str) -> float:
        values: List[float] = []
        for record in records:
            try:
                value = float(record.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else 0.0

    return {
        "episodes": len(records),
        "coverage_init_map_random_count": sum(str(record.get("coverage_init_source", "")) == "map_random" for record in records),
        "coverage_init_inner_random_cluster_count": sum(str(record.get("coverage_init_source", "")) == "inner_random_cluster" for record in records),
        "capture_success_rate": rate("capture_success_bool"),
        "coverage_success_rate": rate("coverage_success_bool"),
        "episode_success_rate": rate("episode_success_bool"),
        "collision_rate": rate("collision_event"),
        "boundary_collision_rate": rate("boundary_collision_event"),
        "soft_oob_rate": rate("soft_oob_event"),
        "zone_demo_episode_rate": rate("zone_demo_enabled"),
        "zone_breach_rate": rate("zone_breach_event"),
        "zone_entered_inner_rate": rate("zone_any_evader_entered_inner"),
        "zone_exit_after_entry_rate": rate("zone_exit_after_entry_event"),
        "zone_pursuer_left_inner_rate": rate("zone_pursuer_left_inner_event"),
        "coverage_geometric_rate": rate("coverage_geometric_success"),
        "coverage_settled_rate": rate("coverage_settled_success"),
        "coverage_cv015_rate": rate("coverage_cv015_success"),
        "coverage_cv015_best_rate": rate("coverage_cv015_best_success"),
        "avg_final_voronoi_cv": mean_value("final_voronoi_cv"),
        "avg_best_voronoi_cv": mean_value("best_voronoi_cv"),
        "coverage_settle_timeout_rate": rate("coverage_settle_timeout"),
        "avg_steps": mean_value("steps"),
        "avg_rollout_wall_seconds": mean_value("rollout_wall_seconds"),
        "avg_settle_elapsed_steps": mean_value("settle_elapsed_steps"),
        "avg_coverage_success_step": mean_value("coverage_success_step"),
        "avg_episode_end_mean_speed": mean_value("episode_end_mean_speed"),
        "avg_episode_end_max_speed": mean_value("episode_end_max_speed"),
        "avg_initial_active_pursuers": mean_value("initial_active_pursuers"),
        "avg_final_active_pursuers": mean_value("final_active_pursuers"),
        "avg_min_active_pursuers": mean_value("min_active_pursuers"),
        "avg_mean_active_pursuers": mean_value("mean_active_pursuers"),
        "avg_initial_radius": mean_value("initial_radius"),
        "avg_final_radius": mean_value("final_radius"),
        "avg_radius_growth": mean_value("radius_growth"),
        "avg_radius_ratio": mean_value("radius_ratio"),
        "avg_mean_path_length": mean_value("mean_path_length"),
        "avg_mean_start_end_displacement": mean_value("mean_start_end_displacement"),
        "avg_final_step_mean_displacement": mean_value("final_step_mean_displacement"),
        "avg_final_step_max_displacement": mean_value("final_step_max_displacement"),
        "stopped_near_start_rate": mean_value("stopped_near_start"),
        "expanded_then_stopped_rate": mean_value("expanded_then_stopped"),
        "rotation_sign_consistency": mean_value("rotation_sign_consistency"),
        "rotation_high_consistency_frac": mean_value("rotation_high_consistency_frac"),
        "mean_abs_angle_delta": mean_value("mean_abs_angle_delta"),
        "mean_centroid_move": mean_value("mean_centroid_move"),
        "mean_radius_cv": mean_value("mean_radius_cv"),
        "avg_final_ce_center_rms": mean_value("final_ce_center_rms"),
        "avg_best_ce_center_rms": mean_value("best_ce_center_rms"),
        "avg_final_ce_center_max": mean_value("final_ce_center_max"),
        "avg_best_ce_center_max": mean_value("best_ce_center_max"),
        "avg_active_speed_rms": mean_value("avg_active_speed_rms"),
        "avg_final_active_speed_rms": mean_value("final_active_speed_rms"),
        "avg_active_speed_mean": mean_value("avg_active_speed_mean"),
        "avg_final_active_speed_mean": mean_value("final_active_speed_mean"),
        "max_active_speed": max(
            [float(record.get("max_active_speed", 0.0)) for record in records] or [0.0]
        ),
        "avg_episode_return_mean": mean_value("episode_return_mean"),
        "avg_episode_return_sum": mean_value("episode_return_sum"),
        "reward_component_means": {
            key: float(np.mean([
                float((record.get("reward_component_means", {}) or {}).get(key, 0.0))
                for record in records
            ]))
            for key in sorted({
                key
                for record in records
                for key in (record.get("reward_component_means", {}) or {})
            })
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gif-count", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--capture-max-steps", type=int, default=600)
    parser.add_argument("--coverage-max-steps", type=int, default=900)
    parser.add_argument("--capture-evaders", type=int, default=None, help="Override capture/mix evader count; defaults to env.num_evaders from the config.")
    parser.add_argument("--max-gif-frames", type=int, default=1000)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--draw-neighbor-edges", dest="draw_neighbor_edges", action="store_true", default=False)
    parser.add_argument("--no-neighbor-edges", dest="draw_neighbor_edges", action="store_false")
    parser.add_argument("--draw-trails", action="store_true", default=False)
    parser.add_argument("--draw-sensing-circles", dest="draw_sensing_circles", action="store_true", default=False)
    parser.add_argument("--no-sensing-circles", dest="draw_sensing_circles", action="store_false")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_wall_start = time.perf_counter()
    cfg = load_config(Path(args.config))
    capture_evaders = int(args.capture_evaders) if args.capture_evaders is not None else int((cfg.get("env", {}) or {}).get("num_evaders", 1))
    model_load_start = time.perf_counter()
    model = CoCapIQN.load(str(Path(args.checkpoint)), device=str(args.device))
    model_load_seconds = time.perf_counter() - model_load_start
    model.eval()
    all_summaries: Dict[str, Any] = {}
    scenario_timings: Dict[str, Any] = {}

    with torch.no_grad():
        for scenario in list(args.scenarios):
            scenario_wall_start = time.perf_counter()
            scenario_dir = output_root / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            records: List[Dict[str, Any]] = []
            for idx in range(int(args.episodes)):
                seed = int(args.seed) + idx
                episode_cfg = cfg
                coverage_init_source = ""
                if scenario == "coverage":
                    episode_cfg, coverage_init_source = coverage_rollout_config(cfg, idx)
                episode_wall_start = time.perf_counter()
                frames, phase_summaries = rollout_scenario(
                    model,
                    episode_cfg,
                    scenario,
                    seed,
                    str(args.device),
                    int(args.max_steps),
                    int(args.capture_max_steps),
                    int(args.coverage_max_steps),
                    capture_evaders,
                )
                rollout_wall_seconds = time.perf_counter() - episode_wall_start
                final_status = frames[-1].get("display_status", {}) if frames else {}
                settle_summary = settle_summary_for_scenario(scenario, phase_summaries)
                zone_summary = next(
                    (
                        item.get("zone_metrics", {}) or {}
                        for item in reversed(phase_summaries)
                        if (item.get("zone_metrics", {}) or {}).get("zone_demo_enabled", False)
                    ),
                    {},
                )
                record = {
                    "scenario": scenario,
                    "seed": seed,
                    "steps": max(len(frames) - 1, 0),
                    "rollout_wall_seconds": rollout_wall_seconds,
                    "capture_success_bool": bool(final_status.get("capture_success_bool", False)),
                    "coverage_success_bool": bool(final_status.get("coverage_success_bool", False)),
                    "episode_success_bool": bool(final_status.get("episode_success_bool", False)),
                    "collision_event": bool(any(frame.get("collision_event", False) for frame in frames)),
                    "boundary_collision_event": bool(any(frame.get("boundary_collision_event", False) for frame in frames)),
                    "soft_oob_event": bool(any(frame.get("soft_oob_event", False) for frame in frames)),
                    "coverage_geometric_success": bool(settle_summary.get("coverage_geometric_success", False)),
                    "coverage_settled_success": bool(settle_summary.get("coverage_settled_success", False)),
                    "coverage_settle_timeout": bool(settle_summary.get("coverage_settle_timeout", False)),
                    "settle_elapsed_steps": int(settle_summary.get("settle_elapsed_steps", 0)),
                    "coverage_init_source": coverage_init_source,
                    "episode_end_mean_speed": float(settle_summary.get("episode_end_mean_speed", 0.0)),
                    "episode_end_max_speed": float(settle_summary.get("episode_end_max_speed", 0.0)),
                    "coverage_success_step": int(settle_summary.get("coverage_success_step", -1)),
                    "episode_return_mean": float(settle_summary.get("episode_return_mean", 0.0)),
                    "episode_return_sum": float(settle_summary.get("episode_return_sum", 0.0)),
                    "reward_component_means": dict(settle_summary.get("reward_component_means", {}) or {}),
                    "zone_demo_enabled": bool(zone_summary.get("zone_demo_enabled", False)),
                    "zone_breach_event": bool(zone_summary.get("zone_breach_event", False)),
                    "zone_any_evader_entered_inner": bool(zone_summary.get("zone_any_evader_entered_inner", False)),
                    "zone_exit_after_entry_event": bool(zone_summary.get("zone_exit_after_entry_event", False)),
                    "zone_pursuer_left_inner_event": bool(zone_summary.get("zone_pursuer_left_inner_event", False)),
                    "phase_summaries": phase_summaries,
                    **coverage_cv_stats(frames),
                    **coverage_quality_stats(frames),
                    **rotation_stats(frames),
                    **trajectory_stats(frames),
                }
                records.append(record)
                if idx < int(args.gif_count):
                    gif_path = scenario_dir / f"rollout_{scenario}_seed_{seed}.gif"
                    render_record = render_gif(
                        frames,
                        gif_path,
                        int(args.max_gif_frames),
                        70,
                        int(args.frame_duration_ms),
                        scenario,
                        bool(args.draw_neighbor_edges),
                        bool(args.draw_trails),
                        bool(args.draw_sensing_circles),
                        scenario == "coverage",
                    )
                    write_outputs(scenario_dir, scenario, seed, Path(args.config), Path(args.checkpoint), frames, phase_summaries, render_record)
                    if scenario == "coverage" and coverage_init_source:
                        summary_path = scenario_dir / f"summary_{scenario}_seed_{seed}.json"
                        episode_path = scenario_dir / f"episode_{scenario}_seed_{seed}.json"
                        if summary_path.is_file():
                            payload = json.loads(summary_path.read_text(encoding="utf-8"))
                            payload["coverage_init_source"] = coverage_init_source
                            summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                        if episode_path.is_file():
                            payload = json.loads(episode_path.read_text(encoding="utf-8"))
                            payload.setdefault("rollout", {})["coverage_init_source"] = coverage_init_source
                            episode_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            scenario_wall_seconds = time.perf_counter() - scenario_wall_start
            payload = {"scenario": scenario, "summary": summarize(records), "records": records}
            scenario_timing = {
                "episodes": len(records),
                "wall_seconds": scenario_wall_seconds,
                "avg_wall_seconds_per_episode": scenario_wall_seconds / max(len(records), 1),
                "avg_rollout_wall_seconds_per_episode": payload["summary"].get("avg_rollout_wall_seconds", 0.0),
            }
            payload["timing"] = scenario_timing
            (scenario_dir / "batch_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            all_summaries[scenario] = payload["summary"]
            scenario_timings[scenario] = scenario_timing
            print(json.dumps({"scenario": scenario, "summary": payload["summary"], "timing": scenario_timing}, ensure_ascii=False), flush=True)
    total_wall_seconds = time.perf_counter() - run_wall_start
    total_episode_count = sum(int(item.get("episodes", 0)) for item in scenario_timings.values())
    timing_payload = {
        "device": str(args.device),
        "checkpoint": str(args.checkpoint),
        "output_root": str(args.output_root),
        "model_load_seconds": model_load_seconds,
        "total_wall_seconds": total_wall_seconds,
        "total_rollout_episodes": total_episode_count,
        "avg_wall_seconds_per_rollout_episode": total_wall_seconds / max(total_episode_count, 1),
        "avg_scenario_wall_seconds_per_rollout_episode": (sum(float(item.get("wall_seconds", 0.0)) for item in scenario_timings.values()) / max(total_episode_count, 1)),
        "scenarios": scenario_timings,
    }
    (output_root / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "timing.json").write_text(json.dumps(timing_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "run_args.json").write_text(json.dumps(vars(args) | {"capture_evaders": capture_evaders, "timing": timing_payload}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
