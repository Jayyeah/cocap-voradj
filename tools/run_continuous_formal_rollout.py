"""Formal 20-rollout / 10-GIF evaluation for continuous P6 checkpoints.

This mirrors the legacy ``batch_rollouts(_parallel)`` + ``rollout_voradj_visual``
output layout, but loads a continuous checkpoint (schema 2) and samples
deterministic continuous actions through the env action adapter. Evaders stay
APF. Scenario names in outputs keep the legacy labels ``capture/coverage/mix``
while the underlying scene configs are the P6 screening scenes
``capture/pure_ce/mixed_crms``.

Output layout (same as legacy formal):
  output_root/
    run_args.json, timing.json, all_summaries.json
    capture|coverage|mix/
      batch_summary.json
      summary_<scenario>_seed_<seed>.json
      episode_<scenario>_seed_<seed>.json
      rollout_<scenario>_seed_<seed>.gif   (first gif_count episodes)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.training.trainer import set_global_config
from tools.batch_rollouts import summarize
from tools.rollout_voradj_visual import (
    apply_display_status,
    coverage_success_from_record,
    phase_summary,
    render_gif,
    snapshot_env,
    status_text,
)
from tools.run_continuous_p6_screening import (
    _make_trainer,
    _sample_actions,
    _scene_config,
)


SCENARIO_ALIASES = {
    "coverage": "pure_ce",
    "capture": "capture",
    "mix": "mixed_crms",
}
DEFAULT_SCENARIOS = ["coverage", "capture", "mix"]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_continuous_trainer(checkpoint: Path, device: str) -> Tuple[Any, Dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = dict(payload["contract"])
    action_mode = (
        "aw"
        if contract["action_mode"] == "acceleration_angular_velocity_body"
        else "axay"
    )
    profile = str(contract.get("trainer_profile", "formal_p6"))
    if profile not in {"formal_p6", "smoke"}:
        raise ValueError(f"unsupported trainer_profile in checkpoint contract: {profile}")
    trainer = _make_trainer(
        "local",
        float(contract["a_max"]),
        device,
        action_mode=action_mode,
        w_max=float(contract.get("w_max", np.pi / 6.0)),
        trainer_profile=profile,
    )
    trainer.load_checkpoint(checkpoint, contract)
    trainer.actor.eval()
    return trainer, contract


def scene_config(
    scene: str,
    contract: Dict[str, Any],
    *,
    alternate_coverage_init: bool,
    episode_index: int,
) -> Dict[str, Any]:
    action_mode = (
        "aw"
        if contract["action_mode"] == "acceleration_angular_velocity_body"
        else "axay"
    )
    cfg = _scene_config(
        scene,
        float(contract["a_max"]),
        action_mode=action_mode,
        w_max=float(contract.get("w_max", np.pi / 6.0)),
        global_evader_visibility=bool(contract.get("global_evader_visibility", False)),
    )
    if scene == "pure_ce" and alternate_coverage_init:
        cfg.setdefault("env", {})["pursuer_spawn_mode"] = (
            "map_random" if int(episode_index) % 2 == 0 else "inner_random_cluster"
        )
        if cfg["env"]["pursuer_spawn_mode"] == "inner_random_cluster":
            cfg["env"]["spawn_cluster_radius"] = 10.0
            cfg["env"]["pursuer_spawn_min_sep"] = 7.0
    return cfg


def act_pursuers(
    trainer: Any,
    obs_list: List[Optional[Dict[str, np.ndarray]]],
    env: VorAdjEnv,
) -> List[Optional[np.ndarray]]:
    active_indices = [idx for idx, obs in enumerate(obs_list) if obs is not None]
    actions: List[Optional[np.ndarray]] = [None] * len(obs_list)
    if not active_indices:
        return actions
    active_obs = [obs_list[idx] for idx in active_indices]
    sampled, _ = _sample_actions(
        trainer,
        active_obs,
        env.action_adapter,
        deterministic=True,
    )
    for position, idx in enumerate(active_indices):
        actions[idx] = sampled[position]
    return actions


def act_evaders(env: VorAdjEnv, apf_agents: List[ApfAgent]) -> List[Optional[int]]:
    if hasattr(env, "configure_evader_apf_agents"):
        env.configure_evader_apf_agents(apf_agents)
    actions: List[Optional[int]] = []
    for evader_obs in env.get_evader_observations_for_apf():
        actions.append(None if evader_obs is None else int(apf_agents[len(actions)].act(evader_obs)))
    return actions


def rollout_phase(
    trainer: Any,
    cfg: Dict[str, Any],
    scenario: str,
    phase: str,
    phase_index: int,
    seed: int,
    max_steps: int,
    global_step_start: int,
    initial_positions: Optional[List[List[float]]] = None,
    initial_active: Optional[List[bool]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[List[float]], List[bool], int]:
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=seed)
    obs_list = env.reset(
        initial_pursuer_positions=initial_positions,
        initial_pursuer_active=initial_active,
    )
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    frames = [snapshot_env(env, scenario, phase, phase_index, 0, global_step_start)]
    action_norms: List[float] = []
    speeds: List[float] = []
    stopped_by_env_done = False
    for local_step in range(1, int(max_steps) + 1):
        set_global_config(cfg)
        actions = act_pursuers(trainer, obs_list, env)
        result = env.step(actions, act_evaders(env, apf_agents))
        for action in actions:
            if action is not None:
                action_norms.append(float(np.linalg.norm(action)))
        speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
        obs_list = result.observations
        frames.append(
            snapshot_env(env, scenario, phase, phase_index, local_step, global_step_start + local_step)
        )
        latest = frames[-1]
        phase_success_reached = bool(
            latest["capture_success"] if phase == "capture" else latest["coverage_success"]
        )
        if phase_success_reached:
            break
        if all(result.dones):
            stopped_by_env_done = True
            break
    record = env.episode_record(task=scenario)
    final_positions = [[float(p.x), float(p.y)] for p in env.pursuers]
    final_active = [bool(not p.deactivated) for p in env.pursuers]
    voradj_metrics = record.get("voradj_metrics", {}) or {}
    capture_success = bool(record.get("captured", False))
    coverage_success = coverage_success_from_record(record, env)
    phase_success = bool(capture_success if phase == "capture" else coverage_success)
    timed_out = bool((not phase_success) and env.episode_step >= int(max_steps))
    phase_status = status_text(phase_success, timed_out=timed_out)
    summary = {
        "phase": phase,
        "seed": int(seed),
        "steps": int(env.episode_step),
        "stopped_by_env_done": bool(stopped_by_env_done),
        "timed_out": timed_out,
        "phase_status": phase_status,
        "capture_success": capture_success,
        "coverage_success": coverage_success,
        "capture_success_status": phase_status if phase == "capture" else status_text(capture_success),
        "coverage_success_status": phase_status if phase == "coverage" else status_text(coverage_success),
        "episode_success": phase_success,
        "collision_event": bool(record.get("collision_event", False)),
        "soft_oob_event": bool(record.get("soft_boundary_out_of_bounds_event", False)),
        "episode_end_mean_speed": float(record.get("episode_end_mean_speed", float("nan"))),
        "episode_end_max_speed": float(record.get("episode_end_max_speed", float("nan"))),
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else float("nan"),
        "action_norm_p95": float(np.percentile(action_norms, 95)) if action_norms else float("nan"),
        "speed_mean": float(np.mean(speeds)) if speeds else float("nan"),
        "speed_max": float(np.max(speeds)) if speeds else float("nan"),
        "zone_metrics": voradj_metrics.get("zone_metrics", {}),
    }
    return frames, summary, final_positions, final_active, global_step_start + len(frames) - 1


def rollout_mix(
    trainer: Any,
    cfg: Dict[str, Any],
    scenario: str,
    seed: int,
    max_steps: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=seed)
    obs_list = env.reset()
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    frames: List[Dict[str, Any]] = []
    action_norms: List[float] = []
    speeds: List[float] = []

    def current_phase() -> str:
        return "capture" if any(not evader.deactivated for evader in env.evaders) else "coverage"

    frames.append(snapshot_env(env, scenario, current_phase(), 0, 0, 0))
    phase_index = 0
    last_phase = frames[-1]["phase"]
    for step in range(1, int(max_steps) + 1):
        set_global_config(cfg)
        actions = act_pursuers(trainer, obs_list, env)
        result = env.step(actions, act_evaders(env, apf_agents))
        for action in actions:
            if action is not None:
                action_norms.append(float(np.linalg.norm(action)))
        speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
        obs_list = result.observations
        phase = current_phase()
        if phase != last_phase:
            phase_index += 1
            last_phase = phase
        frames.append(
            snapshot_env(
                env,
                scenario,
                phase,
                phase_index,
                step if phase_index == 0 else env.post_capture_step,
                step,
            )
        )
        if all(result.dones):
            break
    record = env.episode_record(task=scenario)
    voradj_metrics = record.get("voradj_metrics", {}) or {}
    capture_success = bool(record.get("captured", False))
    coverage_success = coverage_success_from_record(record, env)
    capture_timed_out = bool((not capture_success) and env.episode_step >= int(max_steps))
    coverage_timed_out = bool(
        capture_success
        and (not coverage_success)
        and (
            voradj_metrics.get("post_capture_window_expired", False)
            or env.episode_step >= int(max_steps)
        )
    )
    coverage_not_started = bool(not capture_success)
    summary = [{
        "phase": "mix",
        "seed": int(seed),
        "steps": int(env.episode_step),
        "capture_success": capture_success,
        "coverage_success": coverage_success,
        "capture_success_status": status_text(capture_success, timed_out=capture_timed_out),
        "coverage_success_status": status_text(
            coverage_success, timed_out=coverage_timed_out, not_started=coverage_not_started
        ),
        "episode_success": bool(capture_success and coverage_success),
        "episode_success_status": status_text(bool(capture_success and coverage_success)),
        "collision_event": bool(record.get("collision_event", False)),
        "soft_oob_event": bool(record.get("soft_boundary_out_of_bounds_event", False)),
        "post_capture_window_expired": bool(voradj_metrics.get("post_capture_window_expired", False)),
        "episode_end_mean_speed": float(record.get("episode_end_mean_speed", float("nan"))),
        "episode_end_max_speed": float(record.get("episode_end_max_speed", float("nan"))),
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else float("nan"),
        "action_norm_p95": float(np.percentile(action_norms, 95)) if action_norms else float("nan"),
        "speed_mean": float(np.mean(speeds)) if speeds else float("nan"),
        "speed_max": float(np.max(speeds)) if speeds else float("nan"),
        "zone_metrics": voradj_metrics.get("zone_metrics", {}),
    }]
    stats = {
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else float("nan"),
        "speed_mean": float(np.mean(speeds)) if speeds else float("nan"),
    }
    return frames, summary, stats


def rollout_scenario(
    trainer: Any,
    contract: Dict[str, Any],
    scenario: str,
    seed: int,
    max_steps: int,
    capture_max_steps: int,
    coverage_max_steps: int,
    episode_index: int,
    alternate_coverage_init: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    scene = SCENARIO_ALIASES[str(scenario)]
    stats: Dict[str, float] = {}
    if scenario == "capture":
        cfg = scene_config("capture", contract, alternate_coverage_init=False, episode_index=episode_index)
        frames, summary, *_ = rollout_phase(
            trainer, cfg, scenario, "capture", 0, seed, capture_max_steps, 0
        )
        apply_display_status(frames, scenario, [summary])
        stats = {
            "action_norm_mean": float(summary.get("action_norm_mean", float("nan"))),
            "speed_mean": float(summary.get("speed_mean", float("nan"))),
        }
        return frames, [summary], stats
    if scenario == "coverage":
        cfg = scene_config(
            "pure_ce", contract, alternate_coverage_init=alternate_coverage_init, episode_index=episode_index
        )
        frames, summary, *_ = rollout_phase(
            trainer, cfg, scenario, "coverage", 0, seed, coverage_max_steps, 0
        )
        apply_display_status(frames, scenario, [summary])
        stats = {
            "action_norm_mean": float(summary.get("action_norm_mean", float("nan"))),
            "speed_mean": float(summary.get("speed_mean", float("nan"))),
        }
        return frames, [summary], stats
    if scenario == "mix":
        cfg = scene_config("mixed_crms", contract, alternate_coverage_init=False, episode_index=episode_index)
        frames, summaries, stats = rollout_mix(trainer, cfg, scenario, seed, max_steps)
        apply_display_status(frames, scenario, summaries)
        return frames, summaries, stats
    raise ValueError(f"unsupported scenario: {scenario}")


def policy_metadata(contract: Dict[str, Any], checkpoint: Path) -> Dict[str, Any]:
    step = 0
    for part in checkpoint.stem.split("_"):
        if part.startswith("step") and part[4:].isdigit():
            step = int(part[4:])
    return {
        "algorithm": "psa_masac_local",
        "action_mode": contract["action_mode"],
        "a_max": float(contract["a_max"]),
        "w_max": float(contract.get("w_max", 0.0)),
        "yaw_mode": "hold",
        "yaw_init": "aligned_world_axis",
        "deterministic": True,
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "config_hash": (contract.get("effective_config_hashes", {}) or {}),
        "implementation_hash": contract.get("implementation_hash", ""),
        "resume_contract": contract.get("resume_contract", {}),
    }


def write_episode_outputs(
    output_root: Path,
    scenario: str,
    seed: int,
    config_path: Path,
    checkpoint: Path,
    contract: Dict[str, Any],
    frames: List[Dict[str, Any]],
    phase_summaries: List[Dict[str, Any]],
    render_record: Dict[str, Any],
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    episode_path = output_root / f"episode_{scenario}_seed_{int(seed)}.json"
    final_status = dict(frames[-1].get("display_status", {})) if frames else {}
    summary = {
        "scenario": scenario,
        "seed": int(seed),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "policy": policy_metadata(contract, checkpoint),
        "phase_summaries": phase_summaries,
        "final": {
            "capture_success": bool(
                final_status.get(
                    "capture_success_bool",
                    any(frame.get("capture_success", False) for frame in frames),
                )
            ),
            "coverage_success": bool(
                final_status.get(
                    "coverage_success_bool",
                    any(frame.get("coverage_success", False) for frame in frames),
                )
            ),
            "episode_success": bool(
                final_status.get(
                    "episode_success_bool",
                    frames[-1].get("episode_success", False) if frames else False,
                )
            ),
            "capture_success_status": str(final_status.get("capture_success", status_text(False))),
            "coverage_success_status": str(final_status.get("coverage_success", status_text(False))),
            "episode_success_status": str(final_status.get("episode_success", status_text(False))),
            "collision_event": bool(any(frame.get("collision_event", False) for frame in frames)),
            "soft_oob_event": bool(any(frame.get("soft_oob_event", False) for frame in frames)),
            "frames": int(len(frames)),
            "zone_metrics": dict((frames[-1].get("zone", {}) or {}) if frames else {}),
        },
        **render_record,
    }
    episode_path.write_text(
        json.dumps({"summary": summary, "frames": frames}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / f"summary_{scenario}_seed_{int(seed)}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {**summary, "episode_json": str(episode_path)}


def run_batch(
    checkpoint: Path,
    output_root: Path,
    scenarios: List[str],
    episodes: int,
    gif_count: int,
    seed: int,
    device: str,
    max_steps: int,
    capture_max_steps: int,
    coverage_max_steps: int,
    max_gif_frames: int,
    frame_duration_ms: int,
    draw_neighbor_edges: bool,
    draw_trails: bool,
    draw_sensing_circles: bool,
    alternate_coverage_init: bool,
    seed_offset: int = 0,
    gif_offset: int = 0,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    trainer, contract = load_continuous_trainer(checkpoint, device)
    all_summaries: Dict[str, Any] = {}
    scenario_timings: Dict[str, Any] = {}
    with torch.no_grad():
        for scenario in scenarios:
            scenario_wall_start = time.perf_counter()
            scenario_dir = output_root / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            records: List[Dict[str, Any]] = []
            for idx in range(int(episodes)):
                episode_seed = int(seed) + seed_offset + idx
                episode_wall_start = time.perf_counter()
                frames, phase_summaries, extra_stats = rollout_scenario(
                    trainer,
                    contract,
                    scenario,
                    episode_seed,
                    int(max_steps),
                    int(capture_max_steps),
                    int(coverage_max_steps),
                    idx,
                    bool(alternate_coverage_init),
                )
                rollout_wall_seconds = time.perf_counter() - episode_wall_start
                final_status = frames[-1].get("display_status", {}) if frames else {}
                settle_summary = next(
                    (
                        item
                        for item in reversed(phase_summaries)
                        if "coverage_geometric_success" in item or item.get("phase") == "coverage"
                    ),
                    {},
                )
                record = {
                    "scenario": scenario,
                    "seed": episode_seed,
                    "steps": max(len(frames) - 1, 0),
                    "rollout_wall_seconds": rollout_wall_seconds,
                    "capture_success_bool": bool(final_status.get("capture_success_bool", False)),
                    "coverage_success_bool": bool(final_status.get("coverage_success_bool", False)),
                    "episode_success_bool": bool(final_status.get("episode_success_bool", False)),
                    "collision_event": bool(any(frame.get("collision_event", False) for frame in frames)),
                    "soft_oob_event": bool(any(frame.get("soft_oob_event", False) for frame in frames)),
                    "coverage_geometric_success": bool(settle_summary.get("coverage_geometric_success", False)),
                    "coverage_settled_success": bool(settle_summary.get("coverage_settled_success", False)),
                    "coverage_settle_timeout": bool(settle_summary.get("coverage_settle_timeout", False)),
                    "settle_elapsed_steps": int(settle_summary.get("settle_elapsed_steps", 0)),
                    "coverage_init_source": (
                        "map_random" if idx % 2 == 0 else "inner_random_cluster"
                    )
                    if (scenario == "coverage" and alternate_coverage_init)
                    else "",
                    "episode_end_mean_speed": float(settle_summary.get("episode_end_mean_speed", float("nan"))),
                    "episode_end_max_speed": float(settle_summary.get("episode_end_max_speed", float("nan"))),
                    "action_norm_mean": float(extra_stats.get("action_norm_mean", float("nan"))),
                    "speed_mean": float(extra_stats.get("speed_mean", float("nan"))),
                    "phase_summaries": phase_summaries,
                    **{
                        "final_voronoi_cv": float(
                            frames[-1].get("voronoi_cv", float("nan")) if frames else float("nan")
                        ),
                        "best_voronoi_cv": float(
                            min(
                                (frame.get("voronoi_cv", float("nan")) for frame in frames),
                                default=float("nan"),
                            )
                        ),
                    },
                }
                records.append(record)
                if gif_offset + idx < int(gif_count):
                    gif_path = scenario_dir / f"rollout_{scenario}_seed_{episode_seed}.gif"
                    render_record = render_gif(
                        frames,
                        gif_path,
                        int(max_gif_frames),
                        70,
                        int(frame_duration_ms),
                        scenario,
                        bool(draw_neighbor_edges),
                        bool(draw_trails),
                        bool(draw_sensing_circles),
                        scenario == "coverage",
                    )
                    write_episode_outputs(
                        scenario_dir,
                        scenario,
                        episode_seed,
                        Path("<scene-config-derived>"),
                        checkpoint,
                        contract,
                        frames,
                        phase_summaries,
                        render_record,
                    )
            scenario_wall_seconds = time.perf_counter() - scenario_wall_start
            payload = {"scenario": scenario, "summary": summarize(records), "records": records}
            payload["timing"] = {
                "episodes": len(records),
                "wall_seconds": scenario_wall_seconds,
                "avg_wall_seconds_per_episode": scenario_wall_seconds / max(len(records), 1),
                "avg_rollout_wall_seconds_per_episode": float(
                    payload["summary"].get("avg_rollout_wall_seconds", 0.0)
                ),
            }
            (scenario_dir / "batch_summary.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            all_summaries[scenario] = payload["summary"]
            scenario_timings[scenario] = payload["timing"]
    return {
        "all_summaries": all_summaries,
        "scenario_timings": scenario_timings,
        "contract": contract,
    }


def split_counts(total: int, workers: int) -> List[Tuple[int, int]]:
    workers = max(1, int(workers))
    base = int(total) // workers
    extra = int(total) % workers
    chunks: List[Tuple[int, int]] = []
    offset = 0
    for idx in range(workers):
        count = base + (1 if idx < extra else 0)
        if count <= 0:
            continue
        chunks.append((offset, count))
        offset += count
    return chunks


def _worker_entry(payload: Dict[str, Any]) -> None:
    run_batch(
        Path(payload["checkpoint"]),
        Path(payload["output_root"]),
        payload["scenarios"],
        payload["episodes"],
        payload["gif_count"],
        payload["seed"],
        payload["device"],
        payload["max_steps"],
        payload["capture_max_steps"],
        payload["coverage_max_steps"],
        payload["max_gif_frames"],
        payload["frame_duration_ms"],
        payload["draw_neighbor_edges"],
        payload["draw_trails"],
        payload["draw_sensing_circles"],
        payload["alternate_coverage_init"],
        seed_offset=payload["seed_offset"],
        gif_offset=payload["gif_offset"],
    )


def copy_worker_files(worker_root: Path, output_root: Path, scenarios: List[str]) -> None:
    for scenario in scenarios:
        src_dir = worker_root / scenario
        if not src_dir.is_dir():
            continue
        dst_dir = output_root / scenario
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in src_dir.iterdir():
            if path.name == "batch_summary.json":
                continue
            if path.is_file():
                target = dst_dir / path.name
                if not target.exists():
                    shutil.copy2(path, target)


def merge_worker_outputs(
    output_root: Path,
    worker_roots: List[Path],
    scenarios: List[str],
) -> Dict[str, Any]:
    all_summaries: Dict[str, Any] = {}
    for scenario in scenarios:
        records: List[Dict[str, Any]] = []
        scenario_dir = output_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for worker_root in worker_roots:
            payload_path = worker_root / scenario / "batch_summary.json"
            if not payload_path.is_file():
                continue
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            records.extend(payload.get("records", []))
        records.sort(key=lambda item: int(item.get("seed", 0)))
        summary = summarize(records)
        payload = {"scenario": scenario, "summary": summary, "records": records}
        (scenario_dir / "batch_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        all_summaries[scenario] = summary
    (output_root / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return all_summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIO_ALIASES), default=DEFAULT_SCENARIOS)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gif-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="0 = use checkpoint contract seed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--capture-max-steps", type=int, default=128)
    parser.add_argument("--coverage-max-steps", type=int, default=128)
    parser.add_argument("--max-gif-frames", type=int, default=1000)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--draw-neighbor-edges", dest="draw_neighbor_edges", action="store_true", default=True)
    parser.add_argument("--no-neighbor-edges", dest="draw_neighbor_edges", action="store_false")
    parser.add_argument("--draw-trails", action="store_true", default=False)
    parser.add_argument("--draw-sensing-circles", dest="draw_sensing_circles", action="store_true", default=True)
    parser.add_argument("--no-sensing-circles", dest="draw_sensing_circles", action="store_false")
    parser.add_argument("--alternate-coverage-init", action="store_true", default=False)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = dict(payload["contract"])
    seed = int(args.seed) if args.seed else int(contract["seed"])
    output_root = (
        Path(args.output_root)
        if args.output_root
        else REPO_ROOT
        / "artifacts/2026-08-06_continuous_marl_formal_20rollout10gif"
        / f"{checkpoint.stem}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_args = vars(args) | {
        "created_at": now(),
        "checkpoint_contract": contract,
        "effective_seed": seed,
        "output_root": str(output_root),
    }
    (output_root / "run_args.json").write_text(
        json.dumps(run_args, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    parallel_wall_start = time.perf_counter()
    chunks = split_counts(int(args.episodes), int(args.workers))
    worker_parent = output_root / "_workers"
    worker_parent.mkdir(parents=True, exist_ok=True)
    gif_remaining = int(args.gif_count)
    processes: List[Tuple[multiprocessing.Process, Path]] = []
    for worker_idx, (offset, count) in enumerate(chunks):
        worker_root = worker_parent / f"worker_{worker_idx:02d}"
        worker_root.mkdir(parents=True, exist_ok=True)
        worker_gifs = min(gif_remaining, count)
        gif_remaining -= worker_gifs
        worker_payload = {
            "checkpoint": str(checkpoint),
            "output_root": str(worker_root),
            "scenarios": list(args.scenarios),
            "episodes": count,
            "gif_count": worker_gifs,
            "seed": seed,
            "device": str(args.device),
            "max_steps": int(args.max_steps),
            "capture_max_steps": int(args.capture_max_steps),
            "coverage_max_steps": int(args.coverage_max_steps),
            "max_gif_frames": int(args.max_gif_frames),
            "frame_duration_ms": int(args.frame_duration_ms),
            "draw_neighbor_edges": bool(args.draw_neighbor_edges),
            "draw_trails": bool(args.draw_trails),
            "draw_sensing_circles": bool(args.draw_sensing_circles),
            "alternate_coverage_init": bool(args.alternate_coverage_init),
            "seed_offset": offset,
            "gif_offset": offset,
        }
        process = multiprocessing.Process(target=_worker_entry, args=(worker_payload,))
        process.start()
        processes.append((process, worker_root))

    failures: List[Dict[str, Any]] = []
    for process, worker_root in processes:
        process.join()
        if process.exitcode != 0:
            failures.append(
                {"worker": worker_root.name, "returncode": process.exitcode, "root": str(worker_root)}
            )
    if failures:
        (output_root / "failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise RuntimeError(f"worker failures: {failures}")

    worker_roots = [item[1] for item in processes]
    for worker_root in worker_roots:
        copy_worker_files(worker_root, output_root, list(args.scenarios))
    all_summaries = merge_worker_outputs(output_root, worker_roots, list(args.scenarios))
    total_wall_seconds = time.perf_counter() - parallel_wall_start
    timing_payload = {
        "created_at": now(),
        "device": str(args.device),
        "workers": len(worker_roots),
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "total_wall_seconds": total_wall_seconds,
        "total_rollout_episodes": int(args.episodes) * len(list(args.scenarios)),
        "effective_wall_seconds_per_rollout_episode": total_wall_seconds
        / max(int(args.episodes) * len(list(args.scenarios)), 1),
    }
    (output_root / "timing.json").write_text(
        json.dumps(timing_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "all_summaries": all_summaries,
                "timing": timing_payload,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
