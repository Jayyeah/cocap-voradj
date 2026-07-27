#!/usr/bin/env python3
"""Roll out VorAdj single-head checkpoints and render Voronoi-adjacency GIFs."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
from PIL import Image
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import set_global_config
from cocap_voradj.envs.voronoi_adjacency import SiteKey, VorAdjEnv
from tools.render_coverage_voronoi_gifs import area_cv, clipped_voronoi_cells
from cocap_voradj.control.apf import ApfAgent


PURSUING_COLOR = "#1f77b4"
COVERAGE_COLOR = "#2ca02c"
EVADER_COLOR = "#d62728"
INACTIVE_EVADER_COLOR = "#1f77b4"
PP_EDGE_COLOR = "#6f6f6f"
PE_EDGE_COLOR = "#d62728"
OBSTACLE_COLOR = "#555555"
BOUNDARY_COLOR = "#222222"


def load_config(path: Path) -> Dict[str, Any]:
    config_path = path / "effective_config.yaml" if path.is_dir() else path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def frame_ids(max_len: int, max_frames: int) -> List[int]:
    if max_len <= 1:
        return [0]
    if max_len <= max_frames:
        return list(range(max_len))
    return sorted(set(np.linspace(0, max_len - 1, max_frames, dtype=int).tolist()))


def scenario_config(base: Dict[str, Any], num_evaders: int, mix_mode: bool = False) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["device"] = "cpu"
    cfg.setdefault("env", {})["num_evaders"] = int(num_evaders)
    if mix_mode:
        cfg.setdefault("voradj", {})["capture_episode_ends_on_capture"] = False
        cfg.setdefault("voradj", {})["capture_episode_success_on_capture"] = False
    return cfg


def site_to_json(key: SiteKey) -> Dict[str, Any]:
    return {"type": str(key[0]), "id": int(key[1])}


def site_from_json(item: Dict[str, Any]) -> SiteKey:
    return (str(item["type"]), int(item["id"]))


def unique_edges(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen: set[Tuple[SiteKey, SiteKey]] = set()
    edges: List[Dict[str, Any]] = []
    for key, neighbors in data.get("adjacency", {}).items():
        for other in neighbors:
            pair = tuple(sorted((key, other)))  # type: ignore[arg-type]
            if pair in seen:
                continue
            seen.add(pair)
            if pair[0][0] == "evader" and pair[1][0] == "evader":
                continue
            if pair[0][0] == "pursuer" and pair[1][0] == "pursuer":
                kind = "pursuer_pursuer"
            elif pair[0][0] != pair[1][0]:
                kind = "pursuer_evader"
            else:
                continue
            edges.append({"a": site_to_json(pair[0]), "b": site_to_json(pair[1]), "kind": kind})
    return edges


def snapshot_voronoi_cv(keys: List[SiteKey], sites: np.ndarray, bounds: Tuple[float, float, float, float]) -> float:
    if len(keys) < 2 or len(sites) < 2:
        return float("nan")
    xl, xr, yb, yt = bounds
    cells = clipped_voronoi_cells(np.asarray(sites, dtype=float), xl, xr, yb, yt)
    pursuer_cells = [cell for key, cell in zip(keys, cells) if key[0] == "pursuer"]
    return area_cv(pursuer_cells)


def snapshot_bounds(snapshot: Dict[str, Any]) -> Tuple[float, float, float, float]:
    width = float(snapshot.get("width", 55.0))
    height = float(snapshot.get("height", width))
    raw = snapshot.get("bounds", [0.0, width, 0.0, height])
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def in_snapshot_bounds(snapshot: Dict[str, Any], x: float, y: float, pad: float = 0.0) -> bool:
    xl, xr, yb, yt = snapshot_bounds(snapshot)
    return bool((xl - pad) <= float(x) <= (xr + pad) and (yb - pad) <= float(y) <= (yt + pad))


def settle_enabled(env: VorAdjEnv) -> bool:
    return bool((getattr(env, "reward_cfg", {}) or {}).get("coverage_settle_enabled", False))


def coverage_success_from_record(record: Dict[str, Any], env: VorAdjEnv) -> bool:
    voradj_metrics = record.get("voradj_metrics", {}) or {}
    if settle_enabled(env):
        return bool(record.get("coverage_settled_success", False) or voradj_metrics.get("coverage_settled_success", False))
    return bool(
        record.get("coverage_strict_success", False)
        or voradj_metrics.get("post_capture_coverage_success", False)
        or voradj_metrics.get("pure_coverage_success", False)
    )


def settle_fields_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    voradj_metrics = record.get("voradj_metrics", {}) or {}
    return {
        "coverage_geometric_success": bool(record.get("coverage_geometric_success", False) or voradj_metrics.get("coverage_geometric_success", False)),
        "coverage_settled_success": bool(record.get("coverage_settled_success", False) or voradj_metrics.get("coverage_settled_success", False)),
        "coverage_settle_timeout": bool(record.get("coverage_settle_timeout", False) or voradj_metrics.get("coverage_settle_timeout", False)),
        "settle_hold_steps": int(record.get("settle_hold_steps", voradj_metrics.get("settle_hold_steps", 0))),
        "settle_elapsed_steps": int(voradj_metrics.get("settle_elapsed_steps", 0)),
        "episode_end_mean_speed": float(record.get("episode_end_mean_speed", voradj_metrics.get("episode_end_mean_speed", float("nan")))),
        "episode_end_max_speed": float(record.get("episode_end_max_speed", voradj_metrics.get("episode_end_max_speed", float("nan")))),
    }


def snapshot_env(env: VorAdjEnv, scenario: str, phase: str, phase_index: int, local_step: int, global_step: int) -> Dict[str, Any]:
    data = env._voronoi_map()
    labels = env._task_labels_from_map(data, update_effective=False)
    bounds = env._bounds()
    keys: List[SiteKey] = list(data.get("keys", []))
    sites = np.asarray(data.get("sites", np.zeros((0, 2), dtype=float)), dtype=float)
    record = env.episode_record(task=scenario)
    voradj_metrics = record.get("voradj_metrics", {}) or {}
    coverage_success = coverage_success_from_record(record, env)
    return {
        "scenario": scenario,
        "phase": phase,
        "phase_index": int(phase_index),
        "local_step": int(local_step),
        "global_step": int(global_step),
        "width": float(env.width),
        "height": float(env.height),
        "bounds": [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])],
        "obstacles": [
            {"x": float(obstacle.x), "y": float(obstacle.y), "r": float(obstacle.r)}
            for obstacle in env.obstacles
        ],
        "pursuers": [
            {
                "id": int(i),
                "x": float(p.x),
                "y": float(p.y),
                "r": float(getattr(p, "r", 1.0)),
                "active": bool(not p.deactivated),
                "collision": bool(p.collision),
                "boundary_collision": bool(getattr(p, "boundary_collision", False)),
                "is_pursuing": bool(getattr(p, "is_pursuing", False)),
                "task_label": str(labels[i]) if i < len(labels) else "inactive",
            }
            for i, p in enumerate(env.pursuers)
        ],
        "evaders": [
            {
                "id": int(i),
                "x": float(e.x),
                "y": float(e.y),
                "r": float(getattr(e, "r", 1.0)),
                "active": bool(not e.deactivated),
                "collision": bool(e.collision),
            }
            for i, e in enumerate(env.evaders)
        ],
        "sites": [
            {"type": key[0], "id": int(key[1]), "x": float(point[0]), "y": float(point[1])}
            for key, point in zip(keys, sites)
        ],
        "edges": unique_edges(data),
        "voronoi_cv": float(snapshot_voronoi_cv(keys, sites, bounds)),
        "capture_success": bool(record.get("captured", False)),
        "coverage_success": coverage_success,
        **settle_fields_from_record(record),
        "episode_success": bool(record.get("episode_success", False)),
        "collision_event": bool(record.get("collision_event", False)),
        "soft_oob_event": bool(record.get("soft_boundary_out_of_bounds_event", False)),
    }


def act_pursuers(model: CoCapIQN, obs_list: List[Optional[Dict[str, np.ndarray]]], device: str) -> List[Optional[int]]:
    active = [idx for idx, obs in enumerate(obs_list) if obs is not None]
    actions: List[Optional[int]] = [None] * len(obs_list)
    if not active:
        return actions
    batch = stack_obs([obs_list[idx] for idx in active], device)
    selected = model.act(batch, mode="voradj", epsilon=0.0).detach().cpu().tolist()
    for idx, action in zip(active, selected):
        actions[idx] = int(action)
    return actions


def act_evaders(env: VorAdjEnv, apf_agents: List[ApfAgent]) -> List[Optional[int]]:
    actions: List[Optional[int]] = []
    for idx, evader_obs in enumerate(env.get_evader_observations_for_apf()):
        actions.append(None if evader_obs is None else int(apf_agents[idx].act(evader_obs)))
    return actions



def rollout_phase(
    model: CoCapIQN,
    cfg: Dict[str, Any],
    scenario: str,
    phase: str,
    phase_index: int,
    seed: int,
    device: str,
    max_steps: int,
    global_step_start: int,
    initial_positions: Optional[List[List[float]]] = None,
    initial_active: Optional[List[bool]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[List[float]], List[bool], int]:
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=seed)
    obs_list = env.reset(initial_pursuer_positions=initial_positions, initial_pursuer_active=initial_active)
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    frames = [snapshot_env(env, scenario, phase, phase_index, 0, global_step_start)]
    stopped_by_env_done = False
    phase_success_reached = False
    for local_step in range(1, int(max_steps) + 1):
        set_global_config(cfg)
        actions = act_pursuers(model, obs_list, device)
        result = env.step(actions, act_evaders(env, apf_agents))
        obs_list = result.observations
        frames.append(snapshot_env(env, scenario, phase, phase_index, local_step, global_step_start + local_step))
        latest = frames[-1]
        phase_success_reached = bool(latest["capture_success"] if phase == "capture" else latest["coverage_success"])
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
        **settle_fields_from_record(record),
    }
    return frames, summary, final_positions, final_active, global_step_start + len(frames) - 1


def rollout_mix(
    model: CoCapIQN,
    cfg: Dict[str, Any],
    scenario: str,
    seed: int,
    device: str,
    max_steps: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=seed)
    obs_list = env.reset()
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    frames: List[Dict[str, Any]] = []

    def current_phase() -> str:
        return "capture" if any(not evader.deactivated for evader in env.evaders) else "coverage"

    frames.append(snapshot_env(env, scenario, current_phase(), 0, 0, 0))
    phase_index = 0
    last_phase = frames[-1]["phase"]
    for step in range(1, int(max_steps) + 1):
        set_global_config(cfg)
        actions = act_pursuers(model, obs_list, device)
        result = env.step(actions, act_evaders(env, apf_agents))
        obs_list = result.observations
        phase = current_phase()
        if phase != last_phase:
            phase_index += 1
            last_phase = phase
        frames.append(snapshot_env(env, scenario, phase, phase_index, step if phase_index == 0 else env.post_capture_step, step))
        if all(result.dones):
            break
    record = env.episode_record(task=scenario)
    voradj_metrics = record.get("voradj_metrics", {}) or {}
    capture_success = bool(record.get("captured", False))
    coverage_success = coverage_success_from_record(record, env)
    capture_timed_out = bool((not capture_success) and env.episode_step >= int(max_steps))
    coverage_timed_out = bool(capture_success and (not coverage_success) and (voradj_metrics.get("post_capture_window_expired", False) or env.episode_step >= int(max_steps)))
    coverage_not_started = bool(not capture_success)
    episode_success = bool(capture_success and coverage_success)
    summary = [{
        "phase": "mix",
        "seed": int(seed),
        "steps": int(env.episode_step),
        "capture_success": capture_success,
        "coverage_success": coverage_success,
        "capture_success_status": status_text(capture_success, timed_out=capture_timed_out),
        "coverage_success_status": status_text(coverage_success, timed_out=coverage_timed_out, not_started=coverage_not_started),
        "episode_success": episode_success,
        "episode_success_status": status_text(episode_success),
        "collision_event": bool(record.get("collision_event", False)),
        "soft_oob_event": bool(record.get("soft_boundary_out_of_bounds_event", False)),
        "post_capture_window_expired": bool(voradj_metrics.get("post_capture_window_expired", False)),
        **settle_fields_from_record(record),
    }]
    return frames, summary


def status_text(success: bool, timed_out: bool = False, not_started: bool = False) -> str:
    if success:
        return "true"
    if timed_out:
        return "time out"
    if not_started:
        return "not started"
    return "false"


def phase_summary(summaries: List[Dict[str, Any]], phase: str) -> Dict[str, Any]:
    return next((item for item in summaries if str(item.get("phase")) == phase), {})


def apply_display_status(frames: List[Dict[str, Any]], scenario: str, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if scenario == "capture":
        item = summaries[0] if summaries else {}
        capture_success = bool(item.get("capture_success", False))
        coverage_success = False
        capture_status = str(item.get("capture_success_status", status_text(capture_success)))
        coverage_status = "false"
        episode_success = capture_success
    elif scenario == "coverage":
        item = summaries[0] if summaries else {}
        capture_success = False
        coverage_success = bool(item.get("coverage_success", False))
        capture_status = "false"
        coverage_status = str(item.get("coverage_success_status", status_text(coverage_success)))
        episode_success = coverage_success
    elif scenario in {"ab", "ba"}:
        cap = phase_summary(summaries, "capture")
        cov = phase_summary(summaries, "coverage")
        capture_success = bool(cap.get("capture_success", False))
        coverage_success = bool(cov.get("coverage_success", False))
        capture_status = str(cap.get("capture_success_status", status_text(capture_success)))
        coverage_status = str(cov.get("coverage_success_status", status_text(coverage_success)))
        episode_success = bool(capture_success and coverage_success)
    elif scenario == "mix":
        item = summaries[0] if summaries else {}
        capture_success = bool(item.get("capture_success", False))
        coverage_success = bool(item.get("coverage_success", False))
        capture_status = str(item.get("capture_success_status", status_text(capture_success)))
        coverage_status = str(item.get("coverage_success_status", status_text(coverage_success, not_started=not capture_success)))
        episode_success = bool(capture_success and coverage_success)
    else:
        capture_success = bool(any(frame.get("capture_success", False) for frame in frames))
        coverage_success = bool(any(frame.get("coverage_success", False) for frame in frames))
        capture_status = status_text(capture_success)
        coverage_status = status_text(coverage_success)
        episode_success = bool(frames[-1].get("episode_success", False)) if frames else False
    final_status = {
        "capture_success": capture_status,
        "coverage_success": coverage_status,
        "episode_success": status_text(episode_success),
        "capture_success_bool": bool(capture_success),
        "coverage_success_bool": bool(coverage_success),
        "episode_success_bool": bool(episode_success),
    }
    for frame in frames:
        frame["display_status"] = dict(final_status)
    return final_status


def rollout_scenario(
    model: CoCapIQN,
    base_cfg: Dict[str, Any],
    scenario: str,
    seed: int,
    device: str,
    max_steps: int,
    capture_max_steps: int,
    coverage_max_steps: int,
    capture_evaders: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if scenario == "capture":
        cfg = scenario_config(base_cfg, capture_evaders)
        frames, summary, *_ = rollout_phase(model, cfg, scenario, "capture", 0, seed, device, capture_max_steps, 0)
        summaries = [summary]
        apply_display_status(frames, scenario, summaries)
        return frames, summaries
    if scenario == "coverage":
        cfg = scenario_config(base_cfg, 0)
        frames, summary, *_ = rollout_phase(model, cfg, scenario, "coverage", 0, seed, device, coverage_max_steps, 0)
        summaries = [summary]
        apply_display_status(frames, scenario, summaries)
        return frames, summaries
    if scenario == "mix":
        cfg = scenario_config(base_cfg, capture_evaders, mix_mode=True)
        frames, summaries = rollout_mix(model, cfg, scenario, seed, device, max_steps)
        apply_display_status(frames, scenario, summaries)
        return frames, summaries
    if scenario not in {"ab", "ba"}:
        raise ValueError(f"unknown scenario: {scenario}")

    phases = ["coverage", "capture"] if scenario == "ab" else ["capture", "coverage"]
    all_frames: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    positions: Optional[List[List[float]]] = None
    active: Optional[List[bool]] = None
    global_step = 0
    for phase_index, phase in enumerate(phases):
        cfg = scenario_config(base_cfg, 0 if phase == "coverage" else capture_evaders)
        limit = coverage_max_steps if phase == "coverage" else capture_max_steps
        frames, summary, positions, active, global_step = rollout_phase(
            model,
            cfg,
            scenario,
            phase,
            phase_index,
            seed + 1009 * phase_index,
            device,
            limit,
            global_step if phase_index == 0 else global_step + 1,
            initial_positions=positions,
            initial_active=active,
        )
        all_frames.extend(frames if not all_frames else frames[1:])
        summaries.append(summary)

    final_status = apply_display_status(all_frames, scenario, summaries)
    for frame in all_frames:
        frame["episode_success"] = bool(final_status["episode_success_bool"])
    return all_frames, summaries

def snapshot_site_lookup(snapshot: Dict[str, Any]) -> Dict[SiteKey, Tuple[float, float]]:
    return {
        (str(site["type"]), int(site["id"])): (float(site["x"]), float(site["y"]))
        for site in snapshot.get("sites", [])
    }


def pursuer_color(pursuer: Dict[str, Any]) -> str:
    return PURSUING_COLOR if bool(pursuer.get("is_pursuing", False)) else COVERAGE_COLOR


def plot_trail(axis, frames: List[Dict[str, Any]], frame_idx: int, pursuer_id: int, color: str, trail_window: int) -> None:
    current_phase_index = int(frames[frame_idx].get("phase_index", 0))
    points: List[Tuple[float, float]] = []
    start = max(0, frame_idx - int(trail_window))
    for prior in frames[start : frame_idx + 1]:
        if int(prior.get("phase_index", 0)) != current_phase_index:
            points = []
            continue
        pursuer = next((item for item in prior.get("pursuers", []) if int(item["id"]) == pursuer_id), None)
        if pursuer is None or not bool(pursuer.get("active", False)):
            points = []
            continue
        x = float(pursuer["x"])
        y = float(pursuer["y"])
        if not in_snapshot_bounds(prior, x, y):
            points = []
            continue
        points.append((x, y))
    if len(points) < 2:
        return
    arr = np.asarray(points, dtype=float)
    denominator = max(1, len(arr) - 2)
    for idx in range(len(arr) - 1):
        alpha = 0.12 + 0.52 * (idx / denominator)
        axis.plot(arr[idx : idx + 2, 0], arr[idx : idx + 2, 1], color=color, linewidth=1.6, alpha=alpha, solid_capstyle="round", zorder=3)


def draw_voronoi(axis, snapshot: Dict[str, Any]) -> None:
    # Keep all sites in the tessellation and clip cells to map bounds. Dropping
    # an out-of-bounds site makes the whole Voronoi diagram jump when an agent
    # crosses the soft boundary by a few centimeters. Marker/edge/trail drawing
    # below still hides off-map geometry.
    sites = list(snapshot.get("sites", []))
    if len(sites) < 2:
        return
    keys = [(str(site["type"]), int(site["id"])) for site in sites]
    positions = np.asarray([[float(site["x"]), float(site["y"])] for site in sites], dtype=float)
    xl, xr, yb, yt = snapshot_bounds(snapshot)
    cells = clipped_voronoi_cells(positions, xl, xr, yb, yt)
    pursuers_by_id = {int(item["id"]): item for item in snapshot.get("pursuers", [])}
    for key, cell in zip(keys, cells):
        if len(cell) < 3:
            continue
        if key[0] == "pursuer":
            color = pursuer_color(pursuers_by_id.get(key[1], {"is_pursuing": False}))
            axis.add_patch(Polygon(cell, closed=True, facecolor=color, edgecolor=color, alpha=0.085, linewidth=1.2, zorder=1))
            closed = np.vstack([cell, cell[0]])
            axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.0, alpha=0.72, zorder=2)
        else:
            axis.add_patch(Polygon(cell, closed=True, facecolor="none", edgecolor=EVADER_COLOR, alpha=0.38, linewidth=1.0, zorder=1.5))


def draw_adjacency_edges(axis, snapshot: Dict[str, Any]) -> None:
    lookup = snapshot_site_lookup(snapshot)
    for edge in snapshot.get("edges", []):
        a = site_from_json(edge["a"])
        b = site_from_json(edge["b"])
        if a not in lookup or b not in lookup:
            continue
        if not (in_snapshot_bounds(snapshot, lookup[a][0], lookup[a][1]) and in_snapshot_bounds(snapshot, lookup[b][0], lookup[b][1])):
            continue
        xs = [lookup[a][0], lookup[b][0]]
        ys = [lookup[a][1], lookup[b][1]]
        if edge.get("kind") == "pursuer_evader":
            axis.plot(xs, ys, color=PE_EDGE_COLOR, linewidth=1.5, linestyle="-", alpha=0.75, zorder=2.6)
        elif edge.get("kind") == "pursuer_pursuer":
            axis.plot(xs, ys, color=PP_EDGE_COLOR, linewidth=1.0, linestyle="--", alpha=0.62, zorder=2.4)



def render_frame(
    frames: List[Dict[str, Any]],
    frame_idx: int,
    trail_window: int,
    label: str,
    draw_neighbor_edges: bool,
    draw_trails: bool,
) -> np.ndarray:
    snapshot = frames[frame_idx]
    width = float(snapshot.get("width", 55.0))
    height = float(snapshot.get("height", width))
    xl, xr, yb, yt = snapshot_bounds(snapshot)
    figure, axis = plt.subplots(figsize=(7.0, 6.6), dpi=110)
    axis.set_xlim(xl, xr)
    axis.set_ylim(yb, yt)
    axis.set_autoscale_on(False)
    axis.margins(0.0)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.25, color="#d0d0d0", zorder=0)
    axis.add_patch(Rectangle((xl, yb), xr - xl, yt - yb, fill=False, edgecolor=BOUNDARY_COLOR, linewidth=1.4, zorder=6))
    for obstacle in snapshot.get("obstacles", []):
        axis.add_patch(Circle((float(obstacle["x"]), float(obstacle["y"])), float(obstacle["r"]), color=OBSTACLE_COLOR, alpha=0.38, zorder=4))

    draw_voronoi(axis, snapshot)
    if draw_neighbor_edges:
        draw_adjacency_edges(axis, snapshot)

    if draw_trails:
        for pursuer in snapshot.get("pursuers", []):
            color = pursuer_color(pursuer)
            plot_trail(axis, frames, frame_idx, int(pursuer["id"]), color, trail_window)

    for pursuer in snapshot.get("pursuers", []):
        x = float(pursuer["x"])
        y = float(pursuer["y"])
        if not in_snapshot_bounds(snapshot, x, y):
            continue
        if bool(pursuer.get("active", False)):
            color = pursuer_color(pursuer)
            axis.scatter(x, y, s=66, color=color, edgecolor="white", linewidth=0.9, marker="o", zorder=7)
            axis.text(x + 1.0, y + 1.0, f"P{int(pursuer['id'])}", fontsize=8, color=color, weight="bold", zorder=8)
        else:
            axis.scatter(x, y, s=76, color="red", marker="x", linewidth=2.0, zorder=8)

    for evader in snapshot.get("evaders", []):
        x = float(evader["x"])
        y = float(evader["y"])
        if not in_snapshot_bounds(snapshot, x, y):
            continue
        if bool(evader.get("active", False)):
            axis.scatter(x, y, s=128, color=EVADER_COLOR, edgecolor="#5c1111", linewidth=0.8, marker="*", zorder=7.5)
            axis.text(x + 1.0, y + 1.0, f"E{int(evader['id'])}", fontsize=8, color=EVADER_COLOR, weight="bold", zorder=8)
        else:
            axis.scatter(x, y, s=88, color=INACTIVE_EVADER_COLOR, marker="x", linewidth=2.1, zorder=8)

    cv = float(snapshot.get("voronoi_cv", float("nan")))
    cv_text = "nan" if math.isnan(cv) else f"{cv:.3f}"
    status = snapshot.get("display_status", {}) or {}
    capture_status = str(status.get("capture_success", status_text(bool(snapshot.get("capture_success", False)))))
    coverage_status = str(status.get("coverage_success", status_text(bool(snapshot.get("coverage_success", False)))))
    episode_status = str(status.get("episode_success", status_text(bool(snapshot.get("episode_success", False)))))
    axis.set_title(
        f"{label} | phase {snapshot.get('phase')} | step {int(snapshot.get('global_step', 0))}/{int(frames[-1].get('global_step', len(frames) - 1))}\n"
        f"capture success {capture_status} | coverage success {coverage_status} | "
        f"episode success {episode_status} | voronoi cv {cv_text}",
        fontsize=8.4,
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.89)
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(figure)
    return image


def render_gif(
    frames: List[Dict[str, Any]],
    output_path: Path,
    max_frames: int,
    trail_window: int,
    frame_duration_ms: int,
    label: str,
    draw_neighbor_edges: bool,
    draw_trails: bool,
) -> Dict[str, Any]:
    ids = frame_ids(len(frames), int(max_frames))
    images = [render_frame(frames, idx, trail_window, label, draw_neighbor_edges, draw_trails) for idx in ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], duration=max(10, int(frame_duration_ms)), loop=0, optimize=False, disposal=2)
    return {
        "gif": str(output_path),
        "rendered_frames": len(images),
        "source_frames": len(frames),
        "draw_neighbor_edges": bool(draw_neighbor_edges),
        "draw_trails": bool(draw_trails),
    }


def write_outputs(
    output_root: Path,
    scenario: str,
    seed: int,
    config_path: Path,
    checkpoint_path: Path,
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
        "checkpoint": str(checkpoint_path),
        "phase_summaries": phase_summaries,
        "final": {
            "capture_success": bool(final_status.get("capture_success_bool", any(frame.get("capture_success", False) for frame in frames))),
            "coverage_success": bool(final_status.get("coverage_success_bool", any(frame.get("coverage_success", False) for frame in frames))),
            "episode_success": bool(final_status.get("episode_success_bool", frames[-1].get("episode_success", False) if frames else False)),
            "capture_success_status": str(final_status.get("capture_success", status_text(False))),
            "coverage_success_status": str(final_status.get("coverage_success", status_text(False))),
            "episode_success_status": str(final_status.get("episode_success", status_text(False))),
            "collision_event": bool(any(frame.get("collision_event", False) for frame in frames)),
            "soft_oob_event": bool(any(frame.get("soft_oob_event", False) for frame in frames)),
            "frames": int(len(frames)),
        },
        **render_record,
    }
    episode_path.write_text(json.dumps({"summary": summary, "frames": frames}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / f"summary_{scenario}_seed_{int(seed)}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**summary, "episode_json": str(episode_path)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config YAML or run directory containing effective_config.yaml.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", choices=["capture", "coverage", "ab", "ba", "mix", "on_a_mix", "on-a-mix"], required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--capture-max-steps", type=int, default=600)
    parser.add_argument("--coverage-max-steps", type=int, default=900)
    parser.add_argument("--capture-evaders", type=int, default=None, help="Override capture/mix evader count; defaults to env.num_evaders from the config.")
    parser.add_argument("--max-gif-frames", type=int, default=1000)
    parser.add_argument("--trail-window", type=int, default=70)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--draw-neighbor-edges", dest="draw_neighbor_edges", action="store_true", default=False)
    parser.add_argument("--no-neighbor-edges", dest="draw_neighbor_edges", action="store_false")
    parser.add_argument("--draw-trails", action="store_true", default=False)
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    base_cfg = load_config(config_path)
    capture_evaders = int(args.capture_evaders) if args.capture_evaders is not None else int((base_cfg.get("env", {}) or {}).get("num_evaders", 1))
    device = str(args.device)
    scenario = "mix" if str(args.scenario) in {"mix", "on_a_mix", "on-a-mix"} else str(args.scenario)
    model = CoCapIQN.load(str(checkpoint_path), device=device)
    model.eval()
    with torch.no_grad():
        frames, phase_summaries = rollout_scenario(
            model,
            base_cfg,
            scenario,
            int(args.seed),
            device,
            int(args.max_steps),
            int(args.capture_max_steps),
            int(args.coverage_max_steps),
            capture_evaders,
        )
    gif_path = output_root / f"rollout_{scenario}_seed_{int(args.seed)}.gif"
    render_record = render_gif(
        frames,
        gif_path,
        int(args.max_gif_frames),
        int(args.trail_window),
        int(args.frame_duration_ms),
        scenario,
        bool(args.draw_neighbor_edges),
        bool(args.draw_trails),
    )
    summary = write_outputs(output_root, scenario, int(args.seed), config_path, checkpoint_path, frames, phase_summaries, render_record)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
