#!/usr/bin/env python3
"""Exact-config deterministic MASAC rollouts with a fixed 20-rollout/5-GIF contract.

Unlike the historical P6 visualizer, this entry loads the checkpoint's actual
formal CTDE YAML/effective_config, verifies its manifest hashes, deep-merges the
selected task override, and uses the current central MASAC actor path.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import SCENES, scene_config
from cocap_voradj.training.trainer import load_config, set_global_config
from tools.rollout_voradj_visual import render_gif, snapshot_env, status_text
from tools.run_continuous_ctde_training import (
    _evader_actions_for_env,
    _make_trainer,
    _pad_local_obs_tree,
    _sample_actions,
    _stable_hash,
)


PRESET_SCENES = set(SCENES)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _require_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_preset(path: str | Path) -> Dict[str, Any]:
    preset_path = Path(path)
    payload = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    preset = _require_mapping(payload, "preset")
    if int(preset.get("preset_schema_version", 0)) != 1:
        raise ValueError("preset_schema_version must be 1")
    if int(preset.get("episodes", 0)) <= 0:
        raise ValueError("episodes must be positive")
    gifs = int(preset.get("gif_count", -1))
    if gifs < 0 or gifs > int(preset["episodes"]):
        raise ValueError("gif_count must be between zero and episodes")
    display = _require_mapping(preset.get("display"), "display")
    if int(display.get("frame_duration_ms", 0)) <= 0:
        raise ValueError("display.frame_duration_ms must be positive")
    if int(display.get("max_gif_frames", 0)) <= 0:
        raise ValueError("display.max_gif_frames must be positive")
    scenarios = preset.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list")
    output_names: set[str] = set()
    for index, raw in enumerate(scenarios):
        item = _require_mapping(raw, f"scenarios[{index}]")
        scene = str(item.get("scene", ""))
        if scene not in PRESET_SCENES:
            raise ValueError(f"unsupported scene {scene!r}")
        output_name = str(item.get("output_name", ""))
        if not output_name or output_name in output_names:
            raise ValueError("scenario output_name values must be non-empty and unique")
        output_names.add(output_name)
        if int(item.get("max_steps", 0)) <= 0:
            raise ValueError(f"scenarios[{index}].max_steps must be positive")
    return copy.deepcopy(preset)


def apply_overrides(preset: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    resolved = copy.deepcopy(preset)
    for key in ("episodes", "gif_count", "seed", "workers"):
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = int(value)
    display = resolved["display"]
    if args.frame_duration_ms is not None:
        display["frame_duration_ms"] = int(args.frame_duration_ms)
    if args.max_gif_frames is not None:
        display["max_gif_frames"] = int(args.max_gif_frames)
    if args.horizon_cap is not None:
        cap = int(args.horizon_cap)
        if cap <= 0:
            raise ValueError("--horizon-cap must be positive")
        for item in resolved["scenarios"]:
            item["max_steps"] = min(int(item["max_steps"]), cap)
    episodes = int(resolved["episodes"])
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0 <= int(resolved["gif_count"]) <= episodes:
        raise ValueError("gif_count must be between zero and episodes")
    if int(resolved["workers"]) <= 0:
        raise ValueError("workers must be positive")
    return resolved


def _checkpoint_step(path: Path, payload: Mapping[str, Any]) -> int:
    runtime = payload.get("runtime_state", {}) or {}
    if isinstance(runtime, Mapping) and runtime.get("transition_count") is not None:
        return int(runtime["transition_count"])
    for parent in (path.parent, path.parent.parent):
        digits = "".join(ch for ch in parent.name if ch.isdigit())
        if parent.name.startswith("step_") and digits:
            return int(digits)
    return 0


def verify_config_contract(
    config: Dict[str, Any],
    contract: Mapping[str, Any],
    scenarios: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_root_hash = str(contract.get("config_hash", ""))
    observed_root_hash = _stable_hash(config)
    if expected_root_hash and expected_root_hash != observed_root_hash:
        raise ValueError(
            "checkpoint/effective_config hash mismatch: "
            f"checkpoint={expected_root_hash} config={observed_root_hash}"
        )
    expected_scene_hashes = dict(contract.get("effective_config_hashes", {}) or {})
    observed_scene_hashes: Dict[str, str] = {}
    for item in scenarios:
        scene = str(item["scene"])
        effective = scene_config(config, scene)
        observed = _stable_hash(effective)
        observed_scene_hashes[scene] = observed
        expected = str(expected_scene_hashes.get(scene, ""))
        if expected and expected != observed:
            raise ValueError(
                f"checkpoint/tasks.{scene} hash mismatch: checkpoint={expected} config={observed}"
            )
        configured_horizon = int(effective["env"]["episode_max_length"])
        if int(item["max_steps"]) > configured_horizon:
            raise ValueError(
                f"preset {scene} max_steps={item['max_steps']} exceeds effective env horizon "
                f"{configured_horizon}"
            )
    action = config.get("action", {}) or {}
    expected_action = str(contract.get("action_mode", ""))
    observed_action = str(action.get("mode", ""))
    if expected_action and expected_action != observed_action:
        raise ValueError(
            f"checkpoint action_mode={expected_action!r} != config action.mode={observed_action!r}"
        )
    for key in ("a_max", "w_max"):
        if contract.get(key) is None or action.get(key) is None:
            continue
        if not math.isclose(float(contract[key]), float(action[key]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"checkpoint {key}={contract[key]} != config action.{key}={action[key]}")
    return {
        "root_hash": observed_root_hash,
        "scene_hashes": observed_scene_hashes,
        "action_mode": observed_action,
    }


def build_plan(
    config_path: Path,
    checkpoint: Path,
    preset_path: Path,
    preset: Dict[str, Any],
    device: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config = load_config(str(config_path))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = _require_mapping(payload.get("contract"), "checkpoint.contract")
    verification = verify_config_contract(config, contract, preset["scenarios"])
    plan = {
        "created_at": now(),
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": _checkpoint_step(checkpoint, payload),
        "preset_config": str(preset_path.resolve()),
        "profile": str(preset["profile"]),
        "episodes_per_scenario": int(preset["episodes"]),
        "gif_count_per_scenario": int(preset["gif_count"]),
        "seed": int(preset["seed"]),
        "workers": int(preset["workers"]),
        "device": str(device),
        "deterministic_actor": True,
        "paired_seed_contract": "Use this preset unchanged for last and best checkpoints.",
        "checkpoint_contract": contract,
        "config_verification": verification,
        "display": copy.deepcopy(preset["display"]),
        "scenarios": copy.deepcopy(preset["scenarios"]),
    }
    return plan, config, contract, payload


def _active_positions(env: VorAdjEnv, kind: str) -> np.ndarray:
    entities = env.pursuers if kind == "pursuer" else env.evaders
    return np.asarray(
        [[float(item.x), float(item.y)] for item in entities if not item.deactivated],
        dtype=float,
    ).reshape(-1, 2)


def _min_enemy_distance(env: VorAdjEnv) -> Optional[float]:
    pursuers = _active_positions(env, "pursuer")
    evaders = _active_positions(env, "evader")
    if not len(pursuers) or not len(evaders):
        return None
    return float(min(np.linalg.norm(p - e) for p in pursuers for e in evaders))


def _ce_energy(env: VorAdjEnv) -> Optional[float]:
    pursuers = _active_positions(env, "pursuer")
    if not len(pursuers) or not env._ce_coverage_enabled():
        return None
    evaders = _active_positions(env, "evader")
    values = env._voradj_coverage_potentials(pursuers, evaders)
    return float(np.mean(values)) if len(values) else None


def _mean_finite(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return float(np.mean(values)) if values else None


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(records)

    def rate(key: str) -> float:
        return float(np.mean([bool(item.get(key, False)) for item in records])) if records else 0.0

    return {
        "episodes": count,
        "episode_success_rate": rate("episode_success"),
        "capture_rate": rate("captured"),
        "coverage_strict_rate": rate("coverage_strict_success"),
        "coverage_cv015_rate": rate("coverage_cv015_success"),
        "coverage_cv020_rate": rate("coverage_cv020_success"),
        "collision_rate": rate("collision_event"),
        "detected_rate": rate("detected"),
        "mean_episode_length": _mean_finite(records, "length"),
        "mean_discovery_step": _mean_finite(records, "discovery_step"),
        "mean_initial_min_distance": _mean_finite(records, "initial_min_distance"),
        "mean_final_min_distance": _mean_finite(records, "final_min_distance"),
        "mean_min_min_distance": _mean_finite(records, "min_min_distance"),
        "mean_distance_progress": _mean_finite(records, "distance_progress"),
        "mean_initial_ce_energy": _mean_finite(records, "initial_ce_energy"),
        "mean_final_ce_energy": _mean_finite(records, "final_ce_energy"),
        "mean_ce_energy_progress": _mean_finite(records, "ce_energy_progress"),
        "mean_coverage_area_cv": _mean_finite(records, "coverage_area_cv"),
        "mean_coverage_ce_center_rms": _mean_finite(records, "coverage_ce_center_rms"),
        "mean_coverage_ce_center_max": _mean_finite(records, "coverage_ce_center_max"),
        "mean_action_norm": _mean_finite(records, "action_norm_mean"),
        "mean_speed": _mean_finite(records, "speed_mean"),
        "mean_rollout_wall_seconds": _mean_finite(records, "rollout_wall_seconds"),
    }


def _display_status(scene: str, captured: bool, covered: bool, timed_out: bool) -> Dict[str, Any]:
    if scene == "capture":
        episode_success = captured
        coverage_text = status_text(False, not_started=True)
    elif scene == "pure_ce":
        episode_success = covered
        coverage_text = status_text(covered, timed_out=timed_out and not covered)
    else:
        episode_success = captured and covered
        coverage_text = status_text(
            covered,
            timed_out=timed_out and captured and not covered,
            not_started=not captured,
        )
    return {
        "capture_success": status_text(captured, timed_out=timed_out and not captured),
        "coverage_success": coverage_text,
        "episode_success": status_text(episode_success, timed_out=timed_out and not episode_success),
        "capture_success_bool": bool(captured),
        "coverage_success_bool": bool(covered),
        "episode_success_bool": bool(episode_success),
    }


def rollout_episode(
    trainer: Any,
    root_config: Dict[str, Any],
    scenario_spec: Dict[str, Any],
    seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scene = str(scenario_spec["scene"])
    config = scene_config(root_config, scene)
    horizon = min(int(scenario_spec["max_steps"]), int(config["env"]["episode_max_length"]))
    set_global_config(config)
    env = VorAdjEnv(config, seed=int(seed))
    observations = list(env.reset())
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    initial_distance = _min_enemy_distance(env)
    initial_ce_energy: Optional[float] = None
    min_distances: List[float] = []
    ce_energies: List[float] = []
    action_norms: List[float] = []
    speeds: List[float] = []
    discovery_step: Optional[int] = None

    def phase() -> str:
        if scene == "pure_ce":
            return "coverage"
        return "capture" if any(not e.deactivated for e in env.evaders) else "coverage"

    current_phase = phase()
    if current_phase == "coverage":
        initial_ce_energy = _ce_energy(env)
    phase_index = 0
    frames = [snapshot_env(env, str(scenario_spec["output_name"]), current_phase, phase_index, 0, 0)]
    wall_start = time.perf_counter()
    with torch.no_grad():
        for step in range(1, horizon + 1):
            padded = _pad_local_obs_tree(
                observations,
                int(config["training"]["max_agents"]),
                int(config["actor"]["max_pursuers"]),
                int(config["actor"].get("self_feature_dim", 9)),
            )
            sampled, _ = _sample_actions(
                trainer,
                padded,
                len(env.pursuers),
                env.action_adapter,
                deterministic=True,
            )
            actions: List[Optional[np.ndarray]] = [
                None if observations[index] is None else sampled[index]
                for index in range(len(env.pursuers))
            ]
            outcome = env.step(actions, _evader_actions_for_env(env, apf_agents))
            action_norms.extend(float(np.linalg.norm(value)) for value in actions if value is not None)
            speeds.extend(float(p.speed) for p in env.pursuers if not p.deactivated)
            distance = _min_enemy_distance(env)
            if distance is not None:
                min_distances.append(distance)
            if discovery_step is None and any(
                str((info.get("replay_metadata", {}) or {}).get("task_label", "")) == "capture"
                for info in outcome.infos
            ):
                discovery_step = step
            observations = list(outcome.observations)
            next_phase = phase()
            if next_phase != current_phase:
                phase_index += 1
                current_phase = next_phase
            if current_phase == "coverage":
                energy = _ce_energy(env)
                if energy is not None:
                    if initial_ce_energy is None:
                        initial_ce_energy = energy
                    ce_energies.append(energy)
            frame = snapshot_env(
                env,
                str(scenario_spec["output_name"]),
                current_phase,
                phase_index,
                int(getattr(env, "post_capture_step", step)) if current_phase == "coverage" else step,
                step,
            )
            current_record = env.episode_record(task=scene)
            frame["ce_center_rms"] = float(current_record.get("coverage_ce_center_rms", float("nan")))
            frame["ce_center_max"] = float(current_record.get("coverage_ce_center_max", float("nan")))
            frames.append(frame)
            if all(outcome.dones):
                break
    rollout_wall_seconds = time.perf_counter() - wall_start
    record = env.episode_record(task=scene)
    captured = bool(record.get("captured", False))
    coverage_available = bool(
        scene == "pure_ce" or (scene == "mixed_crms" and any(frame["phase"] == "coverage" for frame in frames))
    )
    area_cv = (
        float(record.get("coverage_strict_area_cv", float("inf")))
        if coverage_available
        else None
    )
    covered = bool(coverage_available and record.get("coverage_strict_success", False))
    timed_out = len(frames) - 1 >= horizon and not all(
        [captured if scene == "capture" else covered]
    )
    display = _display_status(scene, captured, covered, timed_out)
    for frame in frames:
        frame["display_status"] = dict(display)
    final_distance = min_distances[-1] if min_distances else None
    summary = {
        "scenario": scene,
        "output_name": str(scenario_spec["output_name"]),
        "seed": int(seed),
        "max_steps": horizon,
        "length": int(record.get("length", len(frames) - 1)),
        "timed_out": bool(timed_out),
        "episode_success": bool(display["episode_success_bool"]),
        "captured": captured,
        "coverage_strict_success": covered,
        "coverage_available": coverage_available,
        "coverage_cv015_success": bool(
            area_cv is not None and math.isfinite(area_cv) and area_cv <= 0.15
        ),
        "coverage_cv020_success": bool(
            area_cv is not None and math.isfinite(area_cv) and area_cv <= 0.20
        ),
        "coverage_area_cv": area_cv,
        "coverage_ce_center_rms": (
            float(record.get("coverage_ce_center_rms", float("inf")))
            if coverage_available
            else None
        ),
        "coverage_ce_center_max": (
            float(record.get("coverage_ce_center_max", float("inf")))
            if coverage_available
            else None
        ),
        "collision_event": bool(record.get("collision_event", False)),
        "detected": discovery_step is not None,
        "discovery_step": discovery_step,
        "initial_min_distance": initial_distance,
        "final_min_distance": final_distance,
        "min_min_distance": float(min(min_distances)) if min_distances else None,
        "distance_progress": (
            float(initial_distance - final_distance)
            if initial_distance is not None and final_distance is not None
            else None
        ),
        "initial_ce_energy": initial_ce_energy,
        "final_ce_energy": ce_energies[-1] if ce_energies else None,
        "ce_energy_progress": (
            float(initial_ce_energy - ce_energies[-1])
            if initial_ce_energy is not None and ce_energies
            else None
        ),
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
        "speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "rollout_wall_seconds": rollout_wall_seconds,
    }
    return frames, summary


def _run_chunk(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    config = load_config(str(payload["config"]))
    checkpoint = Path(payload["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = dict(saved["contract"])
    trainer = _make_trainer(config, str(payload["device"]))
    trainer.load_checkpoint(checkpoint, contract)
    trainer.actor.eval()
    display = dict(payload["display"])
    results: Dict[str, List[Dict[str, Any]]] = {
        str(item["output_name"]): [] for item in payload["scenarios"]
    }
    for spec in payload["scenarios"]:
        output_name = str(spec["output_name"])
        scenario_dir = Path(payload["output_root"]) / output_name
        for episode_index in payload["episode_indices"]:
            episode_seed = int(payload["seed"]) + episode_index
            frames, summary = rollout_episode(trainer, config, spec, episode_seed)
            if episode_index < int(payload["gif_count"]):
                gif_path = scenario_dir / f"rollout_{output_name}_seed_{episode_seed}.gif"
                render_record = render_gif(
                    frames,
                    gif_path,
                    int(display["max_gif_frames"]),
                    int(display.get("trail_window", 70)),
                    int(display["frame_duration_ms"]),
                    f"MASAC {payload['profile']} / {output_name}",
                    bool(display.get("draw_neighbor_edges", False)),
                    bool(display.get("draw_trails", False)),
                    bool(display.get("draw_sensing_circles", False)),
                    bool(spec.get("draw_ce_targets", False)),
                )
                summary["render"] = render_record
                (scenario_dir / f"episode_{output_name}_seed_{episode_seed}.json").write_text(
                    json.dumps({"summary": summary, "frames": frames}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (scenario_dir / f"summary_{output_name}_seed_{episode_seed}.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            results[output_name].append(summary)
    return results


def split_episode_indices(total: int, workers: int) -> List[List[int]]:
    workers = min(max(1, int(workers)), int(total))
    return [list(range(index, int(total), workers)) for index in range(workers)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset-config", required=True)
    parser.add_argument("--config", required=True, help="Exact source YAML or saved effective_config.yaml.")
    parser.add_argument("--checkpoint", required=True, help="Current CTDE trainer.pt checkpoint.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--gif-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--horizon-cap", type=int, default=None, help="Smoke-only cap applied to every scenario.")
    parser.add_argument("--frame-duration-ms", type=int, default=None)
    parser.add_argument("--max-gif-frames", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preset_path = Path(args.preset_config)
    config_path = Path(args.config)
    checkpoint = Path(args.checkpoint)
    preset = apply_overrides(load_preset(preset_path), args)
    plan, config, _, _ = build_plan(
        config_path,
        checkpoint,
        preset_path,
        preset,
        str(args.device),
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to mix outputs into non-empty directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    for spec in preset["scenarios"]:
        (output_root / str(spec["output_name"])).mkdir(parents=True, exist_ok=True)
    (output_root / "run_args.json").write_text(
        json.dumps(plan | {"output_root": str(output_root.resolve())}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(preset_path, output_root / "preset_source.yaml")
    (output_root / "preset_resolved.yaml").write_text(
        yaml.safe_dump(preset, sort_keys=False), encoding="utf-8"
    )

    common = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "device": str(args.device),
        "profile": str(preset["profile"]),
        "seed": int(preset["seed"]),
        "gif_count": int(preset["gif_count"]),
        "display": preset["display"],
        "scenarios": preset["scenarios"],
    }
    started = time.perf_counter()
    merged: Dict[str, List[Dict[str, Any]]] = {
        str(item["output_name"]): [] for item in preset["scenarios"]
    }
    chunks = split_episode_indices(int(preset["episodes"]), int(preset["workers"]))
    if len(chunks) == 1:
        responses = [_run_chunk(common | {"episode_indices": chunks[0]})]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as executor:
            futures = [
                executor.submit(_run_chunk, common | {"episode_indices": episode_indices})
                for episode_indices in chunks
            ]
            responses = [future.result() for future in as_completed(futures)]
    for response in responses:
        for output_name, records in response.items():
            merged[output_name].extend(records)
    all_summaries: Dict[str, Any] = {}
    for output_name, records in merged.items():
        records.sort(key=lambda item: int(item["seed"]))
        payload = {"scenario": output_name, "summary": summarize(records), "records": records}
        (output_root / output_name / "batch_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        all_summaries[output_name] = payload["summary"]
    (output_root / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    timing = {
        "created_at": now(),
        "wall_seconds": time.perf_counter() - started,
        "episodes_per_scenario": int(preset["episodes"]),
        "scenario_count": len(preset["scenarios"]),
        "workers": len(chunks),
        "device": str(args.device),
    }
    (output_root / "timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "all_summaries": all_summaries, "timing": timing}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
