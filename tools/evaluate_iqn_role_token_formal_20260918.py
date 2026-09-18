#!/usr/bin/env python3
"""Deterministic matched milestone evaluator for the IQN ROLE scratch line."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.evaluation.mission_events import MissionEventTracker, snapshot
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config
from tools.evaluate_vxy_stage2_cross_retention_20260903 import ring_count
from tools.rollout_voradj_visual import act_evaders


SCHEMA = "iqn-role-token-formal-eval-v1"
SCENES = ("coverage", "capture", "mixed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def resolved_config(config_path: Path, scene: str, device: str) -> dict[str, Any]:
    root = load_config(str(config_path))
    task = "voradj_coverage" if scene == "coverage" else "voradj"
    cfg = deep_update(root, root.get("tasks", {}).get(task, {}))
    cfg["device"] = device
    cfg.setdefault("env", {})["num_evaders"] = 0 if scene == "coverage" else 1
    if scene == "capture":
        cfg.setdefault("voradj", {})["capture_episode_ends_on_capture"] = True
        cfg["voradj"]["capture_episode_success_on_capture"] = True
    else:
        cfg.setdefault("voradj", {})["capture_episode_ends_on_capture"] = False
        cfg["voradj"]["capture_episode_success_on_capture"] = False
    return cfg


def assert_role_contract(model: CoCapIQN, config: Mapping[str, Any]) -> None:
    perception = config.get("perception", {}) or {}
    iqn = config.get("iqn", {}) or {}
    if not perception.get("include_is_pursuing") or perception.get("include_z_state"):
        raise AssertionError("formal evaluator received a non-ROLE observation contract")
    if perception.get("friend_ordering_mode") != "physical_only":
        raise AssertionError("ROLE formal evaluator requires physical-only friend ordering")
    if not iqn.get("include_is_pursuing") or iqn.get("include_z_state"):
        raise AssertionError("formal evaluator received a non-ROLE IQN config")
    if iqn.get("pursuing_late_fusion", True) or int(iqn.get("pursuing_embed_dim", 0)) != 0:
        raise AssertionError("ROLE formal evaluator must have no pursuing late-fusion branch")
    if not model.config.include_is_pursuing or model.config.include_z_state:
        raise AssertionError("checkpoint model is not a ROLE model")
    if model.config.pursuing_late_fusion or model.pursuing_embed is not None:
        raise AssertionError("checkpoint retains pursuing late fusion")


def _timing(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "n": len(clean),
        "mean": float(np.mean(clean)) if clean else None,
        "p50": float(np.percentile(clean, 50)) if clean else None,
        "p90": float(np.percentile(clean, 90)) if clean else None,
    }


@torch.no_grad()
def run_episode(
    model: CoCapIQN,
    config_path: Path,
    scene: str,
    seed: int,
    device: str,
) -> dict[str, Any]:
    config = resolved_config(config_path, scene, device)
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=int(seed))
    observations = list(env.reset())
    for observation in observations:
        if observation is None:
            continue
        if observation["self"].shape != (9,) or observation["pursuers"].shape != (8, 7):
            raise AssertionError("ROLE observation shape drift during formal evaluation")
        role_values = np.asarray(observation["self"][-1:]).tolist()
        role_values += np.asarray(observation["pursuers"][:, -1]).tolist()
        if any(float(value) not in {0.0, 1.0} for value in role_values):
            raise AssertionError("ROLE token is not binary")

    dt = float(env.pursuers[0].dt * env.pursuers[0].N)
    apf = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    tracker = MissionEventTracker(dt)
    tracker.observe(snapshot(env, observations), 0)
    capture_step: int | None = None
    ce_step: int | None = None
    capture_types: list[str] = []
    collision = False
    ring2_seen = False
    ring3_seen = False
    action_histogram = [0] * int(model.config.action_size)
    last_step = 0

    for step in range(1, int(env.episode_max_length) + 1):
        last_step = step
        active = [idx for idx, observation in enumerate(observations) if observation is not None]
        if not active:
            break
        batch = stack_obs([observations[idx] for idx in active], device)
        actions = model.act(
            batch,
            mode="voradj",
            epsilon=0.0,
            deterministic_quantiles=True,
        ).detach().cpu().numpy().astype(int).tolist()
        commands: list[int | None] = [None] * len(env.pursuers)
        for idx, action in zip(active, actions):
            commands[idx] = int(action)
            action_histogram[int(action)] += 1
        ring = int(ring_count(env))
        ring2_seen = ring2_seen or ring >= 2
        ring3_seen = ring3_seen or ring >= 3
        outcome = env.step(commands, act_evaders(env, apf))
        events = list(env.last_capture_events)
        capture_types.extend(str(event.get("capture_type", "unknown")) for event in events)
        collision = collision or bool(env.last_collision_events)
        captured_now = bool(
            env.evaders
            and all(evader.deactivated and not evader.collision for evader in env.evaders)
        )
        if captured_now and capture_step is None:
            capture_step = step
        if env.post_capture_coverage_success and ce_step is None:
            ce_step = step
        tracker.observe(snapshot(env, outcome.observations), step, events)
        observations = list(outcome.observations)
        if all(outcome.dones):
            break

    record = env.episode_record(task="coverage" if scene == "coverage" else "mix")
    captured = bool(record.get("captured", False))
    ce_success = bool(env.post_capture_coverage_success or record.get("coverage_strict_success", False))
    collision = bool(collision or record.get("collision_event", False))
    if scene == "capture":
        safe_complete = bool(captured and not collision and record.get("all_pursuers_active", False))
    elif scene == "coverage":
        safe_complete = bool(ce_success and not collision and record.get("all_pursuers_active", False))
    else:
        safe_complete = bool(
            captured and ce_success and not collision and record.get("all_pursuers_active", False)
        )
    return {
        "scene": scene,
        "seed": int(seed),
        "steps": int(last_step),
        "episode_seconds": float(last_step * dt),
        "captured": captured,
        "normal_capture": bool(captured and "loose" in capture_types),
        "stationary_capture": bool(captured and "stationary" in capture_types),
        "collision": collision,
        "post_capture_ce": ce_success if scene == "mixed" else None,
        "ce_success": ce_success,
        "safe_complete": safe_complete,
        "capture_seconds": float(capture_step * dt) if capture_step is not None else None,
        "recovery_seconds": (
            float((ce_step - capture_step) * dt)
            if ce_step is not None and capture_step is not None
            else None
        ),
        "time_to_ce_seconds": float(ce_step * dt) if ce_step is not None and ce_success else None,
        "mission_seconds": (
            float(ce_step * dt)
            if scene == "mixed" and safe_complete and ce_step is not None
            else float(capture_step * dt)
            if scene == "capture" and safe_complete and capture_step is not None
            else float(ce_step * dt)
            if scene == "coverage" and safe_complete and ce_step is not None
            else None
        ),
        "ring2": bool(ring2_seen),
        "ring3": bool(ring3_seen),
        "ce_rms": float(record.get("coverage_ce_center_rms", float("inf"))),
        "area_cv": float(record.get("coverage_strict_area_cv", float("inf"))),
        "action_histogram": action_histogram,
        "episode_record": record,
        "mission_events": tracker.finish(last_step),
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([bool(row.get(key, False)) for row in rows])) if rows else 0.0


def summarize_scene(rows: Sequence[Mapping[str, Any]], scene: str) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "capture_rate": _rate(rows, "captured"),
        "normal_capture_rate": _rate(rows, "normal_capture"),
        "stationary_capture_rate": _rate(rows, "stationary_capture"),
        "collision_rate": _rate(rows, "collision"),
        "ring2_rate": _rate(rows, "ring2"),
        "ring3_rate": _rate(rows, "ring3"),
        "ce_success_rate": _rate(rows, "ce_success"),
        "post_capture_ce_rate": _rate(rows, "post_capture_ce"),
        "safe_complete_rate": _rate(rows, "safe_complete"),
        "capture_seconds": _timing([row.get("capture_seconds") for row in rows]),
        "recovery_seconds": _timing([row.get("recovery_seconds") for row in rows]),
        "time_to_ce_seconds": _timing([row.get("time_to_ce_seconds") for row in rows]),
        "mission_seconds": _timing([row.get("mission_seconds") for row in rows]),
        "ce_rms_mean": _timing([row.get("ce_rms") for row in rows]),
        "area_cv_mean": _timing([row.get("area_cv") for row in rows]),
        "scene_contract": scene,
    }


def evaluate_checkpoint(
    config_path: Path,
    checkpoint: Path,
    output: Path,
    episodes: int = 20,
    seed_base: int = 2026092801,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = CoCapIQN.load(str(checkpoint), device=device).eval()
    root_config = load_config(str(config_path))
    assert_role_contract(model, root_config)
    rows: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(SCENES):
        for episode in range(int(episodes)):
            rows.append(
                run_episode(
                    model,
                    config_path,
                    scene,
                    int(seed_base + scene_index * 100000 + episode),
                    device,
                )
            )
    summary = {
        scene: summarize_scene([row for row in rows if row["scene"] == scene], scene)
        for scene in SCENES
    }
    histogram = np.sum(np.asarray([row["action_histogram"] for row in rows], dtype=np.int64), axis=0)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "episodes_per_scene": int(episodes),
        "seed_base": int(seed_base),
        "deterministic": True,
        "policy_quantiles": "fixed_midpoint_32",
        "summary": summary,
        "action_histogram": {str(index): int(value) for index, value in enumerate(histogram)},
        "records": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2026092801)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    payload = evaluate_checkpoint(
        args.config.resolve(),
        args.checkpoint.resolve(),
        args.out.resolve(),
        episodes=args.episodes,
        seed_base=args.seed_base,
        device=args.device,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
