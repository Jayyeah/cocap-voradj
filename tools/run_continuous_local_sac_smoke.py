"""Short real-environment local-SAC smoke for the P4 gate.

This intentionally does not start a long training job. It collects a small,
deterministic joint replay from pure-CE, capture, and mixed scenes, then runs a
few shared SAC updates and persists the replay with a strict manifest.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from cocap_voradj.config import ConfigManager
from cocap_voradj.dynamics.continuous_action import AccelerationActionAdapter
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer, JointReplaySampler
from cocap_voradj.training.continuous.local_sac import LocalSACConfig, LocalSACTrainer
from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/continuous_marl_20260804/p4_local_sac_smoke_4v1.yaml"
ARTIFACT_ROOT = ROOT / "artifacts/2026-08-04_continuous_marl_refactor"
REPLAY_PATH = ARTIFACT_ROOT / "p4_local_sac_smoke_joint_replay.pkl"
REPORT_PATH = ARTIFACT_ROOT / "p4_local_sac_smoke_report.json"
SEED = 2026080404


def _scene_config(scene: str) -> Dict[str, Any]:
    config = copy.deepcopy(load_config(str(CONFIG_PATH)))
    config["env"]["num_pursuers"] = 4
    config["env"]["num_evaders"] = 0 if scene == "pure_ce" else 1
    config["env"]["episode_max_length"] = 128
    config["env"]["num_obstacles"] = 0
    ConfigManager.get_instance().update_config(config)
    return config


def _stack_obs(observations: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.stack([observation[key] for observation in observations], axis=0) for key in observations[0]}


def _global_state(env: VorAdjEnv) -> np.ndarray:
    pursuer = np.asarray([[p.x, p.y, p.theta, p.speed, p.velocity[0], p.velocity[1]] for p in env.pursuers], dtype=np.float32)
    evader = np.zeros((3, 6), dtype=np.float32)
    for idx, entity in enumerate(env.evaders[:3]):
        evader[idx] = [entity.x, entity.y, float(entity.deactivated), entity.speed, entity.velocity[0], entity.velocity[1]]
    return np.concatenate([pursuer.reshape(-1), evader.reshape(-1)]).astype(np.float32)


def collect_scene(replay: JointReplayBuffer, scene: str, steps: int, rng: np.random.Generator) -> int:
    config = _scene_config(scene)
    env = VorAdjEnv(config, seed=SEED + len(scene))
    env.reset()
    adapter = AccelerationActionAdapter(a_max=0.8, decision_dt=0.5)
    obs = env.get_observations()
    collected = 0
    for step in range(int(steps)):
        before_obs = _stack_obs([item for item in obs if item is not None])
        before_global = _global_state(env)
        actions = []
        for idx, pursuer in enumerate(env.pursuers):
            if pursuer.deactivated:
                actions.append([0.0, 0.0])
                continue
            desired = rng.normal(0.0, 0.25, size=2)
            desired *= min(1.0, 0.8 / max(float(np.linalg.norm(desired)), np.finfo(float).eps))
            action, _ = adapter.validate_with_diagnostics(desired)
            actions.append(action.tolist())
        result = env.step(actions, [None] * len(env.evaders))
        next_obs_list = [item for item in result.observations if item is not None]
        if len(next_obs_list) != len(env.pursuers):
            break
        replay.add(
            local_obs=before_obs,
            next_local_obs=_stack_obs(next_obs_list),
            global_state=before_global,
            next_global_state=_global_state(env),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(result.rewards, dtype=np.float32),
            active_mask=np.asarray([not p.deactivated for p in env.pursuers], dtype=bool),
            terminated=np.asarray(result.dones, dtype=bool),
            truncated=np.zeros(len(env.pursuers), dtype=bool),
            metadata={
                "regime": "coverage_only" if scene == "pure_ce" else "active_target",
                "coverage_only": scene == "pure_ce",
                "active_target": scene != "pure_ce",
                "event_ids": ["discovery"] if scene != "pure_ce" and step == 0 else [],
                "scene": scene,
                "phase": "pure_coverage" if scene == "pure_ce" else "pre_capture",
            },
        )
        obs = next_obs_list
        collected += 1
        if all(result.dones):
            env.reset()
            obs = env.get_observations()
    return collected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-per-scene", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    replay_path = ARTIFACT_ROOT / f"p4_local_sac_smoke_joint_replay{suffix}.pkl"
    report_path = ARTIFACT_ROOT / f"p4_local_sac_smoke_report{suffix}.json"
    replay = JointReplayBuffer(capacity=max(256, args.steps_per_scene * 4), max_agents=4, seed=SEED)
    rng = np.random.default_rng(SEED)
    scene_counts = {scene: collect_scene(replay, scene, args.steps_per_scene, rng) for scene in ("pure_ce", "capture", "mixed_crms")}
    sampler = JointReplaySampler()
    trainer = LocalSACTrainer(
        encoder_config=LocalEntityTokenEncoderConfig(hidden_dim=16, num_heads=4, num_layers=1),
        actor_config=RadialActorConfig(hidden_dim=16, a_max=0.8, decision_dt=0.5),
        config=LocalSACConfig(hidden_dim=16),
    )
    update_metrics = []
    for _ in range(int(args.updates)):
        update_metrics.append(trainer.update(replay.sample(min(int(args.batch_size), len(replay)), sampler=sampler)))
    manifest = {
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "action_mode": "acceleration_2d_body",
        "v_max": 3.0,
        "a_max": 0.8,
        "decision_dt": 0.5,
        "max_agents": 4,
        "encoder_hidden_dim": 16,
    }
    replay.save(replay_path, manifest)
    payload = {
        "schema_version": 1,
        "kind": "continuous_local_sac_real_env_smoke",
        "seed": SEED,
        "scene_counts": scene_counts,
        "replay_size": len(replay),
        "sampler_last_stats": dict(replay.last_sample_stats),
        "update_metrics": update_metrics,
        "all_finite": bool(all(metric["finite"] == 1.0 for metric in update_metrics)),
        "replay_path": str(replay_path),
        "manifest": manifest,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"report={report_path}")
    return 0 if payload["all_finite"] and len(replay) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
