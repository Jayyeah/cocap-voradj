#!/usr/bin/env python3
"""Matched discrete baselines for the local-TD3 Stage-1 capability audit.

This is intentionally a single-task runner.  It reuses the validated
NormSense-V2 environment, historical IQN update, and production categorical
MAPPO update without changing reward, observation, dynamics, or task
semantics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.numeric_validation import assert_finite_numeric_tree
from cocap_voradj.training.runtime_semantics import assert_runtime
from cocap_voradj.training.small_step_ac import tensor_tree
from cocap_voradj.training.td3_stage1_contract import aw9_grid, discrete_task_config
from cocap_voradj.training.trainer import CoCapTrainer, set_global_config, stack_obs
from tools import forward_final_single_task_20260915 as single
from tools import launch_normsense_v2_formal as production
from tools import preflight_forward_final_scratch_20260914 as scratch
from tools import train_forward_final_ppo_20260909 as rollout_tools


SCHEMA = "td3-stage1-discrete-baseline-v1"
SEED = 2026091501
EVAL_SEED_BASE = 2026191501
BUDGET = 100_000
MILESTONES = (25_000, 50_000, 75_000, 100_000)
MATCHED_MAPPO_INITIAL = {
    "actor": "965553646883e512219e85cf7e40036c3b23897136683335ea0c2cc6c5086f4f",
    "value": "a1b1c9091bf1c1b6c1a26cf92f1e515e070e3134d002a731918966f2b666c78b",
    "value_norm": "17051316105d8c8d82e64395f9ef016c2a21c9b358be4fe01a8b445930268515",
}
IQN_HYPERPARAMETERS = {
    "architecture": "voradj_single_head",
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "action_size": 9,
    "self_feature_dim": 9,
    "pursuing_embed_dim": 8,
    "dropout": 0.1,
    "num_quantiles": 32,
    "num_cosine_features": 64,
    "batch_size": 128,
    "replay_capacity": 1_000_000,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "min_replay_size": 3_000,
    "train_freq": 4,
    "target_update_freq": 10_000,
    "checkpoint_freq": 25_000,
    "epsilon_start": 0.6,
    "epsilon_final": 0.05,
    "epsilon_decay_steps": 1_000_000,
    "gradient_clip": 0.5,
    "gradient_clip_norm": 0.5,
    "huber_kappa": 1.0,
    "train_quantiles": 8,
    "target_quantiles": 8,
    "training_quantile_samples": 8,
    "action_quantile_samples": 32,
    "replay_done_mode": "terminated_only",
    "update_rule": "distributional_iqn",
    "log_freq_steps": 1_000,
    "learning_rate_schedule": [{"step": 0, "learning_rate": 1e-4}],
}


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    return scratch.tensor_hash(state)


def implementation_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "src/cocap_voradj/training/trainer.py",
        ROOT / "src/cocap_voradj/models/iqn.py",
        ROOT / "src/cocap_voradj/training/small_step_ac.py",
        ROOT / "src/cocap_voradj/training/td3_stage1_contract.py",
        ROOT / "tools/forward_final_single_task_20260915.py",
        ROOT / "tools/preflight_forward_final_scratch_20260914.py",
        ROOT / "tools/train_forward_final_ppo_20260909.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def check_env(env: VorAdjEnv, task: str, actor=None) -> dict[str, Any]:
    expected = discrete_task_config(task)
    if env.config != expected:
        raise RuntimeError("Discrete baseline environment contract drift")
    facts = single.check_env(env, task, actor)
    radius = float(env._vct_ls_sensing_radius("enemy"))
    runtime = assert_runtime(
        env,
        actor,
        categorical=actor is not None,
        expected={
            "pursuers": 4,
            "evaders": int(task == "capture"),
            "topology": "friendly_voronoi_comm_v0",
            "enemy_token_rule": "surface_radius",
            "global_enemy_flag": False,
            "support_capture_weight": 1.0,
            "support_coverage_weight": 0.0,
            "support_blend_enabled": task == "capture",
            "capture_reward_mode": "ring_importance_ms_v0",
            "capture_radius": 8.0,
            "capture_k": 3,
            "enemy_radius": radius,
            "action_mode": "unicycle_discrete",
            "decision_dt": 0.5,
            "physics_dt": 0.05,
            "a_longitudinal_max": 0.4,
            "omega_max": float(np.pi / 6),
            "v_max": 3.0,
            "drag": 0.4 / 3,
            "collision_semantics": "synchronized_swept_v1",
        },
    )
    np.testing.assert_allclose(env.pursuers[0].action_list, aw9_grid(), atol=1e-12)
    if env.episode_max_length != 3000:
        raise RuntimeError("Discrete baseline horizon drift")
    return {**facts, **runtime, "task": task, "normsense_v2": True}


def make_env(task: str, seed: int) -> tuple[VorAdjEnv, Any]:
    config = discrete_task_config(task)
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=int(seed))
    observations = env.reset()
    check_env(env, task)
    return env, observations


class NormSenseSingleTaskStream(single.SingleTaskStream):
    def __init__(self, task: str, seed: int):
        self.task_name = task
        self.multiplier = 1.0
        self.task = task
        env, self.observations = make_env(task, seed)
        self.envs = {task: env}
        self.apf_agents = {task: [single.p.ApfAgent(e.a, e.w) for e in env.evaders]}
        self.episode = 0
        self.telemetry = single.Telemetry(env, task)
        self.episode_metrics = []
        self.env_steps = 0

    def finish(self):
        row = dict(episode=self.episode, **self.telemetry.finish(self.env))
        self.episode_metrics.append(row)
        self.episode += 1
        set_global_config(self.env.config)
        self.observations = self.env.reset()
        self.apf_agents[self.task] = [single.p.ApfAgent(e.a, e.w) for e in self.env.evaders]
        check_env(self.env, self.task_name)
        self.telemetry = single.Telemetry(self.env, self.task_name)
        return row


def iqn_config(task: str, output: Path, device: str, total_steps: int = BUDGET) -> dict:
    config = copy.deepcopy(discrete_task_config(task))
    config.update(
        seed=SEED,
        device=device,
        output_root=str(output.parent),
        run_name=output.name,
        total_timesteps=int(total_steps),
        train_mode="voradj",
        recent_window=100,
        iqn=copy.deepcopy(IQN_HYPERPARAMETERS),
        checkpointing={"full_resume": False},
    )
    return config


class MatchedIQNTrainer(CoCapTrainer):
    """IQN trainer whose environment remains exactly the pure-task contract."""

    def __init__(self, config: dict, task: str):
        self.stage1_task = task
        self.stage1_wall_started = time.monotonic()
        super().__init__(config)

    def _task_config(self, task: str) -> dict:
        if task != "voradj":
            raise RuntimeError(f"Unexpected IQN task key: {task}")
        return copy.deepcopy(discrete_task_config(self.stage1_task))

    def _make_env(self, task: str) -> VorAdjEnv:
        config = self._task_config(task)
        set_global_config(config)
        return VorAdjEnv(config, seed=self.seed)

    def _set_coverage_ce_control_weights(self) -> float:
        # The current single-task contract runs at the mature 2M reward clock.
        # Generic IQN's training-step schedule would otherwise overwrite the
        # live environment with the historical step-0 value (zero).
        value = float(discrete_task_config(self.stage1_task)["reward"]["coverage_ce_speed_weight"])
        for env in self.envs.values():
            env.reward_cfg["coverage_ce_speed_weight"] = value
        self.current_coverage_ce_speed_weight = value
        return value

    def _select_actions(self, task: str, obs_list):
        self.envs[task].total_steps = 2_000_000 + int(self.global_step)
        return super()._select_actions(task, obs_list)

    def _save_checkpoint(self, name: str) -> Path:
        path = super()._save_checkpoint(name)
        if self.global_step == 0 or self.global_step in MILESTONES or name.startswith("final_"):
            payload = {
                "status": "TRAINING" if self.global_step < self.total_timesteps else "TRAINING_BUDGET_REACHED",
                "step": int(self.global_step),
                "budget": int(self.total_timesteps),
                "wall_seconds": time.monotonic() - self.stage1_wall_started,
                "updates": int(self.update_steps),
                "replay_size": int(len(self.replays["voradj"])),
                "checkpoint": str(path),
            }
            write_json(self.run_dir / "checkpoint_timing" / f"step_{self.global_step:06d}.json", payload)
            write_json(self.run_dir / "status.json", payload)
        return path

    def _reset_task(self, task: str):
        observations = super()._reset_task(task)
        check_env(self.envs[task], self.stage1_task)
        return observations


def _physical_action_stats(actions: list[list[float]]) -> dict[str, float | None]:
    if not actions:
        return {
            "action_a_mean": None,
            "action_a_std": None,
            "action_w_mean": None,
            "action_w_std": None,
            "action_saturation_ratio": None,
        }
    array = np.asarray(actions, dtype=np.float64)
    saturation = np.isclose(np.abs(array[:, 0]), 0.4, atol=1e-9) | np.isclose(
        np.abs(array[:, 1]), np.pi / 6, atol=1e-9
    )
    return {
        "action_a_mean": float(array[:, 0].mean()),
        "action_a_std": float(array[:, 0].std()),
        "action_w_mean": float(array[:, 1].mean()),
        "action_w_std": float(array[:, 1].std()),
        "action_saturation_ratio": float(saturation.mean()),
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


@torch.no_grad()
def evaluate_policy(
    policy,
    algorithm: str,
    task: str,
    output: Path,
    step: int,
    episodes: int = 20,
    seed_base: int = EVAL_SEED_BASE,
    filename: str | None = None,
) -> dict:
    path = output / (filename or f"eval_step_{step:06d}.json")
    actor = policy.model if algorithm == "iqn_trainer" else policy.actor
    actor_hash = tensor_hash(actor.state_dict())
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["actor_sha256"] != actor_hash or existing["episodes"] != episodes:
            raise RuntimeError(f"Existing evaluation mismatch: {path}")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    device = next(actor.parameters()).device
    rng = scratch.rng_state(device)
    records = []
    started = time.monotonic()
    try:
        actor.eval()
        for episode in range(episodes):
            seed = int(seed_base + episode)
            scratch.seed_all(seed)
            stream = NormSenseSingleTaskStream(task, seed)
            check_env(stream.env, task, None if algorithm == "iqn_trainer" else actor)
            observations = stream.observations
            physical_actions: list[list[float]] = []
            histogram = np.zeros(9, dtype=np.int64)
            previous_sign: dict[int, tuple[int, int]] = {}
            switch_count = 0
            switch_denominator = 0
            minimum_distance = math.inf
            ring2_run = ring3_run = ring2_max = ring3_max = 0
            final_geometry: list[dict[str, Any]] = []
            for tick in range(1, stream.env.episode_max_length + 1):
                stream.set_clock(tick - 1)
                active = [i for i, observation in enumerate(observations) if observation is not None]
                local = {key: np.stack([observations[i][key] for i in active])
                         for key in observations[active[0]]}
                if algorithm == "iqn_trainer":
                    batch = stack_obs([observations[i] for i in active], str(device))
                    latent = actor.act(
                        batch, mode="voradj", epsilon=0.0, deterministic_quantiles=True
                    )
                else:
                    latent = actor.distribution(tensor_tree(local, device)).logits.argmax(-1)
                command = np.full(4, 4, dtype=np.int64)
                command[active] = latent.cpu().numpy()
                for focal in active:
                    index = int(command[focal])
                    histogram[index] += 1
                    aw = np.asarray(stream.env.pursuers[focal].action_list[index], dtype=float)
                    physical_actions.append(aw.tolist())
                    sign = (int(np.sign(aw[0])), int(np.sign(aw[1])))
                    if focal in previous_sign:
                        switch_count += int(sign != previous_sign[focal])
                        switch_denominator += 1
                    previous_sign[focal] = sign
                result = stream.step(command)
                if stream.env.evaders:
                    live_evaders = [e for e in stream.env.evaders if not e.deactivated]
                    if live_evaders:
                        distances = [
                            math.hypot(p.x - e.x, p.y - e.y)
                            for p in stream.env.pursuers if not p.deactivated
                            for e in live_evaders
                        ]
                        minimum_distance = min(minimum_distance, min(distances, default=math.inf))
                    sizes = [
                        sum(
                            math.hypot(p.x - e.x, p.y - e.y) <= 8
                            for p in stream.env.pursuers if not p.deactivated
                        )
                        for e in live_evaders
                    ]
                    sizes += [len(event["participants"]) for event in stream.env.last_capture_events]
                    ring = max(sizes, default=0)
                    ring2_run = ring2_run + 1 if ring >= 2 else 0
                    ring3_run = ring3_run + 1 if ring >= 3 else 0
                    ring2_max = max(ring2_max, ring2_run)
                    ring3_max = max(ring3_max, ring3_run)
                    final_geometry.append({
                        "tick": tick,
                        "ring_size": ring,
                        "distances": distances if live_evaders else [],
                        "speeds": [float(p.speed) for p in stream.env.pursuers],
                        "actions": command.tolist(),
                    })
                    final_geometry = final_geometry[-40:]
                observations = result.observations
                if all(result.dones):
                    break
            if not all(result.dones):
                raise RuntimeError("Discrete baseline evaluation did not reach native done")
            row = stream.telemetry.finish(stream.env)
            row.update(
                seed=seed,
                episode=episode,
                mode="argmax",
                native_done=True,
                min_target_distance=(float(minimum_distance) if np.isfinite(minimum_distance) else None),
                repeated_ring2_max=int(ring2_max),
                repeated_ring3_max=int(ring3_max),
                action_histogram=histogram.tolist(),
                action_switch_rate=(switch_count / switch_denominator if switch_denominator else 0.0),
                final_speed_mean=float(np.mean([p.speed for p in stream.env.pursuers])),
                final_speed_max=float(max((p.speed for p in stream.env.pursuers), default=0.0)),
                capture_geometry_last_40=final_geometry,
                **_physical_action_stats(physical_actions),
            )
            records.append(row)
            write_json(
                output / f"{path.stem}_progress.json",
                {
                    "step": step,
                    "complete": len(records),
                    "total": episodes,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        summary = {
            **single.summarize(records),
            "geometric_success_rate": float(np.mean([row["ce_success"] for row in records])),
            "settled_success_rate": float(np.mean([
                bool(row["ce_success"])
                and row["final_speed_mean"] <= 0.15
                and row["final_speed_max"] <= 0.3
                for row in records
            ])),
            "normal_capture_rate": float(np.mean([row["normal_capture"] for row in records])),
            "stationary_capture_rate": float(np.mean([row["stationary_capture"] for row in records])),
            "ring2_visitation": float(np.mean([row["repeated_ring2_max"] > 0 for row in records])),
            "ring3_visitation": float(np.mean([row["repeated_ring3_max"] > 0 for row in records])),
            "repeated_ring2_max": _stats([row["repeated_ring2_max"] for row in records]),
            "repeated_ring3_max": _stats([row["repeated_ring3_max"] for row in records]),
            "min_target_distance": _stats([row["min_target_distance"] for row in records]),
            "final_speed_mean": _stats([row["final_speed_mean"] for row in records]),
            "final_speed_max": _stats([row["final_speed_max"] for row in records]),
            "action_a_mean": _stats([row["action_a_mean"] for row in records]),
            "action_a_std": _stats([row["action_a_std"] for row in records]),
            "action_w_mean": _stats([row["action_w_mean"] for row in records]),
            "action_w_std": _stats([row["action_w_std"] for row in records]),
            "action_saturation_ratio": float(np.mean([row["action_saturation_ratio"] for row in records])),
            "action_switch_rate": float(np.mean([row["action_switch_rate"] for row in records])),
        }
        report = {
            "schema": "td3-stage1-discrete-eval-v1",
            "algorithm": algorithm,
            "task": task,
            "step": int(step),
            "episodes": int(episodes),
            "seed_base": int(seed_base),
            "actor_sha256": actor_hash,
            "deterministic": True,
            "fixed_midpoint_iqn_quantiles": algorithm == "iqn_trainer",
            "records": records,
            "summary": summary,
            "training_rng_preserved": True,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(path, report)
        return report
    finally:
        scratch.restore_rng(rng, device)


def iqn_launch(task: str, trainer: MatchedIQNTrainer, runtime: dict) -> dict:
    return {
        "schema": SCHEMA,
        "id": "D1" if task == "coverage" else "D2",
        "algorithm": "IQN",
        "task": task,
        "initialization": "scratch",
        "seed": SEED,
        "budget": BUDGET,
        "milestones": list(MILESTONES),
        "eval_seed_base": EVAL_SEED_BASE,
        "eval_episodes": 20,
        "resolved_environment": discrete_task_config(task),
        "runtime": runtime,
        "iqn": copy.deepcopy(IQN_HYPERPARAMETERS),
        "actor_initial_sha256": tensor_hash(trainer.model.state_dict()),
        "target_initial_sha256": tensor_hash(trainer.target_model.state_dict()),
        "optimizer_state_initially_empty": not bool(trainer.optimizer.state),
        "teacher_dependency": 0,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": implementation_hashes(),
        "pid": os.getpid(),
        "device": str(trainer.device),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def train_iqn(task: str, output: Path, device: str, total_steps: int = BUDGET) -> None:
    if output.exists():
        raise RuntimeError(f"Fresh output required: {output}")
    preflight_env, _ = make_env(task, SEED)
    runtime = check_env(preflight_env, task)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    trainer = MatchedIQNTrainer(iqn_config(task, output, device, total_steps), task)
    write_json(output / "launch.json", iqn_launch(task, trainer, runtime))
    write_json(output / "status.json", {"status": "RUNNING", "step": 0, "pid": os.getpid()})
    started = time.monotonic()
    final = trainer.train()
    training_wall_seconds = time.monotonic() - started
    evaluation_started = time.monotonic()
    for step in MILESTONES:
        if step > total_steps:
            continue
        checkpoint = output / "checkpoints" / f"step_{step}.pt"
        model = CoCapIQN.load(str(checkpoint), device=str(trainer.device))
        holder = type("IQNHolder", (), {"model": model})()
        evaluate_policy(holder, "iqn_trainer", task, output / "evaluation", step)
        write_json(
            output / "manifests" / f"step_{step:06d}.json",
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "evaluation": str(output / "evaluation" / f"eval_step_{step:06d}.json"),
                "replay_snapshot": False,
                "resume_note": "IQN matched baseline stores immutable policy checkpoints; TD3 owns the replay-snapshot requirement.",
            },
        )
    write_json(
        output / "report.json",
        {
            "status": "COMPLETE_BUDGET",
            "step": total_steps,
            "final_checkpoint": str(final),
            "wall_seconds": time.monotonic() - started,
            "training_wall_seconds": training_wall_seconds,
            "evaluation_wall_seconds": time.monotonic() - evaluation_started,
            "automatic_extension": False,
        },
    )


def mappo_state_hashes(trainer) -> dict[str, str]:
    return {
        "actor": tensor_hash(trainer.actor.state_dict()),
        "value": tensor_hash(trainer.value.state_dict()),
        "value_norm": production.value_norm_hash(trainer.value_norm),
    }


def save_mappo(
    output: Path,
    step: int,
    trainer,
    stream,
    launch: dict,
    rollout: dict,
    metrics: dict,
) -> Path:
    path = output / f"step_{step:06d}.pt"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite immutable checkpoint: {path}")
    payload = {
        "schema": SCHEMA,
        "algorithm": "MAPPO",
        "task": "coverage",
        "step": step,
        "trainer": trainer.state_dict(),
        "stream": stream,
        "launch": launch,
        "rollout": rollout,
        "metrics": metrics,
        "rng": scratch.rng_state(trainer.device),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    verified = torch.load(temporary, map_location="cpu", weights_only=False)
    if verified["schema"] != SCHEMA or verified["step"] != step:
        raise RuntimeError("MAPPO checkpoint round-trip failed")
    os.replace(temporary, path)
    write_json(
        path.with_suffix(".json"),
        {
            "step": step,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "partial_rollout_steps": len(rollout["rewards"]),
            "resume_supported": True,
        },
    )
    return path


def load_mappo(path: Path, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != SCHEMA or payload["algorithm"] != "MAPPO":
        raise RuntimeError("Invalid Stage-1 MAPPO checkpoint")
    trainer = scratch.make_trainer(scratch.load_contract(), SEED, device)
    trainer.load_state_dict(payload["trainer"])
    check_env(payload["stream"].env, "coverage", trainer.actor)
    scratch.restore_rng(payload["rng"], trainer.device)
    return trainer, payload["stream"], payload


def train_mappo(output: Path, device: str, total_steps: int = BUDGET, resume: Path | None = None) -> None:
    if resume is None:
        if output.exists():
            raise RuntimeError(f"Fresh output required: {output}")
        output.mkdir(parents=True)
        scratch.seed_all(SEED)
        trainer = scratch.make_trainer(scratch.load_contract(), SEED, device)
        stream = NormSenseSingleTaskStream("coverage", SEED)
        initial_hashes = mappo_state_hashes(trainer)
        if initial_hashes != MATCHED_MAPPO_INITIAL:
            raise RuntimeError(
                f"D3/D4 scratch initialization mismatch: {initial_hashes}"
            )
        launch = {
            "schema": SCHEMA,
            "id": "D3",
            "algorithm": "categorical_MAPPO_AW9",
            "task": "coverage",
            "initialization": "scratch",
            "seed": SEED,
            "budget": int(total_steps),
            "milestones": list(MILESTONES),
            "eval_seed_base": EVAL_SEED_BASE,
            "eval_episodes": 20,
            "resolved_environment": discrete_task_config("coverage"),
            "runtime": check_env(stream.env, "coverage", trainer.actor),
            "mappo_contract": scratch.load_contract(),
            "initial_hashes": initial_hashes,
            "matched_d4_initial_hashes": MATCHED_MAPPO_INITIAL,
            "teacher_dependency": 0,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "implementation_sha256": implementation_hashes(),
            "pid": os.getpid(),
            "device": str(trainer.device),
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        write_json(output / "launch.json", launch)
        step, rollout, metrics = 0, rollout_tools.empty_rollout(), {}
        save_mappo(output, step, trainer, stream, launch, rollout, metrics)
    else:
        trainer, stream, payload = load_mappo(resume, device)
        step = int(payload["step"])
        rollout = payload["rollout"]
        metrics = payload["metrics"]
        launch = payload["launch"]
        if Path(resume).resolve().parent != output.resolve():
            raise RuntimeError("Resume checkpoint must belong to output")
    started = time.monotonic()
    start_step = step

    def progress(status: str) -> None:
        elapsed = time.monotonic() - started
        speed = (step - start_step) / elapsed if elapsed else 0.0
        write_json(
            output / "status.json",
            {
                "status": status,
                "step": step,
                "budget": total_steps,
                "updates": trainer.update_count,
                "partial_rollout_steps": len(rollout["rewards"]),
                "steps_per_second": speed,
                "eta_seconds": ((total_steps - step) / speed if speed else None),
                "pid": os.getpid(),
            },
        )

    progress("RUNNING")
    while step < total_steps:
        stream.set_clock(step)
        transition, episode = rollout_tools.collect_transition(trainer, stream)
        assert_finite_numeric_tree(transition, path=f"transition[{step}]")
        for key, value in transition.items():
            rollout[key].append(value)
        step += 1
        if episode is not None:
            with (output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **episode}, allow_nan=False) + "\n")
        if len(rollout["rewards"]) == 256:
            metrics = single.update(trainer, rollout, step)
            with (output / "learning.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            rollout = rollout_tools.empty_rollout()
        if step % 100 == 0:
            progress("RUNNING")
        if step in MILESTONES:
            checkpoint = save_mappo(output, step, trainer, stream, launch, rollout, metrics)
            progress("CHECKPOINT_EVAL")
            evaluate_policy(trainer, "mappo", "coverage", output / "evaluation", step)
            write_json(
                output / "manifests" / f"step_{step:06d}.json",
                {
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "evaluation": str(output / "evaluation" / f"eval_step_{step:06d}.json"),
                    "resume_supported": True,
                },
            )
    progress("COMPLETE_BUDGET")
    write_json(
        output / "report.json",
        {
            "status": "COMPLETE_BUDGET",
            "step": step,
            "final_checkpoint": str(output / f"step_{step:06d}.pt"),
            "wall_seconds": time.monotonic() - started,
            "automatic_extension": False,
        },
    )


def audit(output: Path) -> dict:
    rows = {}
    for task in ("coverage", "capture"):
        env, _ = make_env(task, SEED)
        rows[task] = check_env(env, task)
    config_c = iqn_config("coverage", output / "D1", "cpu", 1)
    config_x = iqn_config("capture", output / "D2", "cpu", 1)
    scratch.seed_all(SEED)
    cov = MatchedIQNTrainer(config_c, "coverage")
    scratch.seed_all(SEED)
    cap = MatchedIQNTrainer(config_x, "capture")
    initial_parity = tensor_hash(cov.model.state_dict()) == tensor_hash(cap.model.state_dict())
    if not initial_parity:
        raise RuntimeError("D1/D2 IQN scratch initializations differ")
    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "environment": rows,
        "iqn_initialization_bit_exact_across_tasks": initial_parity,
        "iqn_replay_done_mode": cov.replay_done_mode,
        "mappo_contract": scratch.load_contract(),
        "aw9": aw9_grid().tolist(),
        "budget": BUDGET,
        "milestones": list(MILESTONES),
    }
    return report


def smoke(algorithm: str, task: str, output: Path, device: str) -> None:
    if output.exists():
        raise RuntimeError(f"Fresh output required: {output}")
    if algorithm == "iqn":
        train_iqn(task, output, device, total_steps=16)
        report = {
            "status": "PASS",
            "algorithm": algorithm,
            "task": task,
            "step": 16,
            "note": "Below min replay by design; validates real environment/action/checkpoint path.",
        }
        write_json(output / "smoke.json", report)
        return
    output.mkdir(parents=True)
    scratch.seed_all(SEED)
    trainer = scratch.make_trainer(scratch.load_contract(), SEED, device)
    stream = NormSenseSingleTaskStream("coverage", SEED)
    rollout = rollout_tools.empty_rollout()
    for step in range(256):
        stream.set_clock(step)
        transition, _ = rollout_tools.collect_transition(trainer, stream)
        assert_finite_numeric_tree(transition, path=f"smoke[{step}]")
        for key, value in transition.items():
            rollout[key].append(value)
    metrics = single.update(trainer, rollout, 256)
    launch = {"source": implementation_hashes(), "runtime": check_env(stream.env, "coverage", trainer.actor)}
    checkpoint = save_mappo(output, 256, trainer, stream, launch, rollout_tools.empty_rollout(), metrics)
    clone, _, payload = load_mappo(checkpoint, device)
    if mappo_state_hashes(clone) != mappo_state_hashes(trainer) or payload["step"] != 256:
        raise RuntimeError("MAPPO smoke resume mismatch")
    write_json(
        output / "smoke.json",
        {"status": "PASS", "algorithm": algorithm, "finite": True, "updates": trainer.update_count},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "smoke", "train"), required=True)
    parser.add_argument("--algorithm", choices=("iqn", "mappo"))
    parser.add_argument("--task", choices=("coverage", "capture"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(1)
    if args.mode == "audit":
        report = audit(output)
        write_json(output / "audit.json", report)
        print(json.dumps(report, indent=2))
        return
    if args.algorithm is None or args.task is None:
        parser.error("smoke/train require --algorithm and --task")
    if args.algorithm == "mappo" and args.task != "coverage":
        parser.error("Stage-1 D3 is MAPPO coverage; D4 capture is an existing artifact")
    if args.mode == "smoke":
        smoke(args.algorithm, args.task, output, args.device)
    elif args.algorithm == "iqn":
        if args.resume:
            parser.error("IQN baseline uses immutable policy checkpoints; fresh 100k run required")
        train_iqn(args.task, output, args.device)
    else:
        train_mappo(output, args.device, resume=args.resume.resolve() if args.resume else None)


if __name__ == "__main__":
    main()
