#!/usr/bin/env python3
"""Narrow recovery/evaluation launcher for the two Final NormSense Pure-Capture lines."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import subprocess
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import yaml

from cocap_voradj.envs.density_sensing import POLICY, runtime_metadata
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.forward_final_v2 import new_pure_capture_config
from cocap_voradj.training.numeric_validation import assert_finite_numeric_tree
from cocap_voradj.training.runtime_semantics import assert_runtime
from cocap_voradj.training.small_step_ac import tensor_tree
from tools import forward_final_single_task_20260915 as single_task
from tools import launch_normsense_v2_formal as legacy
from tools import preflight_forward_final_scratch_20260914 as scratch
from tools import train_forward_final_ppo_20260909 as production


SCHEMA = "normsense-pure-capture-v2-checkpoint-v1"
FORMAL_STATUS = "FORMAL_TRAINING"
ARTIFACT_DIR = ROOT / "artifacts/2026-09-15_normsense_v2"
BASELINE_CONFIG = ROOT / "configs/experiments/forward_final_normsense_v2_20260915/NormSense-PureCapture.yaml"
CONSERVATIVE_CONFIG = ROOT / "configs/experiments/forward_final_normsense_v2_20260915/NormSense-Final-PureCapture-Conservative300k.yaml"
BASELINE_INITIAL = {
    "actor": "965553646883e512219e85cf7e40036c3b23897136683335ea0c2cc6c5086f4f",
    "value": "a1b1c9091bf1c1b6c1a26cf92f1e515e070e3134d002a731918966f2b666c78b",
    "value_norm": "17051316105d8c8d82e64395f9ef016c2a21c9b358be4fe01a8b445930268515",
}
LINES = {
    "baseline": {
        "name": "NormSense-Final-PureCapture-Baseline",
        "config": BASELINE_CONFIG,
        "budget": 250000,
        "horizon": 3000,
        "pre_capture_horizon": 0,
        "dropout": 0.1,
    },
    "conservative": {
        "name": "NormSense-Final-PureCapture-Conservative300k",
        "config": CONSERVATIVE_CONFIG,
        "budget": 300000,
        "horizon": 1000,
        "pre_capture_horizon": 1000,
        "dropout": 0.0,
    },
}


def write_json(path: Path, value) -> None:
    legacy.write_json(Path(path), value)


def config_for(line: str) -> dict:
    return yaml.safe_load(Path(LINES[line]["config"]).read_text())


def trainer_contract(line: str) -> dict:
    contract = copy.deepcopy(scratch.load_contract())
    contract["actor"]["dropout"] = float(LINES[line]["dropout"])
    return contract


def env_config(line: str) -> dict:
    config = new_pure_capture_config()
    config["env"]["episode_max_length"] = int(LINES[line]["horizon"])
    if LINES[line]["pre_capture_horizon"]:
        config["env"]["pre_capture_max_length"] = int(LINES[line]["pre_capture_horizon"])
    else:
        config["env"].pop("pre_capture_max_length", None)
    return config


def make_env(line: str, seed: int):
    config = env_config(line)
    single_task.p.set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=int(seed))
    observations = env.reset()
    return env, observations


def actor_dropout_sites(actor) -> list[dict]:
    rows = []
    for name, module in actor.named_modules():
        if isinstance(module, torch.nn.Dropout):
            rows.append({"module": name, "type": "Dropout", "p": float(module.p)})
        elif isinstance(module, torch.nn.MultiheadAttention):
            rows.append({"module": name, "type": "MultiheadAttention", "p": float(module.dropout)})
    return rows


def state_hashes(trainer) -> dict:
    return {
        "actor": scratch.tensor_hash(trainer.actor.state_dict()),
        "value": scratch.tensor_hash(trainer.value.state_dict()),
        "value_norm": legacy.value_norm_hash(trainer.value_norm),
    }


def validate_runtime(line: str, env, actor) -> dict:
    spec = LINES[line]
    expected = env_config(line)
    if env.config != expected:
        raise RuntimeError("Resolved Pure-Capture environment drift")
    sensing = runtime_metadata(env)
    radius = float(sensing["resolved_onboard_radius"])
    facts = assert_runtime(env, actor, categorical=True, expected={
        "pursuers": 4, "evaders": 1, "topology": "friendly_voronoi_comm_v0",
        "enemy_token_rule": "surface_radius", "global_enemy_flag": False,
        "support_capture_weight": 1.0, "support_coverage_weight": 0.0,
        "support_blend_enabled": True, "capture_reward_mode": "ring_importance_ms_v0",
        "capture_radius": 8.0, "capture_k": 3, "enemy_radius": radius,
        "action_mode": "unicycle_discrete", "decision_dt": 0.5, "physics_dt": 0.05,
        "a_longitudinal_max": 0.4, "omega_max": float(np.pi / 6), "v_max": 3.0,
        "drag": 0.4 / 3, "collision_semantics": "synchronized_swept_v1",
    })
    assert env.episode_max_length == spec["horizon"]
    assert int(env.env_cfg.get("pre_capture_max_length", 0)) == spec["pre_capture_horizon"]
    assert env.config["voradj"]["capture_episode_ends_on_capture"] is True
    assert env.reward_cfg["min_active_pursuers"] == 4
    assert env._ring_importance_ms_enabled() and not env._pure_capture_all_capture_enabled()
    assert env._support_reward_capture_component_mode() == "approach_only"
    assert env._support_reward_capture_target_mode() == "neighbor_visible"
    assert env._vct_ls_sensing_radius("enemy") == env._vct_ls_sensing_radius("obstacle") == radius
    assert env.config["perception"]["global_evader_visibility"] is False
    np.testing.assert_allclose(
        env.pursuers[0].action_list,
        [(a, w) for a in (-0.4, 0.0, 0.4) for w in (-np.pi / 6, 0.0, np.pi / 6)],
    )
    sites = actor_dropout_sites(actor)
    expected_dropout = float(spec["dropout"])
    assert sites and all(row["p"] == expected_dropout for row in sites)
    return {
        **facts, "sensing": sensing, "episode_max_length": env.episode_max_length,
        "pre_capture_max_length": int(env.env_cfg.get("pre_capture_max_length", 0)),
        "actor_dropout": expected_dropout, "actor_dropout_sites": sites,
        "actor_training_mode": bool(actor.training),
        "dropout_runtime_note": "MAPPOTrainer keeps Actor in eval mode; configured dropout sites are inactive during rollout and PPO updates.",
    }


class PureCaptureStream(single_task.SingleTaskStream):
    def __init__(self, line: str, seed: int):
        self.line = line
        self.task_name = "capture"
        self.multiplier = 1.0
        self.task = "capture"
        env, self.observations = make_env(line, seed)
        self.envs = {"capture": env}
        self.apf_agents = {"capture": [single_task.p.ApfAgent(e.a, e.w) for e in env.evaders]}
        self.episode = 0
        self.telemetry = single_task.Telemetry(env, "capture")
        self.episode_metrics = []
        self.env_steps = 0
        self._episode_exposure = legacy.exposure_bucket()
        self._episode_step = 0

    def step(self, indices):
        self._episode_step += 1
        result = super().step(indices)
        legacy.observe_exposure(self._episode_exposure, self.env, self._episode_step)
        return result

    def finish(self):
        exposure = legacy.finish_exposure(self._episode_exposure)
        row = dict(episode=self.episode, **self.telemetry.finish(self.env))
        row["perception_exposure"] = exposure
        self.episode_metrics.append(row)
        self.episode += 1
        single_task.p.set_global_config(self.env.config)
        self.observations = self.env.reset()
        self.apf_agents[self.task] = [single_task.p.ApfAgent(e.a, e.w) for e in self.env.evaders]
        self.telemetry = single_task.Telemetry(self.env, self.task_name)
        self._episode_exposure = legacy.exposure_bucket()
        self._episode_step = 0
        return row


def source_hashes(line: str) -> dict:
    paths = [
        Path(__file__), Path(LINES[line]["config"]), scratch.CONFIG,
        Path(scratch.__file__), Path(production.__file__), Path(single_task.__file__),
        ROOT / "src/cocap_voradj/training/forward_final_v2.py",
        ROOT / "src/cocap_voradj/training/numeric_validation.py",
        ROOT / "src/cocap_voradj/training/small_step_ac.py",
        ROOT / "src/cocap_voradj/training/forward_final.py",
        ROOT / "src/cocap_voradj/envs/density_sensing.py",
        ROOT / "src/cocap_voradj/envs/voronoi_adjacency.py",
    ]
    return {str(p.resolve().relative_to(ROOT)): legacy.sha256_file(p) for p in sorted(set(paths))}


def baseline_source_migration(recorded: dict) -> dict:
    current = legacy.source_hashes(BASELINE_CONFIG)
    allowed = {"tools/forward_final_single_task_20260915.py", "tools/launch_normsense_v2_formal.py"}
    differences = []
    for key in sorted(set(recorded) | set(current)):
        if recorded.get(key) != current.get(key):
            differences.append({"path": key, "checkpoint": recorded.get(key), "current": current.get(key)})
            if key not in allowed:
                raise RuntimeError(f"Baseline training-semantic source drift at {key}")
    if {row["path"] for row in differences} != {"tools/forward_final_single_task_20260915.py"}:
        raise RuntimeError(f"Unexpected baseline migration set: {differences}")
    return {
        "status": "PASS_EVALUATOR_ONLY_SOURCE_MIGRATION",
        "differences": differences,
        "unchanged_training_semantic_sources": sorted(set(recorded) - allowed),
    }


def pairwise_distances(positions: np.ndarray) -> list[dict]:
    return [
        {"pair": [i, j], "distance": float(np.linalg.norm(positions[i] - positions[j]))}
        for i, j in itertools.combinations(range(len(positions)), 2)
    ]


def angular_gaps(positions: np.ndarray, enemy: np.ndarray) -> list[float]:
    angles = np.sort(np.mod(np.arctan2(positions[:, 1] - enemy[1], positions[:, 0] - enemy[0]), 2 * np.pi))
    if not len(angles):
        return []
    return [float(x) for x in np.diff(np.r_[angles, angles[0] + 2 * np.pi])]


def trace_row(env, tick: int, command: np.ndarray, result) -> dict:
    pursuers = np.asarray([[p.x, p.y] for p in env.pursuers], dtype=float)
    enemy = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
    return {
        "tick": tick,
        "pursuer_positions": pursuers.tolist(),
        "enemy_position": enemy.tolist(),
        "pairwise_distances": pairwise_distances(pursuers),
        "radial_distance_to_enemy": [float(np.linalg.norm(p - enemy)) for p in pursuers],
        "relative_angular_gaps": angular_gaps(pursuers, enemy),
        "speeds": [float(p.speed) for p in env.pursuers],
        "aw9_indices": [int(x) for x in command],
        "aw9_actions": [list(map(float, env.pursuers[i].action_list[int(a)])) for i, a in enumerate(command)],
        "collision_types": [str(event.get("type", "unknown")) for event in env.last_collision_events],
        "done": bool(all(result.dones)),
    }


def diagnostic_summary(records: list[dict]) -> dict:
    if not records:
        return {}
    valid_steps = sum(r["perception_exposure"]["valid_enemy_steps"] for r in records)
    invisible_steps = sum(
        r["perception_exposure"]["zero_detector_fraction"] * r["perception_exposure"]["valid_enemy_steps"]
        for r in records if r["perception_exposure"]["zero_detector_fraction"] is not None
    )
    latencies = [r["perception_exposure"]["first_detection_latency"] for r in records
                 if r["perception_exposure"]["first_detection_latency"] is not None]
    collision_keys = ("agent_agent", "obstacle", "boundary", "evader_contact")
    categories = {
        key: {
            "episode_rate": float(np.mean([r["collision_types"].get(key, 0) > 0 for r in records])),
            "event_total": int(sum(r["collision_types"].get(key, 0) for r in records)),
        }
        for key in collision_keys
    }
    return {
        "episodes": len(records),
        "enemy_visible_fraction": (1.0 - invisible_steps / valid_steps) if valid_steps else None,
        "first_detection_latency": scratch.stats(latencies),
        "ring2_visitation": float(np.mean([r.get("ring2_steps", 0) > 0 for r in records])),
        "ring3_visitation": float(np.mean([r.get("ring3_steps", 0) > 0 for r in records])),
        "longest_ring3_hold": scratch.stats([r.get("ring3_max_hold", 0) for r in records]),
        "collision_episode_rate": float(np.mean([r["collision"] for r in records])),
        "collision_event_total": int(sum(sum(r["collision_types"].values()) for r in records)),
        "collision_categories": categories,
        "capture_rate": float(np.mean([r["captured"] for r in records])),
        "normal_capture_rate": float(np.mean([r["normal_capture"] for r in records])),
        "stationary_capture_rate": float(np.mean([r["stationary_capture"] for r in records])),
        "capture_time": scratch.stats([r["capture_seconds"] for r in records if r["capture_seconds"] is not None]),
        "total_return": scratch.stats([r["total_return"] for r in records]),
        "discounted_return": scratch.stats([r["discounted_return"] for r in records]),
        "trace_selection": {
            "capture": int(sum(r["captured"] for r in records)),
            "ring3_near_success": int(sum((not r["captured"]) and r.get("ring3_steps", 0) > 0 for r in records)),
            "collision": int(sum(r["collision"] for r in records)),
        },
    }


@torch.no_grad()
def evaluate(actor, line: str, checkpoint_step: int, out: Path, episodes: int, seed_base: int,
             filename: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    path = out / (filename or f"eval_step_{checkpoint_step:06d}.json")
    actor_hash = scratch.tensor_hash(actor.state_dict())
    if path.exists():
        report = json.loads(path.read_text())
        if report["actor_sha256"] != actor_hash or report["episodes_per_mode"] != episodes:
            raise RuntimeError(f"Existing evaluation contract mismatch: {path}")
        return report
    saved_rng = scratch.rng_state(next(actor.parameters()).device)
    records = []
    started = time.monotonic()
    try:
        actor.eval()
        for mode in ("argmax", "sample"):
            for episode in range(episodes):
                seed = seed_base + episode
                scratch.seed_all(seed)
                stream = PureCaptureStream(line, seed)
                validate_runtime(line, stream.env, actor)
                observations = stream.observations
                trace = deque(maxlen=100)
                for tick in range(1, stream.env.episode_max_length + 1):
                    stream.set_clock(tick - 1)
                    active = [i for i, observation in enumerate(observations) if observation is not None]
                    local = tensor_tree({key: np.stack([observations[i][key] for i in active])
                                         for key in observations[active[0]]}, next(actor.parameters()).device)
                    distribution = actor.distribution(local)
                    latent = distribution.logits.argmax(-1) if mode == "argmax" else distribution.sample()
                    command = np.full(4, 4, dtype=int)
                    command[active] = latent.cpu().numpy()
                    result = stream.step(command)
                    trace.append(trace_row(stream.env, tick, command, result))
                    observations = result.observations
                    if all(result.dones):
                        break
                if not all(result.dones):
                    raise RuntimeError("Pure-Capture evaluation failed to reach native terminal")
                row = stream.telemetry.finish(stream.env)
                row.update(mode=mode, seed=seed, episode=episode, native_done=True,
                           perception_exposure=legacy.finish_exposure(stream._episode_exposure))
                if row["captured"] or row.get("ring3_steps", 0) > 0 or row["collision"]:
                    row["diagnostic_trace_last_100"] = list(trace)
                records.append(row)
                write_json(out / f"{path.stem}_progress.json", {
                    "step": checkpoint_step, "complete": len(records), "total": episodes * 2,
                    "elapsed_seconds": time.monotonic() - started,
                })
        report = {
            "schema": "normsense-pure-capture-eval-v2", "line": line,
            "step": checkpoint_step, "actor_sha256": actor_hash,
            "seed_base": seed_base, "episodes_per_mode": episodes,
            "matched_seed_contract": "same seed_base/index for argmax and sample and across checkpoints",
            "records": records,
            "summary": {
                mode: {
                    **single_task.summarize([r for r in records if r["mode"] == mode]),
                    "collision_consolidation": diagnostic_summary([r for r in records if r["mode"] == mode]),
                }
                for mode in ("argmax", "sample")
            },
            "training_rng_preserved": True,
        }
        write_json(path, report)
        return report
    finally:
        scratch.restore_rng(saved_rng, next(actor.parameters()).device)


def save_checkpoint(out: Path, line: str, step: int, trainer, stream, launch: dict,
                    rollout: dict, metrics: dict) -> Path:
    path = out / f"step_{step:06d}.pt"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite immutable checkpoint: {path}")
    payload = {
        "schema": SCHEMA, "formal_status": FORMAL_STATUS, "line": line, "step": step,
        "launch": launch, "trainer": trainer.state_dict(), "stream_state": stream,
        "rollout": rollout, "metrics": metrics, "rng": scratch.rng_state(trainer.device),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if loaded["schema"] != SCHEMA or loaded["step"] != step:
        raise RuntimeError("Checkpoint round-trip failed")
    os.replace(temporary, path)
    write_json(path.with_suffix(".json"), {
        "schema": "normsense-pure-capture-resume-v2", "line": line, "step": step,
        "checkpoint": str(path), "checkpoint_sha256": legacy.sha256_file(path),
        "partial_rollout_steps": len(rollout["rewards"]), "source_sha256": launch["source_sha256"],
        "resume_supported": True,
    })
    return path


def load_checkpoint(path: Path, line: str, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = trainer_contract(line)
    trainer = scratch.make_trainer(contract, 2026091501, device)
    migration = None
    if payload.get("schema") == legacy.CHECKPOINT_SCHEMA:
        if line != "baseline" or payload["kind"] != "pure_capture" or payload["step"] != 100000:
            raise RuntimeError("Only the real 100k baseline checkpoint may use legacy migration")
        migration = baseline_source_migration(payload["launch"]["source_sha256"])
        trainer.load_state_dict(payload["trainer"])
        stream = legacy.InstrumentedPureCaptureStream.__new__(legacy.InstrumentedPureCaptureStream)
        stream.__dict__.update(payload["stream_state"])
    elif payload.get("schema") == SCHEMA:
        if payload["line"] != line or payload["launch"]["source_sha256"] != source_hashes(line):
            raise RuntimeError("Checkpoint line/source mismatch")
        trainer.load_state_dict(payload["trainer"])
        stream = payload["stream_state"]
    else:
        raise RuntimeError("Unknown Pure-Capture checkpoint schema")
    runtime = validate_runtime(line, stream.env, trainer.actor)
    scratch.restore_rng(payload["rng"], trainer.device)
    return trainer, stream, payload, runtime, migration


def audit() -> dict:
    torch.set_num_threads(1)
    baseline_contract = trainer_contract("baseline")
    conservative_contract = trainer_contract("conservative")
    contract_diffs = []
    for key in sorted(single_task.flatten(baseline_contract).keys() | single_task.flatten(conservative_contract).keys()):
        left = single_task.flatten(baseline_contract).get(key)
        right = single_task.flatten(conservative_contract).get(key)
        if left != right:
            contract_diffs.append({"field": key, "baseline": left, "conservative": right})
    baseline_env = single_task.flatten(env_config("baseline"))
    conservative_env = single_task.flatten(env_config("conservative"))
    env_diffs = [
        {"field": key, "baseline": baseline_env.get(key), "conservative": conservative_env.get(key)}
        for key in sorted(baseline_env.keys() | conservative_env.keys())
        if baseline_env.get(key) != conservative_env.get(key)
    ]
    if contract_diffs != [{"field": "actor.dropout", "baseline": 0.1, "conservative": 0.0}]:
        raise RuntimeError(f"Unexpected trainer parity differences: {contract_diffs}")
    expected_env = {"env.episode_max_length", "env.pre_capture_max_length"}
    if {row["field"] for row in env_diffs} != expected_env:
        raise RuntimeError(f"Unexpected environment parity differences: {env_diffs}")
    scratch.seed_all(2026091501)
    baseline = scratch.make_trainer(baseline_contract, 2026091501, "cpu")
    scratch.seed_all(2026091501)
    conservative = scratch.make_trainer(conservative_contract, 2026091501, "cpu")
    b_hash, c_hash = state_hashes(baseline), state_hashes(conservative)
    if b_hash != c_hash or b_hash != BASELINE_INITIAL:
        raise RuntimeError(f"Scratch initialization mismatch: {b_hash} vs {c_hash}")
    b_state, c_state = baseline.actor.state_dict(), conservative.actor.state_dict()
    mismatched = [key for key in b_state if not torch.equal(b_state[key], c_state[key])]
    if mismatched:
        raise RuntimeError(f"Non-dropout Actor parameters differ: {mismatched}")
    parity = {
        "schema": "normsense-pure-capture-parity-v2", "status": "PASS", "UNEXPLAINED": 0,
        "scientific_difference_groups": [
            {"name": "horizon", "fields": env_diffs},
            {"name": "actor_dropout", "fields": contract_diffs},
        ],
        "execution_only_differences": {
            "name": [LINES["baseline"]["name"], LINES["conservative"]["name"]],
            "budget_stop": [250000, 300000],
        },
        "initialization": {
            "seed": 2026091501, "baseline": b_hash, "conservative": c_hash,
            "all_parameter_tensors_bit_exact": True, "mismatched_parameters": [],
        },
        "dropout_sites": {
            "baseline": actor_dropout_sites(baseline.actor),
            "conservative": actor_dropout_sites(conservative.actor),
            "runtime_note": "Actor is held in eval mode by MAPPOTrainer, so these dropout probabilities are configured but inactive during rollout/update.",
            "mlp_note": "The Actor policy MLP and token MLPs contain no Dropout modules; affected sites are the four TransformerEncoder layers and summary MultiheadAttention.",
        },
    }
    write_json(ARTIFACT_DIR / "pure_capture_baseline_vs_conservative300k_parity.json", parity)
    manifest = {
        "schema": "normsense-pure-capture-manifest-v3", "READY": True,
        **{key: value for key, value in config_for("conservative").items() if key != "forbidden_changes"},
        "initial_hashes": c_hash, "parity": parity,
        "resolved_environment": env_config("conservative"),
    }
    write_json(ARTIFACT_DIR / "NormSense-Final-PureCapture-Conservative300k.json", manifest)
    return parity


def train(line: str, out: Path, device: str, resume: Path | None) -> None:
    spec = LINES[line]
    config = config_for(line)
    if int(config["budget"]) != spec["budget"] or int(config.get("episode_max_length", 3000)) != spec["horizon"]:
        raise RuntimeError("Line config/spec mismatch")
    if resume is None and out.exists():
        raise RuntimeError(f"Fresh output required: {out}")
    if resume is not None and not out.exists():
        raise RuntimeError("Resume output directory missing")
    if resume is None:
        out.mkdir(parents=True)
    torch.set_num_threads(1)
    if resume is not None:
        trainer, stream, payload, runtime, migration = load_checkpoint(resume, line, device)
        step = int(payload["step"])
        rollout = payload["rollout"]
        metrics = payload["metrics"]
        if line == "baseline" and (step != 100000 or len(rollout["rewards"]) != 160):
            raise RuntimeError("Baseline is not the real full-resume 100k state")
        launch = {
            **payload["launch"], "schema": "normsense-pure-capture-launch-v2", "line": line,
            "kind": "pure_capture", "name": spec["name"], "budget": spec["budget"],
            "stop_at": spec["budget"], "source_sha256": source_hashes(line),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "pid": os.getpid(), "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "runtime": runtime, "source_migration": migration,
            "resume_parent": {"path": str(resume), "sha256": legacy.sha256_file(resume), "step": step},
        }
        write_json(out / "resume_v2.json", {
            "status": "PASS_FULL_RESUME", "pid": os.getpid(), "step": step,
            "checkpoint": str(resume), "checkpoint_sha256": legacy.sha256_file(resume),
            "partial_rollout_steps": len(rollout["rewards"]), "training_updates": trainer.update_count,
            "state_hashes": state_hashes(trainer), "runtime": runtime, "source_migration": migration,
        })
    else:
        scratch.seed_all(2026091501)
        trainer = scratch.make_trainer(trainer_contract(line), 2026091501, device)
        stream = PureCaptureStream(line, 2026091501)
        runtime = validate_runtime(line, stream.env, trainer.actor)
        initial = state_hashes(trainer)
        if initial != BASELINE_INITIAL:
            raise RuntimeError(f"Initial hashes differ from baseline: {initial}")
        launch = {
            "schema": "normsense-pure-capture-launch-v2", "formal_status": FORMAL_STATUS,
            "line": line, "kind": "pure_capture", "name": spec["name"], "seed": 2026091501,
            "budget": spec["budget"], "checkpoint_interval": 25000, "stop_at": spec["budget"],
            "initialization": "random", "teacher_dependency": 0, "initial_hashes": initial,
            "source_sha256": source_hashes(line),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "pid": os.getpid(), "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": device, "runtime": runtime, "eval_seed_base": 2026191501,
            "eval_episodes_per_mode": 20, "auto_extend": False,
        }
        write_json(out / "launch.json", launch)
        step, rollout, metrics = 0, production.empty_rollout(), {}
        save_checkpoint(out, line, step, trainer, stream, launch, rollout, metrics)
        evaluate(trainer.actor, line, 0, out, 20, 2026191501)
    scratch.forbid_teacher_dependencies(out)
    budget, cadence = spec["budget"], 25000
    start_step, started = step, time.monotonic()
    health_written = False

    def progress(status: str):
        elapsed = time.monotonic() - started
        speed = (step - start_step) / elapsed if elapsed else 0.0
        write_json(out / "progress.json", {
            "status": status, "line": line, "pid": os.getpid(), "step": step, "budget": budget,
            "training_updates": trainer.update_count, "partial_rollout_steps": len(rollout["rewards"]),
            "steps_per_second": speed, "eta_seconds": (budget - step) / speed if speed else None,
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "last_update": metrics,
        })

    progress("RUNNING")
    while step < budget:
        stream.set_clock(step)
        row, episode = production.collect_transition(trainer, stream)
        assert_finite_numeric_tree(row, path=f"transition[{step}]")
        for key, value in row.items():
            rollout[key].append(value)
        step += 1
        if episode is not None:
            with (out / "episodes.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **episode}, ensure_ascii=False, allow_nan=False) + "\n")
        if len(rollout["rewards"]) == 256:
            metrics = legacy.update_one(trainer, rollout, step, "pure_capture")
            with (out / "learning.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
            rollout = production.empty_rollout()
            if not health_written:
                write_json(out / "training_health_gate.json", {
                    "status": "PASS", "step": step, "training_updates": trainer.update_count,
                    "finite": True, "metrics": metrics, "value_norm_finite": all(math.isfinite(metrics[k]) for k in ("value_norm_mean", "value_norm_std")),
                    "entropy_finite": math.isfinite(metrics["entropy"]), "kl_finite": math.isfinite(metrics["exact_full_batch_kl_old_new"]),
                })
                health_written = True
        if step % 100 == 0:
            progress("RUNNING")
        if step % cadence == 0:
            save_checkpoint(out, line, step, trainer, stream, launch, rollout, metrics)
            progress("CHECKPOINT_EVAL")
            evaluate(trainer.actor, line, step, out, 20, 2026191501)
            write_json(out / f"review_{step:06d}.json", {
                "step": step, "review_only": step in (100000, 200000, 250000),
                "continue_to": budget if step < budget else None,
                "stop_for_master_review": step == budget,
            })
    progress("STOP_FOR_MASTER_REVIEW")
    write_json(out / "report.json", {
        "status": "STOP_FOR_MASTER_REVIEW", "line": line, "step": step,
        "budget": budget, "final_checkpoint": str(out / f"step_{step:06d}.pt"),
        "automatic_extension": False,
    })


def eval_checkpoint(line: str, checkpoint: Path, out: Path, episodes: int, filename: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = scratch.make_trainer(trainer_contract(line), 2026091501, "cuda:0" if torch.cuda.is_available() else "cpu")
    trainer.load_state_dict(payload["trainer"])
    evaluate(trainer.actor, line, int(payload["step"]), out, episodes, 2026191501, filename)


def smoke(line: str, out: Path, device: str) -> None:
    if out.exists():
        raise RuntimeError(f"Fresh smoke output required: {out}")
    out.mkdir(parents=True)
    scratch.seed_all(2026091501)
    trainer = scratch.make_trainer(trainer_contract(line), 2026091501, device)
    stream = PureCaptureStream(line, 2026091501)
    runtime = validate_runtime(line, stream.env, trainer.actor)
    rollout = production.empty_rollout()
    for step in range(256):
        stream.set_clock(step)
        row, _ = production.collect_transition(trainer, stream)
        assert_finite_numeric_tree(row, path=f"smoke[{step}]")
        for key, value in row.items():
            rollout[key].append(value)
    metrics = legacy.update_one(trainer, rollout, 256, "pure_capture")
    launch = {"line": line, "source_sha256": source_hashes(line), "runtime": runtime}
    path = save_checkpoint(out, line, 256, trainer, stream, launch, production.empty_rollout(), metrics)
    clone, _, loaded, _, _ = load_checkpoint(path, line, device)
    if state_hashes(clone) != state_hashes(trainer) or loaded["step"] != 256:
        raise RuntimeError("Smoke resume state mismatch")
    write_json(out / "report.json", {
        "status": "PASS", "step_increased": True, "training_update_gt_zero": trainer.update_count > 0,
        "finite": True, "checkpoint_writable": True, "resume_metadata_readable": True,
        "metrics": metrics, "runtime": runtime, "state_hashes": state_hashes(trainer),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "train", "eval", "smoke"), required=True)
    parser.add_argument("--line", choices=tuple(LINES), default="baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--filename")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "audit":
        print(json.dumps(audit(), indent=2))
    elif args.mode == "eval":
        if not args.output or not args.checkpoint or not args.filename:
            parser.error("eval requires --output, --checkpoint, and --filename")
        eval_checkpoint(args.line, args.checkpoint.resolve(), args.output.resolve(), args.episodes, args.filename)
    elif args.mode == "smoke":
        if not args.output:
            parser.error("smoke requires --output")
        smoke(args.line, args.output.resolve(), args.device)
    else:
        if not args.output:
            parser.error("train requires --output")
        train(args.line, args.output.resolve(), args.device, args.resume.resolve() if args.resume else None)


if __name__ == "__main__":
    main()
