#!/usr/bin/env python3
"""Run one formally gated NormSense V2 line.

This is an execution layer only.  It reuses the existing Forward-Final
categorical MAPPO update, reset/recovery stream, and side-channel gradient
logger.  It intentionally has no knobs for k, alpha, optimizer, or budget.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import yaml

from cocap_voradj.envs.density_sensing import POLICY, runtime_metadata
from cocap_voradj.training.forward_final_v2 import (
    NormSensePureCaptureStream,
    check_pure_capture_env,
    make_fullmix_stream,
)
from cocap_voradj.training.gradient_logging import phase_gradient_diagnostics
from cocap_voradj.training.small_step_ac import compute_gae, tensor_tree
from cocap_voradj.training.trainer import CoCapTrainer, set_global_config
from tools import forward_final_single_task_20260915 as single_task
from tools import preflight_forward_final_scratch_20260914 as scratch
from tools import train_forward_final_ppo_20260909 as production


AUDIT_DIR = ROOT / "artifacts/2026-09-15_normsense_v2"
GATE_PATH = AUDIT_DIR / "MASTER.json"
PARITY_PATH = AUDIT_DIR / "fullmix_original_vs_downweight05_parity.json"
SEMANTICS = "terminal-priority-truncation-bootstrap-weighted-ce-v2"
CHECKPOINT_SCHEMA = "forward-final-normsense-formal-checkpoint-v1"
FORMAL_STATUS = "FORMAL_TRAINING"

KIND_INFO = {
    "pure_capture": {
        "name": "NormSense-PureCapture",
        "config": ROOT / "configs/experiments/forward_final_normsense_v2_20260915/NormSense-PureCapture.yaml",
        "manifest": AUDIT_DIR / "NormSense-PureCapture.json",
        "budget": 250000,
        "seed": 2026091501,
        "eval_seed": 2026191501,
        "alpha": 1.0,
    },
    "original_fullmix": {
        "name": "NormSense-Original-FullMix",
        "config": ROOT / "configs/experiments/forward_final_normsense_v2_20260915/NormSense-Original-FullMix.yaml",
        "manifest": AUDIT_DIR / "NormSense-Original-FullMix.json",
        "budget": 100000,
        "seed": 2026091401,
        "eval_seed": 2026092401,
        "alpha": 1.0,
    },
    "downweight05_fullmix": {
        "name": "NormSense-CaptureDownweight05-FullMix",
        "config": ROOT / "configs/experiments/forward_final_normsense_v2_20260915/NormSense-CaptureDownweight05-FullMix.yaml",
        "manifest": AUDIT_DIR / "NormSense-CaptureDownweight05-FullMix.json",
        "budget": 100000,
        "seed": 2026091401,
        "eval_seed": 2026092401,
        "alpha": 0.5,
    },
}


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_norm_hash(value_norm) -> str:
    state = {
        key: (value.detach().cpu().tolist() if torch.is_tensor(value) else value)
        for key, value in value_norm.state_dict().items()
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def finite_tree(value) -> None:
    scratch.finite_tree(value)


def source_hashes(config: Path) -> dict[str, str]:
    files = [
        Path(__file__), config, scratch.CONFIG,
        Path(scratch.__file__), Path(production.__file__),
        Path(single_task.__file__), ROOT / "src/cocap_voradj/training/forward_final_v2.py",
        ROOT / "src/cocap_voradj/training/gradient_logging.py",
        ROOT / "src/cocap_voradj/training/small_step_ac.py",
        ROOT / "src/cocap_voradj/training/forward_final.py",
        ROOT / "src/cocap_voradj/envs/density_sensing.py",
        ROOT / "src/cocap_voradj/envs/voronoi_adjacency.py",
    ]
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted({path.resolve() for path in files})
    }


def gate_and_spec(kind: str) -> tuple[dict, dict, dict, str, dict]:
    info = KIND_INFO[kind]
    gate = json.loads(GATE_PATH.read_text())
    expected_gate = {
        "SENSING_V2_CONTRACT": "PASS",
        "NORM_PURE_CAPTURE_PARITY": "PASS",
        "FULLMIX_ORIGINAL_VS_DOWNWEIGHT05_PARITY": "PASS",
        "ALPHA1_REWARDBALANCED": "SUPERSEDED_NO_EFFECT",
        "THREE_NEW_RUNS": "READY",
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise RuntimeError(f"START GATE FAILED: {key}={gate.get(key)!r}, expected {value!r}")
    parity = json.loads(PARITY_PATH.read_text())
    if parity.get("status") != "PASS" or parity.get("UNEXPLAINED") != 0:
        raise RuntimeError("START GATE FAILED: Full-Mix parity artifact is not PASS")

    config = yaml.safe_load(info["config"].read_text())
    manifest = json.loads(info["manifest"].read_text())
    if not config.get("READY") or not manifest.get("READY"):
        raise RuntimeError(f"Run manifest is not READY: {info['name']}")
    if config["name"] != info["name"] or manifest["name"] != info["name"]:
        raise RuntimeError("Run name/config/manifest mismatch")
    for field in ("seed", "alpha_capture", "budget", "checkpoint_interval"):
        if config[field] != manifest[field]:
            raise RuntimeError(f"Config/manifest mismatch at {field}")
    if manifest["sensing_policy"] != POLICY or config["sensing_policy"] != POLICY:
        raise RuntimeError("NormSense V2 sensing policy missing")
    if config["budget"] != info["budget"] or config["seed"] != info["seed"]:
        raise RuntimeError("Launcher refuses an unexpected budget or seed")
    if config["checkpoint_interval"] != 25000 or config["auto_extend"] is not False:
        raise RuntimeError("Checkpoint/auto-extension contract mismatch")
    if kind == "pure_capture":
        if config["budget"] != 250000 or config["stop_at"] != 250000:
            raise RuntimeError("PureCapture budget contract mismatch")
        if config["alpha_capture"] != 1.0:
            raise RuntimeError("PureCapture alpha contract mismatch")
    else:
        if config["budget"] != 100000 or config["stop_at"] != 100000:
            raise RuntimeError("Full-Mix stop contract mismatch")
        if kind == "original_fullmix" and config["alpha_capture"] != 1.0:
            raise RuntimeError("Original Full-Mix alpha contract mismatch")
        if kind == "downweight05_fullmix":
            if config["alpha_capture"] != 0.5:
                raise RuntimeError("Downweight05 alpha contract mismatch")
            if config["reward_intervention"]["type"] != "predeclared_causal_downweight_ablation":
                raise RuntimeError("Downweight05 intervention contract mismatch")

    # The two Full-Mix lines are paired scratch runs.  Fail closed on the
    # shared initialization and schedule values that matter for causality.
    if kind != "pure_capture":
        other_name = "NormSense-Original-FullMix" if kind == "downweight05_fullmix" else "NormSense-CaptureDownweight05-FullMix"
        other = json.loads((AUDIT_DIR / f"{other_name}.json").read_text())
        for field in ("seed", "initial_hashes", "ppo", "actor", "critic", "actor_head",
                      "rollout_length", "reward_clock_offset", "scene_schedule",
                      "eval_seed_base", "eval_confirmation_seed_base"):
            if manifest[field] != other[field]:
                raise RuntimeError(f"Full-Mix shared contract mismatch at {field}")
    manifest_sha = sha256_file(info["manifest"])
    return gate, config, manifest, manifest_sha, parity


def make_trainer(contract: dict, seed: int, device: str):
    trainer = scratch.make_trainer(contract, seed, device)
    if trainer.update_count != 0 or trainer.actor_optimizer.state or trainer.value_optimizer.state:
        raise RuntimeError("Scratch trainer is not fresh")
    return trainer


def exposure_bucket() -> dict:
    return dict(valid_steps=0, zero_detector_steps=0, detector_sum=0,
                detector_max=0, first_detection_latency=None,
                reset_enemy_visible=None)


def observe_exposure(bucket: dict, env, first_step: int) -> None:
    metrics = getattr(env, "last_voradj_metrics", {}) or {}
    active_enemy = sum(not evader.deactivated for evader in env.evaders)
    if not active_enemy:
        return
    detected = int(metrics.get("vct_ls_detected_evader_count", 0))
    bucket["valid_steps"] += 1
    bucket["detector_sum"] += detected
    bucket["detector_max"] = max(bucket["detector_max"], detected)
    bucket["zero_detector_steps"] += int(metrics.get("vct_ls_all_evaders_invisible", detected == 0))
    if bucket["reset_enemy_visible"] is None:
        bucket["reset_enemy_visible"] = bool(detected > 0)
    if detected > 0 and bucket["first_detection_latency"] is None:
        bucket["first_detection_latency"] = int(first_step)


def finish_exposure(bucket: dict) -> dict:
    valid = bucket["valid_steps"]
    return {
        "reset_enemy_visible": bucket["reset_enemy_visible"],
        "zero_detector_fraction": (bucket["zero_detector_steps"] / valid if valid else None),
        "detector_count_mean": (bucket["detector_sum"] / valid if valid else None),
        "detector_count_max": bucket["detector_max"] if valid else None,
        "first_detection_latency": bucket["first_detection_latency"],
        "valid_enemy_steps": valid,
    }


class InstrumentedFullMixStream(production.FinalMissionStream):
    def __init__(self, seed, run_dir, *, alpha_capture=1.0):
        self._episode_exposure = exposure_bucket()
        self._episode_step = 0
        super().__init__(seed, run_dir, sensing_policy=POLICY, alpha_capture=alpha_capture)

    def step(self, indices):
        self._episode_step += 1
        outcome = super().step(indices)
        observe_exposure(self._episode_exposure, self.env, self._episode_step)
        return outcome

    def finish(self):
        exposure = finish_exposure(self._episode_exposure)
        row = super().finish()
        row["perception_exposure"] = exposure
        self._episode_exposure = exposure_bucket()
        self._episode_step = 0
        return row


class InstrumentedPureCaptureStream(NormSensePureCaptureStream):
    def __init__(self, seed):
        self._episode_exposure = exposure_bucket()
        self._episode_step = 0
        super().__init__(seed)

    def step(self, indices):
        self._episode_step += 1
        outcome = super().step(indices)
        observe_exposure(self._episode_exposure, self.env, self._episode_step)
        return outcome

    def finish(self):
        exposure = finish_exposure(self._episode_exposure)
        row = super().finish()
        row["perception_exposure"] = exposure
        self._episode_exposure = exposure_bucket()
        self._episode_step = 0
        return row


def make_stream(kind: str, seed: int, out: Path, alpha: float):
    if kind == "pure_capture":
        stream = InstrumentedPureCaptureStream(seed)
        stream.set_clock(0)
        check_pure_capture_env(stream.env)
        return stream
    stream = InstrumentedFullMixStream(seed, out / "stream", alpha_capture=alpha)
    for env in stream.envs.values():
        metadata = runtime_metadata(env)
        if metadata["policy"] != POLICY or env.reward_cfg.get("static_capture_scale", 1.0) != alpha:
            raise RuntimeError("Full-Mix runtime sensing or alpha drift")
    return stream


def validate_runtime(kind: str, stream, trainer, config: dict) -> dict:
    if kind == "pure_capture":
        facts = check_pure_capture_env(stream.env)
        metadata = runtime_metadata(stream.env)
        assert stream.env.config["voradj"]["onboard_sensing"]["k"] == 0.8715
        assert stream.env.config["voradj"]["onboard_sensing"]["radius_floor"] == 20.0
        assert stream.env.config["voradj"]["capture_episode_ends_on_capture"] is True
        assert stream.env.config["perception"]["global_evader_visibility"] is False
        return {"capture": facts, "sensing": metadata}
    facts = {}
    for name, env in stream.envs.items():
        facts[name] = dict(production.check_env(env), sensing=runtime_metadata(env),
                           static_capture_scale=env.reward_cfg.get("static_capture_scale", 1.0))
        assert env.config["voradj"]["onboard_sensing"]["k"] == 0.8715
        assert env.config["perception"]["global_evader_visibility"] is False
    assert stream.global_step == config["reward_clock_offset"]
    return facts


def local_advantage_stats(trainer, batch: dict) -> dict:
    device = trainer.device
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    values = trainer._denormalize_values(torch.as_tensor(batch["values"], dtype=torch.float32, device=device))
    next_values = trainer._denormalize_values(torch.as_tensor(batch["next_values"], dtype=torch.float32, device=device))
    terminated = torch.as_tensor(batch["terminated"], dtype=torch.bool, device=device)
    truncated = torch.as_tensor(batch["truncated"], dtype=torch.bool, device=device)
    episode_end = torch.as_tensor(batch["episode_end"], dtype=torch.bool, device=device)
    active = torch.as_tensor(batch["active_mask"], dtype=torch.bool, device=device)
    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, active,
        gamma=trainer.config.gamma, gae_lambda=trainer.config.gae_lambda,
        truncated=truncated, episode_end=episode_end)
    a, r = advantages[active], returns[active]
    normalized = (a - a.mean()) / a.std(unbiased=False).clamp_min(1e-6)
    summary = lambda x: {"mean": float(x.mean()), "std": float(x.std(unbiased=False)),
                         "min": float(x.min()), "max": float(x.max())}
    return {"raw_advantage": summary(a), "normalized_advantage": summary(normalized),
            "gae_target": summary(r), "raw_return": summary(r),
            "active_rows": int(a.numel())}


def update_one(trainer, rollout: dict, step: int, kind: str) -> dict:
    batch = production.stack_rollout(rollout)
    finite_tree(batch)
    zero_error = trainer.assert_behavior_log_probs(batch)
    before = production.rollout_log_probs(trainer, batch)
    advantage = local_advantage_stats(trainer, batch)
    gradient = None
    if kind != "pure_capture":
        gradient = phase_gradient_diagnostics(trainer, batch, minimum_conditioned_rows=32)
    metrics = trainer.update(batch, categorical=True)
    metrics.update(production.exact_update_diagnostics(before, production.rollout_log_probs(trainer, batch)))
    metrics.update({"step": step, "zero_update_max_log_prob_error": zero_error})
    metrics.update(advantage)
    metrics["gradient_logging"] = gradient
    finite_tree(metrics)
    return metrics


def checkpoint(out: Path, kind: str, step: int, trainer, stream, launch: dict,
               rollout: dict, metrics: dict, gradient: dict | None) -> Path:
    path = out / f"step_{step:06d}.pt"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite immutable checkpoint: {path}")
    payload = {
        "schema": CHECKPOINT_SCHEMA, "formal_status": FORMAL_STATUS, "kind": kind,
        "step": step, "launch": launch, "trainer": trainer.state_dict(),
        "stream_state": stream.__dict__, "rollout": rollout, "metrics": metrics,
        "rng": scratch.rng_state(trainer.device),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if loaded["schema"] != CHECKPOINT_SCHEMA or loaded["step"] != step:
        raise RuntimeError("Checkpoint round-trip verification failed")
    os.replace(temporary, path)
    digest = sha256_file(path)
    write_json(out / f"step_{step:06d}.json", {
        "schema": "normsense-resume-metadata-v1", "kind": kind, "step": step,
        "checkpoint": str(path), "checkpoint_sha256": digest,
        "source_sha256": launch["source_sha256"], "resume_supported": True,
        "partial_rollout_steps": len(rollout["rewards"]),
    })
    if gradient is not None:
        write_json(out / f"gradient_step_{step:06d}.json", gradient)
        write_json(out / "gradient_logging_latest.json", {"step": step, **gradient})
    return path


def load_checkpoint(path: Path, kind: str, config: dict, manifest: dict, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != CHECKPOINT_SCHEMA or payload["formal_status"] != FORMAL_STATUS:
        raise RuntimeError("Invalid formal checkpoint schema")
    launch = payload["launch"]
    if launch["kind"] != kind or launch["source_sha256"] != source_hashes(Path(KIND_INFO[kind]["config"])):
        raise RuntimeError("Checkpoint source contract changed")
    if launch["seed"] != manifest["seed"] or launch["budget"] != manifest["budget"]:
        raise RuntimeError("Checkpoint launch contract changed")
    trainer = make_trainer(scratch.load_contract(), manifest["seed"], device)
    trainer.load_state_dict(payload["trainer"])
    stream_cls = InstrumentedPureCaptureStream if kind == "pure_capture" else InstrumentedFullMixStream
    stream = stream_cls.__new__(stream_cls)
    stream.__dict__.update(payload["stream_state"])
    runtime = validate_runtime(kind, stream, trainer, scratch.load_contract())
    scratch.restore_rng(payload["rng"], trainer.device)
    return trainer, stream, payload, runtime


def eval_pure(actor, out: Path, step: int, seed_base: int, episodes: int = 20) -> dict:
    path = out / f"eval_step_{step:06d}.json"
    actor_hash = scratch.tensor_hash(actor.state_dict())
    if path.exists():
        report = json.loads(path.read_text())
        if report["actor_sha256"] != actor_hash:
            raise RuntimeError("Existing eval has a different actor")
        return report
    device = next(actor.parameters()).device
    saved_rng = scratch.rng_state(device)
    records = []
    started = time.monotonic()
    try:
        actor.eval()
        for mode in ("argmax", "sample"):
            for index in range(episodes):
                seed = seed_base + index
                scratch.seed_all(seed)
                stream = InstrumentedPureCaptureStream(seed)
                env, observations = stream.env, stream.observations
                for tick in range(1, env.episode_max_length + 1):
                    stream.set_clock(tick - 1)
                    active = [i for i, observation in enumerate(observations) if observation is not None]
                    local = tensor_tree({key: np.stack([observations[i][key] for i in active])
                                         for key in observations[active[0]]}, device)
                    distribution = actor.distribution(local)
                    latent = distribution.logits.argmax(-1) if mode == "argmax" else distribution.sample()
                    command = np.full(4, 4, dtype=int)
                    command[active] = latent.cpu().numpy()
                    result = stream.step(command)
                    observations = result.observations
                    if all(result.dones):
                        break
                if not all(result.dones):
                    raise RuntimeError("PureCapture evaluation did not reach native terminal")
                row = stream.telemetry.finish(stream.env)
                row.update(mode=mode, seed=seed, episode=index, native_done=True)
                row["perception_exposure"] = finish_exposure(stream._episode_exposure)
                records.append(row)
                write_json(out / "eval_progress.json", {
                    "step": step, "complete": len(records), "total": episodes * 2,
                    "elapsed_seconds": time.monotonic() - started,
                })
        report = {
            "schema": "normsense-formal-eval-v1", "kind": "pure_capture", "step": step,
            "actor_sha256": actor_hash, "seed_base": seed_base, "episodes_per_mode": episodes,
            "records": records,
            "summary": {mode: single_task.summarize([r for r in records if r["mode"] == mode])
                        for mode in ("argmax", "sample")},
            "training_rng_preserved": True,
        }
        write_json(path, report)
        return report
    finally:
        scratch.restore_rng(saved_rng, device)


def eval_fullmix(actor, out: Path, step: int, alpha: float, seed_base: int, episodes: int = 20) -> dict:
    path = out / f"eval_step_{step:06d}.json"
    actor_hash = scratch.tensor_hash(actor.state_dict())
    if path.exists():
        report = json.loads(path.read_text())
        if report["actor_sha256"] != actor_hash:
            raise RuntimeError("Existing eval has a different actor")
        return report
    device = next(actor.parameters()).device
    saved_rng = scratch.rng_state(device)
    records = []
    started = time.monotonic()
    try:
        actor.eval()
        for mode in ("argmax", "sample"):
            for scene in ("mixed", "coverage"):
                for index in range(episodes):
                    seed = seed_base + index + (100000 if scene == "coverage" else 0)
                    scratch.seed_all(seed)
                    env, observations = production.make_env(
                        scene, seed, sensing_policy=POLICY, alpha_capture=alpha)
                    exposure = exposure_bucket()
                    host = type("EvalHost", (), {})()
                    host.envs = {"eval": env}
                    host.apf_agents = {"eval": [single_task.p.ApfAgent(e.a, e.w) for e in env.evaders]}

                    def step_fn(actions):
                        set_global_config(env.config)
                        result = env.step(actions, CoCapTrainer._evader_actions(host, "eval"))
                        observe_exposure(exposure, env, int(exposure["valid_steps"] + 1))
                        return result

                    row = scratch.observe_episode(actor, env, observations, step_fn, scene, mode)
                    row.update(seed=seed, episode=index,
                               reset_source="independent_scene_environment_default",
                               perception_exposure=finish_exposure(exposure))
                    records.append(row)
                    write_json(out / "eval_progress.json", {
                        "step": step, "complete": len(records), "total": episodes * 4,
                        "elapsed_seconds": time.monotonic() - started,
                    })
        report = {
            "schema": "normsense-formal-eval-v1", "kind": "fullmix", "step": step,
            "actor_sha256": actor_hash, "seed_base": seed_base,
            "episodes_per_scene_mode": episodes, "alpha_capture": alpha,
            "records": records,
            "summary": {mode: scratch.summarize([r for r in records if r["mode"] == mode])
                        for mode in ("argmax", "sample")},
            "training_rng_preserved": True,
        }
        write_json(path, report)
        return report
    finally:
        scratch.restore_rng(saved_rng, device)


def train(kind: str, out: Path, device: str, resume: Path | None) -> None:
    gate, config, manifest, manifest_sha, parity = gate_and_spec(kind)
    contract = scratch.load_contract()
    if contract["transition_semantics"] != SEMANTICS:
        raise RuntimeError("Unexpected transition semantics")
    if resume is None:
        if out.exists():
            raise RuntimeError(f"Fresh output required; refusing existing directory {out}")
        out.mkdir(parents=True)
    elif not out.exists():
        raise RuntimeError("Resume output directory does not exist")

    # The guard prevents accidental teacher/checkpoint-bank dependencies in a
    # scratch run.  It is installed only after the committed audit inputs have
    # been read above.
    scratch.forbid_teacher_dependencies(out)
    torch.set_num_threads(1)
    sources = source_hashes(Path(KIND_INFO[kind]["config"]))
    if resume is not None:
        trainer, stream, payload, runtime = load_checkpoint(resume, kind, contract, manifest, device)
        step = int(payload["step"])
        rollout = payload["rollout"]
        metrics = payload["metrics"]
        launch = payload["launch"]
        if step >= manifest["budget"]:
            raise RuntimeError("Completed budget cannot resume")
        write_json(out / "resume.json", {"pid": os.getpid(), "checkpoint": str(resume), "step": step,
                                          "checkpoint_sha256": sha256_file(resume), "valid": True})
    else:
        scratch.seed_all(manifest["seed"])
        trainer = make_trainer(contract, manifest["seed"], device)
        stream = make_stream(kind, manifest["seed"], out, manifest["alpha_capture"])
        runtime = validate_runtime(kind, stream, trainer, contract)
        expected = manifest["initial_hashes"]
        actual = {
            "actor": scratch.tensor_hash(trainer.actor.state_dict()),
            "value": scratch.tensor_hash(trainer.value.state_dict()),
            "value_norm": value_norm_hash(trainer.value_norm),
        }
        if actual != expected:
            raise RuntimeError(f"Fresh initialization hash mismatch: {actual} != {expected}")
        launch = {
            "schema": "normsense-formal-launch-v1", "formal_status": FORMAL_STATUS,
            "kind": kind, "name": manifest["name"], "seed": manifest["seed"],
            "budget": manifest["budget"], "checkpoint_interval": manifest["checkpoint_interval"],
            "stop_at": manifest["stop_at"], "alpha_capture": manifest["alpha_capture"],
            "sensing_policy": POLICY, "k": 0.8715, "radius_floor": 20.0,
            "transition_semantics": SEMANTICS, "initial_hashes": actual,
            "manifest_sha256": manifest_sha, "parity_sha256": sha256_file(PARITY_PATH),
            "source_sha256": sources, "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "pid": os.getpid(), "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": device, "initialization": "random", "teacher_dependency": 0,
            "eval_seed_base": manifest["eval_seed_base"],
            "eval_confirmation_seed_base": manifest.get("eval_confirmation_seed_base"),
            "eval_episodes_per_scene_mode": manifest["eval_episodes_per_scene_mode"],
            "runtime": runtime,
            "gradient_logging": (None if kind == "pure_capture" else {
                "schema": "normsense-gradient-logging-v1",
                "checkpoint_adjacent": True, "phase_order": ["capture", "support", "coverage"],
                "occupancy_weighted": "actual active rows; absent phase is exact zero",
                "phase_conditioned": "actual phase rows; minimum 32 else NA_INSUFFICIENT_ROWS",
                "pairwise_cosine": True,
            }),
            "old_protection": {"legacy_r20_pure_capture": "untouched", "checkpoint_access": "none"},
            "auto_extend": False,
        }
        write_json(out / "launch.json", launch)
        write_json(out / "provenance.json", {
            "formal_status": FORMAL_STATUS, "teacher_dependency": 0,
            "initialization": "random", "manifest_sha256": manifest_sha,
            "initial_hashes": actual, "runtime": runtime,
        })
        step = 0
        rollout = production.empty_rollout()
        metrics = {}
        checkpoint(out, kind, 0, trainer, stream, launch, rollout, metrics, None)

    budget = manifest["budget"]
    cadence = manifest["checkpoint_interval"]
    started = time.monotonic()
    start_step = step
    if step % cadence == 0:
        if kind == "pure_capture":
            eval_pure(trainer.actor, out, step, manifest["eval_seed_base"], 20)
        else:
            eval_fullmix(trainer.actor, out, step, manifest["alpha_capture"], manifest["eval_seed_base"], 20)

    def progress(status: str):
        elapsed = time.monotonic() - started
        speed = (step - start_step) / elapsed if elapsed else 0.0
        write_json(out / "progress.json", {
            "status": status, "formal_status": FORMAL_STATUS, "kind": kind,
            "pid": os.getpid(), "step": step, "budget": budget,
            "steps_per_second": speed, "elapsed_seconds": elapsed,
            "eta_seconds": (budget - step) / speed if speed else None,
            "last_update": metrics, "partial_rollout_steps": len(rollout["rewards"]),
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })

    progress("RUNNING" if step else "INITIAL_CHECKPOINT_EVAL")
    while step < budget:
        stream.set_clock(step)
        row, episode = production.collect_transition(trainer, stream)
        finite_tree(row)
        for key, value in row.items():
            rollout[key].append(value)
        step += 1
        if episode is not None:
            with (out / "episodes.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **episode}, ensure_ascii=False, allow_nan=False) + "\n")
        if len(rollout["rewards"]) == contract["rollout_length"]:
            metrics = update_one(trainer, rollout, step, kind)
            with (out / "learning.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")
            if kind != "pure_capture":
                latest = metrics["gradient_logging"]
                with (out / "gradient_logging.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, **latest}, ensure_ascii=False, allow_nan=False) + "\n")
            rollout = production.empty_rollout()
        if step % 100 == 0:
            progress("RUNNING")
        if step % cadence == 0 or step == budget:
            gradient = metrics.get("gradient_logging") if isinstance(metrics, dict) else None
            checkpoint(out, kind, step, trainer, stream, launch, rollout, metrics, gradient)
            progress("CHECKPOINT_EVAL")
            if kind == "pure_capture":
                eval_pure(trainer.actor, out, step, manifest["eval_seed_base"], 20)
            else:
                eval_fullmix(trainer.actor, out, step, manifest["alpha_capture"], manifest["eval_seed_base"], 20)
    progress("STOP_FOR_MASTER_REVIEW")
    write_json(out / "report.json", {
        "status": "STOP_FOR_MASTER_REVIEW", "formal_status": FORMAL_STATUS,
        "kind": kind, "name": manifest["name"], "step": step, "budget": budget,
        "final_checkpoint": str(out / f"step_{step:06d}.pt"),
        "automatic_extension": False,
        "next_action": "MASTER review",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(KIND_INFO), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        train(args.kind, args.output.resolve(), args.device, args.resume.resolve() if args.resume else None)
    except BaseException:
        if args.output.exists():
            write_json(args.output / "failure.json", {
                "status": "STOP_IMPLEMENTATION", "formal_status": FORMAL_STATUS,
                "kind": args.kind, "pid": os.getpid(),
                "traceback": __import__("traceback").format_exc(),
            })
        raise


if __name__ == "__main__":
    main()
