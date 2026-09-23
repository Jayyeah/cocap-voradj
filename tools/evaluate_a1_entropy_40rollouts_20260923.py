#!/usr/bin/env python3
"""Independent A1 entropy-only Pure Coverage evaluation.

This evaluator is inference-only.  It loads the fresh A1 actor checkpoints,
reconstructs the missing step-0 actor from the recorded fresh seed, and runs
the same Pure Coverage scene with the same reset seeds for argmax and policy
sample evaluation.  It deliberately does not instantiate a trainer or call
any update/resume path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.shared_local_ac import (
    SharedLocalACNetworkConfig,
    SharedLocalActor,
    assert_runtime_aw9,
)
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import deep_update, set_global_config


AW9 = [
    [-0.4, -float(np.pi / 6.0)],
    [-0.4, 0.0],
    [-0.4, float(np.pi / 6.0)],
    [0.0, -float(np.pi / 6.0)],
    [0.0, 0.0],
    [0.0, float(np.pi / 6.0)],
    [0.4, -float(np.pi / 6.0)],
    [0.4, 0.0],
    [0.4, float(np.pi / 6.0)],
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate):
            out.append(candidate)
    return out


def stats(values: Iterable[Any]) -> Dict[str, Any]:
    finite = finite_values(values)
    if not finite:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90, method="linear")),
    }


def load_payload(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "ps-local-discrete-ac-v1":
        raise ValueError(f"unexpected A1 checkpoint payload: {path}")
    return payload


def coverage_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    cfg = deep_update(cfg, (cfg.get("tasks", {}) or {}).get("voradj_coverage", {}) or {})
    cfg.setdefault("env", {})["num_evaders"] = 0
    # The requested independent evaluation contract uses the corrected
    # synchronized-swept collision implementation for every checkpoint. The
    # fresh A1 training artifact records legacy_end_step; retain that fact in
    # the final manifest instead of silently presenting the two as identical.
    cfg["env"]["collision_semantics"] = "synchronized_swept_v1"
    return cfg


def make_actor(
    network_config: Mapping[str, Any],
    device: str,
    checkpoint: Optional[Mapping[str, Any]],
    fresh_seed: int,
) -> SharedLocalActor:
    # The formal trainer creates the actor first, immediately after
    # torch.manual_seed(seed), before creating critic/target modules.  This
    # reproduces its actor-only step-0 weights without running training.
    torch.manual_seed(int(fresh_seed))
    actor = SharedLocalActor(SharedLocalACNetworkConfig(**dict(network_config)))
    if checkpoint is not None:
        actor.load_state_dict(checkpoint["actor"])
    actor = actor.to(device)
    actor.eval()
    return actor


def episode_mode(
    actor: SharedLocalActor,
    cfg: Dict[str, Any],
    *,
    mode: str,
    seed: int,
    sample_seed: int,
    device: str,
    max_steps: int,
) -> Dict[str, Any]:
    if mode not in {"argmax", "sample"}:
        raise ValueError(mode)
    set_global_config(cfg)
    env = VorAdjEnv(cfg, seed=int(seed))
    obs = env.reset()
    assert_runtime_aw9(env)
    decision_seconds = float(env.pursuers[0].dt * max(int(env.pursuers[0].N), 1))
    action_histogram = np.zeros(9, dtype=np.int64)
    entropy_values: List[float] = []
    episode_return = 0.0
    time_to_ce_step: Optional[int] = None
    native_terminated = False
    native_truncated = False
    last_done_infos: List[Dict[str, Any]] = []

    # Separate per-episode action RNG makes sample results reproducible and
    # keeps reset seeds identical across checkpoints and policy modes.
    torch.manual_seed(int(sample_seed))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed(int(sample_seed))

    for _ in range(int(max_steps)):
        active = [idx for idx, item in enumerate(obs) if item is not None]
        actions: List[Optional[int]] = [None] * len(obs)
        if active:
            batch = stack_obs([obs[idx] for idx in active], device)
            with torch.no_grad():
                probabilities = actor.probabilities(batch)
                log_probabilities = probabilities.clamp_min(1e-8).log()
                entropy_values.extend(
                    (-(probabilities * log_probabilities).sum(dim=-1)).detach().cpu().tolist()
                )
                if mode == "argmax":
                    selected = probabilities.argmax(dim=-1)
                else:
                    selected = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            selected_list = selected.detach().cpu().tolist()
            action_histogram += np.bincount(selected.detach().cpu().numpy(), minlength=9)
            for idx, action in zip(active, selected_list):
                actions[idx] = int(action)

        result = env.step(actions, [None] * len(env.evaders))
        episode_return += float(np.sum(result.rewards))
        obs = result.observations
        metrics = dict(getattr(env, "last_distribution_metrics", {}) or {})
        if time_to_ce_step is None and bool(metrics.get("success_after_hold", False)):
            time_to_ce_step = int(env.episode_step)
        if any(result.dones):
            last_done_infos = [dict(info or {}) for info in result.infos]
            native_terminated = any(bool(info.get("terminated", False)) for info in last_done_infos)
            native_truncated = any(bool(info.get("truncated", False)) for info in last_done_infos)
            break

    record = env.episode_record(task="voradj_coverage")
    strict_distribution = dict(record.get("coverage_strict_distribution_metrics", {}) or {})
    strict_success = bool(record.get("coverage_strict_success", False))
    if time_to_ce_step is None and strict_success:
        # Defensive fallback for a future env implementation that does not
        # expose per-step last_distribution_metrics but records native success.
        time_to_ce_step = int(record.get("post_capture_coverage_step", env.episode_step))
    collision = bool(record.get("collision_event", False))
    complete_native = bool(native_terminated or native_truncated)
    censored = bool(native_truncated and not native_terminated and not strict_success)
    failure = bool(complete_native and not strict_success and not censored)
    if not complete_native:
        # This should only happen if the caller supplied a horizon shorter than
        # the native env horizon.  Keep it separately visible as censored.
        censored = True

    return {
        "mode": mode,
        "seed": int(seed),
        "sample_seed": int(sample_seed),
        "decision_seconds": decision_seconds,
        "horizon_steps": int(max_steps),
        "steps": int(env.episode_step),
        "complete_native_rollout": complete_native,
        "terminated": native_terminated,
        "truncated": native_truncated,
        "strict_ce_success": strict_success,
        "collision": collision,
        "failure": failure,
        "censored": censored,
        "time_to_ce_steps": None if time_to_ce_step is None else int(time_to_ce_step),
        "time_to_ce_seconds": None if time_to_ce_step is None else float(time_to_ce_step * decision_seconds),
        "strict_hold_steps": strict_distribution.get("hold_steps"),
        "ce_rms": record.get("coverage_ce_center_rms"),
        "ce_max": record.get("coverage_ce_center_max"),
        "area_cv": record.get("coverage_strict_area_cv"),
        "episode_return": float(episode_return),
        "policy_entropy": stats(entropy_values),
        "action_histogram": [int(value) for value in action_histogram.tolist()],
        "action_count": int(action_histogram.sum()),
        "action_grid": AW9,
        "collision_types": record.get("collision_type_counts", {}),
        "done_infos": last_done_infos,
        "record": record,
    }


def summarize(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    successes = [row for row in episodes if bool(row.get("strict_ce_success", False))]
    action_hist = np.sum(
        np.asarray([row.get("action_histogram", [0] * 9) for row in episodes], dtype=np.int64),
        axis=0,
    ) if episodes else np.zeros(9, dtype=np.int64)
    action_total = int(action_hist.sum())
    return {
        "rollouts": total,
        "strict_ce_success": {
            "n": int(len(successes)),
            "denominator": int(total),
            "rate": float(len(successes) / total) if total else None,
        },
        "collision": {
            "n": int(sum(bool(row.get("collision", False)) for row in episodes)),
            "denominator": int(total),
            "rate": float(sum(bool(row.get("collision", False)) for row in episodes) / total) if total else None,
        },
        "failures": {
            "n": int(sum(bool(row.get("failure", False)) for row in episodes)),
            "denominator": int(total),
        },
        "censored": {
            "n": int(sum(bool(row.get("censored", False)) for row in episodes)),
            "denominator": int(total),
        },
        "ce_rms": stats(row.get("ce_rms") for row in episodes),
        "ce_max": stats(row.get("ce_max") for row in episodes),
        "area_cv": stats(row.get("area_cv") for row in episodes),
        "strict_hold_steps": {
            "all": stats(row.get("strict_hold_steps") for row in episodes),
            "successful_only": stats(row.get("strict_hold_steps") for row in successes),
        },
        "time_to_ce_seconds": stats(row.get("time_to_ce_seconds") for row in successes),
        "time_to_ce_steps": stats(row.get("time_to_ce_steps") for row in successes),
        "episode_return": stats(row.get("episode_return") for row in episodes),
        "policy_entropy": stats(
            (row.get("policy_entropy") or {}).get("mean") for row in episodes
        ),
        "aw9_action_histogram": {
            "counts": [int(value) for value in action_hist.tolist()],
            "fractions": [float(value / action_total) for value in action_hist] if action_total else [0.0] * 9,
            "total_active_agent_actions": action_total,
            "grid": AW9,
        },
    }


def parse_checkpoint(raw: str) -> Tuple[int, Path]:
    step, sep, path = raw.partition("=")
    if not sep:
        raise ValueError(f"checkpoint must be STEP=PATH, got {raw!r}")
    return int(step), Path(path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-checkpoint", required=True, help="25k A1 checkpoint used for config/network metadata and 0k reconstruction")
    parser.add_argument("--checkpoint", action="append", required=True, help="STEP=PATH; repeat for 25k/50k/75k/100k")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed-base", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    if args.episodes != 20:
        raise ValueError("This independent A1 evaluation is fixed at 20 episodes per mode")
    if len(args.checkpoint) != 4:
        raise ValueError("Provide exactly four STEP=PATH arguments for 25k/50k/75k/100k")

    template_path = Path(args.template_checkpoint).resolve()
    template = load_payload(template_path)
    config = copy.deepcopy(template["config"])
    fresh_seed = int(config["seed"])
    eval_seed_base = int(args.eval_seed_base if args.eval_seed_base is not None else fresh_seed + 5000)
    max_steps = int(args.max_steps if args.max_steps is not None else config.get("env", {}).get("episode_max_length", 3000))
    if max_steps != int(config.get("env", {}).get("episode_max_length", max_steps)):
        raise ValueError("max-steps must equal the recorded native episode horizon")
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for corrected CUDA evaluation but torch.cuda.is_available() is false")

    checkpoints: Dict[int, Optional[Path]] = {0: None}
    for raw in args.checkpoint:
        step, path = parse_checkpoint(raw)
        if step not in {25000, 50000, 75000, 100000}:
            raise ValueError(f"unexpected checkpoint step {step}")
        checkpoints[step] = path
    if set(checkpoints) != {0, 25000, 50000, 75000, 100000}:
        raise ValueError("checkpoint grid must be exactly 0/25k/50k/75k/100k")
    for step, path in checkpoints.items():
        if step and (path is None or not path.is_file()):
            raise FileNotFoundError(path)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = output_root / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    initial_actor = make_actor(template["network_config"], "cpu", None, fresh_seed)
    step0_path = model_dir / "model_step_000000000_reconstructed.pt"
    torch.save({
        "schema": "a1-eval-reconstructed-initial-actor-v1",
        "source": "fresh A1 trainer actor initialization; formal runner did not save model_step_000000000.pt",
        "fresh_seed": fresh_seed,
        "network_config": template["network_config"],
        "actor": initial_actor.state_dict(),
        "actor_hash": state_hash(initial_actor),
        "source_template_checkpoint": str(template_path),
    }, step0_path)

    checkpoint_manifest: Dict[str, Any] = {
        "0": {
            "path": str(step0_path),
            "source": "reconstructed from fresh seed/config; no formal model_step_000000000.pt was present",
            "original_eval_artifact": str(template_path.parent.parent / "evaluations" / "eval_step_000000000.json"),
        }
    }
    for step in (25000, 50000, 75000, 100000):
        checkpoint_manifest[str(step)] = {
            "path": str(checkpoints[step]),
            "source": "A1 fresh 2026-09-23 formal actor checkpoint",
        }

    all_results: Dict[str, Any] = {}
    for step in (0, 25000, 50000, 75000, 100000):
        payload = template if step == 0 else load_payload(checkpoints[step])  # type: ignore[arg-type]
        actor = make_actor(template["network_config"], device, None if step == 0 else payload, fresh_seed)
        modes: Dict[str, Any] = {}
        for mode in ("argmax", "sample"):
            episodes: List[Dict[str, Any]] = []
            for index in range(args.episodes):
                reset_seed = eval_seed_base + index
                sample_seed = eval_seed_base + 100000 + index
                episodes.append(
                    episode_mode(
                        actor,
                        coverage_config(config),
                        mode=mode,
                        seed=reset_seed,
                        sample_seed=sample_seed,
                        device=device,
                        max_steps=max_steps,
                    )
                )
                print(f"step={step} mode={mode} episode={index + 1}/{args.episodes} steps={episodes[-1]['steps']} strict={episodes[-1]['strict_ce_success']} collision={episodes[-1]['collision']}", flush=True)
            modes[mode] = {
                "summary": summarize(episodes),
                "episodes": episodes,
            }
        all_results[str(step)] = {
            "checkpoint": checkpoint_manifest[str(step)],
            "actor_hash": state_hash(actor),
            "modes": modes,
        }
        write_json(output_root / f"checkpoint_{step:09d}.json", all_results[str(step)])

    result = {
        "schema": "a1-entropy-localq-cov-independent-eval-v1",
        "branch": "evaluation/a1-entropy-localq-cov-20260923",
        "base_commit": "78adf7889900affa7dd62aa261b36a7c8b55bb15",
        "evaluation_source": "Astra corrected observation/CUDA terminal-row fix present in base commit; inference-only evaluator",
        "scientific_contract": {
            "scene": "Pure Coverage / voradj_coverage",
            "reset_seed_base": eval_seed_base,
            "reset_seeds": [eval_seed_base + i for i in range(args.episodes)],
            "horizon_steps": max_steps,
            "decision_seconds": 0.5,
            "strict_ce_definition": "native coverage_strict_success with centroid-energy RMS/max, area CV, center/inside thresholds, 30-step hold",
            "collision_semantics": "synchronized_swept_v1 (explicit independent-evaluation override)",
            "training_artifact_collision_semantics": str((config.get("env", {}) or {}).get("collision_semantics", "<missing>")),
            "rollouts_per_mode_per_checkpoint": args.episodes,
            "parameter_updates": 0,
        },
        "checkpoint_manifest": checkpoint_manifest,
        "results": all_results,
    }
    write_json(output_root / "a1_independent_eval_20260923.json", result)
    print(json.dumps(json_safe({"output": str(output_root / "a1_independent_eval_20260923.json"), "checkpoint_manifest": checkpoint_manifest}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
