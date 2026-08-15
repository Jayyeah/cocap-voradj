#!/usr/bin/env python3
"""Paired policy-temperature probe for a continuous MASAC capture checkpoint.

This diagnostic deliberately leaves the actor, environment, and reward contract
unchanged.  It only scales the actor's latent Gaussian innovation before the
existing tanh/action-bound transform::

    z_T = mu(obs) + T * sigma(obs) * epsilon,  epsilon ~ Normal(0, I)

``T=0`` is therefore the deterministic mean action, ``T=1`` is the policy's
normal stochastic action, and an intermediate value is a reduced-temperature
stochastic action.  Every temperature uses the same environment seeds and the
same per-episode, per-step epsilon stream.  A fixed ``max_agents x 2`` tensor is
drawn at every step so collision/deactivation differences do not shift the
remaining agents' noise stream.

The tool is capture-only and records normal/stationary capture separately,
2+/3+ ring visitation, 3+ hold, and collision.  It is intentionally independent
of the training runner so using it cannot mutate a resume bundle.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import scene_config
from cocap_voradj.training.trainer import load_config, set_global_config
from tools.run_continuous_ctde_training import (
    _evader_actions_for_env,
    _make_trainer,
    _pad_local_obs_tree,
    _pursuit_step_geometry,
    _tensor_obs,
)
from tools.run_masac_rollout_gifs import verify_config_contract


@dataclass(frozen=True)
class TemperatureMode:
    name: str
    temperature: float


COLLISION_SEMANTICS = ("legacy_end_step", "synchronized_swept_v1")


def apply_collision_semantics_override(
    config: Mapping[str, Any],
    value: str | None,
) -> Dict[str, Any]:
    """Return an evaluation config with an explicit post-verification override.

    The checkpoint's untouched config must be hash-verified before calling this
    helper. Keeping the override here makes a legacy/fixed paired evaluation
    auditable and prevents it being mistaken for checkpoint-contract parity.
    """
    result = copy.deepcopy(dict(config))
    if value is None:
        return result
    normalized = str(value).strip().lower()
    if normalized not in COLLISION_SEMANTICS:
        raise ValueError(
            f"unsupported collision semantics {value!r}; expected one of {COLLISION_SEMANTICS}"
        )
    result.setdefault("env", {})["collision_semantics"] = normalized
    return result


def _mode_name(temperature: float) -> str:
    if math.isclose(temperature, 0.0, abs_tol=1e-12):
        return "deterministic_t0"
    if math.isclose(temperature, 1.0, abs_tol=1e-12):
        return "stochastic_t1"
    text = format(float(temperature), ".6g").replace("-", "m").replace(".", "p")
    return f"reduced_t{text}"


def parse_temperature_modes(values: Iterable[float]) -> list[TemperatureMode]:
    temperatures = [float(value) for value in values]
    if not temperatures:
        raise ValueError("at least one temperature is required")
    if any(not math.isfinite(value) or value < 0.0 for value in temperatures):
        raise ValueError("temperatures must be finite and non-negative")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("temperatures must be unique")
    return [TemperatureMode(_mode_name(value), value) for value in temperatures]


def epsilon_stream(seed: int, max_agents: int):
    """Yield a device-independent fixed-shape standard-normal stream."""
    rng = np.random.default_rng(int(seed))
    while True:
        yield rng.standard_normal((int(max_agents), 2)).astype(np.float32)


@torch.no_grad()
def sample_temperature_actions(
    trainer: Any,
    padded_obs: Mapping[str, np.ndarray],
    active_count: int,
    adapter: Any,
    temperature: float,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Sample bounded ``(a,w)`` actions with explicitly supplied innovation."""
    if not math.isfinite(float(temperature)) or float(temperature) < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    distribution, _ = trainer.actor.distribution(_tensor_obs(padded_obs, trainer.device))
    eps = torch.as_tensor(epsilon, dtype=distribution.loc.dtype, device=trainer.device)
    if eps.shape != distribution.loc.shape:
        raise ValueError(
            f"epsilon shape {tuple(eps.shape)} does not match actor latent shape "
            f"{tuple(distribution.loc.shape)}"
        )
    latent = distribution.loc + float(temperature) * distribution.scale * eps
    scale = torch.as_tensor(
        [float(trainer.actor.config.a_max), float(trainer.actor.config.w_max)],
        dtype=latent.dtype,
        device=latent.device,
    )
    raw = (torch.tanh(latent) * scale).detach().cpu().numpy()
    commands: list[np.ndarray] = []
    rejected = 0
    for action in raw[: int(active_count)]:
        validated, diagnostics = adapter.validate_with_diagnostics(action)
        rejected += int(diagnostics.action_rejected)
        commands.append(validated)
    return np.asarray(commands, dtype=np.float32), float(rejected / max(int(active_count), 1))


def _max_hold(counts: Sequence[int], minimum: int) -> int:
    best = current = 0
    for count in counts:
        current = current + 1 if int(count) >= int(minimum) else 0
        best = max(best, current)
    return int(best)


def run_episode(
    trainer: Any,
    root_config: Dict[str, Any],
    *,
    mode: TemperatureMode,
    seed: int,
    noise_seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    config = scene_config(root_config, "capture")
    set_global_config(config)
    env = VorAdjEnv(config, seed=int(seed))
    observations = list(env.reset())
    apf_agents = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
    max_agents = int(root_config["central_critic"]["max_agents"])
    actor_max_pursuers = int(root_config["actor"]["max_pursuers"])
    self_dim = int(root_config["actor"].get("self_feature_dim", 9))
    noises = epsilon_stream(noise_seed, max_agents)
    ring_counts: list[int] = []
    capture_types: list[str] = []
    rejected_rates: list[float] = []
    steps = 0

    for _ in range(min(int(max_steps), int(config["env"]["episode_max_length"]))):
        padded = _pad_local_obs_tree(
            observations, max_agents, actor_max_pursuers, self_dim,
        )
        sampled, rejected = sample_temperature_actions(
            trainer,
            padded,
            len(env.pursuers),
            env.action_adapter,
            mode.temperature,
            next(noises),
        )
        actions = [
            None if observations[index] is None else sampled[index]
            for index in range(len(env.pursuers))
        ]
        outcome = env.step(actions, _evader_actions_for_env(env, apf_agents))
        steps += 1
        rejected_rates.append(rejected)
        capture_types.extend(
            str(event.get("capture_type", "unknown"))
            for event in getattr(env, "last_capture_events", [])
        )
        geometry = _pursuit_step_geometry(env, sampled)
        if geometry is not None:
            ring_counts.append(int(geometry["num_in_ring_8_10_5"]))
        observations = list(outcome.observations)
        if all(outcome.dones):
            break

    record = env.episode_record(task="capture")
    normal = any(value != "stationary" for value in capture_types)
    stationary = any(value == "stationary" for value in capture_types)
    counted_capture = bool(normal or stationary)
    if bool(record.get("captured", False)) != counted_capture:
        raise RuntimeError(
            "capture record/event mismatch: "
            f"record={record.get('captured')} capture_types={capture_types}"
        )
    ring_steps = max(len(ring_counts), 1)
    collision_events = [
        dict(event) for event in (record.get("collision_events", []) or [])
    ]
    first_collision = min(
        collision_events,
        key=lambda event: (
            int(event.get("step", max_steps)),
            float(event.get("contact_substep") or 0.0),
        ),
        default=None,
    )
    return {
        "seed": int(seed),
        "noise_seed": int(noise_seed),
        "mode": mode.name,
        "temperature": float(mode.temperature),
        "length": int(steps),
        "normal_capture": bool(normal),
        "stationary_capture": bool(stationary),
        "captured": bool(record.get("captured", False)),
        "collision": bool(record.get("collision_event", False)),
        "pursuer_collision": bool(record.get("pursuer_collision_event", False)),
        "evader_collision": bool(record.get("evader_collision_event", False)),
        "collision_type_counts": dict(record.get("collision_type_counts", {}) or {}),
        "first_collision_event": first_collision,
        "max_num_in_ring": int(max(ring_counts, default=0)),
        "visited_2plus_ring": bool(any(value >= 2 for value in ring_counts)),
        "visited_3plus_ring": bool(any(value >= 3 for value in ring_counts)),
        "fraction_steps_2plus_in_ring": float(sum(value >= 2 for value in ring_counts) / ring_steps),
        "fraction_steps_3plus_in_ring": float(sum(value >= 3 for value in ring_counts) / ring_steps),
        "max_2plus_ring_hold_steps": _max_hold(ring_counts, 2),
        "max_3plus_ring_hold_steps": _max_hold(ring_counts, 3),
        "action_rejected_rate": float(np.mean(rejected_rates)) if rejected_rates else 0.0,
        "capture_types": capture_types,
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"episodes": 0}

    def count(key: str) -> int:
        return int(sum(bool(item.get(key, False)) for item in records))

    def mean(key: str) -> float:
        return float(np.mean([float(item.get(key, 0.0)) for item in records]))

    episodes = len(records)
    result: Dict[str, Any] = {
        "episodes": int(episodes),
        "normal_capture_count": count("normal_capture"),
        "stationary_capture_count": count("stationary_capture"),
        "capture_count": count("captured"),
        "collision_count": count("collision"),
        "episodes_with_2plus_ring": count("visited_2plus_ring"),
        "episodes_with_3plus_ring": count("visited_3plus_ring"),
        "normal_capture_rate": count("normal_capture") / episodes,
        "stationary_capture_rate": count("stationary_capture") / episodes,
        "capture_rate": count("captured") / episodes,
        "collision_rate": count("collision") / episodes,
        "episode_2plus_ring_rate": count("visited_2plus_ring") / episodes,
        "episode_3plus_ring_rate": count("visited_3plus_ring") / episodes,
        "mean_fraction_steps_2plus_in_ring": mean("fraction_steps_2plus_in_ring"),
        "mean_fraction_steps_3plus_in_ring": mean("fraction_steps_3plus_in_ring"),
        "mean_max_2plus_ring_hold_steps": mean("max_2plus_ring_hold_steps"),
        "mean_max_3plus_ring_hold_steps": mean("max_3plus_ring_hold_steps"),
        "max_3plus_ring_hold_steps": int(max(int(item.get("max_3plus_ring_hold_steps", 0)) for item in records)),
        "mean_episode_length": mean("length"),
        "mean_action_rejected_rate": mean("action_rejected_rate"),
    }
    collision_types = sorted({
        str(collision_type)
        for item in records
        for collision_type in (item.get("collision_type_counts", {}) or {})
    })
    result["collision_type_episode_counts"] = {
        collision_type: int(sum(
            int((item.get("collision_type_counts", {}) or {}).get(collision_type, 0)) > 0
            for item in records
        ))
        for collision_type in collision_types
    }
    result["collision_type_event_counts"] = {
        collision_type: int(sum(
            int((item.get("collision_type_counts", {}) or {}).get(collision_type, 0))
            for item in records
        ))
        for collision_type in collision_types
    }
    first_steps = [
        int(event["step"])
        for item in records
        for event in [item.get("first_collision_event")]
        if event is not None
    ]
    result["mean_first_collision_step"] = (
        float(np.mean(first_steps)) if first_steps else None
    )
    return result


def paired_deltas(
    records: Sequence[Mapping[str, Any]],
    modes: Sequence[TemperatureMode],
) -> Dict[str, Any]:
    by_mode = {
        mode.name: {int(item["seed"]): item for item in records if item["mode"] == mode.name}
        for mode in modes
    }
    baseline = modes[0]
    metrics = (
        "normal_capture", "stationary_capture", "captured", "collision",
        "visited_2plus_ring", "visited_3plus_ring", "max_3plus_ring_hold_steps",
    )
    result: Dict[str, Any] = {}
    for mode in modes[1:]:
        seeds = sorted(set(by_mode[baseline.name]) & set(by_mode[mode.name]))
        comparison: Dict[str, Any] = {"paired_episodes": len(seeds)}
        for metric in metrics:
            deltas = np.asarray([
                float(by_mode[mode.name][seed][metric])
                - float(by_mode[baseline.name][seed][metric])
                for seed in seeds
            ], dtype=float)
            comparison[metric] = {
                "mean_delta_vs_baseline": float(deltas.mean()) if len(deltas) else 0.0,
                "improved": int(np.sum(deltas > 0)),
                "worsened": int(np.sum(deltas < 0)),
                "tied": int(np.sum(deltas == 0)),
            }
        result[f"{mode.name}_vs_{baseline.name}"] = comparison
    return result


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config = load_config(str(config_path))
    capture_config = scene_config(config, "capture")
    horizon = min(int(args.max_steps), int(capture_config["env"]["episode_max_length"]))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = dict(saved.get("contract", {}) or {})
    verification = verify_config_contract(
        config,
        contract,
        [{"scene": "capture", "max_steps": horizon}],
    )
    # Apply this only after strict checkpoint/config hash verification. It is a
    # simulator intervention, not a claim of saved training-contract parity.
    evaluation_config = apply_collision_semantics_override(
        config, getattr(args, "collision_semantics", None),
    )
    trainer = _make_trainer(config, str(args.device))
    trainer.load_checkpoint(checkpoint, contract)
    trainer.actor.eval()
    modes = parse_temperature_modes(args.temperatures)
    records: list[Dict[str, Any]] = []
    for mode in modes:
        for episode_index in range(int(args.episodes)):
            records.append(run_episode(
                trainer,
                evaluation_config,
                mode=mode,
                seed=int(args.seed) + episode_index,
                noise_seed=int(args.noise_seed) + episode_index,
                max_steps=horizon,
            ))
    return {
        "schema_version": 1,
        "kind": "cf3_paired_policy_temperature_probe",
        "created_at": now(),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "device": str(args.device),
        "episodes_per_mode": int(args.episodes),
        "max_steps": int(horizon),
        "environment_seed_base": int(args.seed),
        "noise_seed_base": int(args.noise_seed),
        "temperature_contract": {
            "equation": "latent = mu(obs) + temperature * sigma(obs) * epsilon; action = bounds * tanh(latent)",
            "epsilon": "same fixed max_agents_x_2 CPU standard-normal stream for matched episode/step across modes",
            "actor_weights_and_reported_sigma": "unchanged",
            "modes": [{"name": mode.name, "temperature": mode.temperature} for mode in modes],
        },
        "collision_semantics_override": {
            "requested": getattr(args, "collision_semantics", None),
            "applied_after_checkpoint_config_verification": True,
            "effective": str(
                scene_config(evaluation_config, "capture").get("env", {}).get(
                    "collision_semantics", "checkpoint_default"
                )
            ),
        },
        "config_verification": verification,
        "summary": {
            mode.name: summarize([item for item in records if item["mode"] == mode.name])
            for mode in modes
        },
        "paired_deltas": paired_deltas(records, modes),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026081201)
    parser.add_argument("--noise-seed", type=int, default=2026081501)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--collision-semantics",
        choices=COLLISION_SEMANTICS,
        default=None,
        help=(
            "Simulator-only override applied after strict checkpoint/config hash verification; "
            "use separate invocations for a legacy/fixed paired audit."
        ),
    )
    args = parser.parse_args()
    if int(args.episodes) <= 0 or int(args.max_steps) <= 0:
        parser.error("--episodes and --max-steps must be positive")
    payload = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
