"""Paired deterministic/stochastic Stage 4 evaluation with behavior diagnostics.

Evaluates a trained MASAC checkpoint plus random/noop/oracle-seek policies on
the SAME initial seeds (paired), and reports capture/collision, d1-d4 geometry,
distance progress, closing fraction, bearing error, ring visitation.

Diagnostic only; the formal gate remains evaluate_ctde_formal eval20.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.trainer import set_global_config
from tools.run_continuous_ctde_training import (
    _evader_actions_for_env,
    _make_trainer,
    _sample_actions,
    _stack_with_batch,
)

ROOT = Path(__file__).resolve().parents[1]


def _wrap(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def seek_actions(env, a_max=0.4, w_max=None):
    w_max = w_max if w_max is not None else math.pi / 6.0
    target = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
    acts = []
    for p in env.pursuers:
        if p.deactivated:
            acts.append([0.0, 0.0])
            continue
        dx, dy = target[0] - p.x, target[1] - p.y
        phi = np.arctan2(dy, dx)
        diff = _wrap(phi - float(p.theta))
        acts.append([float(a_max * max(0.0, math.cos(diff))), float(np.clip(diff, -w_max, w_max))])
    return np.asarray(acts, dtype=np.float32)


def run_episode(env, trainer, adapter, mode, seed, max_steps, config, rng, max_agents, actor_max_pursuers, self_dim):
    env.reset()
    observations = list(env.get_observations())
    epos0 = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
    ppos0 = np.asarray([[p.x, p.y] for p in env.pursuers if not p.deactivated], dtype=float)
    d_init = sorted(np.linalg.norm(ppos0 - epos0, axis=1).tolist()) if len(ppos0) else []
    d_init = (d_init + [99.0] * 4)[:4]
    min_ds = [float("inf")] * 4
    closing_frac_sum = 0.0
    bearing_sum = 0.0
    bearing_n = 0
    turn_ok = 0
    turn_n = 0
    any_within8_steps = 0
    any_ring_steps = 0
    ring2_steps = 0
    ring3_steps = 0
    max_within8 = 0
    max_ring = 0
    steps = 0
    captured = False
    collision = False
    for _ in range(max_steps):
        if any(item is None for item in observations):
            break
        before_obs = _stack_with_batch(observations, max_agents, actor_max_pursuers, self_dim)
        before_obs_padded = {k: v[0] for k, v in before_obs.items()}
        if mode == "random":
            w_max = float(getattr(adapter, "w_max", math.pi / 6.0))
            actions = np.concatenate([
                rng.uniform(-float(config["action"]["a_max"]), float(config["action"]["a_max"]), size=(len(env.pursuers), 1)),
                rng.uniform(-w_max, w_max, size=(len(env.pursuers), 1)),
            ], axis=1).astype(np.float32)
        elif mode == "noop":
            actions = np.zeros((len(env.pursuers), 2), dtype=np.float32)
        elif mode == "oracle":
            actions = seek_actions(env, w_max=float(getattr(adapter, "w_max", math.pi / 6.0)))
        else:
            actions, _ = _sample_actions(
                trainer, before_obs_padded, len(env.pursuers), adapter,
                deterministic=mode == "trained_det",
                rng=rng, a_max=float(config["action"]["a_max"]),
            )
        outcome = env.step(actions.tolist(), _evader_actions_for_env(env))
        observations = list(outcome.observations)
        steps += 1
        if any(info.get("state") == "deactivated after collision" for info in outcome.infos):
            collision = True
        # geometry
        active_idx = [i for i, p in enumerate(env.pursuers) if not p.deactivated]
        if active_idx and env.evaders and not env.evaders[0].deactivated:
            epos = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
            evel = np.asarray(env.evaders[0].velocity, dtype=float)
            ds = []
            close_sum = 0.0
            bear_vals = []
            for i in active_idx:
                p = env.pursuers[i]
                ppos = np.asarray([p.x, p.y], dtype=float)
                rel = epos - ppos
                d = float(np.linalg.norm(rel))
                ds.append(d)
                if d > 1e-6:
                    unit = rel / d
                    pvel = np.asarray(p.velocity, dtype=float)
                    close_sum += float(np.dot(pvel - evel, unit) > 0)
                    phi = math.atan2(rel[1], rel[0])
                    be = abs(_wrap(phi - float(p.theta)))
                    bear_vals.append(be)
                    w = float(actions[i][1]) if i < len(actions) else None
                    if w is not None and be >= 0.05 and abs(w) >= 0.01:
                        turn_n += 1
                        if w * _wrap(phi - float(p.theta)) > 0:
                            turn_ok += 1
            ds_sorted = sorted(ds)
            for i in range(4):
                v = ds_sorted[i] if i < len(ds_sorted) else 99.0
                min_ds[i] = min(min_ds[i], v)
            closing_frac_sum += close_sum / len(ds)
            if bear_vals:
                bearing_sum += float(np.mean(bear_vals))
                bearing_n += 1
            n8 = sum(x < 8.0 for x in ds)
            nring = sum(8.0 <= x < 10.5 for x in ds)
            max_within8 = max(max_within8, n8)
            max_ring = max(max_ring, nring)
            if n8 > 0:
                any_within8_steps += 1
            if nring > 0:
                any_ring_steps += 1
            if nring >= 2:
                ring2_steps += 1
            if nring >= 3:
                ring3_steps += 1
        if all(outcome.dones):
            break
    rec = env.episode_record(task="capture")
    captured = bool(rec.get("captured", False))
    eposf = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float) if env.evaders else epos0
    pposf = np.asarray([[p.x, p.y] for p in env.pursuers if not p.deactivated], dtype=float)
    d_final = sorted(np.linalg.norm(pposf - eposf, axis=1).tolist()) if len(pposf) else []
    d_final = (d_final + [99.0] * 4)[:4]
    return {
        "seed": seed, "mode": mode, "captured": captured, "collision": collision, "length": steps,
        "d1_initial": float(d_init[0]), "d2_initial": float(d_init[1]),
        "d3_initial": float(d_init[2]), "d4_initial": float(d_init[3]),
        "d1_final": float(d_final[0]), "d2_final": float(d_final[1]),
        "d3_final": float(d_final[2]), "d4_final": float(d_final[3]),
        "d1_min": float(min_ds[0]), "d2_min": float(min_ds[1]), "d3_min": float(min_ds[2]), "d4_min": float(min_ds[3]),
        "d1_progress": float(d_init[0] - d_final[0]),
        "fraction_closing": float(closing_frac_sum / max(steps, 1)),
        "abs_bearing_error_mean": float(bearing_sum / max(bearing_n, 1)),
        "turn_direction_correct_rate": float(turn_ok / max(turn_n, 1)),
        "max_num_within_8m": int(max_within8), "max_num_in_ring": int(max_ring),
        "fraction_steps_any_within_8m": float(any_within8_steps / max(steps, 1)),
        "fraction_steps_any_in_ring": float(any_ring_steps / max(steps, 1)),
        "fraction_steps_2plus_in_ring": float(ring2_steps / max(steps, 1)),
        "fraction_steps_3plus_in_ring": float(ring3_steps / max(steps, 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=2026080801)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = resolve_ladder_config(args.config)
    scene = scene_config(config, "capture")
    set_global_config(scene)
    env = VorAdjEnv(scene, seed=args.seed_base)
    adapter = env.action_adapter
    max_agents = int(config["central_critic"]["max_agents"])
    actor_max_pursuers = int(config["actor"]["max_pursuers"])
    self_dim = int(config["actor"]["self_feature_dim"])

    trainer = None
    if args.checkpoint:
        trainer = _make_trainer(config, args.device)
        payload = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        trainer.load_checkpoint(args.checkpoint, payload.get("contract", {}))

    rng = np.random.default_rng(args.seed_base)
    modes = []
    if trainer is not None:
        modes += ["trained_det", "trained_sto"]
    modes += ["random", "noop", "oracle"]
    records = []
    for mode in modes:
        for idx in range(args.episodes):
            seed = args.seed_base + idx
            set_global_config(scene)
            env = VorAdjEnv(scene, seed=seed)
            r = run_episode(env, trainer, adapter, mode, seed, args.max_steps, config, rng,
                            max_agents, actor_max_pursuers, self_dim)
            records.append(r)

    def summarize(sub):
        n = len(sub)
        cap = float(np.mean([r["captured"] for r in sub]))
        coll = float(np.mean([r["collision"] for r in sub]))
        d1p = np.asarray([r["d1_progress"] for r in sub])
        close = np.asarray([r["fraction_closing"] for r in sub])
        bear = np.asarray([r["abs_bearing_error_mean"] for r in sub])
        ring = np.asarray([r["fraction_steps_any_in_ring"] for r in sub])
        return {
            "n": n,
            "capture_rate": cap,
            "collision_rate": coll,
            "d1_progress_mean": float(d1p.mean()), "d1_progress_median": float(np.median(d1p)),
            "d1_progress_positive_ratio": float(np.mean(d1p > 0)),
            "fraction_closing_mean": float(close.mean()),
            "abs_bearing_error_mean": float(bear.mean()),
            "fraction_steps_any_in_ring_mean": float(ring.mean()),
            "max_num_within_8m_mean": float(np.mean([r["max_num_within_8m"] for r in sub])),
            "turn_direction_correct_rate_mean": float(np.mean([r["turn_direction_correct_rate"] for r in sub])),
        }

    summary = {mode: summarize([r for r in records if r["mode"] == mode]) for mode in modes}
    paired = {}
    if trainer is not None:
        for target in ("random", "noop", "oracle"):
            for src in ("trained_det", "trained_sto"):
                key = f"{src}_vs_{target}"
                d = {}
                for metric in ("d1_progress", "fraction_closing", "abs_bearing_error_mean", "fraction_steps_any_in_ring", "collision_rate"):
                    a = np.asarray([r[metric] if metric != "collision_rate" else float(r["collision"]) for r in records if r["mode"] == src])
                    b = np.asarray([r[metric] if metric != "collision_rate" else float(r["collision"]) for r in records if r["mode"] == target])
                    delta = a - b
                    d[metric] = {
                        "mean": float(delta.mean()), "median": float(np.median(delta)),
                        "positive_ratio": float(np.mean(delta > 0)),
                    }
                paired[key] = d
    payload = {
        "kind": "stage4_paired_behavior_eval",
        "config": str(args.config), "checkpoint": args.checkpoint,
        "episodes": args.episodes, "max_steps": args.max_steps,
        "summary": summary, "paired": paired, "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "paired": paired}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
