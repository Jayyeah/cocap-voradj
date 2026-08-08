"""Legacy IQN scratch early-training behavior evaluation (2026-08-08).

Evaluates an IQN checkpoint (discrete (a,w) 3x3 grid) on the original capture
contract with the SAME behavior metrics as the ladder MASAC paired eval:
capture/collision, d1-d4 geometry, distance progress, closing fraction,
bearing error, turn-direction correctness, ring visitation, plus discrete
action histograms.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.trainer import load_config, set_global_config
from tools.run_continuous_ctde_training import _evader_actions_for_env
from cocap_voradj.control.apf import ApfAgent

ROOT = Path(__file__).resolve().parents[1]


def _wrap(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=2026080801)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["num_evaders"] = 1
    set_global_config(env_cfg)
    model = CoCapIQN.load(args.checkpoint, args.device)
    model.eval()
    action_list = env_cfg.get("pursuer", {}).get("a", [-0.4, 0.0, 0.4])
    w_list = env_cfg.get("pursuer", {}).get("w", [-0.5235987755982988, 0.0, 0.5235987755982988])
    grid = [(float(a), float(w)) for a in action_list for w in w_list]

    records = []
    for idx in range(args.episodes):
        seed = args.seed_base + idx
        env = VorAdjEnv(copy.deepcopy(env_cfg), seed=seed + 43)
        env.reset()
        obs_list = list(env.get_observations())
        action_hist = [0] * len(grid)
        active_p = [p for p in env.pursuers if not p.deactivated]
        epos0 = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
        ppos0 = np.asarray([[p.x, p.y] for p in active_p], dtype=float)
        d_init = sorted(np.linalg.norm(ppos0 - epos0, axis=1).tolist()) if len(ppos0) else [99.0] * 4
        min_ds = [float("inf")] * 4
        close_sum = 0.0
        bear_sum = 0.0
        bear_n = 0
        turn_ok = 0
        turn_n = 0
        any8_steps = 0
        any_ring_steps = 0
        ring2 = 0
        ring3 = 0
        max8 = 0
        max_ring = 0
        steps = 0
        captured = False
        collision = False
        apf_agents = [ApfAgent(e.a, e.w) for e in env.evaders] if bool(((cfg.get("evader", {}) or {})).get("autonomous", False)) else None
        for _ in range(args.max_steps):
            active_idx = [i for i, o in enumerate(obs_list) if o is not None]
            if not active_idx:
                break
            batch = stack_obs([obs_list[i] for i in active_idx], args.device)
            with torch.no_grad():
                act = model.act(batch, mode="voradj", epsilon=0.0, deterministic_quantiles=True).detach().cpu().numpy().tolist()
            actions = [None] * len(obs_list)
            for i, a in zip(active_idx, act):
                actions[i] = int(a)
                action_hist[int(a)] += 1
            outcome = env.step(actions, _evader_actions_for_env(env, apf_agents))
            obs_list = list(outcome.observations)
            steps += 1
            if any(info.get("state") == "deactivated after collision" for info in outcome.infos):
                collision = True
            a_idx = [i for i, p in enumerate(env.pursuers) if not p.deactivated]
            if a_idx and env.evaders and not env.evaders[0].deactivated:
                epos = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
                evel = np.asarray(env.evaders[0].velocity, dtype=float)
                ds = []
                csum = 0.0
                bears = []
                for i in a_idx:
                    p = env.pursuers[i]
                    ppos = np.asarray([p.x, p.y], dtype=float)
                    rel = epos - ppos
                    d = float(np.linalg.norm(rel))
                    ds.append(d)
                    if d > 1e-6:
                        unit = rel / d
                        pvel = np.asarray(p.velocity, dtype=float)
                        csum += float(np.dot(pvel - evel, unit) > 0)
                        phi = math.atan2(rel[1], rel[0])
                        be = abs(_wrap(phi - float(p.theta)))
                        bears.append(be)
                        w = grid[actions[i]][1] if actions[i] is not None else None
                        if w is not None and be >= 0.05 and abs(w) >= 0.01:
                            turn_n += 1
                            if w * _wrap(phi - float(p.theta)) > 0:
                                turn_ok += 1
                dsort = sorted(ds)
                for j in range(4):
                    min_ds[j] = min(min_ds[j], dsort[j] if j < len(dsort) else 99.0)
                close_sum += csum / len(ds)
                if bears:
                    bear_sum += float(np.mean(bears))
                    bear_n += 1
                n8 = sum(x < 8.0 for x in ds)
                nring = sum(8.0 <= x < 10.5 for x in ds)
                max8 = max(max8, n8)
                max_ring = max(max_ring, nring)
                if n8 > 0:
                    any8_steps += 1
                if nring > 0:
                    any_ring_steps += 1
                if nring >= 2:
                    ring2 += 1
                if nring >= 3:
                    ring3 += 1
            if all(outcome.dones):
                break
        rec = env.episode_record(task="capture")
        captured = bool(rec.get("captured", False))
        eposf = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
        pposf = np.asarray([[p.x, p.y] for p in env.pursuers if not p.deactivated], dtype=float)
        d_final = sorted(np.linalg.norm(pposf - eposf, axis=1).tolist()) if len(pposf) else [99.0] * 4
        records.append({
            "seed": seed, "captured": captured, "collision": collision, "length": steps,
            "d1_initial": float(d_init[0]), "d2_initial": float(d_init[1]),
            "d3_initial": float(d_init[2]), "d4_initial": float(d_init[3]),
            "d1_final": float(d_final[0]), "d2_final": float(d_final[1]),
            "d3_final": float(d_final[2]), "d4_final": float(d_final[3]),
            "d1_min": float(min_ds[0]), "d2_min": float(min_ds[1]),
            "d3_min": float(min_ds[2]), "d4_min": float(min_ds[3]),
            "d1_progress": float(d_init[0] - d_final[0]),
            "fraction_closing": float(close_sum / max(steps, 1)),
            "abs_bearing_error_mean": float(bear_sum / max(bear_n, 1)),
            "turn_direction_correct_rate": float(turn_ok / max(turn_n, 1)),
            "max_num_within_8m": int(max8), "max_num_in_ring": int(max_ring),
            "fraction_steps_any_within_8m": float(any8_steps / max(steps, 1)),
            "fraction_steps_any_in_ring": float(any_ring_steps / max(steps, 1)),
            "fraction_steps_2plus_in_ring": float(ring2 / max(steps, 1)),
            "fraction_steps_3plus_in_ring": float(ring3 / max(steps, 1)),
            "action_histogram": dict(enumerate(action_hist)),
        })

    n = len(records)
    summary = {
        "n": n,
        "capture_rate": float(np.mean([r["captured"] for r in records])),
        "collision_rate": float(np.mean([r["collision"] for r in records])),
        "avg_length": float(np.mean([r["length"] for r in records])),
        "d1_progress_mean": float(np.mean([r["d1_progress"] for r in records])),
        "d1_progress_positive_ratio": float(np.mean([r["d1_progress"] > 0 for r in records])),
        "d1_min_mean": float(np.mean([r["d1_min"] for r in records])),
        "d2_min_mean": float(np.mean([r["d2_min"] for r in records])),
        "d3_min_mean": float(np.mean([r["d3_min"] for r in records])),
        "fraction_closing_mean": float(np.mean([r["fraction_closing"] for r in records])),
        "abs_bearing_error_mean": float(np.mean([r["abs_bearing_error_mean"] for r in records])),
        "turn_direction_correct_rate_mean": float(np.mean([r["turn_direction_correct_rate"] for r in records])),
        "fraction_steps_any_within_8m_mean": float(np.mean([r["fraction_steps_any_within_8m"] for r in records])),
        "fraction_steps_any_in_ring_mean": float(np.mean([r["fraction_steps_any_in_ring"] for r in records])),
        "fraction_steps_2plus_in_ring_mean": float(np.mean([r["fraction_steps_2plus_in_ring"] for r in records])),
        "action_histogram_total": {str(k): int(sum(r["action_histogram"].get(k, 0) for r in records)) for k in range(len(grid))},
        "action_grid": grid,
    }
    payload = {
        "kind": "iqn_scratch_early_behavior_eval",
        "config": str(args.config), "checkpoint": str(args.checkpoint),
        "episodes": n, "max_steps": args.max_steps, "summary": summary, "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
