"""Counterfactual critic Q-ranking diagnostic for Stage 4 (2026-08-08).

Samples states from live env rollouts with the trained policy, then compares
central-critic Q values for three joint actions at the SAME states:
  A_policy (deterministic actor mean), A_random (uniform), A_seek (oracle seek).
Outputs Q-ranking fractions and mean deltas, bucketed by pursuer-evader d1.
Diagnostic only; does not modify training or the gate.
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
    _tensor_obs,
    build_central_global_obs,
)

ROOT = Path(__file__).resolve().parents[1]


def _wrap(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def seek_actions(positions, evader_pos, theta, a_max=0.4, w_max=None):
    """positions: (n,2), theta: (n,), evader_pos: (2,)."""
    w_max = w_max if w_max is not None else math.pi / 6.0
    acts = []
    for ppos, t in zip(positions, theta):
        dx, dy = evader_pos[0] - ppos[0], evader_pos[1] - ppos[1]
        phi = math.atan2(dy, dx)
        diff = _wrap(phi - float(t))
        acts.append([float(a_max * max(0.0, math.cos(diff))), float(np.clip(diff, -w_max, w_max))])
    return np.asarray(acts, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026080801)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = resolve_ladder_config(args.config)
    scene = scene_config(config, "capture")
    set_global_config(scene)
    env = VorAdjEnv(scene, seed=args.seed)
    env.reset()
    trainer = _make_trainer(config, args.device)
    payload = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.load_checkpoint(args.checkpoint, payload.get("contract", {}))

    max_agents = int(config["central_critic"]["max_agents"])
    actor_max_pursuers = int(config["actor"]["max_pursuers"])
    self_dim = int(config["actor"]["self_feature_dim"])
    adapter = env.action_adapter
    w_max = float(getattr(adapter, "w_max", math.pi / 6.0))
    rng = np.random.default_rng(args.seed)
    states = []
    observations = list(env.get_observations())
    while len(states) < args.samples:
        if any(item is None for item in observations):
            env.reset()
            observations = list(env.get_observations())
            continue
        before_active = np.asarray([not p.deactivated for p in env.pursuers], dtype=bool)
        before_obs = _stack_with_batch(observations, max_agents, actor_max_pursuers, self_dim)
        before_obs_padded = {k: v[0] for k, v in before_obs.items()}
        before_global = build_central_global_obs(
            env, max_agents=max_agents, max_evaders=8, max_obstacles=5, self_feature_dim=self_dim
        )
        ppos = np.asarray([[p.x, p.y] for p in env.pursuers], dtype=float)
        ptheta = np.asarray([p.theta for p in env.pursuers], dtype=float)
        epos = np.asarray([env.evaders[0].x, env.evaders[0].y], dtype=float)
        active_pos = ppos[before_active]
        d1 = float(np.min(np.linalg.norm(active_pos - epos, axis=1))) if len(active_pos) else 99.0
        states.append({
            "local_obs": before_obs_padded,
            "global_obs": before_global,
            "active": before_active,
            "ppos": ppos,
            "ptheta": ptheta,
            "epos": epos,
            "d1": d1,
        })
        actions, _ = _sample_actions(
            trainer, before_obs_padded, len(env.pursuers), adapter,
            deterministic=False, rng=rng,
            a_max=float(config["action"]["a_max"]),
        )
        outcome = env.step(actions.tolist(), _evader_actions_for_env(env))
        observations = list(outcome.observations)
        if all(outcome.dones):
            env.reset()
            observations = list(env.get_observations())

    def q_values(global_obs, joint_np):
        joint = torch.as_tensor(joint_np, dtype=torch.float32, device=trainer.device).unsqueeze(0)
        go = {k: torch.as_tensor(v, dtype=torch.float32, device=trainer.device).unsqueeze(0) for k, v in global_obs.items()}
        with torch.no_grad():
            q1 = trainer.critic1(go, joint)
            q2 = trainer.critic2(go, joint)
        if q1.dim() == 1:
            q1 = q1.view(1, -1)
            q2 = q2.view(1, -1)
        return q1[0].detach().cpu().numpy(), q2[0].detach().cpu().numpy()

    rows = []
    for st in states:
        active = st["active"]
        idx = np.where(active)[0]
        with torch.no_grad():
            policy_acts, _, _ = trainer.actor.sample(_tensor_obs(st["local_obs"], trainer.device), deterministic=True)
        policy_np = policy_acts.detach().cpu().numpy()
        if policy_np.ndim == 3:
            policy_np = policy_np[0]
        policy_full = np.zeros((max_agents, 2), dtype=np.float32)
        policy_full[: policy_np.shape[0]] = policy_np
        random_full = np.concatenate([
            rng.uniform(-float(config["action"]["a_max"]), float(config["action"]["a_max"]), size=(max_agents, 1)),
            rng.uniform(-w_max, w_max, size=(max_agents, 1)),
        ], axis=1).astype(np.float32)
        seek_np = seek_actions(st["ppos"][: len(env.pursuers)], st["epos"], st["ptheta"][: len(env.pursuers)], w_max=w_max)
        seek_full = np.zeros((max_agents, 2), dtype=np.float32)
        seek_full[: len(env.pursuers)] = seek_np

        qp1, qp2 = q_values(st["global_obs"], policy_full)
        qr1, qr2 = q_values(st["global_obs"], random_full)
        qs1, qs2 = q_values(st["global_obs"], seek_full)
        qp = np.minimum(qp1, qp2)[idx].mean()
        qr = np.minimum(qr1, qr2)[idx].mean()
        qs = np.minimum(qs1, qs2)[idx].mean()
        rows.append({
            "d1": st["d1"],
            "bucket": "near" if st["d1"] < 12.0 else ("medium" if st["d1"] < 20.0 else "far"),
            "Q_policy": float(qp), "Q_random": float(qr), "Q_seek": float(qs),
            "q1_policy_mean": float(qp1[idx].mean()), "q2_policy_mean": float(qp2[idx].mean()),
        })

    def agg(sub):
        n = len(sub)
        if not n:
            return None
        qp = np.asarray([r["Q_policy"] for r in sub])
        qr = np.asarray([r["Q_random"] for r in sub])
        qs = np.asarray([r["Q_seek"] for r in sub])
        return {
            "n": n,
            "fraction_Qseek_gt_Qpolicy": float(np.mean(qs > qp)),
            "fraction_Qseek_gt_Qrandom": float(np.mean(qs > qr)),
            "fraction_Qpolicy_gt_Qrandom": float(np.mean(qp > qr)),
            "mean_Qseek_minus_Qpolicy": float(np.mean(qs - qp)),
            "mean_Qseek_minus_Qrandom": float(np.mean(qs - qr)),
            "mean_Qpolicy_minus_Qrandom": float(np.mean(qp - qr)),
            "mean_Q_policy": float(np.mean(qp)),
            "mean_Q_random": float(np.mean(qr)),
            "mean_Q_seek": float(np.mean(qs)),
        }

    summary = {"overall": agg(rows)}
    for bucket in ("near", "medium", "far"):
        summary[bucket] = agg([r for r in rows if r["bucket"] == bucket])
    payload = {
        "kind": "counterfactual_critic_q_ranking",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "samples": len(rows),
        "seed": args.seed,
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
