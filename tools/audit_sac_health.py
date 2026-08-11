#!/usr/bin/env python3
"""Offline SAC scale/gradient audit on fixed uniform replay batches."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from cocap_voradj.training.continuous.joint_replay import (
    JointReplayBuffer,
    UniformJointReplaySampler,
)
from cocap_voradj.training.continuous.local_sac import (
    grad_norm,
    masked_mean,
    sac_bootstrap_mask,
)
from tools.run_continuous_ctde_training import _make_trainer


def summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"n": 0, "min": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def clip_summary(values: list[float], threshold: float) -> dict[str, Any]:
    result = summary(values)
    array = np.asarray(values, dtype=np.float64)
    result.update({
        "threshold": float(threshold),
        "trigger_fraction": float(np.mean(array > threshold)),
        "clip_scale_factor": summary(
            np.minimum(1.0, threshold / np.maximum(array, 1e-12))
        ),
        "no_clip_optimizer_receives": "raw pre-clip gradient",
    })
    return result


def near_collision_flags(batch: dict[str, Any], threshold_m: float = 2.0) -> np.ndarray:
    """Geometry-only warning bucket, not a replay label or training input."""
    state = batch["global_state"]
    pursuers = state["pursuers"][:, 0, :, :2].cpu().numpy()
    obstacles = state["obstacles"].cpu().numpy()
    active = batch["active_mask"].cpu().numpy().astype(bool)
    obstacle_mask = state["obstacle_mask"].cpu().numpy().astype(bool)
    flags = np.zeros(len(active), dtype=bool)
    scale = 120.0
    for row in range(len(active)):
        points = pursuers[row, active[row]]
        if not len(points):
            continue
        boundary = np.min(
            np.concatenate([points, 1.0 - points], axis=1) * scale
        )
        clearance = float(boundary)
        for obstacle in obstacles[row, obstacle_mask[row]]:
            surface = np.linalg.norm(points - obstacle[:2], axis=1) * scale
            surface -= float(obstacle[2]) * scale
            clearance = min(clearance, float(surface.min()))
        if len(points) > 1:
            pairwise = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            pairwise += np.eye(len(points)) * 1e9
            clearance = min(clearance, float(pairwise.min() * scale))
        flags[row] = clearance <= float(threshold_m)
    return flags


def append_active(
    destination: dict[str, list[float]],
    prefix: str,
    mask: torch.Tensor,
    **tensors: torch.Tensor,
) -> None:
    for name, tensor in tensors.items():
        destination[f"{prefix}.{name}"].extend(
            tensor[mask].detach().float().cpu().reshape(-1).tolist()
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clip-threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    trainer = _make_trainer(config, args.device)
    trainer.load_checkpoint(args.checkpoint, manifest)
    replay = JointReplayBuffer.load(args.replay, manifest)
    sampler = UniformJointReplaySampler()
    gradients: dict[str, list[float]] = {
        "critic": [], "actor": [], "alpha": []
    }
    values: dict[str, list[float]] = defaultdict(list)
    transition_ids: list[int] = []
    batch_counts: dict[str, int] = defaultdict(int)

    for _ in range(int(args.batches)):
        batch = replay.sample(int(args.batch_size), sampler=sampler)
        transition_ids.extend(int(item) for item in batch["transition_ids"])
        local_obs = trainer._central_batch(batch, "local_obs")
        next_local_obs = trainer._central_batch(batch, "next_local_obs")
        central_obs = trainer._central_batch(batch, "global_state")
        next_central_obs = trainer._central_batch(batch, "next_global_state")
        active = batch["active_mask"].to(trainer.device).bool()
        rewards = batch["rewards"].to(trainer.device)
        actions = batch["actions"].to(trainer.device)
        bootstrap = sac_bootstrap_mask(
            batch["terminated"].to(trainer.device),
            batch["truncated"].to(trainer.device),
        )
        with torch.no_grad():
            next_actions, next_log_prob, _ = trainer._actor_sample(next_local_obs)
            target_q1 = trainer.target_critic1(next_central_obs, next_actions)
            target_q2 = trainer.target_critic2(next_central_obs, next_actions)
            target_q = torch.minimum(target_q1, target_q2) - trainer.alpha * next_log_prob
            target = rewards + trainer.config.gamma * bootstrap * target_q
        q1 = trainer.critic1(central_obs, actions)
        q2 = trainer.critic2(central_obs, actions)
        critic_loss = masked_mean(
            F.mse_loss(q1, target, reduction="none")
            + F.mse_loss(q2, target, reduction="none"), active
        )
        trainer.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        gradients["critic"].append(grad_norm([
            *trainer.critic1.parameters(), *trainer.critic2.parameters()
        ]))

        policy_actions, log_prob, _ = trainer._actor_sample(local_obs)
        for critic in (trainer.critic1, trainer.critic2):
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
        try:
            trainer.actor_optimizer.zero_grad(set_to_none=True)
            actor_q, actor_backward_loss = trainer._all_agent_actor_backward_terms(
                central_obs, policy_actions, log_prob, active
            )
            actor_backward_loss.backward()
            gradients["actor"].append(grad_norm(trainer.actor.parameters()))
        finally:
            for critic in (trainer.critic1, trainer.critic2):
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)

        alpha_loss = -masked_mean(
            trainer.log_alpha * (log_prob.detach() + trainer.config.target_entropy),
            active,
        )
        trainer.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        gradients["alpha"].append(grad_norm([trainer.log_alpha]))

        twin_gap = (q1 - q2).abs()
        td_error = (q1 - target).abs()
        entropy_term = trainer.alpha.detach() * log_prob
        ratio = entropy_term.abs() / actor_q.abs().clamp_min(1e-6)
        metadata = batch["metadata"]
        near = near_collision_flags(batch)
        collision = np.asarray([
            "collision" in item.get("event_ids", []) for item in metadata
        ])
        bucket_masks: dict[str, torch.Tensor] = {
            "ordinary": active,
            "collision": active & torch.as_tensor(collision, device=active.device)[:, None],
            "near_collision": active & torch.as_tensor(
                near & ~collision, device=active.device
            )[:, None],
            "pursuing": active & torch.as_tensor([
                item.get("phase") == "pre_capture" for item in metadata
            ], device=active.device)[:, None],
            "pure_ce": active & torch.as_tensor([
                item.get("scene") == "pure_ce" for item in metadata
            ], device=active.device)[:, None],
            "pre_capture": active & torch.as_tensor([
                item.get("phase") == "pre_capture" for item in metadata
            ], device=active.device)[:, None],
            "post_capture": active & torch.as_tensor([
                item.get("phase") == "post_capture" for item in metadata
            ], device=active.device)[:, None],
        }
        for bucket, mask in bucket_masks.items():
            batch_counts[bucket] += int(mask.sum().detach())
            append_active(
                values, bucket, mask,
                q1=q1, q2=q2, twin_gap=twin_gap, target_q=target_q,
                td_error=td_error, reward=rewards,
                reward_abs=rewards.abs(), log_prob=log_prob,
                alpha_log_prob=entropy_term,
                alpha_log_prob_abs=entropy_term.abs(), actor_q=actor_q,
                entropy_to_q_abs_ratio=ratio,
            )

    digest = hashlib.sha256(
        np.asarray(transition_ids, dtype=np.int64).tobytes()
    ).hexdigest()
    result = {
        "kind": "fixed_checkpoint_uniform_replay_sac_health_audit",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "replay": str(Path(args.replay).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "sampled_transition_id_sha256": digest,
        "gradients": {
            name: clip_summary(items, float(args.clip_threshold))
            for name, items in gradients.items()
        },
        "buckets": {
            bucket: {
                metric.split(".", 1)[1]: summary(items)
                for metric, items in values.items()
                if metric.startswith(bucket + ".")
            }
            for bucket in bucket_masks
        },
        "bucket_active_agent_counts": dict(batch_counts),
        "near_collision_definition": (
            "non-collision transition with <=2m geometry clearance to boundary, "
            "obstacle surface, or another pursuer"
        ),
        "alpha": float(trainer.alpha.detach()),
        "target_entropy": float(trainer.config.target_entropy),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "gradients": result["gradients"],
        "bucket_active_agent_counts": result["bucket_active_agent_counts"],
        "ordinary": result["buckets"]["ordinary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
