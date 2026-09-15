"""Phase-aware MAPPO gradient diagnostics for the NormSense Full-Mix runs.

The logger is deliberately a side-channel: it replays the current rollout
with the frozen behavior policy, computes diagnostics with ``autograd.grad``
and never steps an optimizer, updates ValueNorm, or consumes RNG.  Occupancy
weighted gradients use the actual active-row denominator.  Conditioned
gradients use all available rows in a phase and report NA when the declared
minimum is not met; no synthetic or equalized rows are fabricated.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from cocap_voradj.training.small_step_ac import compute_gae, flatten_local, tensor_tree


SCHEMA = "normsense-gradient-logging-v1"
PHASES = ("capture", "support", "coverage")
COMPONENT_FIELDS = {
    "capture": "reward_capture_component",
    "support_capture": "reward_support_blend_capture",
    "support_coverage": "reward_support_blend_coverage",
    "coverage": "reward_coverage_component",
    "terminal": "reward_terminal_component",
    "safety": "reward_safety_component",
    "ce_center": "reward_ce_center_component",
    "ce_control": "reward_ce_control_component",
    "ce_pbrs": "reward_ce_pbrs_component",
}


def _stats(values: torch.Tensor | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values.detach().cpu() if torch.is_tensor(values) else values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p50": None, "p90": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _vector_norm(vector: torch.Tensor | None) -> float | None:
    return None if vector is None else float(vector.norm().detach().cpu())


def _cosine(left: torch.Tensor | None, right: torch.Tensor | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = float(left.norm() * right.norm())
    if denominator <= 1e-15:
        return None
    return float((left @ right).detach().cpu()) / denominator


def _explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    if prediction.numel() < 2 or target.numel() < 2:
        return None
    variance = torch.var(target)
    if float(variance) < 1e-8:
        return None
    return float((1.0 - torch.var(target - prediction) / variance).detach().cpu())


def _phase_vector(
    trainer: Any,
    flat_local: Mapping[str, torch.Tensor],
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    normalized_advantage: torch.Tensor,
    mask: torch.Tensor,
    denominator: int,
) -> torch.Tensor | None:
    ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if not len(ids):
        return None
    params = [parameter for parameter in trainer.actor.parameters() if parameter.requires_grad]
    losses = []
    for chunk in ids.split(256):
        log_prob, _ = trainer.actor.evaluate_indices(
            {key: value[chunk] for key, value in flat_local.items()}, actions[chunk]
        )
        ratio = (log_prob - old_log_prob[chunk]).exp()
        advantage = normalized_advantage[chunk]
        clipped = ratio.clamp(1.0 - trainer.config.clip_param, 1.0 + trainer.config.clip_param)
        losses.append(-torch.minimum(ratio * advantage, clipped * advantage).sum())
    loss = torch.stack(losses).sum() / max(int(denominator), 1)
    gradients = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat(
        [gradient.detach().reshape(-1) if gradient is not None else torch.zeros(parameter.numel(), device=trainer.device)
         for parameter, gradient in zip(params, gradients)]
    ).double().cpu()


def phase_gradient_diagnostics(
    trainer: Any,
    batch: Mapping[str, Any],
    *,
    minimum_conditioned_rows: int = 32,
) -> dict[str, Any]:
    """Return occupancy and phase-conditioned gradient/reward diagnostics.

    ``batch`` must contain ``gradient_phase`` and the reward component fields
    emitted by the Full-Mix collector.  The function has no optimizer or RNG
    side effects and is safe to call immediately before ``trainer.update``.
    """
    device = trainer.device
    local = tensor_tree(batch["local_obs"], device)
    flat_local, _ = flatten_local(local)
    active_grid = torch.as_tensor(batch["active_mask"], dtype=torch.bool, device=device)
    active = active_grid.reshape(-1)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    values = torch.as_tensor(batch["values"], dtype=torch.float32, device=device)
    next_values = torch.as_tensor(batch["next_values"], dtype=torch.float32, device=device)
    terminated = torch.as_tensor(batch["terminated"], dtype=torch.bool, device=device)
    truncated = torch.as_tensor(batch.get("truncated", np.zeros_like(batch["terminated"])), dtype=torch.bool, device=device)
    episode_end = torch.as_tensor(batch.get("episode_end", np.asarray(batch["terminated"]) | np.asarray(batch.get("truncated", False))), dtype=torch.bool, device=device)
    raw_values = trainer._denormalize_values(values)
    raw_next_values = trainer._denormalize_values(next_values)
    advantages, value_target = compute_gae(
        rewards,
        raw_values,
        raw_next_values,
        terminated,
        active_grid,
        gamma=float(trainer.config.gamma),
        gae_lambda=float(trainer.config.gae_lambda),
        truncated=truncated,
        episode_end=episode_end,
    )
    valid_advantages = advantages[active_grid]
    normalized_advantage = torch.zeros_like(advantages)
    if valid_advantages.numel():
        normalized_advantage[active_grid] = (valid_advantages - valid_advantages.mean()) / valid_advantages.std(unbiased=False).clamp_min(1e-6)

    phase_array = np.asarray(batch["gradient_phase"]).reshape(-1)
    total_rows = int(active.sum().item())
    actions = torch.as_tensor(batch["latent"], dtype=torch.long, device=device).reshape(-1)
    old_log_prob = torch.as_tensor(batch["log_prob"], dtype=torch.float32, device=device).reshape(-1)
    phase_masks = {phase: active & torch.as_tensor(phase_array == phase, dtype=torch.bool, device=device) for phase in PHASES}

    actor_was_training = bool(trainer.actor.training)
    trainer.actor.eval()
    occupancy_vectors = {
        phase: _phase_vector(trainer, flat_local, actions, old_log_prob, normalized_advantage.reshape(-1), phase_masks[phase], total_rows)
        for phase in PHASES
    }
    conditioned_vectors = {
        phase: (
            _phase_vector(trainer, flat_local, actions, old_log_prob, normalized_advantage.reshape(-1), phase_masks[phase], int(phase_masks[phase].sum().item()))
            if int(phase_masks[phase].sum().item()) >= int(minimum_conditioned_rows)
            else None
        )
        for phase in PHASES
    }
    if actor_was_training:
        trainer.actor.train()

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "phase_order": list(PHASES),
        "row_denominator": "active_rows_in_this_rollout",
        "conditioned_sampling": "all_available_rows_without_replacement",
        "minimum_conditioned_rows": int(minimum_conditioned_rows),
        "global_active_rows": total_rows,
        "phases": {},
        "pairwise_cosine": {"occupancy_weighted": {}, "phase_conditioned": {}},
    }
    for phase in PHASES:
        mask = phase_masks[phase]
        count = int(mask.sum().item())
        conditioned_status = "OK" if count >= int(minimum_conditioned_rows) else "NA_INSUFFICIENT_ROWS"
        component_stats = {}
        for name, field in COMPONENT_FIELDS.items():
            if field in batch:
                component_stats[name] = _stats(torch.as_tensor(batch[field], dtype=torch.float32, device=device).reshape(-1)[mask])
        phase_row = {
            "row_count": count,
            "raw_reward": _stats(rewards.reshape(-1)[mask]),
            "raw_reward_components": component_stats,
            "raw_advantage": _stats(advantages.reshape(-1)[mask]),
            "normalized_advantage": _stats(normalized_advantage.reshape(-1)[mask]),
            "gae_target": _stats(value_target.reshape(-1)[mask]),
            "value_target": _stats(value_target.reshape(-1)[mask]),
            "value_prediction": _stats(raw_values.reshape(-1)[mask]),
            "explained_variance": _explained_variance(raw_values.reshape(-1)[mask], value_target.reshape(-1)[mask]),
            "occupancy_weighted": {
                "gradient_norm": _vector_norm(occupancy_vectors[phase]),
                "normalization": "phase_loss_sum / global_active_rows; absent phase is exact zero contribution",
            },
            "phase_conditioned": {
                "gradient_norm": _vector_norm(conditioned_vectors[phase]),
                "normalization": "phase_loss_sum / phase_active_rows",
                "status": conditioned_status,
            },
        }
        report["phases"][phase] = phase_row
    for left_index, left in enumerate(PHASES):
        for right in PHASES[left_index + 1:]:
            key = f"{left}__{right}"
            report["pairwise_cosine"]["occupancy_weighted"][key] = _cosine(occupancy_vectors[left], occupancy_vectors[right])
            report["pairwise_cosine"]["phase_conditioned"][key] = _cosine(conditioned_vectors[left], conditioned_vectors[right])
    return report
