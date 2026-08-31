"""Auditable MAPPO, TD3 and IQN updates for the small-step migration."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.models.small_step_ac import CentralValueNetwork


def tensor_tree(tree: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in tree.items():
        dtype = torch.long if key == "types" else torch.bool if "mask" in key or key == "masks" else torch.float32
        result[key] = torch.as_tensor(value, dtype=dtype, device=device)
    return result


def flatten_local(tree: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], tuple[int, int]]:
    first = next(iter(tree.values()))
    batch, agents = int(first.shape[0]), int(first.shape[1])
    return {key: value.reshape(batch * agents, *value.shape[2:]) for key, value in tree.items()}, (batch, agents)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> float:
    variance = torch.var(target)
    if float(variance) < 1e-8:
        return 0.0
    return float((1.0 - torch.var(target - prediction) / variance).detach())


class ValueNorm:
    """Running scalar value normalizer used by the MAPPO critic.

    Statistics are updated from detached unnormalized return targets once per
    rollout, while the value network predicts normalized values.  Keeping the
    statistics in trainer state makes checkpoint/resume reproducible.
    """

    def __init__(
        self,
        *,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device: torch.device | str = "cpu",
    ):
        beta = float(beta)
        epsilon = float(epsilon)
        if not 0.0 < beta < 1.0:
            raise ValueError("ValueNorm beta must be in (0, 1)")
        if epsilon <= 0.0:
            raise ValueError("ValueNorm epsilon must be positive")
        self.beta = beta
        self.epsilon = epsilon
        self.running_mean = torch.zeros(1, dtype=torch.float32, device=device)
        self.running_mean_sq = torch.zeros(1, dtype=torch.float32, device=device)
        self.debiasing_term = torch.zeros(1, dtype=torch.float32, device=device)

    @property
    def mean(self) -> torch.Tensor:
        return self.running_mean / self.debiasing_term.clamp_min(self.epsilon)

    @property
    def variance(self) -> torch.Tensor:
        mean = self.mean
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp_min(self.epsilon)
        # Match the reference MAPPO ValueNorm floor.  Tiny early-rollout
        # variance must not amplify critic targets by hundreds of times.
        return (mean_sq - mean.square()).clamp_min(1e-2)

    @property
    def std(self) -> torch.Tensor:
        # Before the first rollout there is no unbiased scale estimate.  A
        # unit fallback keeps the value network's initial output in its
        # natural scale instead of multiplying it by sqrt(epsilon).
        return torch.where(
            self.debiasing_term > 0.0,
            torch.sqrt(self.variance + self.epsilon),
            torch.ones_like(self.running_mean),
        )

    def update(self, values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        values = values.detach().to(dtype=torch.float32).reshape(-1)
        if mask is not None:
            valid = mask.detach().bool().reshape(-1)
            if valid.numel() != values.numel():
                raise ValueError("ValueNorm mask and values have incompatible sizes")
            values = values[valid]
        if not values.numel():
            return
        batch_mean = values.mean().reshape(1)
        batch_mean_sq = values.square().mean().reshape(1)
        self.running_mean.mul_(self.beta).add_(batch_mean, alpha=1.0 - self.beta)
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq, alpha=1.0 - self.beta)
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return (values - mean) / std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return values * std + mean

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "epsilon": self.epsilon,
            "running_mean": self.running_mean.detach().clone(),
            "running_mean_sq": self.running_mean_sq.detach().clone(),
            "debiasing_term": self.debiasing_term.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.running_mean.copy_(torch.as_tensor(state["running_mean"], device=self.running_mean.device))
        self.running_mean_sq.copy_(torch.as_tensor(state["running_mean_sq"], device=self.running_mean_sq.device))
        self.debiasing_term.copy_(torch.as_tensor(state["debiasing_term"], device=self.debiasing_term.device))


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    active: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    truncated: Optional[torch.Tensor] = None,
    episode_end: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked GAE with separate terminal and truncation semantics.

    A true task terminal does not bootstrap.  A time-limit truncation does
    bootstrap from next_values but ends the GAE recursion so advantages never
    leak across the reset into the next episode.  episode_end is retained for
    callers that have an additional rollout boundary; terminal and truncation
    flags are always included in it.
    """

    tensors = [rewards, values, next_values, terminated, active]
    if any(item.shape != rewards.shape for item in tensors[1:]):
        raise ValueError("GAE inputs must have identical [time, agent] shapes")
    terminated = terminated.bool()
    active = active.bool()
    if truncated is None:
        truncated = torch.zeros_like(terminated)
    else:
        truncated = truncated.bool()
        if truncated.shape != rewards.shape:
            raise ValueError("GAE truncated mask has incompatible shape")
    if episode_end is None:
        episode_end = terminated | truncated
    else:
        episode_end = episode_end.bool()
        if episode_end.shape != rewards.shape:
            raise ValueError("GAE episode_end mask has incompatible shape")
        episode_end = episode_end | terminated | truncated

    steps, agents = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(agents, dtype=rewards.dtype, device=rewards.device)
    gamma = float(gamma)
    gae_lambda = float(gae_lambda)
    for index in reversed(range(steps)):
        bootstrap = (~terminated[index]).to(dtype=rewards.dtype)
        delta = rewards[index] + gamma * bootstrap * next_values[index] - values[index]
        continuation = (~episode_end[index]).to(dtype=rewards.dtype)
        gae = delta + gamma * gae_lambda * continuation * gae
        gae = torch.where(active[index], gae, torch.zeros_like(gae))
        advantages[index] = gae
    returns = torch.where(active, advantages + values, torch.zeros_like(values))
    return advantages, returns


@dataclass(frozen=True)
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 10
    minibatches: int = 4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    value_norm: bool = False
    value_norm_beta: float = 0.99999
    value_norm_epsilon: float = 1e-5


class MAPPOTrainer:
    """Standard clipped PPO/GAE using local shared actors and central V."""

    def __init__(self, actor: nn.Module, value: CentralValueNetwork, config: MAPPOConfig, device: str):
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.value = value.to(self.device)
        self.config = config
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.critic_lr, eps=1e-5)
        self.value_normalizer = (
            ValueNorm(beta=config.value_norm_beta, epsilon=config.value_norm_epsilon, device=self.device)
            if config.value_norm
            else None
        )
        self.update_count = 0

    @property
    def value_norm(self) -> Optional[ValueNorm]:
        """Compatibility alias for callers that use the short MAPPO name."""

        return self.value_normalizer

    def _denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        if self.value_normalizer is None:
            return values
        return self.value_normalizer.denormalize(values)

    @torch.no_grad()
    def value_for_gae(self, global_obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return value estimates in the raw return scale used by GAE."""

        return self._denormalize_values(self.value(global_obs))

    @torch.no_grad()
    def act(self, local_obs: Mapping[str, Any], global_obs: Mapping[str, Any], deterministic: bool = False):
        local = tensor_tree(local_obs, self.device)
        global_value = tensor_tree(global_obs, self.device)
        actions, log_prob, latent = self.actor.sample(local, deterministic=deterministic)
        # Store the critic's normalized prediction exactly as produced.  GAE
        # denormalizes it under the pre-update statistics below; PPO value
        # clipping must compare against this original network output, not a
        # value re-normalized after the running statistics have changed.
        values = self.value(global_value)
        return actions.cpu().numpy(), log_prob.cpu().numpy(), latent.cpu().numpy(), values.cpu().numpy()

    def update(self, batch: Mapping[str, Any], categorical: bool) -> dict[str, float]:
        local = tensor_tree(batch["local_obs"], self.device)
        central = tensor_tree(batch["global_obs"], self.device)
        flat_local, (steps, agents) = flatten_local(local)
        active = torch.as_tensor(batch["active_mask"], dtype=torch.bool, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        old_value_predictions = torch.as_tensor(batch["values"], dtype=torch.float32, device=self.device)
        next_value_predictions = torch.as_tensor(batch["next_values"], dtype=torch.float32, device=self.device)
        old_values = self._denormalize_values(old_value_predictions)
        next_values = self._denormalize_values(next_value_predictions)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.bool, device=self.device)
        truncated = torch.as_tensor(
            batch.get("truncated", torch.zeros_like(terminated)), dtype=torch.bool, device=self.device
        )
        episode_end = torch.as_tensor(
            batch.get("episode_end", terminated | truncated), dtype=torch.bool, device=self.device
        )
        old_log_prob = torch.as_tensor(batch["log_prob"], dtype=torch.float32, device=self.device).detach()
        advantages, returns = compute_gae(
            rewards,
            old_values,
            next_values,
            terminated,
            active,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            truncated=truncated,
            episode_end=episode_end,
        )
        valid_adv = advantages[active]
        if valid_adv.numel():
            advantages = (advantages - valid_adv.mean()) / valid_adv.std(unbiased=False).clamp_min(1e-6)
        else:
            advantages = torch.zeros_like(advantages)
        if self.value_normalizer is not None and valid_adv.numel():
            self.value_normalizer.update(returns, active)
            normalized_old_values = old_value_predictions
            normalized_returns = self.value_normalizer.normalize(returns)
        else:
            normalized_old_values = old_value_predictions
            normalized_returns = returns
        flat_active = active.reshape(-1)
        valid_indices = torch.nonzero(flat_active, as_tuple=False).squeeze(-1)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        latent = torch.as_tensor(batch["latent"], device=self.device)
        if not len(valid_indices):
            self.update_count += 1
            return {
                "actor_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "approx_kl": 0.0,
                "actor_grad_norm": 0.0,
                "value_grad_norm": 0.0,
                "explained_variance": 0.0,
                "kl_early_stop": 0.0,
                "ppo_epochs_completed": 0.0,
                "minibatch_updates": 0.0,
                "actor_update_l2": 0.0,
                "actor_update_relative_l2": 0.0,
                "update_count": float(self.update_count),
            }
        actor_before = [parameter.detach().clone() for parameter in self.actor.parameters()]
        metric_rows = []
        stopped_early = False
        epochs_completed = 0
        for _ in range(self.config.ppo_epochs):
            permutation = valid_indices[torch.randperm(len(valid_indices), device=self.device)]
            for indices in torch.chunk(permutation, max(1, self.config.minibatches)):
                if not len(indices):
                    continue
                obs_mb = {key: value[indices] for key, value in flat_local.items()}
                if categorical:
                    log_prob, entropy = self.actor.evaluate_indices(obs_mb, latent.reshape(-1)[indices].long())
                    base_entropy = entropy
                else:
                    physical = actions.reshape(steps * agents, 2)[indices].float()
                    latent_mb = latent.reshape(steps * agents, 2)[indices].float()
                    # The rollout's pre-tanh sample is authoritative.  A
                    # saturated float32 physical action cannot be inverted
                    # losslessly and previously produced artificial PPO KL.
                    log_prob, entropy, base_entropy, distribution, log_std = (
                        self.actor.evaluate_latent(obs_mb, latent_mb)
                    )
                log_ratio = log_prob - old_log_prob.reshape(-1)[indices]
                ratio = torch.exp(log_ratio)
                adv = advantages.reshape(-1)[indices]
                surrogate = torch.minimum(ratio * adv, ratio.clamp(1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * adv)
                actor_loss = -surrogate.mean() - self.config.entropy_coef * entropy.mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm))
                self.actor_optimizer.step()

                values_all = self.value(central).reshape(-1)
                value_pred = values_all[indices]
                value_old = normalized_old_values.reshape(-1)[indices]
                value_target = normalized_returns.reshape(-1)[indices]
                clipped_value = value_old + (value_pred - value_old).clamp(-self.config.clip_param, self.config.clip_param)
                value_loss = 0.5 * torch.maximum((value_pred - value_target).square(), (clipped_value - value_target).square()).mean()
                self.value_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * value_loss).backward()
                value_grad = float(torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm))
                self.value_optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    row = {
                        "actor_loss": float(actor_loss), "value_loss": float(value_loss),
                        "entropy": float(entropy.mean()),
                        "clip_fraction": float((torch.abs(ratio - 1.0) > self.config.clip_param).float().mean()),
                        "approx_kl": float(approx_kl),
                        "actor_grad_norm": actor_grad, "value_grad_norm": value_grad,
                    }
                    if not categorical:
                        normalized_action = torch.tanh(latent_mb)
                        threshold = float(getattr(self.actor.config, "saturation_threshold", 0.99))
                        saturated = normalized_action.abs() >= threshold
                        action_magnitude = torch.linalg.vector_norm(physical, dim=-1)
                        mean_abs = distribution.loc.abs()
                        latent_abs = latent_mb.abs()
                        row.update({
                            # Keep historical names while adding unambiguous
                            # pre-tanh diagnostics that cannot cancel signs.
                            "gaussian_mean_a": float(distribution.loc[:, 0].mean()),
                            "gaussian_mean_w": float(distribution.loc[:, 1].mean()),
                            "pre_tanh_mean_a": float(distribution.loc[:, 0].mean()),
                            "pre_tanh_mean_w": float(distribution.loc[:, 1].mean()),
                            "pre_tanh_mean_abs_a": float(mean_abs[:, 0].mean()),
                            "pre_tanh_mean_abs_w": float(mean_abs[:, 1].mean()),
                            "pre_tanh_mean_abs_max": float(mean_abs.max()),
                            "sampled_latent_abs_mean": float(latent_abs.mean()),
                            "sampled_latent_abs_max": float(latent_abs.max()),
                            "log_std_a_mean": float(log_std[:, 0].mean()),
                            "log_std_w_mean": float(log_std[:, 1].mean()),
                            "log_std_min": float(log_std.min()),
                            "log_std_max": float(log_std.max()),
                            "gaussian_std_a": float(distribution.scale[:, 0].mean()),
                            "gaussian_std_w": float(distribution.scale[:, 1].mean()),
                            "std_a_mean": float(distribution.scale[:, 0].mean()),
                            "std_w_mean": float(distribution.scale[:, 1].mean()),
                            "entropy_base_gaussian": float(base_entropy.mean()),
                            "entropy_squashed_physical_mc": float(entropy.mean()),
                            "post_tanh_saturation_ratio": float(saturated.float().mean()),
                            "post_tanh_saturation_any_ratio": float(saturated.any(dim=-1).float().mean()),
                            "post_tanh_saturation_a_ratio": float(saturated[:, 0].float().mean()),
                            "post_tanh_saturation_w_ratio": float(saturated[:, 1].float().mean()),
                            "action_abs_a_mean": float(physical[:, 0].abs().mean()),
                            "action_abs_w_mean": float(physical[:, 1].abs().mean()),
                            "action_magnitude_mean": float(action_magnitude.mean()),
                            "action_magnitude_max": float(action_magnitude.max()),
                            "action_log_prob_mean": float(log_prob.mean()),
                            "action_log_prob_min": float(log_prob.min()),
                            "action_log_prob_max": float(log_prob.max()),
                        })
                    metric_rows.append(row)
                    if (
                        self.config.target_kl is not None
                        and float(self.config.target_kl) > 0.0
                        and float(approx_kl) > float(self.config.target_kl)
                    ):
                        stopped_early = True
                if stopped_early:
                    break
            epochs_completed += 1
            if stopped_early:
                break
        self.update_count += 1
        result = {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}
        if not categorical:
            for key in (
                "pre_tanh_mean_abs_max",
                "sampled_latent_abs_max",
                "log_std_max",
                "action_magnitude_max",
                "action_log_prob_max",
            ):
                result[key] = float(max(row[key] for row in metric_rows))
            for key in ("log_std_min", "action_log_prob_min"):
                result[key] = float(min(row[key] for row in metric_rows))
            result["critic_grad_norm"] = result["value_grad_norm"]
        with torch.no_grad():
            prediction = self._denormalize_values(self.value(central))[active]
            result["explained_variance"] = explained_variance(prediction, returns[active])
            if self.value_normalizer is not None:
                result["value_norm_mean"] = float(self.value_normalizer.mean)
                result["value_norm_std"] = float(self.value_normalizer.std)
        result["kl_early_stop"] = float(stopped_early)
        result["ppo_epochs_completed"] = float(epochs_completed)
        result["minibatch_updates"] = float(len(metric_rows))
        with torch.no_grad():
            update_sq = sum(
                float((parameter - before).square().sum())
                for parameter, before in zip(self.actor.parameters(), actor_before)
            )
            parameter_sq = sum(float(before.square().sum()) for before in actor_before)
            result["actor_update_l2"] = float(np.sqrt(update_sq))
            result["actor_update_relative_l2"] = float(
                np.sqrt(update_sq) / max(np.sqrt(parameter_sq), 1e-12)
            )
            result["approx_kl_max"] = float(max(row["approx_kl"] for row in metric_rows))
        result["update_count"] = float(self.update_count)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "value": self.value.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(), "value_optimizer": self.value_optimizer.state_dict(),
                "value_normalizer": self.value_normalizer.state_dict() if self.value_normalizer is not None else None,
                "update_count": self.update_count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"]); self.value.load_state_dict(state["value"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"]); self.value_optimizer.load_state_dict(state["value_optimizer"])
        if self.value_normalizer is not None and state.get("value_normalizer") is not None:
            self.value_normalizer.load_state_dict(state["value_normalizer"])
        self.update_count = int(state["update_count"])


@dataclass(frozen=True)
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    policy_noise: float = 0.1
    noise_clip: float = 0.2
    policy_delay: int = 2
    max_grad_norm: float = 0.5


class TD3Trainer:
    def __init__(self, actor: nn.Module, critics: nn.Module, config: TD3Config, device: str):
        self.device = torch.device(device); self.actor = actor.to(self.device); self.critics = critics.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device).requires_grad_(False)
        self.target_critics = copy.deepcopy(self.critics).to(self.device).requires_grad_(False)
        self.config = config
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.critic_lr)
        self.update_count = 0
        self.last_actor_update = {
            "last_actor_loss": 0.0,
            "last_actor_grad_norm": 0.0,
            "last_actor_update_count": 0.0,
        }

    @staticmethod
    def _soft(source: nn.Module, target: nn.Module, tau: float) -> None:
        for src, dst in zip(source.parameters(), target.parameters()):
            dst.data.lerp_(src.data, tau)

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        local = tensor_tree(batch["local_obs"], self.device); next_local = tensor_tree(batch["next_local_obs"], self.device)
        central = tensor_tree(batch["global_state"], self.device); next_central = tensor_tree(batch["next_global_state"], self.device)
        flat_local, shape = flatten_local(local); flat_next, _ = flatten_local(next_local)
        active = batch["active_mask"].to(self.device).bool(); actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device); bootstrap = (~batch["terminated"].to(self.device).bool()).float()
        with torch.no_grad():
            next_action = self.target_actor(flat_next).reshape(*shape, 2)
            noise = torch.randn_like(next_action) * self.config.policy_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            scale = self.target_actor.scale.view(1, 1, 2)
            next_action = (next_action + noise * scale).clamp(-scale, scale)
            tq1, tq2 = self.target_critics(next_central, next_action)
            target = rewards + self.config.gamma * bootstrap * torch.minimum(tq1, tq2)
        q1, q2 = self.critics(central, actions)
        critic_loss = masked_mean((q1 - target).square() + (q2 - target).square(), active)
        self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
        critic_grad = float(torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.max_grad_norm)); self.critic_optimizer.step()
        actor_loss_value = 0.0; actor_grad = 0.0
        self.update_count += 1
        if self.update_count % self.config.policy_delay == 0:
            policy_action = self.actor(flat_local).reshape(*shape, 2)
            for parameter in self.critics.parameters(): parameter.requires_grad_(False)
            actor_loss = -masked_mean(self.critics.critic1(central, policy_action), active)
            self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
            actor_grad = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)); self.actor_optimizer.step()
            for parameter in self.critics.parameters(): parameter.requires_grad_(True)
            self._soft(self.actor, self.target_actor, self.config.tau); self._soft(self.critics, self.target_critics, self.config.tau)
            actor_loss_value = float(actor_loss.detach())
            self.last_actor_update = {
                "last_actor_loss": actor_loss_value,
                "last_actor_grad_norm": actor_grad,
                "last_actor_update_count": float(self.update_count),
            }
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": actor_loss_value,
                "q1_mean": float(q1[active].mean().detach()), "q2_mean": float(q2[active].mean().detach()),
                "twin_q_gap": float((q1[active] - q2[active]).abs().mean().detach()),
                "critic_grad_norm": critic_grad, "actor_grad_norm": actor_grad,
                "actor_updated_this_step": float(self.update_count % self.config.policy_delay == 0),
                **self.last_actor_update, "update_count": float(self.update_count)}

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critics": self.critics.state_dict(), "target_actor": self.target_actor.state_dict(),
                "target_critics": self.target_critics.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(), "update_count": self.update_count,
                "last_actor_update": dict(self.last_actor_update)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key, module in (("actor", self.actor), ("critics", self.critics), ("target_actor", self.target_actor), ("target_critics", self.target_critics)):
            module.load_state_dict(state[key])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"]); self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_count = int(state["update_count"])
        self.last_actor_update = dict(state.get("last_actor_update", self.last_actor_update))


class IQNTrainer:
    """The repository's distributional IQN update, isolated from action meaning."""
    def __init__(self, model: CoCapIQN, lr: float, gamma: float, kappa: float, quantiles: int, target_freq: int, device: str):
        self.device = torch.device(device); self.model = model.to(self.device); self.target = copy.deepcopy(model).to(self.device).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr)); self.gamma = float(gamma)
        self.kappa = float(kappa); self.quantiles = int(quantiles); self.target_freq = int(target_freq); self.update_count = 0

    def update(self, batch: Mapping[str, Any], global_step: int | None = None) -> dict[str, float]:
        obs = batch["obs"]; next_obs = batch["next_obs"]; actions = batch["actions"]; rewards = batch["rewards"]; dones = batch["dones"]
        size = int(actions.shape[0]); n = self.quantiles
        target_tau = torch.rand(size, n, device=self.device)
        with torch.no_grad():
            next_q = self.target(next_obs, num_tau=n, mode="voradj", tau=target_tau)["q_values"].max(2)[0].unsqueeze(1)
            target = rewards.view(size, 1, 1) + self.gamma * (1.0 - dones.view(size, 1, 1)) * next_q
        tau = torch.rand(size, n, device=self.device)
        output = self.model(obs, num_tau=n, mode="voradj", tau=tau)
        prediction = output["q_values"].gather(-1, actions[:, None, None].expand(-1, n, 1))
        error = target - prediction
        huber = torch.where(error.abs() <= self.kappa, 0.5 * error.square(), self.kappa * (error.abs() - 0.5 * self.kappa))
        weight = (output["taus"] - (error.detach() < 0).float()).abs()
        loss = (weight * huber / self.kappa).sum(1).mean(1).mean()
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)); self.optimizer.step(); self.update_count += 1
        target_due = (global_step is not None and int(global_step) % self.target_freq == 0) or (global_step is None and self.update_count % self.target_freq == 0)
        if target_due: self.target.load_state_dict(self.model.state_dict())
        return {"iqn_loss": float(loss.detach()), "grad_norm": grad, "update_count": float(self.update_count)}

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(), "update_count": self.update_count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"]); self.target.load_state_dict(state["target"]); self.optimizer.load_state_dict(state["optimizer"]); self.update_count = int(state["update_count"])
