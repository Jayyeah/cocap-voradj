"""Auditable MAPPO, TD3 and IQN updates for the small-step migration."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

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


class MAPPOTrainer:
    """Standard clipped PPO/GAE using local shared actors and central V."""

    def __init__(self, actor: nn.Module, value: CentralValueNetwork, config: MAPPOConfig, device: str):
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.value = value.to(self.device)
        self.config = config
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.critic_lr, eps=1e-5)
        self.update_count = 0

    @torch.no_grad()
    def act(self, local_obs: Mapping[str, Any], global_obs: Mapping[str, Any], deterministic: bool = False):
        local = tensor_tree(local_obs, self.device)
        global_value = tensor_tree(global_obs, self.device)
        actions, log_prob, latent = self.actor.sample(local, deterministic=deterministic)
        values = self.value(global_value)
        return actions.cpu().numpy(), log_prob.cpu().numpy(), latent.cpu().numpy(), values.cpu().numpy()

    def update(self, batch: Mapping[str, Any], categorical: bool) -> dict[str, float]:
        local = tensor_tree(batch["local_obs"], self.device)
        central = tensor_tree(batch["global_obs"], self.device)
        flat_local, (steps, agents) = flatten_local(local)
        active = torch.as_tensor(batch["active_mask"], dtype=torch.bool, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch["values"], dtype=torch.float32, device=self.device)
        next_values = torch.as_tensor(batch["next_values"], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.bool, device=self.device)
        episode_end = torch.as_tensor(batch["episode_end"], dtype=torch.bool, device=self.device)
        old_log_prob = torch.as_tensor(batch["log_prob"], dtype=torch.float32, device=self.device)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(agents, device=self.device)
        for index in reversed(range(steps)):
            bootstrap = (~terminated[index]).float()
            delta = rewards[index] + self.config.gamma * bootstrap * next_values[index] - old_values[index]
            continuation = (~episode_end[index]).float()
            gae = delta + self.config.gamma * self.config.gae_lambda * continuation * gae
            advantages[index] = gae
        returns = advantages + old_values
        valid_adv = advantages[active]
        advantages = (advantages - valid_adv.mean()) / valid_adv.std(unbiased=False).clamp_min(1e-6)
        flat_active = active.reshape(-1)
        valid_indices = torch.nonzero(flat_active, as_tuple=False).squeeze(-1)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        latent = torch.as_tensor(batch["latent"], device=self.device)
        metric_rows = []
        for _ in range(self.config.ppo_epochs):
            permutation = valid_indices[torch.randperm(len(valid_indices), device=self.device)]
            for indices in torch.chunk(permutation, max(1, self.config.minibatches)):
                if not len(indices):
                    continue
                obs_mb = {key: value[indices] for key, value in flat_local.items()}
                if categorical:
                    log_prob, entropy = self.actor.evaluate_indices(obs_mb, latent.reshape(-1)[indices].long())
                else:
                    physical = actions.reshape(steps * agents, 2)[indices].float()
                    log_prob = self.actor.log_prob(obs_mb, physical)
                    distribution, _ = self.actor.distribution(obs_mb)
                    entropy = distribution.entropy().sum(dim=-1)
                ratio = torch.exp(log_prob - old_log_prob.reshape(-1)[indices])
                adv = advantages.reshape(-1)[indices]
                surrogate = torch.minimum(ratio * adv, ratio.clamp(1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * adv)
                actor_loss = -surrogate.mean() - self.config.entropy_coef * entropy.mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm))
                self.actor_optimizer.step()

                values_all = self.value(central).reshape(-1)
                value_pred = values_all[indices]
                value_old = old_values.reshape(-1)[indices]
                value_target = returns.reshape(-1)[indices]
                clipped_value = value_old + (value_pred - value_old).clamp(-self.config.clip_param, self.config.clip_param)
                value_loss = 0.5 * torch.maximum((value_pred - value_target).square(), (clipped_value - value_target).square()).mean()
                self.value_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coef * value_loss).backward()
                value_grad = float(torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm))
                self.value_optimizer.step()
                with torch.no_grad():
                    row = {
                        "actor_loss": float(actor_loss), "value_loss": float(value_loss),
                        "entropy": float(entropy.mean()),
                        "clip_fraction": float((torch.abs(ratio - 1.0) > self.config.clip_param).float().mean()),
                        "approx_kl": float((old_log_prob.reshape(-1)[indices] - log_prob).mean()),
                        "actor_grad_norm": actor_grad, "value_grad_norm": value_grad,
                    }
                    if not categorical:
                        row.update({"gaussian_mean_a": float(distribution.loc[:, 0].mean()),
                                    "gaussian_mean_w": float(distribution.loc[:, 1].mean()),
                                    "gaussian_std_a": float(distribution.scale[:, 0].mean()),
                                    "gaussian_std_w": float(distribution.scale[:, 1].mean())})
                    metric_rows.append(row)
        self.update_count += 1
        result = {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}
        with torch.no_grad():
            prediction = self.value(central)[active]
            result["explained_variance"] = explained_variance(prediction, returns[active])
        result["update_count"] = float(self.update_count)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "value": self.value.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(), "value_optimizer": self.value_optimizer.state_dict(),
                "update_count": self.update_count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"]); self.value.load_state_dict(state["value"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"]); self.value_optimizer.load_state_dict(state["value_optimizer"])
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
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": actor_loss_value,
                "q1_mean": float(q1[active].mean().detach()), "q2_mean": float(q2[active].mean().detach()),
                "twin_q_gap": float((q1[active] - q2[active]).abs().mean().detach()),
                "critic_grad_norm": critic_grad, "actor_grad_norm": actor_grad, "update_count": float(self.update_count)}

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critics": self.critics.state_dict(), "target_actor": self.target_actor.state_dict(),
                "target_critics": self.target_critics.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(), "update_count": self.update_count}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key, module in (("actor", self.actor), ("critics", self.critics), ("target_actor", self.target_actor), ("target_critics", self.target_critics)):
            module.load_state_dict(state[key])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"]); self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_count = int(state["update_count"])


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
