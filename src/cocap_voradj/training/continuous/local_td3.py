"""Strictly local TD3 update for continuous acceleration/angular velocity."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from cocap_voradj.models.continuous.local_td3 import (
    LocalTD3Actor,
    LocalTD3Critic,
    assert_no_parameter_sharing,
)
from cocap_voradj.training.continuous.local_sac import grad_norm


@dataclass(frozen=True)
class LocalTD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    max_grad_norm: float = 0.5
    saturation_threshold: float = 0.99


def _select_obs(
    local_obs: Mapping[str, torch.Tensor], mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Flatten [batch,agent,...] and keep only selected focal rows."""

    flat_mask = mask.reshape(-1)
    return {
        key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])[flat_mask]
        for key, value in local_obs.items()
    }


def td3_bootstrap_mask(
    terminated: torch.Tensor, truncated: torch.Tensor
) -> torch.Tensor:
    """Bootstrap through time limits and stop only at true task terminals."""

    if terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated must have identical shapes")
    return ~terminated.bool()


class LocalTD3Trainer:
    """Twin local critics and delayed deterministic actor update.

    Dropout modules stay in evaluation mode even while autograd is enabled.
    This makes the actor truly deterministic and preserves the legacy IQN
    decision-feature parity boundary used by warm start.
    """

    schema_version = 1

    def __init__(
        self,
        actor: LocalTD3Actor,
        critic1: LocalTD3Critic,
        critic2: LocalTD3Critic,
        config: LocalTD3Config | None = None,
        device: str = "cpu",
    ) -> None:
        assert_no_parameter_sharing(actor, critic1, critic2)
        self.device = torch.device(device)
        self.config = config or LocalTD3Config()
        if self.config.policy_delay <= 0:
            raise ValueError("policy_delay must be positive")
        if not 0.0 < self.config.tau <= 1.0:
            raise ValueError("tau must be in (0,1]")
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device).requires_grad_(False)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device).requires_grad_(False)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.config.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.config.critic_lr)
        self.update_count = 0
        self.policy_update_count = 0
        self._deterministic_modes()

    def _deterministic_modes(self) -> None:
        for module in (
            self.actor,
            self.critic1,
            self.critic2,
            self.target_actor,
            self.target_critic1,
            self.target_critic2,
        ):
            module.eval()

    @staticmethod
    @torch.no_grad()
    def _soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)

    def smoothed_target_action(
        self, next_obs: Mapping[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply clipped Gaussian noise in normalized action coordinates."""

        base = self.target_actor(next_obs)
        normalized_noise = (
            torch.randn_like(base) * float(self.config.policy_noise)
            if noise is None
            else noise.to(device=base.device, dtype=base.dtype)
        )
        normalized_noise = normalized_noise.clamp(
            -float(self.config.noise_clip), float(self.config.noise_clip)
        )
        scale = self.target_actor.scale
        action = (base + normalized_noise * scale).clamp(-scale, scale)
        return action, normalized_noise

    def act(self, local_obs: Mapping[str, Any]) -> torch.Tensor:
        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in local_obs.items()
        }
        self.actor.eval()
        with torch.no_grad():
            return self.actor(tensors)

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        self._deterministic_modes()
        local = {key: value.to(self.device) for key, value in batch["local_obs"].items()}
        next_local = {
            key: value.to(self.device) for key, value in batch["next_local_obs"].items()
        }
        active = batch["active_mask"].to(self.device).bool()
        if not bool(active.any()):
            raise ValueError("local TD3 batch contains no active focal rows")
        obs = _select_obs(local, active)
        action = batch["actions"].to(self.device)[active]
        reward = batch["rewards"].to(self.device)[active]
        terminated = batch["terminated"].to(self.device).bool()[active]
        truncated = batch["truncated"].to(self.device).bool()[active]

        # A time-limit truncation remains bootstrappable; only a true terminal
        # removes the Bellman successor. Avoid forwarding absent terminal next
        # observations, since 0 * NaN would still poison the target.
        bootstrap = td3_bootstrap_mask(terminated, truncated)
        target_q = torch.zeros_like(reward)
        smoothing_noise = torch.empty((0, 2), device=self.device)
        if bool(bootstrap.any()):
            flat_next_mask = active.clone()
            flat_next_mask[active] = bootstrap
            next_obs = _select_obs(next_local, flat_next_mask)
            with torch.no_grad():
                next_action, smoothing_noise = self.smoothed_target_action(next_obs)
                next_min = torch.minimum(
                    self.target_critic1(next_obs, next_action),
                    self.target_critic2(next_obs, next_action),
                )
            target_q[bootstrap] = next_min
        target = reward + float(self.config.gamma) * bootstrap.float() * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        critic1_grad = grad_norm(self.critic1.parameters())
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.max_grad_norm)
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        critic2_grad = grad_norm(self.critic2.parameters())
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.max_grad_norm)
        self.critic2_optimizer.step()

        self.update_count += 1
        actor_updated = self.update_count % int(self.config.policy_delay) == 0
        actor_loss_value = 0.0
        actor_grad = 0.0
        actor_action = self.actor(obs).detach()
        if actor_updated:
            for parameter in self.critic1.parameters():
                parameter.requires_grad_(False)
            policy_action = self.actor(obs)
            actor_loss = -self.critic1(obs, policy_action).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = grad_norm(self.actor.parameters())
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            for parameter in self.critic1.parameters():
                parameter.requires_grad_(True)
            self.policy_update_count += 1
            self._soft_update(self.actor, self.target_actor, self.config.tau)
            self._soft_update(self.critic1, self.target_critic1, self.config.tau)
            self._soft_update(self.critic2, self.target_critic2, self.config.tau)
            actor_loss_value = float(actor_loss.detach())
            actor_action = policy_action.detach()

        normalized_action = actor_action / self.actor.scale
        saturation = normalized_action.abs() >= float(self.config.saturation_threshold)
        q_gap = q1.detach() - q2.detach()
        td1 = q1.detach() - target.detach()
        td2 = q2.detach() - target.detach()
        metrics = {
            "critic1_loss": float(critic1_loss.detach()),
            "critic2_loss": float(critic2_loss.detach()),
            "critic_loss": float((critic1_loss + critic2_loss).detach()),
            "actor_loss": actor_loss_value,
            "actor_grad_norm": actor_grad,
            "critic1_grad_norm": critic1_grad,
            "critic2_grad_norm": critic2_grad,
            "update_count": float(self.update_count),
            "policy_update_count": float(self.policy_update_count),
            "actor_updated_this_step": float(actor_updated),
            "q1_mean": float(q1.detach().mean()),
            "q1_std": float(q1.detach().std(unbiased=False)),
            "q2_mean": float(q2.detach().mean()),
            "q2_std": float(q2.detach().std(unbiased=False)),
            "q1_minus_q2_mean": float(q_gap.mean()),
            "q_gap_abs_mean": float(q_gap.abs().mean()),
            "q_min_mean": float(torch.minimum(q1.detach(), q2.detach()).mean()),
            "bellman_target_mean": float(target.detach().mean()),
            "bellman_target_std": float(target.detach().std(unbiased=False)),
            "td_error1_abs_mean": float(td1.abs().mean()),
            "td_error2_abs_mean": float(td2.abs().mean()),
            "bootstrap_fraction": float(bootstrap.float().mean()),
            "truncation_fraction": float(truncated.float().mean()),
            "target_noise_normalized_abs_mean": (
                float(smoothing_noise.abs().mean()) if smoothing_noise.numel() else 0.0
            ),
            "action_a_mean": float(actor_action[:, 0].mean()),
            "action_a_std": float(actor_action[:, 0].std(unbiased=False)),
            "action_w_mean": float(actor_action[:, 1].mean()),
            "action_w_std": float(actor_action[:, 1].std(unbiased=False)),
            "action_saturation_ratio": float(saturation.float().mean()),
            "active_focal_rows": float(action.shape[0]),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("local TD3 update produced NaN or Inf")
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "update_count": self.update_count,
            "policy_update_count": self.policy_update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.schema_version:
            raise ValueError("local TD3 trainer schema mismatch")
        for key, module in (
            ("actor", self.actor),
            ("critic1", self.critic1),
            ("critic2", self.critic2),
            ("target_actor", self.target_actor),
            ("target_critic1", self.target_critic1),
            ("target_critic2", self.target_critic2),
        ):
            module.load_state_dict(state[key])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(state["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
        self.update_count = int(state["update_count"])
        self.policy_update_count = int(state["policy_update_count"])
        self._deterministic_modes()
