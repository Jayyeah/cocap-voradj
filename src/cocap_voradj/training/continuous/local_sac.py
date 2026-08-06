"""Minimal shared local-SAC update used only for the P4 data-chain smoke."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LocalEntityTokenEncoder,
    LocalEntityTokenEncoderConfig,
)
from cocap_voradj.models.continuous.radial_actor import (
    AccelerationActorConfig,
    RadialSquashedGaussianAccelerationActor,
)
from cocap_voradj.models.continuous.box_actor import (
    BoxActorConfig,
    SquashedGaussianAccelerationAngularVelocityActor,
)


@dataclass(frozen=True)
class LocalSACConfig:
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 0.2
    target_entropy: float = -2.0
    grad_clip_norm: float | None = None


class LocalQCritic(nn.Module):
    def __init__(self, encoder: LocalEntityTokenEncoder, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        feature_dim = 2 * encoder.config.hidden_dim + 2
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: Mapping[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        fused = torch.cat([features["self_token"], features["mean_context"], action], dim=-1)
        return self.head(fused).squeeze(-1)


def flatten_joint_local_obs(local_obs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Flatten [batch, agent, ...] local observations to [batch*agent, ...]."""
    flattened: Dict[str, torch.Tensor] = {}
    for key, value in local_obs.items():
        if value.dim() < 2:
            raise ValueError(f"local observation {key} must include batch and agent dimensions")
        flattened[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
    return flattened


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def sac_bootstrap_mask(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    """Return the per-agent Bellman bootstrap multiplier.

    ``truncated`` is intentionally accepted as part of the replay contract,
    but a time-limit truncation is not an absorbing terminal. Only
    ``terminated`` disables bootstrap.
    """
    if terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated must have identical shapes")
    return (~terminated.bool()).to(dtype=torch.float32)


def grad_norm(parameters: Any) -> float:
    """Return the finite global L2 norm of currently accumulated gradients."""
    squared = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    if not squared:
        return 0.0
    return float(torch.sqrt(torch.stack(squared).sum()).detach())


class LocalSACTrainer:
    """Shared-parameter SAC over all active pursuers in a joint batch."""

    def __init__(
        self,
        encoder_config: LocalEntityTokenEncoderConfig | None = None,
        actor_config: AccelerationActorConfig | BoxActorConfig | None = None,
        config: LocalSACConfig | None = None,
        device: str = "cpu",
        action_mode: str = "axay",
    ):
        self.device = torch.device(device)
        self.config = config or LocalSACConfig()
        encoder_config = encoder_config or LocalEntityTokenEncoderConfig(hidden_dim=self.config.hidden_dim)
        if str(action_mode).lower() in {"aw", "acceleration_angular_velocity_body"}:
            actor_config = actor_config or BoxActorConfig(hidden_dim=encoder_config.hidden_dim)
            self.actor = SquashedGaussianAccelerationAngularVelocityActor(
                LocalEntityTokenEncoder(encoder_config), actor_config
            ).to(self.device)
        else:
            actor_config = actor_config or AccelerationActorConfig(hidden_dim=encoder_config.hidden_dim)
            self.actor = RadialSquashedGaussianAccelerationActor(
                LocalEntityTokenEncoder(encoder_config), actor_config
            ).to(self.device)
        self.action_mode = str(action_mode)
        self.critic1 = LocalQCritic(LocalEntityTokenEncoder(encoder_config), encoder_config.hidden_dim).to(self.device)
        self.critic2 = LocalQCritic(LocalEntityTokenEncoder(encoder_config), encoder_config.hidden_dim).to(self.device)
        self.target_critic1 = LocalQCritic(LocalEntityTokenEncoder(encoder_config), encoder_config.hidden_dim).to(self.device)
        self.target_critic2 = LocalQCritic(LocalEntityTokenEncoder(encoder_config), encoder_config.hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(self.config.alpha_init), device=self.device)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.config.critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.resume_runtime_state: Dict[str, Any] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp_min(1e-6)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1.0 - self.config.tau).add_(self.config.tau * source_parameter)

    def save_checkpoint(
        self,
        path: str | Path,
        contract: Mapping[str, Any],
        runtime_state: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": 2,
            "contract": dict(contract),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": self._cuda_rng_state_all(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "runtime_state": dict(runtime_state or {}),
        }
        torch.save(payload, Path(path))

    @staticmethod
    def _cuda_rng_state_all():
        if not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.get_rng_state_all()
        except Exception:
            return None

    def load_checkpoint(self, path: str | Path, expected_contract: Mapping[str, Any]) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("schema_version") != 2:
            raise ValueError("local SAC checkpoint schema version mismatch")
        if dict(payload.get("contract", {})) != dict(expected_contract):
            raise ValueError("local SAC checkpoint contract mismatch; refusing unsafe resume")
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.target_critic1.load_state_dict(payload["target_critic1"])
        self.target_critic2.load_state_dict(payload["target_critic2"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_rng_state = payload.get("cuda_rng_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            states = [item.cpu() for item in cuda_rng_state] if isinstance(cuda_rng_state, (list, tuple)) else [cuda_rng_state.cpu()]
            torch.cuda.set_rng_state_all(states[: torch.cuda.device_count()])
        if payload.get("python_rng_state") is not None:
            random.setstate(payload["python_rng_state"])
        if payload.get("numpy_rng_state") is not None:
            np.random.set_state(payload["numpy_rng_state"])
        self.resume_runtime_state = dict(payload.get("runtime_state", {}) or {})

    def update(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        local_obs = {key: value.to(self.device) for key, value in batch["local_obs"].items()}
        next_local_obs = {key: value.to(self.device) for key, value in batch["next_local_obs"].items()}
        active = batch["active_mask"].to(self.device).bool()
        rewards = batch["rewards"].to(self.device)
        bootstrap = sac_bootstrap_mask(
            batch["terminated"].to(self.device),
            batch["truncated"].to(self.device),
        )
        actions = batch["actions"].to(self.device)
        flat_obs = flatten_joint_local_obs(local_obs)
        flat_next_obs = flatten_joint_local_obs(next_local_obs)
        flat_actions = actions.reshape(-1, 2)
        flat_rewards = rewards.reshape(-1)
        flat_bootstrap = bootstrap.reshape(-1)
        flat_active = active.reshape(-1)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(flat_next_obs)
            target_q = torch.minimum(
                self.target_critic1(flat_next_obs, next_action),
                self.target_critic2(flat_next_obs, next_action),
            ) - self.alpha.detach() * next_log_prob
            target = flat_rewards + self.config.gamma * flat_bootstrap * target_q
        q1 = self.critic1(flat_obs, flat_actions)
        q2 = self.critic2(flat_obs, flat_actions)
        critic_loss = masked_mean(F.mse_loss(q1, target, reduction="none") + F.mse_loss(q2, target, reduction="none"), flat_active)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = grad_norm([*self.critic1.parameters(), *self.critic2.parameters()])
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [*self.critic1.parameters(), *self.critic2.parameters()],
                float(self.config.grad_clip_norm),
            )
        self.critic_optimizer.step()

        action, log_prob, latent = self.actor.sample(flat_obs)
        _, log_std = self.actor.distribution(flat_obs)
        actor_q = torch.minimum(self.critic1(flat_obs, action), self.critic2(flat_obs, action))
        actor_loss = masked_mean(self.alpha.detach() * log_prob - actor_q, flat_active)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = grad_norm(self.actor.parameters())
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config.grad_clip_norm))
        self.actor_optimizer.step()

        alpha_loss = -masked_mean(self.log_alpha * (log_prob.detach() + self.config.target_entropy), flat_active)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_grad_norm = grad_norm([self.log_alpha])
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([self.log_alpha], float(self.config.grad_clip_norm))
        self.alpha_optimizer.step()
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        metrics = {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "active_count": float(flat_active.sum().detach()),
            "q1_mean": float(q1.detach().mean()),
            "q2_mean": float(q2.detach().mean()),
            "target_q_mean": float(target_q.detach().mean()),
            "td_error_abs_mean": float((q1.detach() - target.detach()).abs().mean()),
            "log_prob_mean": float(log_prob.detach().mean()),
            "entropy_proxy_mean": float((-log_prob.detach()).mean()),
            "log_std_mean": float(log_std.detach().mean()),
            "log_std_min": float(log_std.detach().min()),
            "log_std_max": float(log_std.detach().max()),
            "latent_norm_mean": float(torch.linalg.vector_norm(latent.detach(), dim=-1).mean()),
            "critic_grad_norm": critic_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "alpha_grad_norm": alpha_grad_norm,
            "finite": float(all(torch.isfinite(value).all() for value in (critic_loss, actor_loss, alpha_loss, self.alpha))),
        }
        if not bool(metrics["finite"]):
            raise FloatingPointError("local SAC update produced NaN or Inf")
        return metrics
