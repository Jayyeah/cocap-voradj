"""Centralized-twin-critic SAC update for the PSA-MASAC experiment path."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping
import random

import numpy as np
import torch
import torch.nn.functional as F

from cocap_voradj.models.continuous.central_attention_critic import (
    CentralAttentionCritic,
    CentralCriticConfig,
)
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoderConfig
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoder
from cocap_voradj.models.continuous.radial_actor import (
    AccelerationActorConfig,
    RadialSquashedGaussianAccelerationActor,
)
from cocap_voradj.models.continuous.box_actor import (
    BoxActorConfig,
    SquashedGaussianAccelerationAngularVelocityActor,
)
from cocap_voradj.training.continuous.local_sac import (
    flatten_joint_local_obs,
    grad_norm,
    masked_mean,
    sac_bootstrap_mask,
)


def _quantile_summary(tensor: torch.Tensor) -> Dict[str, float]:
    value = tensor.detach().float().reshape(-1)
    if value.numel() == 0:
        return {"min": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(value.min()),
        "p5": float(value.quantile(0.05)),
        "p50": float(value.quantile(0.50)),
        "p95": float(value.quantile(0.95)),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
    }


@dataclass(frozen=True)
class CentralSACConfig:
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 0.2
    target_entropy: float = -2.0
    grad_clip_norm: float | None = None


class CentralSACTrainer:
    """Shared decentralized actor with two independent global critics.

    The actor consumes only local observations. The critics consume the
    training-only global entity schema and produce one Q value per focal
    pursuer. All reductions are masked means over active agents.
    """

    def __init__(
        self,
        encoder_config: LocalEntityTokenEncoderConfig | None = None,
        actor_config: AccelerationActorConfig | BoxActorConfig | None = None,
        critic_config: CentralCriticConfig | None = None,
        config: CentralSACConfig | None = None,
        device: str = "cpu",
        action_mode: str = "axay",
    ):
        self.device = torch.device(device)
        self.config = config or CentralSACConfig()
        encoder_config = encoder_config or LocalEntityTokenEncoderConfig(hidden_dim=self.config.hidden_dim)
        if str(action_mode).lower() in {"aw", "acceleration_angular_velocity_body"}:
            actor_config = actor_config or BoxActorConfig(hidden_dim=encoder_config.hidden_dim)
        else:
            actor_config = actor_config or AccelerationActorConfig(hidden_dim=encoder_config.hidden_dim)
        critic_config = critic_config or CentralCriticConfig(hidden_dim=self.config.hidden_dim)
        # Actor and critics have independent parameters even when their token
        # widths happen to match.
        if str(action_mode).lower() in {"aw", "acceleration_angular_velocity_body"}:
            self.actor = SquashedGaussianAccelerationAngularVelocityActor(
                LocalEntityTokenEncoder(encoder_config), actor_config
            ).to(self.device)
        else:
            self.actor = RadialSquashedGaussianAccelerationActor(
                LocalEntityTokenEncoder(encoder_config), actor_config
            ).to(self.device)
        self.action_mode = str(action_mode)
        self.critic1 = CentralAttentionCritic(critic_config).to(self.device)
        self.critic2 = CentralAttentionCritic(critic_config).to(self.device)
        self.target_critic1 = CentralAttentionCritic(critic_config).to(self.device)
        self.target_critic2 = CentralAttentionCritic(critic_config).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()
        for parameter in [*self.target_critic1.parameters(), *self.target_critic2.parameters()]:
            parameter.requires_grad_(False)
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(float(self.config.alpha_init), device=self.device)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.config.critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.resume_runtime_state: Dict[str, Any] = {}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp_min(1e-6)

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1.0 - self.config.tau).add_(self.config.tau * source_parameter)

    @staticmethod
    def _to_device_tree(value: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {key: tensor.to(device) for key, tensor in value.items()}

    def _central_batch(self, batch: Mapping[str, Any], key: str) -> Dict[str, torch.Tensor]:
        value = batch[key]
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be the structured central entity schema")
        return self._to_device_tree(value, self.device)

    def _actor_sample(self, local_obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = flatten_joint_local_obs(local_obs)
        action, log_prob, latent = self.actor.sample(flat)
        batch, agents = local_obs["self"].shape[:2]
        return action.reshape(batch, agents, 2), log_prob.reshape(batch, agents), latent.reshape(batch, agents, 2)

    def _focal_actor_q(
        self,
        central_obs: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Q_i with all other-agent action branches detached."""
        values = []
        for focal in range(actions.shape[1]):
            focal_actions = actions.detach().clone()
            focal_actions[:, focal] = actions[:, focal]
            q1 = self.critic1(central_obs, focal_actions)[:, focal]
            q2 = self.critic2(central_obs, focal_actions)[:, focal]
            values.append(torch.minimum(q1, q2))
        result = torch.stack(values, dim=1)
        return result * active.to(dtype=result.dtype)

    def _focal_actor_q_for_ids(
        self,
        central_obs: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        focal_agent_id: torch.Tensor,
    ) -> torch.Tensor:
        """Q_i for each focal item while detaching every teammate action branch."""
        batch = actions.shape[0]
        detached = actions.detach().clone()
        row = torch.arange(batch, device=actions.device)
        detached[row, focal_agent_id] = actions[row, focal_agent_id]
        q1 = self.critic1(central_obs, detached).gather(1, focal_agent_id[:, None]).squeeze(1)
        q2 = self.critic2(central_obs, detached).gather(1, focal_agent_id[:, None]).squeeze(1)
        return torch.minimum(q1, q2)

    def update(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        if "focal_agent_id" in batch:
            return self._update_focal(batch)
        local_obs = self._central_batch(batch, "local_obs")
        next_local_obs = self._central_batch(batch, "next_local_obs")
        central_obs = self._central_batch(batch, "global_state")
        next_central_obs = self._central_batch(batch, "next_global_state")
        active = batch["active_mask"].to(self.device).bool()
        rewards = batch["rewards"].to(self.device)
        bootstrap = sac_bootstrap_mask(
            batch["terminated"].to(self.device),
            batch["truncated"].to(self.device),
        )
        actions = batch["actions"].to(self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self._actor_sample(next_local_obs)
            target_q1 = self.target_critic1(next_central_obs, next_actions)
            target_q2 = self.target_critic2(next_central_obs, next_actions)
            target_q = torch.minimum(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target = rewards + self.config.gamma * bootstrap * target_q

        q1 = self.critic1(central_obs, actions)
        q2 = self.critic2(central_obs, actions)
        critic_loss = masked_mean(
            F.mse_loss(q1, target, reduction="none") + F.mse_loss(q2, target, reduction="none"), active
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = grad_norm([*self.critic1.parameters(), *self.critic2.parameters()])
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [*self.critic1.parameters(), *self.critic2.parameters()],
                float(self.config.grad_clip_norm),
            )
        self.critic_optimizer.step()

        policy_actions, log_prob, latent = self._actor_sample(local_obs)
        flat_local_obs = flatten_joint_local_obs(local_obs)
        _, log_std = self.actor.distribution(flat_local_obs)
        actor_q = self._focal_actor_q(central_obs, policy_actions, active)
        actor_loss = masked_mean(self.alpha.detach() * log_prob - actor_q, active)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = grad_norm(self.actor.parameters())
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config.grad_clip_norm))
        self.actor_optimizer.step()

        alpha_loss = -masked_mean(self.log_alpha * (log_prob.detach() + self.config.target_entropy), active)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_grad_norm = grad_norm([self.log_alpha])
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([self.log_alpha], float(self.config.grad_clip_norm))
        self.alpha_optimizer.step()
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        finite = all(torch.isfinite(item).all() for item in (critic_loss, actor_loss, alpha_loss, self.alpha, q1, q2))
        metrics = {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "active_count": float(active.sum().detach()),
            "q1_mean": float(q1.detach().mean()),
            "q2_mean": float(q2.detach().mean()),
            "twin_q_gap": float((q1.detach() - q2.detach()).abs().mean()),
            "target_q_mean": float(target_q.detach().mean()),
            "td_error_abs_mean": float((q1.detach() - target.detach()).abs().mean()),
            "log_prob_mean": float(log_prob.detach().mean()),
            "entropy_proxy_mean": float((-log_prob.detach()).mean()),
            # ---- E0 entropy calibration diagnostics (2026-08-08) ----
            "log_alpha": float(self.log_alpha.detach()),
            "log_prob_std": float(log_prob.detach().std()),
            "log_prob_p5": float(torch.quantile(log_prob.detach(), 0.05)),
            "log_prob_p50": float(torch.quantile(log_prob.detach(), 0.50)),
            "log_prob_p95": float(torch.quantile(log_prob.detach(), 0.95)),
            "entropy_residual_mean": float((log_prob.detach() + self.config.target_entropy).mean()),
            "entropy_residual_p5": float(torch.quantile(log_prob.detach() + self.config.target_entropy, 0.05)),
            "entropy_residual_p50": float(torch.quantile(log_prob.detach() + self.config.target_entropy, 0.50)),
            "entropy_residual_p95": float(torch.quantile(log_prob.detach() + self.config.target_entropy, 0.95)),
            "log_std_a_mean": float(log_std.detach()[..., 0].mean()),
            "log_std_omega_mean": float(log_std.detach()[..., 1].mean()),
            "std_a_mean": float(log_std.detach()[..., 0].exp().mean()),
            "std_omega_mean": float(log_std.detach()[..., 1].exp().mean()),
            "log_std_mean": float(log_std.detach().mean()),
            "log_std_min": float(log_std.detach().min()),
            "log_std_max": float(log_std.detach().max()),
            "latent_norm_mean": float(torch.linalg.vector_norm(latent.detach(), dim=-1).mean()),
            "critic_grad_norm": critic_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "alpha_grad_norm": alpha_grad_norm,
            "finite": float(finite),
        }
        if not finite:
            raise FloatingPointError("central SAC update produced NaN or Inf")
        return metrics

    def _update_focal(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        local_obs = self._central_batch(batch, "local_obs")
        next_local_obs = self._central_batch(batch, "next_local_obs")
        central_obs = self._central_batch(batch, "global_state")
        next_central_obs = self._central_batch(batch, "next_global_state")
        active = batch["active_mask"].to(self.device).bool()
        focal_id = batch["focal_agent_id"].to(self.device).long()
        focal_reward = batch["focal_reward"].to(self.device)
        focal_terminated = batch["focal_terminated"].to(self.device)
        focal_truncated = batch["focal_truncated"].to(self.device)
        bootstrap = sac_bootstrap_mask(focal_terminated, focal_truncated)
        actions = batch["actions"].to(self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self._actor_sample(next_local_obs)
            target_q = torch.minimum(
                self.target_critic1(next_central_obs, next_actions),
                self.target_critic2(next_central_obs, next_actions),
            ) - self.alpha.detach() * next_log_prob
            target_q_focal = target_q.gather(1, focal_id[:, None]).squeeze(1)
            target = focal_reward + self.config.gamma * bootstrap * target_q_focal

        q1 = self.critic1(central_obs, actions)
        q2 = self.critic2(central_obs, actions)
        q1_focal = q1.gather(1, focal_id[:, None]).squeeze(1)
        q2_focal = q2.gather(1, focal_id[:, None]).squeeze(1)
        critic_loss = F.mse_loss(q1_focal, target) + F.mse_loss(q2_focal, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_pre_clip_grad_norm = grad_norm([*self.critic1.parameters(), *self.critic2.parameters()])
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [*self.critic1.parameters(), *self.critic2.parameters()],
                float(self.config.grad_clip_norm),
            )
            critic_post_clip_grad_norm = grad_norm([*self.critic1.parameters(), *self.critic2.parameters()])
        else:
            critic_post_clip_grad_norm = critic_pre_clip_grad_norm
        critic_clip_ratio = (
            min(1.0, float(self.config.grad_clip_norm) / max(critic_pre_clip_grad_norm, 1e-8))
            if self.config.grad_clip_norm is not None
            else 1.0
        )
        self.critic_optimizer.step()

        policy_actions, log_prob, latent = self._actor_sample(local_obs)
        log_prob_focal = log_prob.gather(1, focal_id[:, None]).squeeze(1)
        flat_local_obs = flatten_joint_local_obs(local_obs)
        _, log_std = self.actor.distribution(flat_local_obs)
        log_std = log_std.reshape(log_prob.shape[0], log_prob.shape[1], -1)
        log_std_focal = log_std.gather(1, focal_id[:, None, None].expand(-1, -1, log_std.shape[-1])).squeeze(1)
        latent_focal = latent.gather(1, focal_id[:, None, None].expand(-1, -1, latent.shape[-1])).squeeze(1)
        actor_q = self._focal_actor_q_for_ids(central_obs, policy_actions, focal_id)
        actor_loss = (self.alpha.detach() * log_prob_focal - actor_q).mean()
        for critic in (self.critic1, self.critic2):
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_pre_clip_grad_norm = grad_norm(self.actor.parameters())
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config.grad_clip_norm))
            actor_post_clip_grad_norm = grad_norm(self.actor.parameters())
        else:
            actor_post_clip_grad_norm = actor_pre_clip_grad_norm
        actor_clip_ratio = (
            min(1.0, float(self.config.grad_clip_norm) / max(actor_pre_clip_grad_norm, 1e-8))
            if self.config.grad_clip_norm is not None
            else 1.0
        )
        self.actor_optimizer.step()
        for critic in (self.critic1, self.critic2):
            for parameter in critic.parameters():
                parameter.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_prob_focal.detach() + self.config.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_pre_clip_grad_norm = grad_norm([self.log_alpha])
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([self.log_alpha], float(self.config.grad_clip_norm))
            alpha_post_clip_grad_norm = grad_norm([self.log_alpha])
        else:
            alpha_post_clip_grad_norm = alpha_pre_clip_grad_norm
        alpha_clip_ratio = (
            min(1.0, float(self.config.grad_clip_norm) / max(alpha_pre_clip_grad_norm, 1e-8))
            if self.config.grad_clip_norm is not None
            else 1.0
        )
        self.alpha_optimizer.step()
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        finite = all(
            torch.isfinite(item).all()
            for item in (
                critic_loss,
                actor_loss,
                alpha_loss,
                self.alpha,
                q1_focal,
                q2_focal,
                target,
                log_prob_focal,
            )
        )
        metrics = {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "active_count": float(active.sum().detach()),
            "focal_count": float(len(focal_id)),
            "q1_mean": float(q1_focal.detach().mean()),
            "q2_mean": float(q2_focal.detach().mean()),
            "twin_q_gap": float((q1_focal.detach() - q2_focal.detach()).abs().mean()),
            "target_q_mean": float(target_q_focal.detach().mean()),
            "td_error_abs_mean": float((q1_focal.detach() - target.detach()).abs().mean()),
            "q1_summary": _quantile_summary(q1_focal),
            "q2_summary": _quantile_summary(q2_focal),
            "target_q_summary": _quantile_summary(target_q_focal),
            "td_error_summary": _quantile_summary((q1_focal.detach() - target.detach()).abs()),
            "log_prob_mean": float(log_prob_focal.detach().mean()),
            "entropy_proxy_mean": float((-log_prob_focal.detach()).mean()),
            # ---- E0 entropy calibration diagnostics (2026-08-08) ----
            "log_alpha": float(self.log_alpha.detach()),
            "log_prob_std": float(log_prob_focal.detach().std()),
            "log_prob_p5": float(torch.quantile(log_prob_focal.detach(), 0.05)),
            "log_prob_p50": float(torch.quantile(log_prob_focal.detach(), 0.50)),
            "log_prob_p95": float(torch.quantile(log_prob_focal.detach(), 0.95)),
            "entropy_residual_mean": float((log_prob_focal.detach() + self.config.target_entropy).mean()),
            "entropy_residual_p5": float(torch.quantile(log_prob_focal.detach() + self.config.target_entropy, 0.05)),
            "entropy_residual_p50": float(torch.quantile(log_prob_focal.detach() + self.config.target_entropy, 0.50)),
            "entropy_residual_p95": float(torch.quantile(log_prob_focal.detach() + self.config.target_entropy, 0.95)),
            "log_std_a_mean": float(log_std_focal.detach()[..., 0].mean()),
            "log_std_omega_mean": float(log_std_focal.detach()[..., 1].mean()),
            "std_a_mean": float(log_std_focal.detach()[..., 0].exp().mean()),
            "std_omega_mean": float(log_std_focal.detach()[..., 1].exp().mean()),
            "log_std_mean": float(log_std_focal.detach().mean()),
            "log_std_min": float(log_std_focal.detach().min()),
            "log_std_max": float(log_std_focal.detach().max()),
            "latent_norm_mean": float(torch.linalg.vector_norm(latent_focal.detach(), dim=-1).mean()),
            "critic_grad_norm": critic_pre_clip_grad_norm,
            "critic_pre_clip_grad_norm": critic_pre_clip_grad_norm,
            "critic_post_clip_grad_norm": critic_post_clip_grad_norm,
            "critic_clip_ratio": critic_clip_ratio,
            "actor_grad_norm": actor_pre_clip_grad_norm,
            "actor_pre_clip_grad_norm": actor_pre_clip_grad_norm,
            "actor_post_clip_grad_norm": actor_post_clip_grad_norm,
            "actor_clip_ratio": actor_clip_ratio,
            "alpha_grad_norm": alpha_pre_clip_grad_norm,
            "alpha_pre_clip_grad_norm": alpha_pre_clip_grad_norm,
            "alpha_post_clip_grad_norm": alpha_post_clip_grad_norm,
            "alpha_clip_ratio": alpha_clip_ratio,
            "finite": float(finite),
        }
        if not finite:
            raise FloatingPointError("focal central SAC update produced NaN or Inf")
        return metrics

    def save_checkpoint(
        self,
        path: str | Path,
        contract: Mapping[str, Any],
        runtime_state: Mapping[str, Any] | None = None,
    ) -> None:
        torch.save(
            {
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
            },
            Path(path),
        )

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
            raise ValueError("central SAC checkpoint schema version mismatch")
        if dict(payload.get("contract", {})) != dict(expected_contract):
            raise ValueError("central SAC checkpoint contract mismatch; refusing unsafe resume")
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
