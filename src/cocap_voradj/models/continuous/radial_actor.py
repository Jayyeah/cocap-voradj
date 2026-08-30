"""Radial-squashed Gaussian actor for a two-dimensional acceleration disk."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from cocap_voradj.dynamics.continuous_action import AccelerationActionAdapter
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoder, policy_context


@dataclass(frozen=True)
class AccelerationActorConfig:
    hidden_dim: int = 128
    a_max: float = 0.8
    decision_dt: float = 0.5
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    dropout: float = 0.0
    context_pooling: str = "mean"


class RadialSquashedGaussianAccelerationActor(nn.Module):
    """Gaussian latent policy mapped to the closed acceleration disk.

    For latent ``z`` in R², the action is
    ``a = a_max * tanh(||z||) * z / max(||z||, eps)``. The radial Jacobian is
    included in ``log_prob``; unlike component-wise tanh this guarantees
    ``||a|| <= a_max``. The environment, not this actor, enforces ``v_max``
    while integrating the velocity.
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: AccelerationActorConfig | None = None,
        adapter: Optional[AccelerationActionAdapter] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config or AccelerationActorConfig(hidden_dim=encoder.config.hidden_dim)
        h = int(self.config.hidden_dim)
        pooling = str(self.config.context_pooling).strip().lower()
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"unsupported actor context_pooling: {self.config.context_pooling!r}")
        self.context_pooling = pooling
        input_dim = int(getattr(encoder, "decision_feature_dim", (3 if pooling == "mean_max" else 2) * h))
        self.policy = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(float(self.config.dropout)),
        )
        self.mean_head = nn.Linear(h, 2)
        self.log_std_head = nn.Linear(h, 2)
        self.adapter = adapter or AccelerationActionAdapter(
            self.config.a_max,
            self.config.decision_dt,
        )

    def distribution(self, obs: Mapping[str, torch.Tensor]) -> Tuple[Normal, torch.Tensor]:
        features = policy_context(self.encoder, obs, context_pooling=self.context_pooling)
        hidden = self.policy(features)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.config.log_std_min, self.config.log_std_max)
        return Normal(mean, log_std.exp()), log_std

    @staticmethod
    def _squash(z: torch.Tensor, a_max: float, eps: float = 1e-8) -> torch.Tensor:
        radius = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        direction = z / radius.clamp_min(eps)
        return float(a_max) * torch.tanh(radius) * direction

    def _log_abs_det_jacobian(self, z: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.vector_norm(z, dim=-1).clamp_min(1e-8)
        tanh_radius = torch.tanh(radius).clamp_min(1e-8)
        sech2 = (1.0 - torch.tanh(radius).square()).clamp_min(1e-8)
        logdet = (
            2.0 * math_log(self.config.a_max)
            + torch.log(tanh_radius)
            + torch.log(sech2)
            - torch.log(radius)
        )
        return torch.where(radius < 1e-4, torch.full_like(logdet, 2.0 * math_log(self.config.a_max)), logdet)

    def _inverse(self, action: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
        if torch.any(radius > float(self.config.a_max) + 1e-5):
            raise ValueError("action lies outside radial acceleration disk")
        normalized = (radius / float(self.config.a_max)).clamp(0.0, 1.0 - 1e-6)
        latent_radius = torch.atanh(normalized)
        return action / radius.clamp_min(1e-8) * latent_radius

    def log_prob(self, obs: Mapping[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        distribution, _ = self.distribution(obs)
        latent = self._inverse(action)
        return distribution.log_prob(latent).sum(dim=-1) - self._log_abs_det_jacobian(latent)

    def sample(
        self,
        obs: Mapping[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, _ = self.distribution(obs)
        latent = distribution.loc if deterministic else distribution.rsample()
        action = self._squash(latent, self.config.a_max)
        log_prob = distribution.log_prob(latent).sum(dim=-1) - self._log_abs_det_jacobian(latent)
        return action, log_prob, latent

    @torch.no_grad()
    def act_with_acceleration_contract(
        self,
        obs: Mapping[str, torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, list[Dict[str, float | bool]]]:
        """Deploy a bounded acceleration action through the contract adapter."""
        raw, _, _ = self.sample(obs, deterministic=deterministic)
        final = []
        diagnostics = []
        for idx, command in enumerate(raw.detach().cpu().numpy()):
            action, diag = self.adapter.validate_with_diagnostics(command)
            final.append(action)
            diagnostics.append(diag.as_dict())
        return torch.as_tensor(np.asarray(final), device=raw.device, dtype=raw.dtype), diagnostics


# Compatibility aliases. New experiments use the acceleration names.
RadialActorConfig = AccelerationActorConfig
RadialSquashedGaussianActor = RadialSquashedGaussianAccelerationActor


def math_log(value: float) -> float:
    return float(np.log(float(value)))
