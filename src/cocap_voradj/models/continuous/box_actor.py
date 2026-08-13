"""Factorized squashed Gaussian actor for the continuous ``(a,w)`` bridge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from cocap_voradj.dynamics.continuous_action import AccelerationAngularVelocityActionAdapter
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoder


@dataclass(frozen=True)
class BoxActorConfig:
    hidden_dim: int = 128
    a_max: float = 0.4
    w_max: float = float(np.pi / 6.0)
    decision_dt: float = 0.5
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    dropout: float = 0.0
    context_pooling: str = "mean"


class SquashedGaussianAccelerationAngularVelocityActor(nn.Module):
    """Independent bounded Gaussian coordinates for ``(a,w)``.

    The affine tanh Jacobian is included in ``log_prob`` so the actor can be
    used directly by SAC.  Component bounds intentionally match the legacy
    IQN grid endpoints and are checked again by the environment adapter.
    """

    def __init__(
        self,
        encoder: LocalEntityTokenEncoder,
        config: BoxActorConfig | None = None,
        adapter: Optional[AccelerationAngularVelocityActionAdapter] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config or BoxActorConfig(hidden_dim=encoder.config.hidden_dim)
        h = int(self.config.hidden_dim)
        pooling = str(self.config.context_pooling).strip().lower()
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"unsupported actor context_pooling: {self.config.context_pooling!r}")
        self.context_pooling = pooling
        input_dim = (3 if pooling == "mean_max" else 2) * h
        self.policy = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(float(self.config.dropout)),
        )
        self.mean_head = nn.Linear(h, 2)
        self.log_std_head = nn.Linear(h, 2)
        self.adapter = adapter or AccelerationAngularVelocityActionAdapter(
            self.config.a_max,
            self.config.w_max,
            self.config.decision_dt,
        )

    def distribution(self, obs: Mapping[str, torch.Tensor]) -> Tuple[Normal, torch.Tensor]:
        features = self.encoder(obs)
        context = [features["self_token"], features["mean_context"]]
        if self.context_pooling == "mean_max":
            context.append(features["max_context"])
        hidden = self.policy(torch.cat(context, dim=-1))
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(self.config.log_std_min, self.config.log_std_max)
        return Normal(mean, log_std.exp()), log_std

    def _scale(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor([self.config.a_max, self.config.w_max], device=device, dtype=dtype)

    def _squash(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent) * self._scale(latent.device, latent.dtype)

    def _log_abs_det_jacobian(self, latent: torch.Tensor) -> torch.Tensor:
        scale = self._scale(latent.device, latent.dtype)
        # log(scale * (1 - tanh(z)^2)) using the numerically stable identity.
        correction = 2.0 * (np.log(2.0) - latent - nn.functional.softplus(-2.0 * latent))
        return (torch.log(scale) + correction).sum(dim=-1)

    def _inverse(self, action: torch.Tensor) -> torch.Tensor:
        scale = self._scale(action.device, action.dtype)
        normalized = (action / scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.atanh(normalized)

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
        action = self._squash(latent)
        log_prob = distribution.log_prob(latent).sum(dim=-1) - self._log_abs_det_jacobian(latent)
        return action, log_prob, latent

    @torch.no_grad()
    def act_with_contract(
        self,
        obs: Mapping[str, torch.Tensor],
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, list[Dict[str, float | bool]]]:
        raw, _, _ = self.sample(obs, deterministic=deterministic)
        final = []
        diagnostics = []
        for command in raw.detach().cpu().numpy():
            action, diag = self.adapter.validate_with_diagnostics(command)
            final.append(action)
            diagnostics.append(diag.as_dict())
        return torch.as_tensor(np.asarray(final), device=raw.device, dtype=raw.dtype), diagnostics
