"""Factorized squashed Gaussian actor for the continuous ``(a,w)`` bridge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from cocap_voradj.dynamics.continuous_action import AccelerationAngularVelocityActionAdapter
from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoder, policy_context


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
    orthogonal_policy_head: bool = False
    mean_output_gain: float = 0.01
    initial_log_std: Optional[float] = None
    saturation_threshold: float = 0.99


class SquashedGaussianAccelerationAngularVelocityActor(nn.Module):
    """Independent bounded Gaussian coordinates for ``(a,w)``.

    The affine tanh Jacobian is included in ``log_prob`` so the actor can be
    used directly by SAC.  Component bounds intentionally match the legacy
    IQN grid endpoints and are checked again by the environment adapter.
    """

    def __init__(
        self,
        encoder: nn.Module,
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
        input_dim = int(getattr(encoder, "decision_feature_dim", (3 if pooling == "mean_max" else 2) * h))
        self.policy = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(float(self.config.dropout)),
        )
        self.mean_head = nn.Linear(h, 2)
        self.log_std_head = nn.Linear(h, 2)
        if float(self.config.a_max) <= 0.0 or float(self.config.w_max) <= 0.0:
            raise ValueError("continuous AW physical scales must be positive")
        if self.config.log_std_min > self.config.log_std_max:
            raise ValueError("log_std_min must not exceed log_std_max")
        if not 0.0 < float(self.config.saturation_threshold) < 1.0:
            raise ValueError("saturation_threshold must be in (0, 1)")
        if self.config.orthogonal_policy_head:
            nn.init.orthogonal_(self.policy[0].weight, gain=np.sqrt(2.0))
            nn.init.zeros_(self.policy[0].bias)
            nn.init.orthogonal_(self.mean_head.weight, gain=float(self.config.mean_output_gain))
            nn.init.zeros_(self.mean_head.bias)
        if self.config.initial_log_std is not None:
            initial_log_std = float(self.config.initial_log_std)
            if not self.config.log_std_min <= initial_log_std <= self.config.log_std_max:
                raise ValueError("initial_log_std must lie inside the configured clamp")
            # A zero state-dependent weight is the standard, auditable MAPPO
            # starting point.  The head remains trainable after initialization.
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, initial_log_std)
        self.adapter = adapter or AccelerationAngularVelocityActionAdapter(
            self.config.a_max,
            self.config.w_max,
            self.config.decision_dt,
        )

    def distribution(self, obs: Mapping[str, torch.Tensor]) -> Tuple[Normal, torch.Tensor]:
        features = policy_context(self.encoder, obs, context_pooling=self.context_pooling)
        hidden = self.policy(features)
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

    def log_prob_from_distribution(self, distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
        """Density of an authoritative pre-tanh sample in physical action space."""

        if latent.shape[-1:] != (2,):
            raise ValueError("continuous AW latent must have final dimension 2")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("continuous AW latent must be finite")
        return distribution.log_prob(latent).sum(dim=-1) - self._log_abs_det_jacobian(latent)

    def log_prob_from_latent(
        self,
        obs: Mapping[str, torch.Tensor],
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Re-evaluate a rollout's exact pre-tanh latent without an inverse tanh.

        PPO must use this path.  A saturated float32 physical action can no
        longer encode whether its original latent was, for example, 8 or 10;
        reconstructing it with ``atanh`` would therefore create an artificial
        likelihood ratio and KL spike.
        """

        distribution, _ = self.distribution(obs)
        return self.log_prob_from_distribution(distribution, latent)

    def evaluate_latent(
        self,
        obs: Mapping[str, torch.Tensor],
        latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Normal, torch.Tensor]:
        """Return PPO log-prob and both bounded/base entropy diagnostics.

        The bounded entropy is a one-sample reparameterized Monte-Carlo
        estimate of the actual physical-action policy entropy.  It uses the
        same distribution forward pass as the PPO likelihood, which also
        avoids an additional legacy-backbone dropout draw.
        """

        distribution, log_std = self.distribution(obs)
        log_prob = self.log_prob_from_distribution(distribution, latent)
        entropy_latent = distribution.rsample()
        bounded_entropy = -self.log_prob_from_distribution(distribution, entropy_latent)
        base_entropy = distribution.entropy().sum(dim=-1)
        return log_prob, bounded_entropy, base_entropy, distribution, log_std

    def log_prob(self, obs: Mapping[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1:] != (2,):
            raise ValueError("continuous AW action must have final dimension 2")
        if not bool(torch.isfinite(action).all()):
            raise ValueError("continuous AW action must be finite")
        scale = self._scale(action.device, action.dtype)
        if bool((action.abs() > scale * (1.0 + 1e-6)).any()):
            raise ValueError("continuous AW action lies outside physical bounds")
        distribution, _ = self.distribution(obs)
        latent = self._inverse(action)
        return self.log_prob_from_distribution(distribution, latent)

    def sample(
        self,
        obs: Mapping[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, _ = self.distribution(obs)
        latent = distribution.loc if deterministic else distribution.rsample()
        action = self._squash(latent)
        log_prob = self.log_prob_from_distribution(distribution, latent)
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
