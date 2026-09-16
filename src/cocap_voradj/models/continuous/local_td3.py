"""Local deterministic actor and twin critics for isolated TD3-AW Stage 1.

Every network owns an independent legacy-compatible decision backbone.  The
critics intentionally consume only the focal agent's local observation and
physical ``(a, omega)`` command; no global state or teammate action enters this
module.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
)


class LocalTD3Actor(nn.Module):
    """Deterministic local policy with component-wise physical AW bounds."""

    def __init__(
        self,
        backbone: LegacyVorAdjFeatureBackbone,
        a_max: float = 0.4,
        w_max: float = torch.pi / 6.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.output_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        # A small final layer is the conventional deterministic-policy start
        # and leaves exploration magnitude under the explicit noise contract.
        nn.init.uniform_(self.head[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.head[-1].bias, -3e-3, 3e-3)
        self.register_buffer(
            "scale", torch.tensor([float(a_max), float(w_max)], dtype=torch.float32)
        )

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.tanh(self.head(self.backbone(obs))) * self.scale

    def normalized_action(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(obs) / self.scale


class LocalTD3Critic(nn.Module):
    """Shared per-agent local critic ``Q_i(o_i, a_i)``."""

    def __init__(self, backbone: LegacyVorAdjFeatureBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.output_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.head[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.head[-1].bias, -3e-3, 3e-3)

    def forward(
        self, obs: Mapping[str, torch.Tensor], action_aw: torch.Tensor
    ) -> torch.Tensor:
        if action_aw.ndim != 2 or action_aw.shape[-1] != 2:
            raise ValueError("local TD3 critic action must have shape [batch,2]")
        feature = self.backbone(obs)
        if feature.shape[0] != action_aw.shape[0]:
            raise ValueError("local observation/action batch mismatch")
        return self.head(torch.cat([feature, action_aw], dim=-1)).squeeze(-1)


def assert_no_parameter_sharing(*modules: nn.Module) -> None:
    """Fail fast if independently optimized TD3 modules alias a parameter."""

    seen: set[int] = set()
    for module in modules:
        current = {id(parameter) for parameter in module.parameters()}
        if seen.intersection(current):
            raise ValueError("TD3 actor/Q1/Q2 must not share parameters")
        seen.update(current)
