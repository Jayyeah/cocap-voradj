"""Matched critic families for the frozen-actor identifiability audit.

All Q networks consume the same physical action contract and produce one
scalar per focal agent. NeighborQ uses only local observations from the focal
agent and its observed VorAdj neighbors; it never receives central state. The
same local backbone instance is shared across focal and neighbor rows, making
the neighbor set permutation invariant without a second observation encoder.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
    LegacyVorAdjFeatureBackboneConfig,
)


class LocalQ(nn.Module):
    """TD3/SAC-style scalar local Q."""

    def __init__(self, config: LegacyVorAdjFeatureBackboneConfig):
        super().__init__()
        self.backbone = LegacyVorAdjFeatureBackbone(config)
        h = int(config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(h + 2, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1),
        )

    def forward(self, obs: Mapping[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.backbone(obs), action], dim=-1)).squeeze(-1)


class NeighborQ(nn.Module):
    """Permutation-invariant local-neighbor Q without global state."""

    def __init__(self, config: LegacyVorAdjFeatureBackboneConfig, max_neighbors: int = 8):
        super().__init__()
        self.backbone = LegacyVorAdjFeatureBackbone(config)
        self.max_neighbors = int(max_neighbors)
        h = int(config.hidden_dim)
        self.neighbor_fusion = nn.Sequential(
            nn.Linear(h + 2, h), nn.LayerNorm(h), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * h + 2, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1),
        )

    def forward(
        self,
        obs: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        neighbor_obs: Mapping[str, torch.Tensor],
        neighbor_action: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(action.shape[0])
        mask = neighbor_mask.bool()
        if tuple(mask.shape) != (batch, self.max_neighbors):
            raise ValueError("neighbor mask shape mismatch")
        pooled = torch.zeros(batch, self.max_neighbors, self.backbone.output_dim,
                             device=action.device, dtype=action.dtype)
        flat_mask = mask.reshape(-1)
        if bool(flat_mask.any()):
            flat_obs = {
                key: value.reshape(batch * self.max_neighbors, *value.shape[2:])[flat_mask]
                for key, value in neighbor_obs.items()
            }
            flat_action = neighbor_action.reshape(batch * self.max_neighbors, 2)[flat_mask]
            pooled.reshape(batch * self.max_neighbors, -1)[flat_mask] = self.neighbor_fusion(
                torch.cat([self.backbone(flat_obs), flat_action], dim=-1)
            )
        valid = mask.unsqueeze(-1)
        count = valid.sum(1).clamp_min(1).to(action.dtype)
        mean = (pooled * valid).sum(1) / count
        maximum = pooled.masked_fill(~valid, -1e9).max(1).values
        maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        focal = self.backbone(obs)
        return self.head(torch.cat([focal, action, mean, maximum], dim=-1)).squeeze(-1)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
