"""Centralized entity-attention critics for the isolated PSA-MASAC path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CentralCriticConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    self_feature_dim: int = 9
    max_agents: int = 12
    max_evaders: int = 8
    max_obstacles: int = 5
    dropout: float = 0.0


def _embed(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())


class CentralAttentionCritic(nn.Module):
    """One independent critic producing Q for every active focal pursuer.

    The central input is a padded global entity set plus each focal agent's
    body-frame acceleration action. No phase/event/origin labels are accepted.
    """

    def __init__(self, config: CentralCriticConfig | None = None):
        super().__init__()
        self.config = config or CentralCriticConfig()
        h = self.config.hidden_dim
        self.self_encoder = _embed(self.config.self_feature_dim, h)
        self.pursuer_encoder = _embed(7, h)
        self.evader_encoder = _embed(7, h)
        self.obstacle_encoder = _embed(5, h)
        self.action_encoder = _embed(2, h)
        self.type_embedding = nn.Embedding(4, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=self.config.num_heads,
            dim_feedforward=4 * h,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(layer, num_layers=self.config.num_layers, enable_nested_tensor=False)
        self.q_head = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, 1))

    @property
    def token_count(self) -> int:
        return 1 + self.config.max_agents + self.config.max_evaders + self.config.max_obstacles

    def _validate(self, global_obs: Mapping[str, torch.Tensor], joint_actions: torch.Tensor) -> Tuple[int, int]:
        required = {"self", "pursuers", "evaders", "obstacles", "pursuer_mask", "evader_mask", "obstacle_mask", "active_mask"}
        missing = required.difference(global_obs)
        if missing:
            raise ValueError(f"central critic missing global keys: {sorted(missing)}")
        if joint_actions.dim() != 3 or tuple(joint_actions.shape[-2:]) != (self.config.max_agents, 2):
            raise ValueError("joint_actions must have shape [B,max_agents,2]")
        batch = int(joint_actions.shape[0])
        expected = {
            "self": (batch, self.config.max_agents, self.config.self_feature_dim),
            "pursuers": (batch, self.config.max_agents, self.config.max_agents, 7),
            "evaders": (batch, self.config.max_evaders, 7),
            "obstacles": (batch, self.config.max_obstacles, 5),
            "pursuer_mask": (batch, self.config.max_agents, self.config.max_agents),
            "evader_mask": (batch, self.config.max_evaders),
            "obstacle_mask": (batch, self.config.max_obstacles),
            "active_mask": (batch, self.config.max_agents),
        }
        for key, shape in expected.items():
            if tuple(global_obs[key].shape) != shape:
                raise ValueError(f"central {key} shape mismatch: expected {shape}, got {tuple(global_obs[key].shape)}")
        if not bool(global_obs["active_mask"].bool().any()):
            raise ValueError("central critic requires at least one active focal pursuer")
        return batch, self.config.max_agents

    def forward(self, global_obs: Mapping[str, torch.Tensor], joint_actions: torch.Tensor) -> torch.Tensor:
        batch, agents = self._validate(global_obs, joint_actions)
        h = self.config.hidden_dim
        focal_self = global_obs["self"]
        # Expand shared entity sets once per focal agent; no focal-specific
        # learned feature is reused across the two critic instances.
        pursuers = global_obs["pursuers"]
        evaders = global_obs["evaders"][:, None].expand(batch, agents, -1, -1)
        obstacles = global_obs["obstacles"][:, None].expand(batch, agents, -1, -1)
        focal = self.self_encoder(focal_self) + self.action_encoder(joint_actions)
        pursuer_tokens = self.pursuer_encoder(pursuers.reshape(batch * agents, agents, 7))
        # Bind every acceleration action to its corresponding pursuer
        # entity token. The focal query also receives its own command below;
        # this token-level binding is what lets Q_i respond to teammates'
        # commands while the actor can stop-gradient those other branches.
        joint_action_tokens = self.action_encoder(
            joint_actions[:, None].expand(batch, agents, agents, 2).reshape(batch * agents, agents, 2)
        )
        pursuer_tokens = pursuer_tokens + joint_action_tokens
        evader_tokens = self.evader_encoder(evaders.reshape(batch * agents, self.config.max_evaders, 7))
        obstacle_tokens = self.obstacle_encoder(obstacles.reshape(batch * agents, self.config.max_obstacles, 5))
        tokens = torch.cat([focal.reshape(batch * agents, 1, h), pursuer_tokens, evader_tokens, obstacle_tokens], dim=1)
        type_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=tokens.device),
                torch.ones(self.config.max_agents, dtype=torch.long, device=tokens.device),
                torch.full((self.config.max_evaders,), 2, dtype=torch.long, device=tokens.device),
                torch.full((self.config.max_obstacles,), 3, dtype=torch.long, device=tokens.device),
            ]
        ).view(1, -1).expand(batch * agents, -1)
        tokens = tokens + self.type_embedding(type_ids)
        pursuer_mask = global_obs["pursuer_mask"].reshape(batch * agents, agents)
        evader_mask = global_obs["evader_mask"][:, None].expand(batch, agents, -1).reshape(batch * agents, self.config.max_evaders)
        obstacle_mask = global_obs["obstacle_mask"][:, None].expand(batch, agents, -1).reshape(batch * agents, self.config.max_obstacles)
        # Keep the focal token visible even for padded/inactive rows. If an
        # inactive row were entirely masked, Transformer attention can produce
        # NaNs; the final active-mask multiplication still makes its Q zero.
        focal_mask = torch.ones_like(global_obs["active_mask"], dtype=torch.bool).reshape(batch * agents, 1)
        mask = torch.cat([focal_mask, pursuer_mask, evader_mask, obstacle_mask], dim=1).bool()
        encoded = self.attention(tokens, src_key_padding_mask=~mask)
        q = self.q_head(encoded[:, 0]).reshape(batch, agents)
        return q * global_obs["active_mask"].to(dtype=q.dtype)


class CentralTwinCritics(nn.Module):
    """Two fully independent centralized critics; no encoder parameter sharing."""

    def __init__(self, config: CentralCriticConfig | None = None):
        super().__init__()
        self.critic1 = CentralAttentionCritic(config)
        self.critic2 = CentralAttentionCritic(config)

    def forward(self, global_obs: Mapping[str, torch.Tensor], joint_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.critic1(global_obs, joint_actions), self.critic2(global_obs, joint_actions)
