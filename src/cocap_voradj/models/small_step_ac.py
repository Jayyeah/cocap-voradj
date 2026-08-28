"""Policy/value modules for the isolated small-step Actor-Critic migration."""
from __future__ import annotations

import copy
from typing import Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from cocap_voradj.models.continuous.local_entity_token_encoder import LocalEntityTokenEncoder
from cocap_voradj.models.iqn import CoCapIQN


def _context(encoder: LocalEntityTokenEncoder, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    features = encoder(obs)
    return torch.cat([features["self_token"], features["mean_context"]], dim=-1)


class CategoricalGridActor(nn.Module):
    """Shared decentralized categorical actor with an auditable physical grid."""

    def __init__(self, encoder: LocalEntityTokenEncoder, grid: np.ndarray):
        super().__init__()
        self.encoder = encoder
        h = int(encoder.config.hidden_dim)
        self.policy = nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU())
        self.logits_head = nn.Linear(h, 9)
        value = np.asarray(grid, dtype=np.float32)
        if value.shape != (9, 2) or not np.all(np.isfinite(value)):
            raise ValueError("categorical action grid must have shape [9,2]")
        self.register_buffer("action_grid", torch.as_tensor(value))

    def distribution(self, obs: Mapping[str, torch.Tensor]) -> Categorical:
        return Categorical(logits=self.logits_head(self.policy(_context(self.encoder, obs))))

    def sample(self, obs: Mapping[str, torch.Tensor], deterministic: bool = False):
        distribution = self.distribution(obs)
        index = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return self.action_grid[index], distribution.log_prob(index), index

    def evaluate_indices(self, obs: Mapping[str, torch.Tensor], index: torch.Tensor):
        distribution = self.distribution(obs)
        return distribution.log_prob(index.long()), distribution.entropy()


class DeterministicAWActor(nn.Module):
    """Shared local TD3 actor producing bounded continuous ``(a,w)``."""

    def __init__(self, encoder: LocalEntityTokenEncoder, a_max: float, w_max: float):
        super().__init__()
        self.encoder = encoder
        h = int(encoder.config.hidden_dim)
        self.policy = nn.Sequential(
            nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, 2)
        )
        self.register_buffer("scale", torch.tensor([float(a_max), float(w_max)]))

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.tanh(self.policy(_context(self.encoder, obs))) * self.scale

    def sample(self, obs: Mapping[str, torch.Tensor], deterministic: bool = False):
        del deterministic
        action = self.forward(obs)
        zeros = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
        return action, zeros, action


class IQNGridPolicy(nn.Module):
    """Deployment wrapper retaining IQN while mapping indices onto VXY9."""

    def __init__(self, model: CoCapIQN, grid: np.ndarray, quantiles: int = 32, epsilon: float = 0.05):
        super().__init__()
        self.model = model
        self.quantiles = int(quantiles)
        self.epsilon = float(epsilon)
        self.register_buffer("action_grid", torch.as_tensor(np.asarray(grid), dtype=torch.float32))

    def sample(self, obs: Mapping[str, torch.Tensor], deterministic: bool = False):
        q = self.model(dict(obs), num_tau=self.quantiles, mode="voradj")["q_values"].mean(dim=1)
        if deterministic:
            index = q.argmax(dim=-1)
        else:
            index = q.argmax(dim=-1)
            explore = torch.rand(index.shape, device=index.device) < self.epsilon
            index = torch.where(explore, torch.randint(0, 9, index.shape, device=index.device), index)
        zeros = torch.zeros(index.shape[0], device=index.device, dtype=q.dtype)
        return self.action_grid[index], zeros, index


def _embed(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())


class CentralValueNetwork(nn.Module):
    """Training-only action-free centralized value function, one V per pursuer."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        self_feature_dim: int = 9,
        max_agents: int = 12,
        max_evaders: int = 8,
        max_obstacles: int = 5,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_agents = int(max_agents)
        self.max_evaders = int(max_evaders)
        self.max_obstacles = int(max_obstacles)
        self.self_feature_dim = int(self_feature_dim)
        h = self.hidden_dim
        self.self_encoder = _embed(self.self_feature_dim, h)
        self.pursuer_encoder = _embed(7, h)
        self.evader_encoder = _embed(7, h)
        self.obstacle_encoder = _embed(5, h)
        self.type_embedding = nn.Embedding(4, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=int(num_heads), dim_feedforward=4 * h,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.attention = nn.TransformerEncoder(layer, int(num_layers), enable_nested_tensor=False)
        self.value_head = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, global_obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        active = global_obs["active_mask"].bool()
        batch, agents = active.shape
        if agents != self.max_agents:
            raise ValueError("central value active mask has wrong agent dimension")
        h = self.hidden_dim
        focal = self.self_encoder(global_obs["self"])
        pursuers = self.pursuer_encoder(global_obs["pursuers"].reshape(batch * agents, agents, 7))
        evaders = global_obs["evaders"][:, None].expand(batch, agents, -1, -1)
        evaders = self.evader_encoder(evaders.reshape(batch * agents, self.max_evaders, 7))
        obstacles = global_obs["obstacles"][:, None].expand(batch, agents, -1, -1)
        obstacles = self.obstacle_encoder(obstacles.reshape(batch * agents, self.max_obstacles, 5))
        tokens = torch.cat([focal.reshape(batch * agents, 1, h), pursuers, evaders, obstacles], dim=1)
        type_ids = torch.cat([
            torch.zeros(1, device=tokens.device, dtype=torch.long),
            torch.ones(agents, device=tokens.device, dtype=torch.long),
            torch.full((self.max_evaders,), 2, device=tokens.device, dtype=torch.long),
            torch.full((self.max_obstacles,), 3, device=tokens.device, dtype=torch.long),
        ]).view(1, -1).expand(batch * agents, -1)
        tokens = tokens + self.type_embedding(type_ids)
        pursuer_mask = global_obs["pursuer_mask"].reshape(batch * agents, agents).bool()
        evader_mask = global_obs["evader_mask"][:, None].expand(batch, agents, -1).reshape(batch * agents, -1).bool()
        obstacle_mask = global_obs["obstacle_mask"][:, None].expand(batch, agents, -1).reshape(batch * agents, -1).bool()
        focal_mask = torch.ones(batch * agents, 1, device=tokens.device, dtype=torch.bool)
        mask = torch.cat([focal_mask, pursuer_mask, evader_mask, obstacle_mask], dim=1)
        encoded = self.attention(tokens, src_key_padding_mask=~mask)
        value = self.value_head(encoded[:, 0]).reshape(batch, agents)
        return value * active.to(value.dtype)


def hard_copy(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    target.requires_grad_(False)
    return target
