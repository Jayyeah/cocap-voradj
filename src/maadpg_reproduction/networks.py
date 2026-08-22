"""Independent local actors and genuinely joint centralized critics."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def project_action_tensor(actions: Tensor, epsilon: float = 1.0e-12) -> Tensor:
    """Differentiably project the final actor box output onto the L2 unit disk."""

    norms = torch.linalg.vector_norm(actions, dim=-1, keepdim=True)
    scale = torch.clamp(norms, min=1.0)
    return actions / torch.clamp(scale, min=epsilon)


def _init_hidden(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5.0), nonlinearity="relu")
    nn.init.zeros_(layer.bias)


def _init_output(layer: nn.Linear) -> None:
    nn.init.uniform_(layer.weight, -3.0e-3, 3.0e-3)
    nn.init.uniform_(layer.bias, -3.0e-3, 3.0e-3)


class Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int = 26,
        hidden_dims: tuple[int, int] = (128, 128),
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(observation_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, action_dim),
            nn.Tanh(),
        )
        _init_hidden(self.net[0])
        _init_hidden(self.net[2])
        _init_output(self.net[4])

    def forward(self, local_observation: Tensor) -> Tensor:
        return project_action_tensor(self.net(local_observation))


class JointCritic(nn.Module):
    def __init__(
        self,
        agent_count: int = 3,
        observation_dim: int = 26,
        action_dim: int = 2,
        hidden_dims: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.agent_count = agent_count
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        input_dim = agent_count * (observation_dim + action_dim)
        first, second = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, 1),
        )
        _init_hidden(self.net[0])
        _init_hidden(self.net[2])
        _init_output(self.net[4])

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("critic expects batched joint observations/actions")
        if observations.shape[1:] != (self.agent_count, self.observation_dim):
            raise ValueError(f"invalid observation shape {tuple(observations.shape)}")
        if actions.shape[1:] != (self.agent_count, self.action_dim):
            raise ValueError(f"invalid action shape {tuple(actions.shape)}")
        joint = torch.cat(
            [observations.flatten(start_dim=1), actions.flatten(start_dim=1)], dim=-1
        )
        return self.net(joint).squeeze(-1)
