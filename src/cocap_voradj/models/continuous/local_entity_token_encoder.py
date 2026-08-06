"""Local entity-token encoder for the isolated continuous-action namespace."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LocalEntityTokenEncoderConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    self_feature_dim: int = 9
    max_pursuers: int = 8
    max_evaders: int = 8
    max_obstacles: int = 5
    dropout: float = 0.0


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())


class LocalEntityTokenEncoder(nn.Module):
    """Encode the same local entity-token contract as the legacy IQN front end.

    The module deliberately keeps the legacy encoder key names so a checkpoint
    can be copied with an explicit, auditable key mapping. It has no critic or
    global-state input and is safe to use for actor deployment.
    """

    def __init__(self, config: LocalEntityTokenEncoderConfig | None = None):
        super().__init__()
        self.config = config or LocalEntityTokenEncoderConfig()
        h = int(self.config.hidden_dim)
        self.encoders = nn.ModuleDict(
            {
                "self": _mlp(self.config.self_feature_dim, h),
                "pursuers": _mlp(7, h),
                "evaders": _mlp(7, h),
                "obstacles": _mlp(5, h),
            }
        )
        self.type_embedding = nn.Embedding(4, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=self.config.num_heads,
            dim_feedforward=4 * h,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False,
        )

    @property
    def token_count(self) -> int:
        return 1 + self.config.max_pursuers + self.config.max_evaders + self.config.max_obstacles

    def _validate_observation(self, obs: Mapping[str, torch.Tensor]) -> None:
        required = {"self", "pursuers", "evaders", "obstacles", "masks", "types"}
        missing = required.difference(obs)
        if missing:
            raise ValueError(f"continuous encoder observation missing keys: {sorted(missing)}")
        batch = obs["self"].shape[0]
        if obs["self"].dim() != 2 or obs["self"].shape[1] != self.config.self_feature_dim:
            raise ValueError("self observation has incompatible shape")
        expected = {
            "pursuers": (batch, self.config.max_pursuers, 7),
            "evaders": (batch, self.config.max_evaders, 7),
            "obstacles": (batch, self.config.max_obstacles, 5),
            "masks": (batch, self.token_count),
            "types": (batch, self.token_count),
        }
        for key, shape in expected.items():
            if tuple(obs[key].shape) != shape:
                raise ValueError(f"{key} shape mismatch: expected {shape}, got {tuple(obs[key].shape)}")

    def forward(self, obs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._validate_observation(obs)
        encoded = [self.encoders["self"](obs["self"]).unsqueeze(1)]
        encoded.extend(
            [
                self.encoders["pursuers"](obs["pursuers"]),
                self.encoders["evaders"](obs["evaders"]),
                self.encoders["obstacles"](obs["obstacles"]),
            ]
        )
        tokens = torch.cat(encoded, dim=1) + self.type_embedding(obs["types"].long())
        mask = obs["masks"].bool()
        transformed = self.transformer(tokens, src_key_padding_mask=~mask)
        masked = transformed.masked_fill(~mask.unsqueeze(-1), 0.0)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(transformed.dtype)
        mean_context = masked.sum(dim=1) / count
        max_context = transformed.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_context = torch.where(torch.isfinite(max_context), max_context, torch.zeros_like(max_context))
        start = 1 + self.config.max_pursuers
        end = start + self.config.max_evaders
        evader_mask = mask[:, start:end] & (obs["types"][:, start:end] == 2)
        return {
            "tokens": transformed,
            "mask": mask,
            "self_token": transformed[:, 0],
            "mean_context": mean_context,
            "max_context": max_context,
            "evader_features": transformed[:, start:end],
            "evader_mask": evader_mask,
        }

    @staticmethod
    def map_legacy_iqn_encoder_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return only front-end keys shared with ``CoCapIQN``.

        The names are intentionally unchanged; the returned mapping is the
        auditable boundary between the old checkpoint and this new namespace.
        """
        prefixes = ("encoders.", "type_embedding.", "transformer.")
        return {key: value for key, value in state_dict.items() if key.startswith(prefixes)}

    def load_legacy_iqn_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        mapped = self.map_legacy_iqn_encoder_keys(state_dict)
        if not mapped:
            raise ValueError("legacy state dict contains no IQN encoder keys")
        return self.load_state_dict(mapped, strict=strict)
