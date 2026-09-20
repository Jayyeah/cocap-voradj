"""Shared-local categorical Actor-Critic modules for the AC capability baseline.

The encoder intentionally mirrors the Final IQN local decision representation
but has independent weights for the actor and critic.  No agent id is accepted
by either network; the row index only remains replay provenance.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


EXPECTED_AW9 = tuple(
    (float(acc), float(ang))
    for acc in (-0.4, 0.0, 0.4)
    for ang in (-math.pi / 6.0, 0.0, math.pi / 6.0)
)


def canonical_aw9_grid(config: Optional[Mapping] = None) -> np.ndarray:
    """Return the canonical Final IQN AW9 grid and optionally check config."""

    if config is not None:
        pursuer = config.get("pursuer", {}) or {}
        acc = tuple(float(value) for value in pursuer.get("a", (-0.4, 0.0, 0.4)))
        ang = tuple(float(value) for value in pursuer.get("w", (-math.pi / 6.0, 0.0, math.pi / 6.0)))
        if acc != (-0.4, 0.0, 0.4) or not np.allclose(
            ang, (-math.pi / 6.0, 0.0, math.pi / 6.0), atol=1e-8
        ):
            raise AssertionError(f"AW9 runtime config changed: a={acc}, w={ang}")
    return np.asarray(EXPECTED_AW9, dtype=np.float32)


def assert_runtime_aw9(env) -> np.ndarray:
    """Check the live Robot.action_list rather than trusting YAML ordering."""

    if not getattr(env, "pursuers", None):
        raise AssertionError("AW9 runtime assertion requires a reset environment")
    actual = tuple((float(a), float(w)) for a, w in env.pursuers[0].action_list)
    if len(actual) != 9 or not np.allclose(np.asarray(actual), np.asarray(EXPECTED_AW9), atol=1e-8):
        raise AssertionError(f"live AW9 mapping mismatch: {actual}")
    return np.asarray(actual, dtype=np.float32)


@dataclass(frozen=True)
class SharedLocalACNetworkConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    action_size: int = 9
    self_feature_dim: int = 9
    max_pursuers: int = 8
    max_evaders: int = 8
    max_obstacles: int = 5
    pursuing_embed_dim: int = 8


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())


class LocalObservationEncoder(nn.Module):
    """Final-IQN-compatible local observation encoder without IQN quantiles."""

    def __init__(self, config: SharedLocalACNetworkConfig):
        super().__init__()
        self.config = config
        h = int(config.hidden_dim)
        self.encoders = nn.ModuleDict(
            {
                "self": _mlp(config.self_feature_dim, h),
                "pursuers": _mlp(7, h),
                "evaders": _mlp(7, h),
                "obstacles": _mlp(5, h),
            }
        )
        self.type_embedding = nn.Embedding(4, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=int(config.num_heads),
            dim_feedforward=4 * h,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(config.num_layers),
            enable_nested_tensor=False,
        )
        self.target_query = nn.Linear(2 * h, h)
        self.target_key = nn.Linear(h, h)
        self.target_value = nn.Linear(h, h)
        self.summary_role_embedding = nn.Embedding(3, h)
        self.summary_attention = nn.MultiheadAttention(h, int(config.num_heads), dropout=0.1, batch_first=True)
        self.summary_fusion = nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU())
        self.pursuing_embed = nn.Linear(1, int(config.pursuing_embed_dim))
        self.single_action_feature = nn.Sequential(
            nn.Linear(h + int(config.pursuing_embed_dim), h), nn.LayerNorm(h), nn.ReLU()
        )
        self.output_dim = h

    def features(self, obs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = [self.encoders["self"](obs["self"]).unsqueeze(1)]
        encoded.append(self.encoders["pursuers"](obs["pursuers"]))
        encoded.append(self.encoders["evaders"](obs["evaders"]))
        encoded.append(self.encoders["obstacles"](obs["obstacles"]))
        tokens = torch.cat(encoded, dim=1) + self.type_embedding(obs["types"].long())
        mask = obs["masks"].bool()
        trans = self.transformer(tokens, src_key_padding_mask=~mask)
        masked = trans.masked_fill(~mask.unsqueeze(-1), 0.0)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(trans.dtype)
        mean_context = masked.sum(dim=1) / count
        max_context = trans.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_context = torch.where(torch.isfinite(max_context), max_context, torch.zeros_like(max_context))
        evader_start = 1 + int(self.config.max_pursuers)
        evader_end = evader_start + int(self.config.max_evaders)
        evader_features = trans[:, evader_start:evader_end]
        evader_mask = mask[:, evader_start:evader_end] & (obs["types"][:, evader_start:evader_end] == 2)
        return {
            "self_token": trans[:, 0],
            "mean_context": mean_context,
            "max_context": max_context,
            "evader_features": evader_features,
            "evader_mask": evader_mask,
        }

    def decision_feature(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        feat = self.features(obs)
        base = torch.cat([feat["self_token"], feat["max_context"]], dim=-1)
        query = self.target_query(base).unsqueeze(1)
        keys = self.target_key(feat["evader_features"])
        values = self.target_value(feat["evader_features"])
        scores = torch.matmul(query, keys.transpose(-2, -1)) / max(math.sqrt(self.config.hidden_dim), 1.0)
        valid = feat["evader_mask"].unsqueeze(1)
        has_evader = valid.any(dim=-1).squeeze(1)
        scores = scores.masked_fill(~valid, -1e9)
        scores = torch.where(has_evader[:, None, None], scores, torch.zeros_like(scores))
        attn = torch.softmax(scores, dim=-1) * valid.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        target_context = torch.matmul(attn, values).squeeze(1)
        target_context = torch.where(has_evader[:, None], target_context, torch.zeros_like(target_context))
        batch = obs["self"].shape[0]
        roles = torch.arange(3, device=obs["self"].device).view(1, 3).expand(batch, -1)
        summary_tokens = torch.stack([feat["mean_context"], feat["max_context"], target_context], dim=1)
        summary_tokens = summary_tokens + self.summary_role_embedding(roles)
        summary_mask = torch.stack(
            [torch.zeros_like(has_evader, dtype=torch.bool), torch.zeros_like(has_evader, dtype=torch.bool), ~has_evader],
            dim=1,
        )
        summary_context, _ = self.summary_attention(
            feat["self_token"].unsqueeze(1),
            summary_tokens,
            summary_tokens,
            key_padding_mask=summary_mask,
            need_weights=False,
        )
        fused = self.summary_fusion(torch.cat([feat["self_token"], summary_context.squeeze(1)], dim=-1))
        pursuing_scalar = obs["self"][:, -1:] if obs["self"].shape[-1] >= 8 else torch.zeros(batch, 1, device=obs["self"].device)
        pursuing_embed = self.pursuing_embed(pursuing_scalar)
        return self.single_action_feature(torch.cat([fused, pursuing_embed], dim=-1))


class SharedLocalActor(nn.Module):
    """One categorical AW9 actor shared by every active pursuer row."""

    def __init__(self, config: SharedLocalACNetworkConfig):
        super().__init__()
        self.config = config
        self.encoder = LocalObservationEncoder(config)
        h = int(config.hidden_dim)
        self.policy = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU())
        self.logits_head = nn.Linear(h, int(config.action_size))
        self.register_buffer("action_grid", torch.as_tensor(canonical_aw9_grid(), dtype=torch.float32))

    def logits(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.logits_head(self.policy(self.encoder.decision_feature(obs)))

    def distribution(self, obs: Mapping[str, torch.Tensor]) -> Categorical:
        return Categorical(logits=self.logits(obs))

    def probabilities(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.softmax(self.logits(obs), dim=-1)

    def behavior_probabilities(self, obs: Mapping[str, torch.Tensor], epsilon: float) -> torch.Tensor:
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon must be in [0,1]")
        pi = self.probabilities(obs)
        return (1.0 - float(epsilon)) * pi + float(epsilon) / float(self.config.action_size)

    @torch.no_grad()
    def sample_behavior(self, obs: Mapping[str, torch.Tensor], epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = self.probabilities(obs)
        mu = (1.0 - float(epsilon)) * pi + float(epsilon) / float(self.config.action_size)
        action = torch.multinomial(mu, num_samples=1).squeeze(-1)
        return action, pi.gather(1, action[:, None]).squeeze(1), mu.gather(1, action[:, None]).squeeze(1)

    @torch.no_grad()
    def deterministic_action(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.logits(obs).argmax(dim=-1)

    def save_config(self) -> dict:
        return asdict(self.config)


class SharedLocalQ(nn.Module):
    """One local Q(o,a0..a8) shared by every active pursuer row."""

    def __init__(self, config: SharedLocalACNetworkConfig):
        super().__init__()
        self.config = config
        self.encoder = LocalObservationEncoder(config)
        h = int(config.hidden_dim)
        self.q_head = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, int(config.action_size)))

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.q_head(self.encoder.decision_feature(obs))

    def save_config(self) -> dict:
        return asdict(self.config)


def hard_copy(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    target.requires_grad_(False)
    target.eval()
    return target


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))
