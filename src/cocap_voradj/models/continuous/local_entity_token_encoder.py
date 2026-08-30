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


@dataclass(frozen=True)
class LegacyVorAdjFeatureBackboneConfig:
    """Shape and dropout contract for the legacy IQN decision feature.

    CoCapIQN historically hard-coded 0.1 dropout in both its entity
    transformer and summary attention.  Keeping that default here is
    intentional: loading an old IQN checkpoint must reproduce its feature in
    evaluation mode.  The actor head is owned by the caller and is therefore
    not part of this config or state dict.
    """

    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    self_feature_dim: int = 9
    max_pursuers: int = 8
    max_evaders: int = 8
    max_obstacles: int = 5
    pursuing_embed_dim: int = 8
    dropout: float = 0.1


class LegacyVorAdjFeatureBackbone(nn.Module):
    """Reusable CoCapIQN.voradj_single_feature representation.

    The old IQN actor/Q trunk is deliberately kept intact in
    cocap_voradj.models.iqn.  This standalone module mirrors only the modules
    needed to compute its decision feature and exposes an explicit state-dict
    boundary.  An old IQN checkpoint can therefore be copied to an
    Actor-Critic actor without loading IQN quantile heads or changing IQN's own
    checkpoint-loading path.

    In eval mode, forward(obs) is numerically identical to
    CoCapIQN.voradj_single_feature(obs, CoCapIQN.features(obs)) after the
    shared legacy keys have been loaded.  During training it intentionally
    retains the legacy dropout probabilities, matching the original trunk.
    """

    _LEGACY_DECISION_PREFIXES = (
        "encoders.",
        "type_embedding.",
        "transformer.",
        "target_query.",
        "target_key.",
        "target_value.",
        "summary_role_embedding.",
        "summary_attention.",
        "summary_fusion.",
        "pursuing_embed.",
        "single_action_feature.",
    )

    def __init__(self, config: LegacyVorAdjFeatureBackboneConfig | None = None):
        super().__init__()
        self.config = config or LegacyVorAdjFeatureBackboneConfig()
        h = int(self.config.hidden_dim)
        self.decision_feature_dim = h
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
            nhead=int(self.config.num_heads),
            dim_feedforward=4 * h,
            dropout=float(self.config.dropout),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(self.config.num_layers),
            enable_nested_tensor=False,
        )
        self.target_query = nn.Linear(2 * h, h)
        self.target_key = nn.Linear(h, h)
        self.target_value = nn.Linear(h, h)
        self.summary_role_embedding = nn.Embedding(3, h)
        self.summary_attention = nn.MultiheadAttention(
            h, int(self.config.num_heads), dropout=float(self.config.dropout), batch_first=True
        )
        self.summary_fusion = nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU())
        self.pursuing_embed = nn.Linear(1, int(self.config.pursuing_embed_dim))
        self.single_action_feature = nn.Sequential(
            nn.Linear(h + int(self.config.pursuing_embed_dim), h), nn.LayerNorm(h), nn.ReLU()
        )

    @property
    def output_dim(self) -> int:
        """Dimension consumed by a policy head."""

        return self.decision_feature_dim

    def features(self, obs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the exact legacy entity summary used by the IQN trunk."""

        encoded = [self.encoders["self"](obs["self"]).unsqueeze(1)]
        encoded.append(self.encoders["pursuers"](obs["pursuers"]))
        encoded.append(self.encoders["evaders"](obs["evaders"]))
        encoded.append(self.encoders["obstacles"](obs["obstacles"]))
        tokens = torch.cat(encoded, dim=1) + self.type_embedding(obs["types"].long())
        mask = obs["masks"].bool()
        transformed = self.transformer(tokens, src_key_padding_mask=~mask)
        masked = transformed.masked_fill(~mask.unsqueeze(-1), 0.0)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(transformed.dtype)
        mean_context = masked.sum(dim=1) / count
        max_context = transformed.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_context = torch.where(torch.isfinite(max_context), max_context, torch.zeros_like(max_context))
        start = 1 + int(self.config.max_pursuers)
        end = start + int(self.config.max_evaders)
        evader_mask = mask[:, start:end] & (obs["types"][:, start:end] == 2)
        return {
            "self_token": transformed[:, 0],
            "mean_context": mean_context,
            "max_context": max_context,
            "evader_features": transformed[:, start:end],
            "evader_mask": evader_mask,
        }

    def voradj_single_feature(
        self, obs: Mapping[str, torch.Tensor], feat: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply the exact legacy target-summary and pursuing-role path."""

        h = int(self.config.hidden_dim)
        base = torch.cat([feat["self_token"], feat["max_context"]], dim=-1)
        query = self.target_query(base).unsqueeze(1)
        keys = self.target_key(feat["evader_features"])
        values = self.target_value(feat["evader_features"])
        scores = torch.matmul(query, keys.transpose(-2, -1)) / max(h ** 0.5, 1.0)
        valid = feat["evader_mask"].unsqueeze(1)
        has_evader = valid.any(dim=-1).squeeze(1)
        scores = scores.masked_fill(~valid, -1e9)
        scores = torch.where(has_evader[:, None, None], scores, torch.zeros_like(scores))
        attn = torch.softmax(scores, dim=-1) * valid.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        target_context = torch.matmul(attn, values).squeeze(1)
        target_context = torch.where(has_evader[:, None], target_context, torch.zeros_like(target_context))

        roles = torch.arange(3, device=obs["self"].device).view(1, 3).expand(obs["self"].shape[0], -1)
        summary_tokens = torch.stack([feat["mean_context"], feat["max_context"], target_context], dim=1)
        summary_tokens = summary_tokens + self.summary_role_embedding(roles)
        summary_mask = torch.stack(
            [
                torch.zeros_like(has_evader, dtype=torch.bool),
                torch.zeros_like(has_evader, dtype=torch.bool),
                ~has_evader,
            ],
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
        if obs["self"].shape[-1] >= 8:
            pursuing_scalar = obs["self"][:, -1:]
        else:
            pursuing_scalar = torch.zeros(
                obs["self"].shape[0], 1, device=obs["self"].device, dtype=obs["self"].dtype
            )
        pursuing_embed = self.pursuing_embed(pursuing_scalar)
        return self.single_action_feature(torch.cat([fused, pursuing_embed], dim=-1))

    def decision_feature(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Named alias used by Actor-Critic policy-head adapters."""

        return self.forward(obs)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.voradj_single_feature(obs, self.features(obs))

    @classmethod
    def from_iqn_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        config: LegacyVorAdjFeatureBackboneConfig | None = None,
    ) -> "LegacyVorAdjFeatureBackbone":
        """Construct and strictly load only the reusable IQN decision keys."""

        backbone = cls(config)
        backbone.load_legacy_iqn_state_dict(state_dict, strict=True)
        return backbone

    @classmethod
    def map_legacy_iqn_decision_keys(
        cls, state_dict: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(cls._LEGACY_DECISION_PREFIXES)
        }

    def load_legacy_iqn_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        mapped = self.map_legacy_iqn_decision_keys(state_dict)
        if not mapped:
            raise ValueError("legacy state dict contains no IQN decision-backbone keys")
        return self.load_state_dict(mapped, strict=strict)


def policy_context_from_features(
    features: Mapping[str, torch.Tensor] | torch.Tensor,
    context_pooling: str = "mean",
) -> torch.Tensor:
    """Adapt either the old pooled encoder or the legacy decision backbone.

    Existing actors continue to consume self + mean (or mean_max).
    A LegacyVorAdjFeatureBackbone returns its already-fused decision tensor,
    which must pass directly to the new policy head.
    """

    if torch.is_tensor(features):
        return features
    if "decision_feature" in features:
        return features["decision_feature"]
    context = [features["self_token"], features["mean_context"]]
    if str(context_pooling).strip().lower() == "mean_max":
        context.append(features["max_context"])
    return torch.cat(context, dim=-1)


def policy_context(
    encoder: nn.Module,
    obs: Mapping[str, torch.Tensor],
    context_pooling: str = "mean",
) -> torch.Tensor:
    """Run an encoder and produce a policy-head input of the right contract."""

    return policy_context_from_features(encoder(obs), context_pooling=context_pooling)
