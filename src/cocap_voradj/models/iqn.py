
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoCapNetConfig:
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    action_size: int = 9
    self_feature_dim: int = 7
    max_pursuers: int = 12
    max_evaders: int = 8
    max_obstacles: int = 5
    num_quantiles: int = 16
    num_cosine_features: int = 32
    gate_hidden_dim: int = 64
    architecture: str = "dual_head"
    pursuing_embed_dim: int = 8


def mlp_encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())


class CoCapIQN(nn.Module):
    def __init__(self, config: CoCapNetConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.encoders = nn.ModuleDict({
            "self": mlp_encoder(config.self_feature_dim, h),
            "pursuers": mlp_encoder(7, h),
            "evaders": mlp_encoder(7, h),
            "obstacles": mlp_encoder(5, h),
        })
        self.type_embedding = nn.Embedding(4, h)
        layer = nn.TransformerEncoderLayer(d_model=h, nhead=config.num_heads, dim_feedforward=4 * h, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.cos_embedding = nn.Linear(config.num_cosine_features, h)
        self.register_buffer("pis", torch.arange(config.num_cosine_features).float().view(1, 1, -1) * torch.pi)
        self.coverage_fusion = nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU())
        self.encirclement_fusion = nn.Sequential(nn.Linear(3 * h, h), nn.LayerNorm(h), nn.ReLU())
        self.target_query = nn.Linear(2 * h, h)
        self.target_key = nn.Linear(h, h)
        self.target_value = nn.Linear(h, h)
        self.coverage_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, config.action_size))
        self.encirclement_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, config.action_size))
        self.gate_head = nn.Sequential(nn.Linear(h, config.gate_hidden_dim), nn.ReLU(), nn.Linear(config.gate_hidden_dim, 1))
        self.summary_role_embedding = nn.Embedding(3, h)
        self.summary_attention = nn.MultiheadAttention(h, config.num_heads, dropout=0.1, batch_first=True)
        self.summary_fusion = nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU())
        self.pursuing_embed = nn.Linear(1, config.pursuing_embed_dim)
        self.single_action_feature = nn.Sequential(nn.Linear(h + config.pursuing_embed_dim, h), nn.LayerNorm(h), nn.ReLU())
        self.single_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.LayerNorm(h), nn.Linear(h, config.action_size))

    def features(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        evader_slots = obs["types"] == 2
        evader_mask_full = evader_slots & mask
        evader_features = trans[:, 1 + self.config.max_pursuers: 1 + self.config.max_pursuers + self.config.max_evaders]
        evader_mask = evader_mask_full[:, 1 + self.config.max_pursuers: 1 + self.config.max_pursuers + self.config.max_evaders]
        return {"self_token": trans[:, 0], "mean_context": mean_context, "max_context": max_context, "evader_features": evader_features, "evader_mask": evader_mask}

    def quantile_embedding(self, batch: int, num_tau: int, device, tau: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if tau is None:
            tau = torch.rand(batch, num_tau, 1, device=device)
        else:
            tau = tau.to(device=device, dtype=torch.float32)
            if tau.dim() == 1:
                tau = tau.view(1, -1, 1).expand(batch, -1, -1)
            elif tau.dim() == 2:
                tau = tau.unsqueeze(-1)
                if tau.shape[0] == 1 and batch > 1:
                    tau = tau.expand(batch, -1, -1)
            elif tau.dim() != 3:
                raise ValueError("tau must have shape (N,), (B, N), or (B, N, 1)")
            if tau.shape[0] != batch:
                raise ValueError(f"tau batch mismatch: expected {batch}, got {tau.shape[0]}")
            num_tau = int(tau.shape[1])
        cos = torch.cos(tau * self.pis.to(device))
        return F.relu(self.cos_embedding(cos)), tau

    def encirclement_feature(self, feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        base = torch.cat([feat["self_token"], feat["max_context"]], dim=-1)
        query = self.target_query(base).unsqueeze(1)
        keys = self.target_key(feat["evader_features"])
        values = self.target_value(feat["evader_features"])
        scores = torch.matmul(query, keys.transpose(-2, -1)) / max(self.config.hidden_dim ** 0.5, 1.0)
        valid = feat["evader_mask"].unsqueeze(1)
        has = valid.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~valid, -1e9)
        scores = torch.where(has, scores, torch.zeros_like(scores))
        attn = torch.softmax(scores, dim=-1) * valid.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        target = torch.matmul(attn, values).squeeze(1)
        return self.encirclement_fusion(torch.cat([feat["self_token"], feat["max_context"], target], dim=-1))

    def voradj_single_feature(self, obs: Dict[str, torch.Tensor], feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        base = torch.cat([feat["self_token"], feat["max_context"]], dim=-1)
        query = self.target_query(base).unsqueeze(1)
        keys = self.target_key(feat["evader_features"])
        values = self.target_value(feat["evader_features"])
        scores = torch.matmul(query, keys.transpose(-2, -1)) / max(self.config.hidden_dim ** 0.5, 1.0)
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
        summary_mask = torch.stack([
            torch.zeros_like(has_evader, dtype=torch.bool),
            torch.zeros_like(has_evader, dtype=torch.bool),
            ~has_evader,
        ], dim=1)
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
            pursuing_scalar = torch.zeros(obs["self"].shape[0], 1, device=obs["self"].device, dtype=obs["self"].dtype)
        pursuing_embed = self.pursuing_embed(pursuing_scalar)
        return self.single_action_feature(torch.cat([fused, pursuing_embed], dim=-1))

    def forward(self, obs: Dict[str, torch.Tensor], num_tau: int = 8, mode: str = "mixed", tau: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat = self.features(obs)
        qemb, tau = self.quantile_embedding(obs["self"].shape[0], num_tau, obs["self"].device, tau=tau)
        num_tau = int(tau.shape[1])
        if self.config.architecture == "voradj_single_head":
            single_feature = self.voradj_single_feature(obs, feat)
            q = self.single_head((single_feature.unsqueeze(1) * qemb).reshape(-1, self.config.hidden_dim)).reshape(obs["self"].shape[0], num_tau, -1)
            gate = torch.zeros(obs["self"].shape[0], device=obs["self"].device, dtype=q.dtype)
            return {"q_values": q, "q_coverage": q, "q_encirclement": q, "gate": gate, "tau": tau, "taus": tau}
        cov_feature = self.coverage_fusion(torch.cat([feat["self_token"], feat["mean_context"]], dim=-1))
        enc_feature = self.encirclement_feature(feat)
        q_cov = self.coverage_head((cov_feature.unsqueeze(1) * qemb).reshape(-1, self.config.hidden_dim)).reshape(obs["self"].shape[0], num_tau, -1)
        q_enc = self.encirclement_head((enc_feature.unsqueeze(1) * qemb).reshape(-1, self.config.hidden_dim)).reshape(obs["self"].shape[0], num_tau, -1)
        gate = torch.sigmoid(self.gate_head(feat["self_token"]).squeeze(-1))
        if mode == "coverage":
            q = q_cov
        elif mode == "encirclement":
            q = q_enc
        else:
            q = (1.0 - gate[:, None, None]) * q_cov + gate[:, None, None] * q_enc
        return {"q_values": q, "q_coverage": q_cov, "q_encirclement": q_enc, "gate": gate, "tau": tau, "taus": tau}

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], mode: str, epsilon: float = 0.0) -> torch.Tensor:
        q = self.forward(obs, num_tau=32, mode=mode)["q_values"].mean(dim=1)
        greedy = q.argmax(dim=-1)
        if epsilon <= 0:
            return greedy
        random_actions = torch.randint(0, self.config.action_size, greedy.shape, device=greedy.device)
        choose_random = torch.rand(greedy.shape, device=greedy.device) < epsilon
        return torch.where(choose_random, random_actions, greedy)

    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        torch.save({"state_dict": self.state_dict(), "config": asdict(self.config), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CoCapIQN":
        payload = torch.load(path, map_location=device)
        model = cls(CoCapNetConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model.to(device)
