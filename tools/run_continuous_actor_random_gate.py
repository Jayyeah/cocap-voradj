"""Run the P3 10k random forward/backward numerical gate."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LocalEntityTokenEncoder,
    LocalEntityTokenEncoderConfig,
)
from cocap_voradj.models.continuous.radial_actor import RadialActorConfig, RadialSquashedGaussianActor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/2026-08-04_continuous_marl_refactor/p3_random_gate.json"


def main() -> int:
    torch.manual_seed(2026080403)
    count = 10_000
    encoder_config = LocalEntityTokenEncoderConfig(hidden_dim=32, num_heads=4, num_layers=2)
    actor = RadialSquashedGaussianActor(
        LocalEntityTokenEncoder(encoder_config),
        RadialActorConfig(hidden_dim=32, a_max=0.8, decision_dt=0.5, dropout=0.0),
    )
    tokens = actor.encoder.token_count
    types = torch.zeros(count, tokens, dtype=torch.long)
    types[:, 1 : 1 + encoder_config.max_pursuers] = 1
    types[:, 1 + encoder_config.max_pursuers : 1 + encoder_config.max_pursuers + encoder_config.max_evaders] = 2
    types[:, 1 + encoder_config.max_pursuers + encoder_config.max_evaders :] = 3
    obs = {
        "self": torch.randn(count, encoder_config.self_feature_dim),
        "pursuers": torch.randn(count, encoder_config.max_pursuers, 7),
        "evaders": torch.randn(count, encoder_config.max_evaders, 7),
        "obstacles": torch.randn(count, encoder_config.max_obstacles, 5),
        "masks": torch.ones(count, tokens, dtype=torch.bool),
        "types": types,
    }
    action, log_prob, _ = actor.sample(obs)
    loss = (-log_prob + action.square().mean(dim=-1)).mean()
    loss.backward()
    finite_outputs = bool(torch.isfinite(action).all() and torch.isfinite(log_prob).all() and torch.isfinite(loss))
    finite_grads = bool(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in actor.parameters()))
    payload = {
        "schema_version": 1,
        "kind": "continuous_actor_10k_random_forward_backward_gate",
        "seed": 2026080403,
        "samples": count,
        "hidden_dim": encoder_config.hidden_dim,
        "loss": float(loss.detach()),
        "max_action_norm": float(torch.linalg.vector_norm(action, dim=-1).max().detach()),
        "finite_outputs": finite_outputs,
        "finite_grads": finite_grads,
        "gate": bool(finite_outputs and finite_grads),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"artifact={OUTPUT}")
    return 0 if payload["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
