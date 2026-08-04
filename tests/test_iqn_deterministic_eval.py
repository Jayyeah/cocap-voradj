from __future__ import annotations

import types

import torch

from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig


def test_act_uses_fixed_quantile_midpoints_when_requested() -> None:
    model = CoCapIQN(CoCapNetConfig())
    observed_tau = []

    def fake_forward(self, obs, num_tau=8, mode="mixed", tau=None):
        observed_tau.append(None if tau is None else tau.detach().cpu().clone())
        q_values = torch.zeros(obs["self"].shape[0], num_tau, self.config.action_size)
        q_values[..., 3] = 1.0
        return {"q_values": q_values}

    model.forward = types.MethodType(fake_forward, model)
    obs = {"self": torch.zeros(2, model.config.self_feature_dim)}

    actions = model.act(
        obs,
        mode="voradj",
        epsilon=0.0,
        deterministic_quantiles=True,
    )

    expected = (torch.arange(32, dtype=torch.float32) + 0.5) / 32.0
    assert torch.equal(actions, torch.tensor([3, 3]))
    assert len(observed_tau) == 1
    assert torch.equal(observed_tau[0], expected)
