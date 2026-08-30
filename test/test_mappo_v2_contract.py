from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LegacyVorAdjFeatureBackbone,
    LegacyVorAdjFeatureBackboneConfig,
)
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.models.small_step_ac import CategoricalGridActor, CentralValueNetwork, IQNGridPolicy
from cocap_voradj.training.small_step_ac import MAPPOConfig, MAPPOTrainer, ValueNorm, compute_gae
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config
from tools.run_small_step_ac_migration import configure_environment


ROOT = Path(__file__).resolve().parents[1]
V2_CONFIG_DIR = ROOT / "configs/experiments/mappo9_v2_20260830"
GOLDEN_CONFIG = ROOT / "configs/experiments/small_step_ac_migration_20260828/iqn_vxy9_seed1.yaml"


def _local_obs(batch: int, max_pursuers: int = 2, max_evaders: int = 1, max_obstacles: int = 1):
    token_count = 1 + max_pursuers + max_evaders + max_obstacles
    masks = torch.ones(batch, token_count, dtype=torch.bool)
    types = torch.tensor(
        [0] + [1] * max_pursuers + [2] * max_evaders + [3] * max_obstacles,
        dtype=torch.long,
    ).repeat(batch, 1)
    return {
        "self": torch.randn(batch, 9),
        "pursuers": torch.randn(batch, max_pursuers, 7),
        "evaders": torch.randn(batch, max_evaders, 7),
        "obstacles": torch.randn(batch, max_obstacles, 5),
        "masks": masks,
        "types": types,
    }


def _global_obs(steps: int, agents: int = 2):
    active = torch.ones(steps, agents, dtype=torch.bool)
    return {
        "self": torch.randn(steps, agents, 9),
        "pursuers": torch.randn(steps, agents, agents, 7),
        "evaders": torch.randn(steps, 1, 7),
        "obstacles": torch.randn(steps, 1, 5),
        "pursuer_mask": torch.ones(steps, agents, agents, dtype=torch.bool),
        "evader_mask": torch.ones(steps, 1, dtype=torch.bool),
        "obstacle_mask": torch.ones(steps, 1, dtype=torch.bool),
        "active_mask": active,
    }


def test_mappo_v2_configs_keep_the_corrected_aw9_task_and_conservative_recipe() -> None:
    golden_root = resolve_ladder_config(GOLDEN_CONFIG)
    golden = configure_environment(golden_root, "mappo9")
    fingerprints = []
    for path in sorted(V2_CONFIG_DIR.glob("seed*.yaml")):
        root = resolve_ladder_config(path)
        configured = configure_environment(root, "mappo9_v2")
        fingerprints.append(
            (
                configured["env"]["collision_semantics"],
                configured["env"]["episode_max_length"],
                configured["reward"]["k_required"],
                configured["reward"]["min_active_pursuers"],
                configured["voradj"]["is_pursuing_release_delay_steps"],
                configured["perception"]["global_evader_visibility"],
                configured["action"]["mode"],
            )
        )
        assert root["small_step_ac"]["algorithm"] == "mappo9_v2"
        assert root["small_step_ac"]["total_env_steps"] == 400_000
        assert root["small_step_ac"]["checkpoint_interval"] == 25_000
        recipe = root["small_step_ac"]["mappo"]
        assert recipe["ppo_epochs"] == 3
        assert recipe["minibatches"] == 2
        assert recipe["actor_lr"] == 3e-5
        assert recipe["target_kl"] == 0.02
        assert recipe["value_norm"] is True
    assert len(fingerprints) == 3 and len(set(fingerprints)) == 1
    assert fingerprints[0][:-1] == (
        golden["env"]["collision_semantics"],
        golden["env"]["episode_max_length"],
        golden["reward"]["k_required"],
        golden["reward"]["min_active_pursuers"],
        golden["voradj"]["is_pursuing_release_delay_steps"],
        golden["perception"]["global_evader_visibility"],
    )
    assert fingerprints[0][-1] == "acceleration_angular_velocity_body"


def test_legacy_voradj_backbone_is_exact_iqn_decision_feature() -> None:
    torch.manual_seed(20260830)
    iqn_config = CoCapNetConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        action_size=9,
        self_feature_dim=9,
        max_pursuers=2,
        max_evaders=1,
        max_obstacles=1,
        num_quantiles=8,
        num_cosine_features=16,
        architecture="voradj_single_head",
        pursuing_embed_dim=8,
    )
    iqn = CoCapIQN(iqn_config).eval()
    original_keys = tuple(iqn.state_dict())
    backbone = LegacyVorAdjFeatureBackbone.from_iqn_state_dict(
        iqn.state_dict(),
        LegacyVorAdjFeatureBackboneConfig(
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            self_feature_dim=9,
            max_pursuers=2,
            max_evaders=1,
            max_obstacles=1,
            pursuing_embed_dim=8,
            dropout=0.1,
        ),
    ).eval()
    obs = _local_obs(5)
    with torch.no_grad():
        expected = iqn.voradj_single_feature(obs, iqn.features(obs))
        actual = backbone(obs)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert tuple(iqn.state_dict()) == original_keys


def test_gae_bootstraps_truncation_but_not_true_terminal() -> None:
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[2.0], [2.0]])
    next_values = torch.tensor([[10.0], [10.0]])
    active = torch.ones(2, 1, dtype=torch.bool)

    terminal_adv, _ = compute_gae(
        rewards,
        values,
        next_values,
        torch.tensor([[True], [True]]),
        active,
        gamma=0.9,
        gae_lambda=0.95,
    )
    truncated_adv, _ = compute_gae(
        rewards,
        values,
        next_values,
        torch.zeros(2, 1, dtype=torch.bool),
        active,
        gamma=0.9,
        gae_lambda=0.95,
        truncated=torch.ones(2, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(terminal_adv, torch.full((2, 1), -1.0))
    torch.testing.assert_close(truncated_adv, torch.full((2, 1), 8.0))


def test_value_norm_roundtrip_floor_and_resume_state() -> None:
    normalizer = ValueNorm(beta=0.9)
    values = torch.tensor([3.0, 3.0, 3.0])
    normalizer.update(values)
    assert float(normalizer.variance) >= 1e-2 - 1e-8
    torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(values)), values)
    restored = ValueNorm(beta=0.9)
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored.mean, normalizer.mean)
    torch.testing.assert_close(restored.std, normalizer.std)


def test_mappo_v2_target_kl_value_norm_and_actor_update_metrics() -> None:
    torch.manual_seed(20260831)
    backbone = LegacyVorAdjFeatureBackbone(
        LegacyVorAdjFeatureBackboneConfig(
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            self_feature_dim=9,
            max_pursuers=2,
            max_evaders=1,
            max_obstacles=1,
            pursuing_embed_dim=4,
            dropout=0.0,
        )
    )
    component = 0.4
    grid = np.asarray(
        [(a, w) for a in (-component, 0.0, component) for w in (-math.pi / 6, 0.0, math.pi / 6)],
        dtype=np.float32,
    )
    actor = CategoricalGridActor(backbone, grid, orthogonal_policy_head=True)
    value = CentralValueNetwork(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        max_agents=2,
        max_evaders=1,
        max_obstacles=1,
    )
    trainer = MAPPOTrainer(
        actor,
        value,
        MAPPOConfig(
            ppo_epochs=4,
            minibatches=1,
            actor_lr=3e-4,
            critic_lr=3e-4,
            target_kl=1e-4,
            value_norm=True,
        ),
        "cpu",
    )
    steps, agents = 4, 2
    local_flat = _local_obs(steps * agents)
    local = {key: item.reshape(steps, agents, *item.shape[1:]) for key, item in local_flat.items()}
    with torch.no_grad():
        _, current_log_prob, latent = actor.sample(local_flat, deterministic=False)
    batch = {
        "local_obs": local,
        "global_obs": _global_obs(steps, agents),
        "actions": actor.action_grid[latent].reshape(steps, agents, 2),
        "latent": latent.reshape(steps, agents),
        "log_prob": (current_log_prob + 2.0).reshape(steps, agents),
        "values": torch.zeros(steps, agents),
        "next_values": torch.zeros(steps, agents),
        "rewards": torch.arange(steps * agents, dtype=torch.float32).reshape(steps, agents) / 10.0,
        "active_mask": torch.ones(steps, agents, dtype=torch.bool),
        "terminated": torch.zeros(steps, agents, dtype=torch.bool),
        "truncated": torch.zeros(steps, agents, dtype=torch.bool),
        "episode_end": torch.zeros(steps, agents, dtype=torch.bool),
    }
    metrics = trainer.update(batch, categorical=True)
    assert metrics["kl_early_stop"] == 1.0
    assert metrics["ppo_epochs_completed"] == 1.0
    assert metrics["minibatch_updates"] == 1.0
    assert metrics["approx_kl_max"] > 1e-4
    assert metrics["actor_update_l2"] > 0.0
    assert metrics["actor_update_relative_l2"] > 0.0
    assert metrics["value_grad_norm"] > 0.0
    assert "value_norm_mean" in metrics and "value_norm_std" in metrics


class _TauSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau = None

    def forward(self, obs, num_tau, mode, tau=None):
        del mode
        self.tau = tau
        batch = obs["self"].shape[0]
        count = num_tau if tau is None else int(tau.numel())
        return {"q_values": torch.zeros(batch, count, 9)}


def test_iqn_deterministic_policy_uses_fixed_midpoint_quantiles() -> None:
    spy = _TauSpy()
    policy = IQNGridPolicy(spy, np.zeros((9, 2), dtype=np.float32), quantiles=8)
    policy.sample({"self": torch.zeros(2, 9)}, deterministic=True)
    expected = (torch.arange(8, dtype=torch.float32) + 0.5) / 8.0
    torch.testing.assert_close(spy.tau, expected)
    policy.sample({"self": torch.zeros(2, 9)}, deterministic=False)
    assert spy.tau is None
