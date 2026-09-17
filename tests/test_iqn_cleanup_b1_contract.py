from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
TEACHER = ROOT / "artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt"
NO_BIT = ROOT / "configs/experiments/iqn_cleanup_emergent_b1_20260917/no_is_pursuing.yaml"


def _obs(batch: int, self_dim: int, friend_dim: int) -> dict[str, torch.Tensor]:
    return {
        "self": torch.zeros(batch, self_dim),
        "pursuers": torch.zeros(batch, 8, friend_dim),
        "evaders": torch.zeros(batch, 8, 7),
        "obstacles": torch.zeros(batch, 5, 5),
        "types": torch.cat(
            [
                torch.zeros(batch, 1),
                torch.ones(batch, 8),
                torch.full((batch, 8), 2.0),
                torch.full((batch, 5), 3.0),
            ],
            dim=1,
        ).long(),
        "masks": torch.ones(batch, 22, dtype=torch.bool),
    }


def test_legacy_final_iqn_loader_keeps_old_contract():
    model = CoCapIQN.load(str(TEACHER), device="cpu").eval()
    assert model.config.include_is_pursuing is True
    assert model.config.self_feature_dim == 9
    assert model.config.pursuer_feature_dim == 7
    assert "pursuing_embed.weight" in model.state_dict()
    output = model(_obs(2, 9, 7), num_tau=2, mode="voradj")
    assert output["q_values"].shape == (2, 2, 9)


def test_legacy_seven_dimensional_iqn_keeps_zero_role_fallback():
    model = CoCapIQN(CoCapNetConfig(self_feature_dim=7, pursuer_feature_dim=7)).eval()
    output = model(_obs(2, 7, 7), num_tau=2, mode="voradj")
    assert output["q_values"].shape == (2, 2, 9)


def test_no_bit_student_removes_all_policy_role_paths_and_transfers_weights():
    teacher = CoCapIQN.load(str(TEACHER), device="cpu").eval()
    student, report = CoCapIQN.make_no_is_pursuing_student(teacher)
    assert student.config.include_is_pursuing is False
    assert student.config.self_feature_dim == 8
    assert student.config.pursuer_feature_dim == 6
    assert not any(key.startswith("pursuing_embed.") for key in student.state_dict())
    assert report["late_fusion_removed"] is True
    assert len(report["copied_exact_keys"]) >= 100
    assert torch.equal(
        student.state_dict()["transformer.layers.0.self_attn.in_proj_weight"],
        teacher.state_dict()["transformer.layers.0.self_attn.in_proj_weight"],
    )
    output = student(_obs(2, 8, 6), num_tau=2, mode="voradj")
    assert output["q_values"].shape == (2, 2, 9)


def test_no_bit_config_changes_policy_observation_only():
    standard = load_config("configs/experiments/forward_final_mappo_20260908/stage1_4v1.yaml")
    no_bit = load_config(str(NO_BIT))
    assert no_bit["perception"]["include_is_pursuing"] is False
    assert no_bit["perception"]["self_feature_dim"] == 8
    assert no_bit["iqn"]["include_is_pursuing"] is False
    assert no_bit["iqn"]["pursuer_feature_dim"] == 6
    assert no_bit["reward"] == standard["reward"]
    assert no_bit["voradj"] == standard["voradj"]
    assert no_bit["env"] == standard["env"]


def test_same_env_state_exposes_teacher_and_role_free_views_without_changing_role_metadata():
    config = load_config(str(NO_BIT))
    config = deep_update(config, config["tasks"]["voradj"])
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026091701)
    no_bit = env.reset()
    roleful = env.get_policy_observations(True)
    explicit_no_bit = env.get_policy_observations(False)
    assert no_bit[0]["self"].shape == (8,)
    assert no_bit[0]["pursuers"].shape == (8, 6)
    assert roleful[0]["self"].shape == (9,)
    assert roleful[0]["pursuers"].shape == (8, 7)
    assert np.array_equal(roleful[0]["masks"], explicit_no_bit[0]["masks"])
    result = env.step([0] * len(env.pursuers), [None] * len(env.evaders))
    assert all("replay_metadata" in info for info in result.infos)
    assert all("effective_pursuing" in info["replay_metadata"] for info in result.infos)
    assert all("reward_role" in info["replay_metadata"] for info in result.infos)
