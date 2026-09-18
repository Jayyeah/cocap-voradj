from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
ROLE = ROOT / "configs/experiments/iqn_role_token_scratch_20260918/role_token.yaml"
Z = ROOT / "configs/experiments/iqn_zstate_b3_20260918/z_state.yaml"


def _scene(config: dict, task: str) -> dict:
    return deep_update(config, config.get("tasks", {}).get(task, {}))


def _model(config: dict) -> CoCapIQN:
    iqn = config["iqn"]
    return CoCapIQN(
        CoCapNetConfig(
            hidden_dim=int(iqn["hidden_dim"]),
            num_heads=int(iqn["num_heads"]),
            num_layers=int(iqn["num_layers"]),
            action_size=int(iqn["action_size"]),
            self_feature_dim=int(iqn["self_feature_dim"]),
            max_pursuers=int(config["perception"]["max_pursuer_num"]),
            max_evaders=int(config["perception"]["max_evader_num"]),
            max_obstacles=int(config["perception"]["max_obstacle_num"]),
            num_quantiles=int(iqn["num_quantiles"]),
            num_cosine_features=int(iqn["num_cosine_features"]),
            architecture=str(iqn["architecture"]),
            pursuing_embed_dim=int(iqn["pursuing_embed_dim"]),
            pursuer_feature_dim=int(iqn["pursuer_feature_dim"]),
            include_is_pursuing=bool(iqn["include_is_pursuing"]),
            include_z_state=bool(iqn["include_z_state"]),
            pursuing_late_fusion=bool(iqn["pursuing_late_fusion"]),
        )
    )


def test_role_config_is_matched_to_z_except_declared_observation_fields():
    role = load_config(str(ROLE))
    z = load_config(str(Z))
    assert role["reward"] == z["reward"]
    assert role["env"] == z["env"]
    assert role["voradj"] == z["voradj"]
    assert role["perception"]["include_is_pursuing"] is True
    assert role["perception"]["include_z_state"] is False
    assert role["perception"]["friend_ordering_mode"] == "physical_only"
    assert role["iqn"]["include_is_pursuing"] is True
    assert role["iqn"]["include_z_state"] is False
    assert role["iqn"]["pursuing_late_fusion"] is False
    assert role["iqn"]["pursuing_embed_dim"] == 0
    assert not (role.get("pretrained") or {}).get("path")


def test_role_model_has_tokens_but_no_late_fusion():
    model = _model(load_config(str(ROLE)))
    assert model.config.include_is_pursuing is True
    assert model.config.include_z_state is False
    assert model.config.pursuing_late_fusion is False
    assert model.pursuing_embed is None
    assert not any(key.startswith("pursuing_embed.") for key in model.state_dict())
    assert model.single_action_feature[0].in_features == model.config.hidden_dim


def test_role_and_z_have_identical_physical_friend_ordering():
    role = load_config(str(ROLE))
    z = load_config(str(Z))
    role_scene = _scene(role, "voradj")
    z_scene = _scene(z, "voradj")
    set_global_config(role_scene)
    role_env = VorAdjEnv(copy.deepcopy(role_scene), seed=2026091801)
    role_obs = role_env.reset()
    set_global_config(z_scene)
    z_env = VorAdjEnv(copy.deepcopy(z_scene), seed=2026091801)
    z_obs = z_env.reset()
    for role_item, z_item in zip(role_obs, z_obs):
        if role_item is None:
            assert z_item is None
            continue
        assert role_item["self"].shape == (9,)
        assert role_item["pursuers"].shape == (8, 7)
        assert np.array_equal(role_item["types"], z_item["types"])
        assert np.array_equal(role_item["masks"], z_item["masks"])
        assert np.array_equal(role_item["self"][:8], z_item["self"][:8])
        assert np.array_equal(role_item["pursuers"][:, :6], z_item["pursuers"][:, :6])
        assert np.all(np.isin(role_item["self"][-1], [0.0, 1.0]))
        assert np.all(np.isin(role_item["pursuers"][:, -1], [0.0, 1.0]))
