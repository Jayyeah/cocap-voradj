from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from cocap_voradj.config import ConfigManager
from cocap_voradj.dynamics.continuous_action import ActionContractError
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/continuous_marl_20260804/p1_acceleration_contract_4v1.yaml"


def build_env() -> VorAdjEnv:
    config = load_config(str(CONFIG_PATH))
    ConfigManager.get_instance().update_config(config)
    return VorAdjEnv(config, seed=2026080401)


def test_p1_config_is_explicit_and_reset_holds_world_axis_yaw() -> None:
    config = load_config(str(CONFIG_PATH))
    assert config["algorithm"] == "p1_contract_smoke"
    assert config["action_mode"] == "acceleration_2d_body"
    assert config["pursuer"]["action_mode"] == "acceleration_2d_body"
    assert config["yaw"] == {"mode": "hold", "init": "aligned_world_axis"}
    env = build_env()
    env.reset()
    assert env.action_mode == "acceleration_2d_body"
    assert env.action_size == 2
    assert env.v_max == pytest.approx(3.0)
    assert all(p.max_speed == pytest.approx(env.v_max) for p in env.pursuers)
    assert np.allclose([p.theta for p in env.pursuers], 0.0)


def test_p1_valid_command_moves_with_yaw_hold_and_updates_diagnostics() -> None:
    env = build_env()
    env.reset()
    before = np.asarray([[p.x, p.y] for p in env.pursuers], dtype=float)
    result = env.step([[0.2, 0.0]] * 4, [None])
    after = np.asarray([[p.x, p.y] for p in env.pursuers], dtype=float)
    assert result.rewards.shape == (4,)
    assert np.all(after[:, 0] > before[:, 0])
    assert np.allclose([p.theta for p in env.pursuers], 0.0)
    assert all(p.last_action_diagnostics["commanded_acceleration"] <= 0.8 + 1e-8 for p in env.pursuers)
    for info in result.infos:
        diagnostics = info["action_diagnostics"]
        assert diagnostics["actual_acceleration"] <= 0.8 + 1e-8
        assert diagnostics["jerk"] >= 0.0
        assert diagnostics["speed_after"] >= 0.0
        assert diagnostics["action_rejected"] is False
        assert diagnostics["validation_delta"] == 0.0
        assert info["replay_metadata"]["action_diagnostics"] == diagnostics


def test_p1_env_rejects_raw_unfeasible_command_instead_of_clipping() -> None:
    env = build_env()
    env.reset()
    env.step([[0.4, 0.0]] * 4, [None])
    with pytest.raises(ActionContractError):
        env.step([[1.0, 0.0]] * 4, [None])


@pytest.mark.parametrize("num_pursuers,num_evaders", [(8, 2), (12, 3)])
def test_p1_scales_and_mixed_evader_interface_keep_contract(num_pursuers: int, num_evaders: int) -> None:
    config = copy.deepcopy(load_config(str(CONFIG_PATH)))
    config["env"]["num_pursuers"] = num_pursuers
    config["env"]["num_evaders"] = num_evaders
    ConfigManager.get_instance().update_config(config)
    env = VorAdjEnv(config, seed=2026080401 + num_pursuers)
    env.reset(initial_pursuer_active=[True] * (num_pursuers - 1) + [False])
    result = env.step([[0.05, 0.0]] * num_pursuers, [None] * num_evaders)
    assert env.action_size == 2
    assert len(result.infos) == num_pursuers
    assert env.pursuers[-1].deactivated
    assert result.infos[-1]["action_diagnostics"] == {}
    assert all("jerk" in info["action_diagnostics"] for info in result.infos[:-1])
