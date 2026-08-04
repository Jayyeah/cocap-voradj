from pathlib import Path

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
FINAL_MAIN = ROOT / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml"
ZONE_DEMO = ROOT / "configs/demos/zonedemo_v0/zone_v0_8v2_goal50_b1_exp15_vk1_dynbase.yaml"
ZONE_ADAPT = ROOT / "configs/experiments/zonedemo_adapt_20260728"


def test_zone_is_opt_in_and_not_a_main_training_scene() -> None:
    main = load_config(str(FINAL_MAIN))
    assert main.get("zone_demo", {}).get("enabled", False) is False
    assert main["voradj"]["scenes"] == ["voradj", "voradj_coverage"]


def test_zone_b1_generalization_config_resets_in_outer_ring() -> None:
    config = load_config(str(ZONE_DEMO))
    config["device"] = "cpu"
    config["reward"]["voradj_grid_size"] = 20
    assert config["zone_demo"]["enabled"] is True
    assert config["env"]["evader_spawn_mode"] == "outer_ring_random"
    assert config["apf"]["force_exponent"] == 1.5
    assert config["apf"]["velocity_k"] == 1.0
    assert config["apf"]["dynamic_position_repulsion_requires_closing"] is False

    set_global_config(config)
    env = VorAdjEnv(config, seed=20260804)
    env.reset()
    assert env._zone_enabled() is True
    assert all(not env._zone_point_in_inner(env._position(evader)) for evader in env.evaders)
    assert len(env.zone_evader_targets) == config["env"]["num_evaders"]


def test_zone_adaptation_branch_keeps_train_and_control_configs() -> None:
    train_configs = sorted(ZONE_ADAPT.glob("zonedemo_b1_adapt_*_500k.yaml"))
    control_configs = sorted(ZONE_ADAPT.glob("zonedemo_b1_oldmix_eval_*.yaml"))
    assert len(train_configs) == 2
    assert len(control_configs) == 2
    assert all(load_config(str(path))["zone_demo"]["enabled"] is True for path in train_configs)
    assert all(load_config(str(path))["zone_demo"]["enabled"] is False for path in control_configs)
