from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from cocap_voradj.training.continuous.formal_config import resolve_ladder_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/mappo_aw_v2_20260831"
PARENT_CONFIG_DIR = ROOT / "configs/experiments/mappo9_v2_20260830"

EXPECTED_SEEDS = (2026083001, 2026083002, 2026083003)
EVALUATION_SEED = 2026082900
TOTAL_ENV_STEPS = 400_000
CHECKPOINT_INTERVAL = 25_000
AW_MAX = (0.4, math.pi / 6.0)

# Every resolved difference from the corresponding MAPPO-9-v2 seed must be
# an algorithm/head selection or an experiment/run identity.  In particular,
# environment, reward, masks semantics, critic and PPO recipe are not in this
# allowlist and therefore cannot drift silently.
ALLOWED_RESOLVED_DIFFS = frozenset(
    {
        "run_name",
        "device",
        "small_step_ac.algorithm",
        "small_step_ac.algorithm_line",
        "small_step_ac.run_dir",
        "small_step_ac.mappo.log_std_min",
        "small_step_ac.mappo.log_std_max",
        "small_step_ac.mappo.initial_log_std",
        "small_step_ac.mappo.saturation_threshold",
        "actor.backbone",
        "actor.policy_head",
        "actor.action_dim",
        "experiment_metadata.series",
        "experiment_metadata.algorithm_line",
        "experiment_metadata.hypothesis",
        "experiment_metadata.only_changed_variable",
        "experiment_metadata.actor_contract",
    }
)


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_leaves(child, path))
        return output
    return {prefix: value}


def _resolved_differences(parent: dict[str, Any], candidate: dict[str, Any]) -> set[str]:
    parent_leaves = _leaves(parent)
    candidate_leaves = _leaves(candidate)
    return {
        path
        for path in parent_leaves.keys() | candidate_leaves.keys()
        if parent_leaves.get(path) != candidate_leaves.get(path)
    }


def _assert_aw_bounds(config: dict[str, Any]) -> None:
    action = config["action"]
    assert action["mode"] == "acceleration_angular_velocity_body"
    assert action["dimension"] == 2
    assert action["bound_type"] == "independent_box"
    assert math.isclose(float(action["a_max"]), AW_MAX[0], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(action["w_max"]), AW_MAX[1], rel_tol=0.0, abs_tol=1e-12)
    assert len(action["low"]) == len(action["high"]) == 2
    for actual, expected in zip(action["low"], (-AW_MAX[0], -AW_MAX[1])):
        assert math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    for actual, expected in zip(action["high"], AW_MAX):
        assert math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)

    actor = config["actor"]
    assert actor["backbone"] == "legacy_iqn_decision_feature"
    assert actor["policy_head"] == "factorized_squashed_gaussian_aw"
    assert actor["action_dim"] == 2
    assert math.isclose(float(actor["a_max"]), AW_MAX[0], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(actor["w_max"]), AW_MAX[1], rel_tol=0.0, abs_tol=1e-12)


def _assert_task_contract(config: dict[str, Any]) -> None:
    env = config["env"]
    assert (env["width"], env["height"]) == (120.0, 120.0)
    assert (env["num_pursuers"], env["num_evaders"], env["num_obstacles"]) == (4, 1, 1)
    assert env["episode_max_length"] == 1000
    assert env["pre_capture_max_length"] == 1000
    assert env["pursuer_spawn_mode"] == env["evader_spawn_mode"] == "map_random"
    assert env["pursuer_spawn_min_sep"] == 15.0
    assert env["collision_semantics"] == "synchronized_swept_v1"

    perception = config["perception"]
    assert perception["observation_mode"] == "voronoi_adjacency"
    assert perception["topology"] == "legacy_voradj"
    assert perception["global_evader_visibility"] is False
    assert perception["local_sensing_uses_surface_distance"] is True
    assert perception["range"] == perception["enemy_sensing_radius"] == 20.0

    voradj = config["voradj"]
    assert voradj["is_pursuing_release_delay_steps"] == 10
    assert voradj["pure_capture_all_capture_enabled"] is True
    assert voradj["support_reward_blend_enabled"] is False

    reward = config["reward"]
    assert reward["capture_reward_mode"] == "legacy"
    assert reward["k_required"] == 3
    assert reward["min_active_pursuers"] == 2
    assert reward["capture_stationary_enabled"] is True
    assert reward["capture_stationary_speed_threshold"] == 0.2
    assert reward["capture_stationary_hold_steps"] == 10
    assert reward["capture_stationary_min_pursuers"] == 2
    assert (reward["omega_approach"], reward["omega_mean_shift"], reward["omega_front"]) == (1.0, 2.0, 0.5)


def test_mappo_aw_v2_resolved_config_is_strict_parent_parity() -> None:
    paths = sorted(CONFIG_DIR.glob("seed*.yaml"))
    assert len(paths) == 3
    for path in paths:
        index = int(path.stem.removeprefix("seed"))
        parent_path = PARENT_CONFIG_DIR / f"seed{index}.yaml"
        assert path.read_text(encoding="utf-8").startswith("extends: common.yaml")
        raw_common = yaml.safe_load((CONFIG_DIR / "common.yaml").read_text(encoding="utf-8"))
        assert raw_common["extends"] == "../mappo9_v2_20260830/common.yaml"

        parent = resolve_ladder_config(parent_path)
        candidate = resolve_ladder_config(path)
        differences = _resolved_differences(parent, candidate)
        assert differences <= ALLOWED_RESOLVED_DIFFS, sorted(
            differences - ALLOWED_RESOLVED_DIFFS
        )
        assert candidate["seed"] == EXPECTED_SEEDS[index - 1]
        assert candidate["seed"] == parent["seed"]
        assert candidate["device"] == "cuda:1"
        assert candidate["run_name"] == f"mappo_aw_v2_seed{index}_20260831"
        assert candidate["small_step_ac"]["run_dir"] == (
            f"artifacts/2026-08-31_mappo_aw_v2/mappo_aw_v2_seed{index}"
        )


def test_mappo_aw_v2_keeps_eval_checkpoint_and_ppo_recipe() -> None:
    for index in range(1, 4):
        parent = resolve_ladder_config(PARENT_CONFIG_DIR / f"seed{index}.yaml")
        config = resolve_ladder_config(CONFIG_DIR / f"seed{index}.yaml")
        spec = config["small_step_ac"]
        parent_spec = parent["small_step_ac"]
        assert spec["algorithm"] == "mappo_aw_v2"
        assert spec["algorithm_line"] == "mappo_aw_v2_legacy_iqn_decision_feature"
        assert spec["total_env_steps"] == TOTAL_ENV_STEPS
        assert spec["checkpoint_interval"] == CHECKPOINT_INTERVAL
        assert spec["eval_episodes"] == 20
        assert spec["evaluation_seed"] == EVALUATION_SEED
        assert spec["retain_final_full_resume"] is True
        assert spec["mappo"] == {
            **parent_spec["mappo"],
            "log_std_min": -5.0,
            "log_std_max": 1.0,
            "initial_log_std": 0.0,
            "saturation_threshold": 0.99,
        }
        recipe = spec["mappo"]
        assert recipe["rollout_length"] == 256
        assert recipe["gamma"] == 0.99
        assert recipe["gae_lambda"] == 0.95
        assert recipe["clip_param"] == 0.2
        assert recipe["ppo_epochs"] == 3
        assert recipe["minibatches"] == 2
        assert recipe["actor_lr"] == 3e-5
        assert recipe["critic_lr"] == 1e-4
        assert recipe["target_kl"] == 0.02
        assert recipe["entropy_coef"] == 0.01
        assert recipe["value_coef"] == 1.0
        assert recipe["max_grad_norm"] == 0.5
        assert recipe["value_norm"] is True
        assert recipe["value_norm_beta"] == 0.99999
        assert recipe["value_norm_epsilon"] == 1e-5
        assert recipe["output_gain"] == 0.01


def test_mappo_aw_v2_keeps_environment_reward_and_action_bounds() -> None:
    parent = resolve_ladder_config(PARENT_CONFIG_DIR / "seed1.yaml")
    for index in range(1, 4):
        config = resolve_ladder_config(CONFIG_DIR / f"seed{index}.yaml")
        assert config["env"] == parent["env"]
        assert config["perception"] == parent["perception"]
        assert config["dynamics"] == parent["dynamics"]
        assert config["reward"] == parent["reward"]
        assert config["voradj"] == parent["voradj"]
        assert config["central_critic"] == parent["central_critic"]
        assert config["iqn"] == parent["iqn"]
        assert config["training"] == parent["training"]
        _assert_task_contract(config)
        _assert_aw_bounds(config)
