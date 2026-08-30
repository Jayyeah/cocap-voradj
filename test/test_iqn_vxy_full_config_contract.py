from __future__ import annotations

from pathlib import Path

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
AW_DIR = ROOT / "configs/experiments/cr_ms_support_approach_ce_curriculum_20260802"
VXY_DIR = ROOT / "configs/experiments/iqn_vxy_full_migration_20260830"
STAGES = (
    "stage1_4p1e1obs_scratch2m.yaml",
    "stage2_8p2e2obs_700k.yaml",
    "stage3_12p3e3obs_700k.yaml",
)


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, path))
        return result
    return {prefix: value}


def _allowed_difference(path: str) -> bool:
    exact = {
        "action_mode", "v_max", "decision_dt", "output_root", "run_name",
        "env.action_mode", "env.collision_semantics", "pursuer.action_mode",
        "iqn.checkpoint_freq",
    }
    return path in exact or path.startswith(
        ("action.", "dynamics.", "yaw.", "checkpointing.", "experiment_metadata.")
    )


def test_full_vxy_stages_differ_from_final_aw_only_on_audited_paths() -> None:
    for stage in STAGES:
        aw = _flatten(load_config(str(AW_DIR / stage)))
        vxy = _flatten(load_config(str(VXY_DIR / stage)))
        sentinel = object()
        differences = {
            key for key in set(aw) | set(vxy) if aw.get(key, sentinel) != vxy.get(key, sentinel)
        }
        assert sorted(key for key in differences if not _allowed_difference(key)) == []
        assert "action_mode" in differences
        assert "iqn.checkpoint_freq" in differences


def test_full_vxy_preserves_historical_collision_and_stage_curriculum() -> None:
    expected = ((4, 1, 1, 2_000_000), (8, 2, 2, 700_000), (12, 3, 3, 700_000))
    for stage, (pursuers, evaders, obstacles, steps) in zip(STAGES, expected):
        aw = load_config(str(AW_DIR / stage))
        vxy = load_config(str(VXY_DIR / stage))
        set_global_config(aw)
        aw_env = VorAdjEnv(aw, seed=int(aw["seed"]))
        set_global_config(vxy)
        vxy_env = VorAdjEnv(vxy, seed=int(vxy["seed"]))
        assert aw_env.collision_semantics == vxy_env.collision_semantics == "legacy_end_step"
        assert vxy_env.action_size == 9
        assert vxy["total_timesteps"] == steps
        task = dict(vxy["env"])
        task.update(vxy.get("tasks", {}).get("voradj", {}).get("env", {}))
        assert (task["num_pursuers"], task["num_evaders"], task["num_obstacles"]) == (
            pursuers, evaders, obstacles,
        )
        assert vxy["iqn"]["checkpoint_freq"] == 25_000
        assert vxy["checkpointing"]["full_resume"] is True


def test_full_vxy_keeps_network_replay_optimizer_and_reward_equal() -> None:
    for stage in STAGES:
        aw = load_config(str(AW_DIR / stage))
        vxy = load_config(str(VXY_DIR / stage))
        for key in (
            "architecture", "hidden_dim", "num_heads", "num_layers", "action_size",
            "batch_size", "replay_capacity", "learning_rate", "gamma", "min_replay_size",
            "train_freq", "target_update_freq", "epsilon_start", "epsilon_final",
            "epsilon_decay_steps", "update_rule",
        ):
            assert vxy["iqn"][key] == aw["iqn"][key]
        assert vxy["reward"] == aw["reward"]
        assert vxy["voradj"] == aw["voradj"]
