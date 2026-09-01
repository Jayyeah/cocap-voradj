from __future__ import annotations

import hashlib
from pathlib import Path

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.trainer import load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs/experiments/iqn_vxy_full_migration_20260830/stage2_8p2e2obs_700k.yaml"
PURE = ROOT / "configs/experiments/iqn_vxy_pure_coverage_20260901"
SELECTED = ROOT / "artifacts/2026-08-30_iqn_vxy_full/stage2_8p2e2obs_700k/checkpoints/step_600000.pt"


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, path))
        return result
    return {prefix: value}


def test_pure_coverage_changes_only_task_distribution_budget_and_infra() -> None:
    parent = load_config(str(PARENT))
    pure = load_config(str(PURE / "common.yaml"))
    parent_flat = _flatten(parent)
    pure_flat = _flatten(pure)
    sentinel = object()
    differences = {
        key
        for key in set(parent_flat) | set(pure_flat)
        if parent_flat.get(key, sentinel) != pure_flat.get(key, sentinel)
    }
    allowed_prefixes = (
        "env.",
        "tasks.voradj.env.",
        "tasks.voradj_coverage.env.",
        "experiment_metadata.",
    )
    allowed_exact = {
        "output_root",
        "train_mode",
        "total_timesteps",
        "voradj.scenes",
    }
    assert sorted(
        key
        for key in differences
        if key not in allowed_exact and not key.startswith(allowed_prefixes)
    ) == []
    assert pure["reward"] == parent["reward"]
    assert pure["action"] == parent["action"]
    assert pure["dynamics"] == parent["dynamics"]
    assert pure["perception"] == parent["perception"]
    assert pure["iqn"] == parent["iqn"]
    assert pure["total_timesteps"] == 500_000
    assert pure["train_mode"] == "voradj"


def test_warm_and_scratch_are_paired_except_initialization_and_paths() -> None:
    warm = _flatten(load_config(str(PURE / "warm_start.yaml")))
    scratch = _flatten(load_config(str(PURE / "scratch.yaml")))
    sentinel = object()
    differences = {
        key
        for key in set(warm) | set(scratch)
        if warm.get(key, sentinel) != scratch.get(key, sentinel)
    }
    assert differences == {
        "run_name",
        "checkpointing.full_resume_path",
        "pretrained.path",
        "experiment_metadata.initialization",
        "experiment_metadata.selected_checkpoint_sha256",
    }
    assert warm["seed"] == scratch["seed"] == 2026090101
    assert warm["pretrained.path"].endswith("step_600000.pt")
    assert scratch["pretrained.path"] is None


def test_pure_coverage_env_is_final_8p0e2obs_vxy9_ce_contract() -> None:
    config = load_config(str(PURE / "scratch.yaml"))
    task = dict(config["env"])
    task.update(config["tasks"]["voradj"]["env"])
    config = dict(config)
    config["env"] = task
    set_global_config(config)
    env = VorAdjEnv(config, seed=int(config["seed"]))
    env.reset()
    assert len(env.pursuers) == 8
    assert len(env.evaders) == 0
    assert len(env.obstacles) == 2
    assert env.action_size == 9
    assert env.collision_semantics == "legacy_end_step"
    assert config["env"]["episode_max_length"] == 3000
    assert config["reward"]["coverage_objective_version"] == "centroid_energy_v0"
    assert config["reward"]["coverage_ce_success_rms_threshold"] == 0.05
    assert config["reward"]["coverage_ce_success_max_threshold"] == 0.10
    assert config["reward"]["coverage_ce_success_hold_steps"] == 30
    assert config["reward"]["coverage_ce_min_active_pursuers"] == 8
    assert config["reward"]["coverage_settle_enabled"] is False


def test_warm_start_selected_checkpoint_hash() -> None:
    digest = hashlib.sha256(SELECTED.read_bytes()).hexdigest()
    assert digest == "1388ac6813fa6c6ee9f0cc6041446e04556a01ef611a07d0afe3071340a86940"
