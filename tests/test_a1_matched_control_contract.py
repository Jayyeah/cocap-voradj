from __future__ import annotations

from pathlib import Path

import yaml

from cocap_voradj.training.trainer import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/experiments/ac_entropy_localq_cov_20260921/stage1.yaml"
CONTROL = ROOT / "configs/experiments/ac_entropy_localq_cov_20260921/stage1_no_entropy_control.yaml"


def _without_identity(config: dict) -> dict:
    value = yaml.safe_load(yaml.safe_dump(config, sort_keys=True))
    value.pop("run_name", None)
    value.pop("output_root", None)
    value.setdefault("ac", {}).pop("actor_entropy_alpha", None)
    metadata = value.get("experiment_metadata", {})
    for key in ("child_task", "actor_entropy_alpha", "matched_control_of"):
        metadata.pop(key, None)
    return value


def test_control_config_is_exact_match_except_registered_entropy_override() -> None:
    base = load_config(str(BASE))
    control = load_config(str(CONTROL))
    assert _without_identity(base) == _without_identity(control)
    assert base["ac"]["actor_entropy_alpha"] == 0.05
    assert control["ac"]["actor_entropy_alpha"] == 0.0
    assert control["seed"] == base["seed"] == 2026092101
    assert control["ac"]["mode"] == base["ac"]["mode"] == "coverage"


def test_control_contract_keeps_formal_gates_and_runtime_identity() -> None:
    base = load_config(str(BASE))
    control = load_config(str(CONTROL))
    assert control["total_timesteps"] == base["total_timesteps"] == 100000
    assert control["training"] == base["training"]
    assert control["ac"]["checkpoint"] == base["ac"]["checkpoint"]
    assert control["ac"]["evaluation"] == base["ac"]["evaluation"]
    assert control["ac"]["behavior_policy"] == base["ac"]["behavior_policy"]
