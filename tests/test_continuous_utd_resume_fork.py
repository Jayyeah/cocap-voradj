from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import tools.run_continuous_ctde_training as runner


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / (
    "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_oldmix_4p1e1obs_200k_aw_allagent_noclip_scratch.yaml"
)
TARGET_CONFIG = ROOT / (
    "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_oldmix_4p1e1obs_300k_aw_allagent_noclip_c1_utd05.yaml"
)
SOURCE_MANIFEST = ROOT / (
    "artifacts/2026-08-11_allagent_noclip_scratch/"
    "legacy_voradj_oldmix_allagent_noclip_scratch_4p1e1obs_200k_aw_20260811/"
    "resume_frozen_p1_step_000200000/manifest.json"
)


def target_manifest(source: dict) -> dict:
    target = copy.deepcopy(source)
    target["config"] = str(TARGET_CONFIG.relative_to(ROOT))
    target["config_hash"] = "target"
    target["effective_config_hashes"] = {"target": "target"}
    target["implementation_hash"] = "runner-wrapper-change-only"
    target["initialization"] = "continuation_from_p1_200k_frozen_bundle"
    return target


@pytest.mark.skipif(not SOURCE_MANIFEST.is_file(), reason="P1 frozen manifest unavailable")
def test_c1_config_is_strict_utd_only_fork() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    config = runner.resolve_ladder_config(TARGET_CONFIG)
    audit = runner._validate_utd_only_resume_fork(source, target_manifest(source), config)
    diffs = {row["path"]: (row["before"], row["after"]) for row in audit["config_diffs"]}
    assert diffs["training.update_every_env_steps"] == (4, 2)
    assert diffs["training.total_env_steps"] == (200000, 300000)
    assert "masac.actor_lr" not in diffs
    assert "masac.critic_lr" not in diffs
    assert config["training"]["gradient_steps"] == 1
    assert config["training"]["grad_clip_norm"] is None


@pytest.mark.skipif(not SOURCE_MANIFEST.is_file(), reason="P1 frozen manifest unavailable")
def test_c1_rejects_extra_hyperparameter_change() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    config = runner.resolve_ladder_config(TARGET_CONFIG)
    config["masac"]["actor_lr"] = 3e-4
