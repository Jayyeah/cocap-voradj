from __future__ import annotations

import copy
import random
from pathlib import Path

import pytest
import torch

from cocap_voradj.training.trainer import CoCapTrainer, FULL_RESUME_SCHEMA, load_config


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "configs/experiments/parallel_ce_legacy_voradj_20260809"
    / "legacy_voradj_b0_pure_capture_local_k10_4p1e1obs_200k_aw.yaml"
)


def _config(tmp_path: Path, run_name: str) -> dict:
    config = load_config(str(BASE_CONFIG))
    config["output_root"] = str(tmp_path)
    config["run_name"] = run_name
    config["device"] = "cpu"
    config["total_timesteps"] = 0
    config["checkpointing"] = {"full_resume": True}
    return config


def _close_logs(trainer: CoCapTrainer) -> None:
    if not trainer.episode_log.closed:
        trainer.episode_log.close()
    if not trainer.metric_log.closed:
        trainer.metric_log.close()


def test_iqn_rolling_resume_contains_and_restores_full_runtime(tmp_path: Path) -> None:
    source_config = _config(tmp_path, "source")
    source = CoCapTrainer(source_config)
    source.train()
    resume_path = tmp_path / "source/checkpoints/resume_latest.pt"
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert payload["schema"] == FULL_RESUME_SCHEMA
    assert set(payload) >= {
        "model", "target_model", "optimizer", "runtime", "rng", "contract_hash", "contract"
    }
    assert set(payload["runtime"]) >= {
        "replays",
        "envs",
        "apf_agents",
        "current_observations",
        "current_task",
        "global_step",
        "update_steps",
    }
    random.setstate(payload["rng"]["python"])
    expected_next_random = random.random()

    resumed_config = _config(tmp_path, "resumed")
    resumed_config["checkpointing"]["resume_path"] = str(resume_path)
    resumed = CoCapTrainer(resumed_config)
    try:
        assert resumed.global_step == source.global_step == 0
        assert resumed.current_task == source.current_task
        assert resumed.current_observations[0].keys() == source.current_observations[0].keys()
        assert random.random() == expected_next_random
        for key, value in source.model.state_dict().items():
            assert torch.equal(value, resumed.model.state_dict()[key])
        for key, value in source.target_model.state_dict().items():
            assert torch.equal(value, resumed.target_model.state_dict()[key])
    finally:
        _close_logs(resumed)


def test_iqn_resume_rejects_changed_training_contract(tmp_path: Path) -> None:
    source = CoCapTrainer(_config(tmp_path, "source"))
    source.train()
    resume_path = tmp_path / "source/checkpoints/resume_latest.pt"
    changed = _config(tmp_path, "changed")
    changed["checkpointing"]["resume_path"] = str(resume_path)
    changed.setdefault("discount", {})["gamma"] = 0.98
    with pytest.raises(ValueError, match="training contract mismatch"):
        trainer = CoCapTrainer(copy.deepcopy(changed))
        _close_logs(trainer)


def test_iqn_full_resume_can_use_external_storage_and_move_it_on_restore(
    tmp_path: Path,
) -> None:
    external_path = tmp_path / "external" / "resume_latest.pt"
    source_config = _config(tmp_path, "external_source")
    source_config["checkpointing"]["full_resume_path"] = str(external_path)
    source = CoCapTrainer(source_config)
    try:
        source.train()
        assert source.full_resume_path == external_path
        assert external_path.is_file()
        assert not (
            tmp_path / "external_source/checkpoints/resume_latest.pt"
        ).exists()
        payload = torch.load(external_path, map_location="cpu", weights_only=False)
        assert payload["schema"] == FULL_RESUME_SCHEMA
    finally:
        _close_logs(source)

    moved_path = tmp_path / "moved" / "resume_latest.pt"
    resumed_config = _config(tmp_path, "external_resumed")
    resumed_config["checkpointing"]["resume_path"] = str(external_path)
    resumed_config["checkpointing"]["full_resume_path"] = str(moved_path)
    resumed = CoCapTrainer(resumed_config)
    try:
        assert resumed.global_step == 0
        assert resumed.full_resume_path == moved_path
        resumed.train()
        assert moved_path.is_file()
    finally:
        _close_logs(resumed)
