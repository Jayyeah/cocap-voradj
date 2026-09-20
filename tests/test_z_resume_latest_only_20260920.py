from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cocap_voradj.training.trainer import CoCapTrainer, load_config
from tools import iqn_z_unified_decay_curriculum_20260919 as run


BASE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/experiments/parallel_ce_legacy_voradj_20260809"
    / "legacy_voradj_b0_pure_capture_local_k10_4p1e1obs_200k_aw.yaml"
)


class FakeTrainer:
    instances: list["FakeTrainer"] = []

    def __init__(self, config: dict) -> None:
        self.config = copy.deepcopy(config)
        self.run_dir = Path(config["output_root"]) / config["run_name"]
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        resume_path = Path(str((config.get("checkpointing") or {}).get("resume_path", "")))
        self.global_step = run._resume_global_step(resume_path) if resume_path.is_file() else 0
        self.replays = {"main": [1, 2]}
        self.recovery_init_pool = [1]
        self.optimizer = SimpleNamespace(state={"main": 1})
        self.episode_log = SimpleNamespace(closed=True)
        self.metric_log = SimpleNamespace(closed=True)
        FakeTrainer.instances.append(self)

    def train(self) -> Path:
        self.global_step = int(self.config["total_timesteps"])
        checkpoint = self.ckpt_dir / f"step_{self.global_step}.pt"
        checkpoint.write_bytes(f"ordinary-{self.global_step}".encode())
        (self.ckpt_dir / "latest.pt").write_bytes(f"latest-{self.global_step}".encode())
        payload = {
            "schema": "fake-full-resume",
            "runtime": {
                "global_step": self.global_step,
                "update_steps": self.global_step * 2,
                "target_update_count": self.global_step // 2,
                "replays": self.replays,
                "recovery_init_pool": self.recovery_init_pool,
            },
        }
        torch.save(payload, self.ckpt_dir / "resume_latest.pt")
        (self.run_dir / "metrics.jsonl").write_text(
            json.dumps({"global_step": self.global_step, "update_steps": self.global_step * 2}) + "\n",
            encoding="utf-8",
        )
        return checkpoint


def _fake_config(_arm: str, _stage: str) -> dict:
    return {
        "total_timesteps": 3,
        "checkpointing": {"full_resume": True},
        "formal_evaluation": {"episodes_per_scene": 1, "seed_base": 0},
    }


def _install_fake_curriculum(monkeypatch: pytest.MonkeyPatch, milestones: tuple[int, ...]) -> dict[str, list[Path]]:
    calls: dict[str, list[Path]] = {"evaluations": []}
    monkeypatch.setattr(run, "CoCapTrainer", FakeTrainer)
    monkeypatch.setattr(run, "_close", lambda _trainer: None)
    monkeypatch.setattr(run, "_fresh_stage_gate", lambda _trainer, _stage, _warm_start: {"fake": True})
    monkeypatch.setattr(run, "resolved", _fake_config)
    monkeypatch.setattr(run, "MILESTONES", {"stage1": milestones, "stage2": milestones, "stage3": milestones})

    def evaluate(checkpoint: Path, eval_dir: Path, *_args, **_kwargs) -> dict:
        calls["evaluations"].append(eval_dir / "report.json")
        eval_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "summary": {
                "coverage": {
                    "strict_ce_rate": 0.5,
                    "collision_rate": 0.0,
                    "ce_rms": {"mean": 0.2},
                    "area_cv": {"mean": 0.3},
                },
                "capture": {
                    "normal_capture_rate": 0.8,
                    "collision_rate": 0.0,
                    "capture_seconds": {"mean": 20.0},
                },
                "mixed": {
                    "capture_rate": 0.7,
                    "safe_complete_rate": 0.6,
                    "post_capture_ce_rate": 0.5,
                    "collision_rate": 0.0,
                },
            },
        }
        report_path = eval_dir / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(run.matched, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(
        run,
        "select_balanced",
        lambda stage_dir, registered: {
            "selected": {
                "checkpoint": str(stage_dir / "training" / "checkpoints" / f"step_{registered[-1]}.pt")
            }
        },
    )
    return calls


def _run_fake_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, milestones: tuple[int, ...] = (1, 2, 3)) -> tuple[Path, dict]:
    _install_fake_curriculum(monkeypatch, milestones)
    output = tmp_path / "z05"
    state: dict = {}
    run.run_stage("z05", "stage1", output, "cpu", None, state)
    return output, state


def _fake_payload(path: Path, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "fake-full-resume",
            "runtime": {
                "global_step": step,
                "update_steps": step * 2,
                "target_update_count": step // 2,
                "replays": {"main": [1, 2]},
                "recovery_init_pool": [1],
            },
        },
        path,
    )


def test_t1_latest_only_retention_and_milestone_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, _state = _run_fake_stage(tmp_path, monkeypatch)
    checkpoint_dir = output / "stages/stage1/training/checkpoints"

    assert (checkpoint_dir / "resume_latest.pt").is_file()
    assert list(checkpoint_dir.glob("resume_step_*.pt")) == []
    assert {path.name for path in checkpoint_dir.glob("step_*.pt")} == {"step_1.pt", "step_2.pt", "step_3.pt"}
    assert len(list((output / "stages/stage1/evaluations").glob("*/report.json"))) == 3
    assert len(list((output / "stages/stage1/runtime_evidence").glob("step_*.json"))) == 3

    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["last_full_resume"] == str(checkpoint_dir / "resume_latest.pt")
    assert status["last_full_resume_global_step"] == 3
    assert status["full_resume_retention"] == "latest_only"


def test_t2_atomic_resume_replacement_has_no_historical_link(tmp_path: Path) -> None:
    config = load_config(str(BASE_CONFIG))
    config.update(output_root=str(tmp_path), run_name="atomic", device="cpu", total_timesteps=0)
    config["checkpointing"] = {"full_resume": True}
    trainer = CoCapTrainer(config)
    try:
        trainer.train()
        resume = tmp_path / "atomic/checkpoints/resume_latest.pt"
        old_inode = resume.stat().st_ino
        trainer.global_step = 1
        trainer._save_full_resume()
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        assert payload["runtime"]["global_step"] == 1
        assert resume.stat().st_ino != old_inode
        assert list(resume.parent.glob("resume_step_*.pt")) == []
    finally:
        if not trainer.episode_log.closed:
            trainer.episode_log.close()
        if not trainer.metric_log.closed:
            trainer.metric_log.close()


def _state_digest(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_t3_exact_resume_restores_model_optimizer_replay_rng_z_and_step(tmp_path: Path) -> None:
    config = run.resolved("z05", "stage1")
    config.update(output_root=str(tmp_path), run_name="source", device="cpu", total_timesteps=3)
    config["iqn"]["checkpoint_freq"] = 3
    config["iqn"]["log_freq_steps"] = 1
    config["checkpointing"] = {"full_resume": True}
    source = CoCapTrainer(config)
    try:
        source.train()
        resume = tmp_path / "source/checkpoints/resume_latest.pt"
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        source_config = copy.deepcopy(config)
        source_config["output_root"] = str(tmp_path)
        source_config["run_name"] = "resumed"
        source_config["checkpointing"]["resume_path"] = str(resume)
        resumed = CoCapTrainer(source_config)
        try:
            assert resumed.global_step == payload["runtime"]["global_step"] == 3
            assert _state_digest(source.model.state_dict()) == _state_digest(resumed.model.state_dict())
            assert _state_digest(source.target_model.state_dict()) == _state_digest(resumed.target_model.state_dict())
            assert _state_digest(source.optimizer.state_dict()) == _state_digest(resumed.optimizer.state_dict())
            assert _state_digest(payload["runtime"]["replays"]) == _state_digest(resumed.replays)
            assert {
                name: len(buffer) for name, buffer in payload["runtime"]["replays"].items()
            } == {name: len(buffer) for name, buffer in resumed.replays.items()}
            assert payload["runtime"]["recovery_init_pool"] == resumed.recovery_init_pool
            assert payload["z_states"] == {
                task: env.z_state_dict()
                for task, env in resumed.envs.items()
                if getattr(env, "_z_state_enabled", False)
            }
            assert resumed.global_step == payload["runtime"]["global_step"]
            assert torch.equal(torch.get_rng_state(), payload["rng"]["torch"])
            assert np.array_equal(np.random.get_state()[1], payload["rng"]["numpy"][1])
        finally:
            if not resumed.episode_log.closed:
                resumed.episode_log.close()
            if not resumed.metric_log.closed:
                resumed.metric_log.close()
    finally:
        if not source.episode_log.closed:
            source.episode_log.close()
        if not source.metric_log.closed:
            source.metric_log.close()


def test_t4_restart_skips_historical_milestones_without_rewriting_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _state = _run_fake_stage(tmp_path, monkeypatch)
    stage_dir = output / "stages/stage1"
    tracked = [
        stage_dir / "evaluations/step_000000001/report.json",
        stage_dir / "runtime_evidence/step_000000001.json",
        stage_dir / "evaluations/step_000000003/report.json",
        stage_dir / "runtime_evidence/step_000000003.json",
    ]
    before = {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in tracked}
    FakeTrainer.instances.clear()
    calls = {"evaluations": []}
    monkeypatch.setattr(run.matched, "evaluate_checkpoint", lambda *args, **kwargs: calls["evaluations"].append(Path(args[1]) / "report.json"))
    run.run_stage("z05", "stage1", output, "cpu", None, {})

    assert FakeTrainer.instances == []
    assert calls["evaluations"] == []
    assert {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in tracked} == before


def test_t5_equal_step_recovers_missing_report_and_evidence_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _state = _run_fake_stage(tmp_path, monkeypatch)
    stage_dir = output / "stages/stage1"
    report = stage_dir / "evaluations/step_000000003/report.json"
    evidence = stage_dir / "runtime_evidence/step_000000003.json"
    report.unlink()
    evidence.unlink()
    FakeTrainer.instances.clear()
    calls = _install_fake_curriculum(monkeypatch, (1, 2, 3))
    run.run_stage("z05", "stage1", output, "cpu", None, {})

    assert FakeTrainer.instances == []
    assert calls["evaluations"] == [report]
    assert report.is_file() and evidence.is_file()
    assert json.loads(evidence.read_text(encoding="utf-8"))["global_step"] == 3


def test_t6_historical_provenance_gap_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, _state = _run_fake_stage(tmp_path, monkeypatch)
    evidence = output / "stages/stage1/runtime_evidence/step_000000001.json"
    evidence.unlink()
    calls = _install_fake_curriculum(monkeypatch, (1, 2, 3))

    with pytest.raises(RuntimeError, match="provenance is incomplete"):
        run.run_stage("z05", "stage1", output, "cpu", None, {})
    assert calls["evaluations"] == []


def test_t7_balanced_selection_is_independent_of_historical_full_resumes(tmp_path: Path) -> None:
    milestones = (100, 200, 300)
    summary = {
        "coverage": {"strict_ce_rate": 0.6, "collision_rate": 0.0, "ce_rms": {"mean": 0.2}, "area_cv": {"mean": 0.3}},
        "capture": {"normal_capture_rate": 0.8, "collision_rate": 0.0, "capture_seconds": {"mean": 20.0}},
        "mixed": {"capture_rate": 0.7, "safe_complete_rate": 0.6, "post_capture_ce_rate": 0.5, "collision_rate": 0.0},
    }
    for step in milestones:
        checkpoint = tmp_path / "training/checkpoints" / f"step_{step}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"ordinary-{step}".encode())
        report = tmp_path / "evaluations" / f"step_{step:09d}" / "report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "summary": summary,
                }
            ),
            encoding="utf-8",
        )
        _fake_payload(tmp_path / "training/checkpoints" / f"resume_step_{step:09d}.pt", step)
    before = run.select_balanced(tmp_path, milestones)["selected"]["step"]
    for path in (tmp_path / "training/checkpoints").glob("resume_step_*.pt"):
        path.unlink()
    after = run.select_balanced(tmp_path, milestones)["selected"]["step"]
    assert after == before


def test_t8_stage_promotion_uses_selected_ordinary_checkpoint_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_curriculum(monkeypatch, (1,))
    warm_start = tmp_path / "stage1/checkpoints/step_1.pt"
    warm_start.parent.mkdir(parents=True)
    warm_start.write_bytes(b"selected ordinary model")
    output = tmp_path / "z05"
    state: dict = {}
    run.run_stage("z05", "stage2", output, "cpu", warm_start, state)

    trainer = FakeTrainer.instances[-1]
    assert trainer.config["pretrained"]["path"] == str(warm_start)
    assert trainer.config["pretrained"]["compatibility_mode"] == "shape_compatible"
    assert not list((output / "stages/stage2/training/checkpoints").glob("resume_step_*.pt"))
