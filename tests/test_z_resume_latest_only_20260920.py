from __future__ import annotations

import copy
import hashlib
import io
import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cocap_voradj.training.trainer import CoCapTrainer, load_config
from tools import iqn_z_unified_decay_curriculum_20260919 as run
from tools import supervise_iqn_z_unified_decay_dual_20260919 as dual


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
        self.initial_step = self.global_step
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
                "z": {
                    "max_lineage_hop": 1,
                    "post_capture_never_release_episodes": 0,
                },
            },
        }
        report_path = eval_dir / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(run.matched, "evaluate_checkpoint", evaluate)
    def select_balanced(stage_dir: Path, registered: tuple[int, ...]) -> dict:
        checkpoint = stage_dir / "training" / "checkpoints" / f"step_{registered[-1]}.pt"
        report = {
            "schema": run.SCHEMA,
            "selection_reason": "fake selection",
            "fallback_used": False,
            "selected": {
                "step": registered[-1],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            },
        }
        run.matched.atomic_json(stage_dir / "selection_report.json", report)
        return report

    monkeypatch.setattr(run, "select_balanced", select_balanced)
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


def _semantic_digest(value: object) -> str:
    """Digest state content without pickle object-address/serialization artifacts."""

    digest = hashlib.sha256()

    def feed(item: object) -> None:
        digest.update(type(item).__qualname__.encode("utf-8") + b"\0")
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode() + tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode() + repr(item.shape).encode() + item.tobytes())
        elif isinstance(item, np.random.RandomState):
            feed(item.get_state())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=repr):
                feed(key)
                feed(item[key])
        elif isinstance(item, (list, tuple, deque)):
            if isinstance(item, deque):
                feed(item.maxlen)
            for child in item:
                feed(child)
        elif isinstance(item, set):
            for child in sorted(item, key=repr):
                feed(child)
        elif hasattr(item, "__dict__"):
            state = dict(item.__dict__)
            config = state.get("config")
            if isinstance(config, dict):
                config = dict(config)
                config["output_root"] = "<normalized-output-root>"
                config["run_name"] = "<normalized-run-name>"
                state["config"] = config
            feed(state)
        else:
            digest.update(repr(item).encode("utf-8"))

    feed(value)
    return digest.hexdigest()


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


def _continuation_config(root: Path, run_name: str, total_timesteps: int) -> dict:
    config = run.resolved("z05", "stage1")
    config.update(output_root=str(root), run_name=run_name, device="cpu", total_timesteps=total_timesteps)
    config["train_mode"] = "voradj"
    config["iqn"].update(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_quantiles=4,
        num_cosine_features=8,
        train_quantiles=2,
        target_quantiles=2,
        training_quantile_samples=2,
        batch_size=1,
        replay_capacity=64,
        min_replay_size=1,
        train_freq=1,
        target_update_freq=2,
        checkpoint_freq=6,
        log_freq_steps=1,
    )
    config["checkpointing"] = {"full_resume": True}
    return config


def _close_real_trainer(trainer: CoCapTrainer) -> None:
    if not trainer.episode_log.closed:
        trainer.episode_log.close()
    if not trainer.metric_log.closed:
        trainer.metric_log.close()


def test_t9_continuation_equivalence_bit_exact_after_process_restart(tmp_path: Path) -> None:
    split_step, final_step = 6, 12
    uninterrupted = CoCapTrainer(_continuation_config(tmp_path / "path_a", "training", final_step))
    uninterrupted.recovery_init_pool = deque(
        [{"step": 0, "positions": [[0.1, 0.2]], "active_mask": [True]}], maxlen=8
    )
    try:
        uninterrupted.train()
    finally:
        _close_real_trainer(uninterrupted)
    uninterrupted_payload = torch.load(
        tmp_path / "path_a/training/checkpoints/resume_latest.pt", map_location="cpu", weights_only=False
    )

    split = CoCapTrainer(_continuation_config(tmp_path / "path_b", "training", final_step))
    split.recovery_init_pool = deque(
        [{"step": 0, "positions": [[0.1, 0.2]], "active_mask": [True]}], maxlen=8
    )
    split.total_timesteps = split_step
    try:
        split.train()
    finally:
        _close_real_trainer(split)
    split_resume = tmp_path / "path_b/training/checkpoints/resume_latest.pt"
    resumed_config = _continuation_config(tmp_path / "path_b", "training", final_step)
    resumed_config["checkpointing"]["resume_path"] = str(split_resume)
    restarted = CoCapTrainer(resumed_config)
    try:
        restarted.train()
    finally:
        _close_real_trainer(restarted)
    restarted_payload = torch.load(
        tmp_path / "path_b/training/checkpoints/resume_latest.pt", map_location="cpu", weights_only=False
    )

    assert uninterrupted_payload["runtime"]["global_step"] == restarted_payload["runtime"]["global_step"] == final_step
    for key in ("model", "target_model", "optimizer", "z_states"):
        assert _state_digest(uninterrupted_payload[key]) == _state_digest(restarted_payload[key]), key
    assert _semantic_digest(uninterrupted_payload["runtime"]) == _semantic_digest(restarted_payload["runtime"])
    assert uninterrupted_payload["runtime"]["update_steps"] == restarted_payload["runtime"]["update_steps"]
    assert uninterrupted_payload["runtime"]["target_update_count"] == restarted_payload["runtime"]["target_update_count"]
    assert _semantic_digest(uninterrupted_payload["runtime"]["replays"]) == _semantic_digest(restarted_payload["runtime"]["replays"])
    assert "recovery_init_pool" in uninterrupted_payload["runtime"]
    assert _semantic_digest(uninterrupted_payload["runtime"]["recovery_init_pool"]) == _semantic_digest(
        restarted_payload["runtime"]["recovery_init_pool"]
    )
    for key in ("current_task", "task_cursor", "action_histogram", "current_observations"):
        assert _state_digest(uninterrupted_payload["runtime"][key]) == _state_digest(restarted_payload["runtime"][key]), key
    for key in ("current_learning_rate", "epsilon_decay_steps"):
        assert uninterrupted_payload["runtime"][key] == restarted_payload["runtime"][key]
    assert uninterrupted_payload["rng"]["python"] == restarted_payload["rng"]["python"]
    assert uninterrupted_payload["rng"]["numpy"][0] == restarted_payload["rng"]["numpy"][0]
    assert np.array_equal(uninterrupted_payload["rng"]["numpy"][1], restarted_payload["rng"]["numpy"][1])
    assert uninterrupted_payload["rng"]["numpy"][2:] == restarted_payload["rng"]["numpy"][2:]
    assert torch.equal(uninterrupted_payload["rng"]["torch"], restarted_payload["rng"]["torch"])


def _fake_summary() -> dict:
    return {
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
        "z": {"max_lineage_hop": 1, "post_capture_never_release_episodes": 0},
    }


def _seed_selected_stage(output: Path, stage: str, step: int = 1) -> Path:
    stage_dir = output / "stages" / stage
    checkpoint = stage_dir / "training/checkpoints" / f"step_{step}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"selected-{stage}-{step}".encode())
    report = stage_dir / "evaluations" / f"step_{step:09d}" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "summary": _fake_summary()}),
        encoding="utf-8",
    )
    selected = {
        "step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    run.matched.atomic_json(
        stage_dir / "selection_report.json",
        {"schema": run.SCHEMA, "selection_reason": "seeded selection", "fallback_used": False, "selected": selected},
    )
    return checkpoint


def _seed_partial_stage_resume(output: Path, stage: str, step: int = 1) -> Path:
    stage_dir = output / "stages" / stage
    checkpoint = stage_dir / "training/checkpoints" / f"step_{step}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"ordinary-{stage}-{step}".encode())
    resume = stage_dir / "training/checkpoints/resume_latest.pt"
    _fake_payload(resume, step)
    (stage_dir / "training/metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (stage_dir / "training/metrics.jsonl").write_text(json.dumps({"global_step": step}) + "\n", encoding="utf-8")
    report = stage_dir / "evaluations" / f"step_{step:09d}" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": "seed", "summary": _fake_summary()}), encoding="utf-8")
    evidence = stage_dir / "runtime_evidence" / f"step_{step:09d}.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({"step": step, "global_step": step}), encoding="utf-8")
    return resume


def test_t10_stage_restart_chain_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "artifacts/2026-09-19_iqn_z_unified_decay_curriculum/preflight").mkdir(parents=True)
    (tmp_path / "artifacts/2026-09-19_iqn_z_unified_decay_curriculum/preflight/startup_sanity.json").write_text(
        json.dumps({"status": "pass"}), encoding="utf-8"
    )
    monkeypatch.setattr(run.matched, "git", lambda *_args: "test-head")

    # Scenario A: Stage1 selected, then a fresh Stage2 and Stage3 chain.
    output_a = tmp_path / "scenario_a"
    stage1_checkpoint = _seed_selected_stage(output_a, "stage1")
    _install_fake_curriculum(monkeypatch, (1, 2))
    FakeTrainer.instances.clear()
    assert run.supervise_arm("z05", output_a, "cpu") == 0
    a_instances = list(FakeTrainer.instances)
    assert all("stage1" not in str(item.run_dir) for item in a_instances)
    stage2_first = next(item for item in a_instances if "stage2" in str(item.run_dir))
    stage3_first = next(item for item in a_instances if "stage3" in str(item.run_dir))
    assert stage2_first.initial_step == stage3_first.initial_step == 0
    assert stage2_first.config["pretrained"]["path"] == str(stage1_checkpoint)
    stage2_selected = Path(json.loads((output_a / "stages/stage2/selection_report.json").read_text())["selected"]["checkpoint"])
    assert stage3_first.config["pretrained"]["path"] == str(stage2_selected)

    # Scenario B: Stage2 has a durable partial resume; restart must continue it.
    output_b = tmp_path / "scenario_b"
    _seed_selected_stage(output_b, "stage1")
    _seed_partial_stage_resume(output_b, "stage2", step=1)
    _seed_selected_stage(output_b, "stage3")
    _install_fake_curriculum(monkeypatch, (1, 2))
    FakeTrainer.instances.clear()
    assert run.supervise_arm("z05", output_b, "cpu") == 0
    b_instances = list(FakeTrainer.instances)
    assert len(b_instances) == 1
    assert "stage2" in str(b_instances[0].run_dir)
    assert b_instances[0].initial_step == 1
    assert b_instances[0].config["checkpointing"]["resume_path"].endswith("stage2/training/checkpoints/resume_latest.pt")

    # Scenario C: Stage1 and Stage2 selected; Stage3 must start fresh from Stage2 ordinary selection.
    output_c = tmp_path / "scenario_c"
    _seed_selected_stage(output_c, "stage1")
    stage2_checkpoint = _seed_selected_stage(output_c, "stage2")
    _install_fake_curriculum(monkeypatch, (1, 2))
    FakeTrainer.instances.clear()
    assert run.supervise_arm("z05", output_c, "cpu") == 0
    c_instances = list(FakeTrainer.instances)
    assert c_instances and all("stage3" in str(item.run_dir) for item in c_instances)
    assert c_instances[0].initial_step == 0
    assert c_instances[0].config["pretrained"]["path"] == str(stage2_checkpoint)

    # Scenario D: Stage3 itself has a durable partial resume; restart must continue Stage3.
    output_d = tmp_path / "scenario_d"
    _seed_selected_stage(output_d, "stage1")
    _seed_selected_stage(output_d, "stage2")
    _seed_partial_stage_resume(output_d, "stage3", step=1)
    _install_fake_curriculum(monkeypatch, (1, 2))
    FakeTrainer.instances.clear()
    assert run.supervise_arm("z05", output_d, "cpu") == 0
    d_instances = list(FakeTrainer.instances)
    assert len(d_instances) == 1
    assert "stage3" in str(d_instances[0].run_dir)
    assert d_instances[0].initial_step == 1
    assert d_instances[0].config["checkpointing"]["resume_path"].endswith(
        "stage3/training/checkpoints/resume_latest.pt"
    )


def test_t11_coordinator_relaunch_contract_and_gpu_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(dual, "ROOT", tmp_path)
    monkeypatch.setattr(dual, "tmux_present", lambda _session: False)

    def fake_run(command: list[str], **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dual.subprocess, "run", fake_run)
    storage = tmp_path / "runtime"
    sentinel = storage / "z05/sentinel.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")
    dual.launch_arm("z05", tmp_path / "coord", storage, 0)
    dual.launch_arm("z07", tmp_path / "coord", storage, 1)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert len(commands) == 2
    assert "tools/iqn_z_unified_decay_curriculum_20260919.py" in commands[0][-1]
    assert "--arm z05" in commands[0][-1] and "--output" in commands[0][-1]
    assert "--arm z07" in commands[1][-1]
    assert "runtime/z05" in commands[0][-1] and "runtime/z07" in commands[1][-1]

    assert dual.needs_launch({"status": "running", "pid_alive": False, "tmux_present": False}) is True
    assert dual.needs_launch({"status": "running", "pid_alive": True, "tmux_present": False}) is False
    arm_status = {
        "z05": {"status": "running", "pid_alive": False, "tmux_present": False},
        "z07": {"status": "running", "pid_alive": False, "tmux_present": False},
    }
    launchable = [arm for arm in dual.ARMS if dual.needs_launch(arm_status[arm])]
    free_gpus = [0, 1]
    assert dict(zip(launchable, free_gpus)) == {"z05": 0, "z07": 1}


def test_formal_eval_report_is_atomic_and_final_comparison_has_no_resume_dependency() -> None:
    evaluator_source = (Path(run.__file__).with_name("iqn_token_matched_20260919.py")).read_text(encoding="utf-8")
    coordinator_source = Path(dual.__file__).read_text(encoding="utf-8")
    runner_source = Path(run.__file__).read_text(encoding="utf-8")
    assert 'atomic_json(output / "report.json", report)' in evaluator_source
    assert "resume_step_" not in runner_source
    assert "resume_step_" not in coordinator_source


def test_final_comparison_automation_ignores_historical_resume_files(tmp_path: Path) -> None:
    for arm, alpha in (("z05", 0.5), ("z07", 0.7)):
        arm_dir = tmp_path / arm
        stages = []
        for index, stage in enumerate(("stage1", "stage2", "stage3"), start=1):
            checkpoint = arm_dir / stage / "training/checkpoints" / f"step_{index}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{arm}-{stage}".encode())
            stages.append(
                {
                    "stage": stage,
                    "selected_step": index,
                    "selected_checkpoint": str(checkpoint),
                    "formal_summary": _fake_summary(),
                }
            )
        (arm_dir / f"{arm.upper()}_CURRICULUM_FINAL_REPORT.json").write_text(
            json.dumps({"arm": arm, "alpha": alpha, "stages": stages}), encoding="utf-8"
        )

    dual.write_final_comparison(tmp_path)
    before = json.loads((tmp_path / "Z05_VS_Z07_FINAL_COMPARISON.json").read_text(encoding="utf-8"))["stage_comparison"]
    for arm in ("z05", "z07"):
        historical = tmp_path / arm / "stage1/training/checkpoints/resume_step_000000001.pt"
        historical.parent.mkdir(parents=True, exist_ok=True)
        historical.write_bytes(b"historical full resume")
    dual.write_final_comparison(tmp_path)
    after = json.loads((tmp_path / "Z05_VS_Z07_FINAL_COMPARISON.json").read_text(encoding="utf-8"))["stage_comparison"]
    assert after == before
