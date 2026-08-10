from __future__ import annotations

import json
from pathlib import Path

from tools import supervise_allagent_oldmix_ablation as supervisor


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _legacy_final(tmp_path: Path) -> tuple[dict, Path]:
    run_dir = tmp_path / "old_line"
    bundle = run_dir / "checkpoints/step_000200000"
    bundle.mkdir(parents=True)
    for name in (
        "trainer.pt",
        "replay.pkl",
        "runtime_state.pkl",
        "effective_config.yaml",
        "manifest.json",
    ):
        (bundle / name).write_bytes(name.encode("utf-8"))
    tag = "legacy_old_line"
    _write_json(
        run_dir / f"{tag}_report.json",
        {"transition_count": 200000, "all_finite": True},
    )
    return {"name": "Legacy", "tag": tag, "run_dir": run_dir}, bundle


def test_legacy_final_without_storage_sidecar_is_migrated_after_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    line, bundle = _legacy_final(tmp_path)
    assert supervisor.raw_final_bundle(line) == bundle

    calls = []

    def fake_validate(candidate: dict, candidate_bundle: Path) -> dict:
        calls.append((candidate, candidate_bundle))
        return {"transition_count": 200000, "replay_size": 200000}

    monkeypatch.setattr(supervisor, "validate_full_resume_bundle", fake_validate)
    prepared = supervisor.prepare_verified_completion(line)

    assert calls == [(line, bundle)]
    assert prepared is not None
    storage = json.loads((bundle / "checkpoint_storage.json").read_text())
    assert storage["kind"] == "full_resume"
    assert storage["step"] == 200000
    assert storage["contains_replay"] is True
    resume = Path(prepared["resume_latest"])
    assert (resume / "replay.pkl").samefile(bundle / "replay.pkl")
    assert supervisor.completion_bundle(line) == bundle


def test_invalid_existing_storage_sidecar_is_not_accepted(tmp_path: Path) -> None:
    line, bundle = _legacy_final(tmp_path)
    _write_json(
        bundle / "checkpoint_storage.json",
        {"step": 175000, "contains_replay": False},
    )
    assert supervisor.raw_final_bundle(line) is None


def test_prune_keeps_verified_final_and_resume_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    line, final_bundle = _legacy_final(tmp_path)
    monkeypatch.setattr(supervisor, "ARTIFACT_ROOT", tmp_path / "audit")
    monkeypatch.setattr(
        supervisor,
        "validate_full_resume_bundle",
        lambda *_: {"transition_count": 200000, "replay_size": 200000},
    )
    prepared = supervisor.prepare_verified_completion(line)
    assert prepared is not None

    milestone = Path(line["run_dir"]) / "checkpoints/step_000025000"
    milestone.mkdir(parents=True)
    (milestone / "trainer.pt").write_bytes(b"model")
    (milestone / "replay.pkl").write_bytes(b"obsolete replay")

    audit = supervisor.prune_obsolete_milestone_replays(
        line,
        final_bundle,
        dict(prepared["validation"]),
    )

    assert not (milestone / "replay.pkl").exists()
    storage = json.loads((milestone / "checkpoint_storage.json").read_text())
    assert storage["kind"] == "evaluation_model_only"
    assert storage["contains_replay"] is False
    assert (final_bundle / "replay.pkl").is_file()
    resume_replay = Path(line["run_dir"]) / "resume_latest/replay.pkl"
    assert resume_replay.samefile(final_bundle / "replay.pkl")
    assert audit["removed_bytes"] == len(b"obsolete replay")
