from pathlib import Path

import yaml

from cocap_voradj.training.trainer import load_config
from tools import supervise_crms_support_approach_curriculum as supervisor


def test_stage_contract_covers_full_scratch_curriculum() -> None:
    assert [stage["stage"] for stage in supervisor.STAGES] == [1, 2, 3]
    assert [stage["capture_evaders"] for stage in supervisor.STAGES] == [1, 2, 3]
    assert [stage["coverage_max_steps"] for stage in supervisor.STAGES] == [1200, 1500, 1800]
    assert [stage["mix_max_steps"] for stage in supervisor.STAGES] == [2200, 2500, 2800]
    assert [stage["total_steps"] for stage in supervisor.STAGES] == [2_000_000, 700_000, 700_000]
    assert "stage1_4p1e1obs_scratch2m.yaml" in supervisor.STAGES[0]["config"]


def test_runtime_pretrained_wrapper_preserves_source_and_compatibility_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    config = tmp_path / "stage2.yaml"
    runtime = tmp_path / "runs/_curriculum_runtime/stage2.yaml"
    checkpoint = tmp_path / "runs/stage1/checkpoints/step_900000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config.write_text(
        yaml.safe_dump(
            {
                "run_name": "stage2",
                "pretrained": {"compatibility_mode": "shape_compatible"},
                "experiment_metadata": {"stage": "stage2"},
            }
        ),
        encoding="utf-8",
    )
    original = config.read_text(encoding="utf-8")
    supervisor.write_pretrained_wrapper(config, runtime, checkpoint)
    updated = load_config(str(runtime))
    assert config.read_text(encoding="utf-8") == original
    assert updated["pretrained"]["path"] == "runs/stage1/checkpoints/step_900000.pt"
    assert updated["pretrained"]["compatibility_mode"] == "shape_compatible"
    assert updated["experiment_metadata"]["selected_from_previous_stage"] is True
