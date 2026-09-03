from pathlib import Path

from cocap_voradj.training.trainer import load_config
from tools import supervise_iqn_vxy_full_20260830 as supervisor


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/experiments/iqn_vxy_full_migration_20260830"
CANDIDATE = ROOT / "configs/experiments/iqn_vxy_support11_1m_20260903"
PAIRS = (
    ("stage1_4p1e1obs_scratch2m.yaml", "stage1_4p1e1obs_1m.yaml"),
    ("stage2_8p2e2obs_700k.yaml", "stage2_8p2e2obs_1m.yaml"),
    ("stage3_12p3e3obs_700k.yaml", "stage3_12p3e3obs_1m.yaml"),
)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, path))
        return result
    return {prefix: value}


def test_support11_resolved_configs_change_only_authorized_paths() -> None:
    exact_allowed = {
        "output_root",
        "run_name",
        "total_timesteps",
        "reward.support_reward_capture_weight",
        "reward.support_reward_coverage_weight",
        "checkpointing.full_resume_path",
        "experiment_metadata.series",
        "experiment_metadata.only_changed_variable",
        "experiment_metadata.infrastructure_only_changes",
        "experiment_metadata.stage_timesteps",
        "experiment_metadata.historical_stage_source",
        "experiment_metadata.baseline_support_reward_capture_weight",
        "experiment_metadata.baseline_support_reward_coverage_weight",
        "experiment_metadata.candidate_support_reward_capture_weight",
        "experiment_metadata.candidate_support_reward_coverage_weight",
    }
    for baseline_name, candidate_name in PAIRS:
        baseline = flatten(load_config(str(BASE / baseline_name)))
        candidate = flatten(load_config(str(CANDIDATE / candidate_name)))
        sentinel = object()
        differences = {
            key for key in set(baseline) | set(candidate)
            if baseline.get(key, sentinel) != candidate.get(key, sentinel)
        }
        assert differences <= exact_allowed, sorted(differences - exact_allowed)
        assert {
            "total_timesteps",
            "reward.support_reward_capture_weight",
            "reward.support_reward_coverage_weight",
        } <= differences
        assert candidate["total_timesteps"] == 1_000_000
        assert candidate["reward.support_reward_capture_weight"] == 1.0
        assert candidate["reward.support_reward_coverage_weight"] == 1.0


def test_support11_profile_owns_paths_sessions_and_resume() -> None:
    try:
        supervisor.activate_profile("support11")
        assert supervisor.AUTO_GATE_OVERRIDE_ALL is True
        assert len(supervisor.STAGES) == 3
        assert all(stage["steps"] == 1_000_000 for stage in supervisor.STAGES)
        assert len({str(stage["rolling_resume_path"]) for stage in supervisor.STAGES}) == 3
        assert all(str(stage["rolling_resume_path"]).startswith("/dev/shm/") for stage in supervisor.STAGES)
        paths = supervisor.stage_paths(supervisor.STAGES[0])
        assert "2026-09-03_iqn_vxy_support11_1m" in str(paths["run"])
        assert "iqn_vxy_support11_1m" in paths["selection"].name
    finally:
        supervisor.activate_profile("full")
    assert supervisor.AUTO_GATE_OVERRIDE_ALL is False
    assert supervisor.STAGES[0]["steps"] == 2_000_000
    assert supervisor.SESSION_PREFIX == "cocap_vxy_full"


def test_authorized_override_still_requires_selected_checkpoint() -> None:
    stage = {"number": 1}
    assert supervisor.can_override_stage_gate(
        stage, {"checks": {"checkpoint": True}}, False, allow_all_stages=True
    )
    assert not supervisor.can_override_stage_gate(
        stage, {"checks": {"checkpoint": False}}, False, allow_all_stages=True
    )
