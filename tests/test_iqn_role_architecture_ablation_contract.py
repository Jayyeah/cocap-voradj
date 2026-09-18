from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config


ROOT = Path(__file__).resolve().parents[1]
TEACHER = ROOT / "artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt"
C1 = ROOT / "configs/experiments/iqn_role_architecture_ablation_20260918/token_only_is_pursuing.yaml"
B3_STUDENT = ROOT / "artifacts/2026-09-18_iqn_zstate_b3/student_zstate.pt"
ARTIFACT = ROOT / "artifacts/2026-09-18_iqn_role_arch_ablation"


def test_original_final_checkpoint_loads_with_historical_late_fusion():
    teacher = CoCapIQN.load(str(TEACHER), device="cpu").eval()
    assert teacher.config.include_is_pursuing is True
    assert teacher.config.include_z_state is False
    assert teacher.config.pursuing_late_fusion is True
    assert teacher.pursuing_embed is not None
    assert teacher.single_action_feature[0].in_features == (
        teacher.config.hidden_dim + teacher.config.pursuing_embed_dim
    )


def test_c1_transfer_removes_only_late_role_branch():
    teacher = CoCapIQN.load(str(TEACHER), device="cpu").eval()
    student, report = CoCapIQN.make_token_only_is_pursuing_student(teacher)
    assert student.config.include_is_pursuing is True
    assert student.config.include_z_state is False
    assert student.config.pursuing_late_fusion is False
    assert student.config.self_feature_dim == teacher.config.self_feature_dim == 9
    assert student.config.pursuer_feature_dim == teacher.config.pursuer_feature_dim == 7
    assert student.pursuing_embed is None
    assert not any(key.startswith("pursuing_embed.") for key in student.state_dict())
    assert student.single_action_feature[0].in_features == teacher.config.hidden_dim
    assert report["role_token_columns_preserved"] is True
    assert report["role_dependent_friend_ordering_preserved"] is True
    for key in (
        "encoders.self.0.weight",
        "encoders.pursuers.0.weight",
        "transformer.layers.0.self_attn.in_proj_weight",
        "summary_fusion.0.weight",
    ):
        assert torch.equal(student.state_dict()[key], teacher.state_dict()[key])
    assert torch.equal(
        student.single_action_feature[0].weight,
        teacher.single_action_feature[0].weight[:, : teacher.config.hidden_dim],
    )


def test_c1_config_preserves_role_tokens_and_original_friend_ordering():
    config = load_config(str(C1))
    config = deep_update(config, config["tasks"]["voradj"])
    assert config["perception"]["include_is_pursuing"] is True
    assert config["perception"]["include_z_state"] is False
    assert config["iqn"]["include_is_pursuing"] is True
    assert config["iqn"]["pursuing_late_fusion"] is False
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=2026091801)
    observations = env.reset()
    explicit = env.get_policy_observations(True, False)
    for configured, roleful in zip(observations, explicit):
        if configured is None:
            assert roleful is None
            continue
        assert roleful is not None
        for key in configured:
            assert torch.equal(torch.as_tensor(configured[key]), torch.as_tensor(roleful[key]))
        assert configured["self"].shape == (9,)
        assert configured["pursuers"].shape[-1] == 7


def test_existing_b3_checkpoint_still_loads_without_late_fusion():
    student = CoCapIQN.load(str(B3_STUDENT), device="cpu").eval()
    assert student.config.include_is_pursuing is False
    assert student.config.include_z_state is True
    assert student.config.pursuing_late_fusion is False
    assert student.pursuing_embed is None


def test_frozen_role_dataset_is_strictly_matched_to_b3():
    manifest = json.loads((ARTIFACT / "dataset/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["episodes_per_scene"] == 20
    assert manifest["row_count"] == 29864
    assert len(manifest["shards"]) == 41
    assert manifest["b3_match"]["strict"] is True
    assert manifest["b3_match"]["all_metadata_exact"] is True
    assert manifest["b3_match"]["max_teacher_q_abs_delta"] == 0
    assert manifest["b3_match"]["max_reward_abs_delta"] == 0


def test_positive_control_records_case_c_and_complete_rollouts():
    audit = json.loads((ARTIFACT / "FINAL_AUDIT.json").read_text())
    assert audit["status"] == "complete"
    assert audit["decision"] == {
        "case": "CASE_C",
        "verdict": "CURRENT_BC_PIPELINE_FAILS_POSITIVE_CONTROL",
        "pass_rule": "all three formal safe-complete rates >= 0.85",
        "c0_postbc_pass": False,
        "c1_postbc_pass": False,
    }
    assert audit["arms"]["C0_step0"]["offline"]["overall"]["action_agreement"] > 0.9999
    assert audit["arms"]["C0_step0"]["offline"]["overall"]["q_regret"] == 0
    assert audit["arms"]["C0_postbc"]["offline"]["overall"]["action_agreement"] < 0.5
    summary = audit["rollout"]["summary"]
    for arm in ("C0_postbc", "C1_postbc"):
        assert all(summary[arm][scene]["episodes"] == 20 for scene in ("coverage", "capture", "mixed"))
        assert summary[arm]["coverage"]["ce_success_rate"] == 0
        assert summary[arm]["mixed"]["ce_success_rate"] == 0
    for arm in ("C0_step0", "C1_step0"):
        assert all(summary[arm][scene]["episodes"] == 3 for scene in ("coverage", "capture", "mixed"))
        assert all(summary[arm][scene]["safe_complete_rate"] == 1 for scene in ("coverage", "capture", "mixed"))
    assert audit["future_z_todo"]["implemented"] is False
    assert audit["next_stage_training_launched"] is False
