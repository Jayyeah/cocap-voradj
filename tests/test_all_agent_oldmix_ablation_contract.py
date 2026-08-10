from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from cocap_voradj.models.continuous.box_actor import BoxActorConfig
from cocap_voradj.models.continuous.central_attention_critic import CentralCriticConfig
from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LocalEntityTokenEncoderConfig,
)
from cocap_voradj.training.continuous.central_sac import (
    CentralSACConfig,
    CentralSACTrainer,
)
from cocap_voradj.training.continuous.formal_config import (
    resolve_ladder_config,
    training_mode_contract,
)
from cocap_voradj.training.continuous.joint_replay import (
    JointReplayBuffer,
    UniformJointReplaySampler,
)
from tools.run_continuous_ctde_training import (
    _link_bundle_tree_atomic,
    _make_trainer,
    _manifest,
    _runtime_state,
    _save_checkpoint_bundle,
    _save_rolling_resume_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_A = ROOT / (
    "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_oldmix_4p1e1obs_200k_aw.yaml"
)
BASELINE_B = ROOT / (
    "configs/experiments/parallel_ce_legacy_voradj_20260809/"
    "legacy_voradj_oldmix_4p1e1obs_200k_aw_allagent.yaml"
)


def _without_ablation_fields(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("run_name", None)
    training = result["training"]
    for key in (
        "optimizer_unit",
        "replay_sampling",
        "focal_training",
        "focal_quota",
    ):
        training.pop(key, None)
    result.pop("experiment_metadata", None)
    return result


def _small_replay(count: int = 160, agents: int = 4) -> JointReplayBuffer:
    replay = JointReplayBuffer(capacity=max(count, 8), max_agents=agents, seed=17)
    for index in range(count):
        category = index % 3
        phase = ("pre_capture", "post_capture", "pure_coverage")[category]
        scene = ("mixed_crms", "mixed_crms", "pure_ce")[category]
        role = 1 if category == 0 else 3
        roles = np.full(agents, role, dtype=np.uint8)
        active = np.ones(agents, dtype=bool)
        local = {"x": np.full((agents, 1), index, dtype=np.float32)}
        replay.add(
            local_obs=local,
            next_local_obs={"x": local["x"] + 1.0},
            global_state={"g": np.asarray([index], dtype=np.float32)},
            next_global_state={"g": np.asarray([index + 1], dtype=np.float32)},
            actions=np.zeros((agents, 2), dtype=np.float32),
            rewards=np.zeros(agents, dtype=np.float32),
            active_mask=active,
            terminated=np.zeros(agents, dtype=bool),
            truncated=np.zeros(agents, dtype=bool),
            metadata={
                "phase": phase,
                "scene": scene,
                "origin": "test",
                "regime": "active_target" if category == 0 else "coverage_only",
                "coverage_only": category != 0,
                "active_target": category == 0,
                "event_ids": [],
            },
            agent_role_id=roles,
        )
    return replay


def _central_batch(batch_size: int = 3, agents: int = 6) -> dict[str, object]:
    torch.manual_seed(2026081001)
    pursuer_slots, evader_slots, obstacle_slots = agents, 2, 1
    token_count = 1 + pursuer_slots + evader_slots + obstacle_slots
    local = {
        "self": torch.randn(batch_size, agents, 9),
        "pursuers": torch.randn(batch_size, agents, pursuer_slots, 7),
        "evaders": torch.randn(batch_size, agents, evader_slots, 7),
        "obstacles": torch.randn(batch_size, agents, obstacle_slots, 5),
        "masks": torch.ones(batch_size, agents, token_count, dtype=torch.bool),
        "types": torch.zeros(batch_size, agents, token_count, dtype=torch.long),
    }
    local["types"][:, :, 1 : 1 + pursuer_slots] = 1
    local["types"][:, :, 1 + pursuer_slots : 1 + pursuer_slots + evader_slots] = 2
    local["types"][:, :, 1 + pursuer_slots + evader_slots :] = 3
    active = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
        ],
        dtype=torch.bool,
    )[:batch_size, :agents]
    central = {
        "self": torch.randn(batch_size, agents, 9),
        "pursuers": torch.randn(batch_size, agents, agents, 7),
        "evaders": torch.randn(batch_size, evader_slots, 7),
        "obstacles": torch.randn(batch_size, obstacle_slots, 5),
        "pursuer_mask": active[:, :, None] & active[:, None, :],
        "evader_mask": torch.ones(batch_size, evader_slots, dtype=torch.bool),
        "obstacle_mask": torch.ones(batch_size, obstacle_slots, dtype=torch.bool),
        "active_mask": active,
    }
    next_local = {
        key: value.clone() if not value.is_floating_point() else value + 0.01
        for key, value in local.items()
    }
    next_central = {
        key: value.clone() if not value.is_floating_point() else value + 0.01
        for key, value in central.items()
    }
    return {
        "local_obs": local,
        "next_local_obs": next_local,
        "global_state": central,
        "next_global_state": next_central,
        "actions": torch.randn(batch_size, agents, 2).clamp(-0.4, 0.4),
        "rewards": torch.randn(batch_size, agents),
        "active_mask": active,
        "terminated": torch.zeros(batch_size, agents, dtype=torch.bool),
        "truncated": torch.zeros(batch_size, agents, dtype=torch.bool),
    }


def _small_trainer(agents: int = 6) -> CentralSACTrainer:
    return CentralSACTrainer(
        encoder_config=LocalEntityTokenEncoderConfig(
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            max_pursuers=agents,
            max_evaders=2,
            max_obstacles=1,
        ),
        actor_config=BoxActorConfig(hidden_dim=16),
        critic_config=CentralCriticConfig(
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            max_agents=agents,
            max_evaders=2,
            max_obstacles=1,
        ),
        config=CentralSACConfig(hidden_dim=16, grad_clip_norm=0.5),
        device="cpu",
        action_mode="aw",
    )


def test_effective_config_changes_only_training_unit() -> None:
    baseline_a = resolve_ladder_config(BASELINE_A)
    baseline_b = resolve_ladder_config(BASELINE_B)
    assert training_mode_contract(baseline_a) == {
        "optimizer_unit": "focal_agent_item",
        "replay_sampling": "focal_role_balanced",
        "focal_training": True,
    }
    assert training_mode_contract(baseline_b) == {
        "optimizer_unit": "joint_transition_all_active_agents",
        "replay_sampling": "uniform_joint",
        "focal_training": False,
    }
    assert _without_ablation_fields(baseline_a) == _without_ablation_fields(baseline_b)
    assert baseline_b["seed"] == baseline_a["seed"] == 2026080902
    assert baseline_b["training"]["scene_cycle"] == ["mixed_crms", "pure_ce"]
    assert baseline_b["training"]["periodic_checkpoint_replay_mode"] == "rolling_latest"


def test_manifest_strictly_separates_focal_and_all_agent() -> None:
    baseline_a = resolve_ladder_config(BASELINE_A)
    baseline_b = resolve_ladder_config(BASELINE_B)
    trainer = _make_trainer(baseline_b, "cpu")
    scenes = {"capture": "a", "pure_ce": "b", "mixed_crms": "c"}
    manifest_a = _manifest(
        baseline_a,
        2026080902,
        "baseline-a",
        trainer,
        scenes,
        config_path=str(BASELINE_A),
    )
    manifest_b = _manifest(
        baseline_b,
        2026080902,
        "baseline-b",
        trainer,
        scenes,
        config_path=str(BASELINE_B),
    )
    assert manifest_a["optimizer_unit"] == "focal_agent_item"
    assert manifest_a["replay_sampling"] == "focal_role_balanced"
    assert manifest_a["focal_training"] is True
    assert "focal_sampler_numpy" in manifest_a["resume_contract"]["rng"]
    assert manifest_b["optimizer_unit"] == "joint_transition_all_active_agents"
    assert manifest_b["replay_sampling"] == "uniform_joint"
    assert manifest_b["focal_training"] is False
    assert manifest_b["scene_cycle"] == ["mixed_crms", "pure_ce"]
    assert "focal_sampler_numpy" not in manifest_b["resume_contract"]["rng"]
    assert manifest_a != manifest_b


def test_uniform_sampler_returns_joint_batch_without_focal_optimizer_fields() -> None:
    replay = _small_replay()
    batch = replay.sample(128, sampler=UniformJointReplaySampler())
    assert len(batch["transition_ids"]) == 128
    assert len(set(batch["transition_ids"].tolist())) == 128
    for forbidden in (
        "focal_agent_id",
        "focal_reward",
        "focal_terminated",
        "focal_truncated",
        "focal_mask",
        "bucket_id",
    ):
        assert forbidden not in batch
    stats = batch["sampling_stats"]
    assert stats["sampler"] == "uniform_joint"
    assert stats["unique_joint_transitions"] == 128
    assert stats["replacement_count"] == 0
    assert stats["role_metadata_used_for_sampling"] is False
    assert stats["active_agent_loss_terms"] == 512
    assert sum(stats["sampled_phase_counts"].values()) == 128
    assert sum(stats["sampled_active_agent_role_counts"].values()) == 512


def test_active_slot_actor_q_matches_all_slot_reference() -> None:
    trainer = _small_trainer()
    batch = _central_batch()
    central = trainer._central_batch(batch, "global_state")
    active = batch["active_mask"]
    actions_reference = torch.randn(3, 6, 2, requires_grad=True)
    actions_optimized = actions_reference.detach().clone().requires_grad_(True)
    reference = trainer._focal_actor_q_reference(central, actions_reference, active)
    optimized = trainer._focal_actor_q(central, actions_optimized, active)
    assert float((reference - optimized).abs().max()) < 1e-6
    reference_loss = (reference * active).sum() / active.sum()
    optimized_loss = (optimized * active).sum() / active.sum()
    grad_reference = torch.autograd.grad(reference_loss, actions_reference)[0]
    grad_optimized = torch.autograd.grad(optimized_loss, actions_optimized)[0]
    assert torch.allclose(grad_reference, grad_optimized, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(grad_optimized[:, 4:]) == 0


def test_all_agent_losses_and_actor_gradients_match_reference() -> None:
    reference = _small_trainer()
    optimized = copy.deepcopy(reference)
    reference._focal_actor_q = reference._focal_actor_q_reference
    batch = _central_batch()
    rng_state = torch.get_rng_state()
    torch.set_rng_state(rng_state)
    reference_metrics = reference.update(batch)
    torch.set_rng_state(rng_state)
    optimized_metrics = optimized.update(batch)
    for key in ("critic_loss", "actor_loss", "alpha_loss"):
        assert abs(reference_metrics[key] - optimized_metrics[key]) < 1e-6
    for reference_parameter, optimized_parameter in zip(
        reference.actor.parameters(),
        optimized.actor.parameters(),
    ):
        assert torch.allclose(
            reference_parameter,
            optimized_parameter,
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            reference_parameter.grad,
            optimized_parameter.grad,
            atol=1e-6,
            rtol=1e-6,
        )
    assert optimized_metrics["active_agent_loss_terms_per_update"] == 11.0
    assert optimized_metrics["active_actor_q_slots"] == 4.0
    assert optimized_metrics["active_agent_loss_terms_per_second"] > 0.0
    assert optimized_metrics["updates_per_second"] > 0.0


def test_all_agent_runtime_state_has_no_focal_sampler_input() -> None:
    state = _runtime_state(
        transition_count=25000,
        update_count=10,
        scene_index=4,
        current_scene="pure_ce",
        recovery_pool=__import__("collections").deque(),
        metrics_history=[],
        runner_rng=np.random.default_rng(1),
        focal_sampler=None,
        scene_counts={"mixed_crms": 1, "pure_ce": 1},
        origin_counts={"map_random": 1},
        sampling_stats={"sampler": "uniform_joint"},
        all_finite=True,
        update_metrics_tail=[],
        training_mode={
            "optimizer_unit": "joint_transition_all_active_agents",
            "replay_sampling": "uniform_joint",
            "focal_training": False,
        },
    )
    assert "focal_sampler_state" not in state
    assert state["training_mode"]["focal_training"] is False


def test_rolling_latest_replaces_replay_and_final_is_full(tmp_path: Path) -> None:
    config = resolve_ladder_config(BASELINE_B)
    trainer = _small_trainer(agents=4)
    replay = _small_replay(count=8, agents=4)
    manifest = {"mode": "all-agent", "seed": 2026080902}

    for step in (25000, 50000):
        runtime = {"transition_count": step}
        milestone = _save_checkpoint_bundle(
            tmp_path,
            step,
            config,
            manifest,
            trainer,
            replay,
            runtime,
            [],
            {},
            include_replay=False,
        )
        _save_rolling_resume_bundle(
            tmp_path,
            step,
            config,
            manifest,
            trainer,
            replay,
            runtime,
            [],
            {},
        )
        assert not (milestone / "replay.pkl").exists()
        storage = json.loads((milestone / "checkpoint_storage.json").read_text())
        assert storage["kind"] == "evaluation_model_only"
        assert storage["contains_replay"] is False

    replay_paths = list(tmp_path.glob("**/replay.pkl"))
    assert replay_paths == [tmp_path / "resume_latest" / "replay.pkl"]
    resume_storage = json.loads(
        (tmp_path / "resume_latest" / "checkpoint_storage.json").read_text()
    )
    assert resume_storage["step"] == 50000
    assert resume_storage["kind"] == "rolling_latest_full_resume"

    final_bundle = _save_checkpoint_bundle(
        tmp_path,
        200000,
        config,
        manifest,
        trainer,
        replay,
        {"transition_count": 200000},
        [],
        {},
        include_replay=True,
    )
    linked = _link_bundle_tree_atomic(final_bundle, tmp_path / "resume_latest")
    assert (final_bundle / "replay.pkl").is_file()
    assert (linked / "replay.pkl").samefile(final_bundle / "replay.pkl")
