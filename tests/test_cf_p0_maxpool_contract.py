from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.continuous.formal_config import resolve_ladder_config, scene_config
from cocap_voradj.training.trainer import set_global_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiments/parallel_ce_legacy_voradj_20260809"
CF3 = CONFIG_DIR / "legacy_voradj_cf3_capture_first_local_support_full_4p1e1obs_100k_aw.yaml"
P1 = CONFIG_DIR / "legacy_voradj_p1_capture_first_local_support_maxpool_4p1e1obs_100k_aw.yaml"
PC0 = CONFIG_DIR / "legacy_voradj_pc0_cf3_actor_warmstart_postcapture300_4p1e1obs_100k_aw.yaml"



def _safe_cf_env(seed: int) -> VorAdjEnv:
    config = scene_config(resolve_ladder_config(CF3), "capture")
    set_global_config(config)
    env = VorAdjEnv(copy.deepcopy(config), seed=seed)
    env.reset()
    env.obstacles = []
    for index, pursuer in enumerate(env.pursuers):
        env._reset_robot(pursuer, np.asarray((25.0 + 15.0 * index, 30.0), dtype=float), theta=0.0)
        pursuer.deactivated = False
    env._reset_robot(env.evaders[0], np.asarray((15.0, 70.0), dtype=float), theta=0.0)
    env.evaders[0].velocity = np.zeros(2, dtype=float)
    env._invalidate_voronoi_cache()
    data = env._capture_voronoi_map()
    raw = env._raw_task_labels_from_map(data)
    env.last_task_labels = env._task_labels_from_map(data, update_effective=True, raw_labels=raw)
    return env


def test_min_active_keeps_three_and_two_but_terminates_at_one() -> None:
    for active_count, should_end in ((3, False), (2, False), (1, True)):
        env = _safe_cf_env(2026081310 + active_count)
        for pursuer in env.pursuers[active_count:]:
            pursuer.deactivated = True
        result = env.step([[0.0, 0.0]] * 4, [None])
        assert bool(result.dones[0]) is should_end


def test_k10_effective_role_matches_obs_reward_replay_and_hold() -> None:
    env = _safe_cf_env(2026081314)
    env._reset_robot(env.pursuers[0], np.asarray((28.0, 30.0), dtype=float), theta=0.0)
    env._reset_robot(env.pursuers[1], np.asarray((40.0, 30.0), dtype=float), theta=0.0)
    env._reset_robot(env.evaders[0], np.asarray((20.0, 30.0), dtype=float), theta=0.0)
    env._invalidate_voronoi_cache()
    data = env._capture_voronoi_map()
    raw = env._raw_task_labels_from_map(data)
    assert raw[0] == "capture"
    env.last_task_labels = env._task_labels_from_map(data, update_effective=True, raw_labels=raw)
    assert env._pursuing_release_counters[0] == 10
    moved = copy.deepcopy(data)
    moved["adjacency"][("pursuer", 0)].discard(("evader", 0))
    moved["adjacency"][("evader", 0)].discard(("pursuer", 0))
    env._capture_voronoi_map = lambda *_args, **_kwargs: moved
    moved_raw = env._raw_task_labels_from_map(moved)
    assert moved_raw[0] == "coverage"
    assert env._effective_task_labels(moved_raw)[0] == "capture"
    obs = env.get_observations()[0]
    assert obs is not None and obs["self"][8] == 1.0
    result = env.step([[0.0, 0.0]] * 4, [None])
    meta = result.infos[0]["replay_metadata"]
    assert meta["raw_task_label"] == "coverage"
    assert meta["task_label"] == "capture"
    assert meta["effective_pursuing"] is True
    assert meta["reward_role"] == "capture"
    assert meta["reward_capture"] != 0.0
    assert meta["reward_coverage"] == 0.0
    expected_hold = sum(info["replay_metadata"]["reward_role"] in {"support", "coverage"} for info in result.infos)
    assert env.last_reward_terms["hold_reward_eligible_count"] == expected_hold


def test_p0_resume_fork_accepts_only_two_contract_changes(tmp_path: Path) -> None:
    from tools.run_continuous_ctde_training import _validate_p0_semantics_resume_fork
    target = resolve_ladder_config(CF3)
    source = copy.deepcopy(target)
    source["reward"]["min_active_pursuers"] = 4
    source["reward"]["coverage_ce_min_active_pursuers"] = 4
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "effective_config.yaml").write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    source_manifest = {"algorithm": "masac_ctde", "config": "old.yaml", "config_hash": "old", "effective_config_hashes": {"capture": "old"}, "implementation_hash": "old"}
    target_manifest = {**source_manifest, "config": "new.yaml", "config_hash": "new", "effective_config_hashes": {"capture": "new"}, "implementation_hash": "new"}
    audit = _validate_p0_semantics_resume_fork(source_manifest, target_manifest, target, bundle / "trainer.pt")
    assert audit["kind"] == "p0_semantics_resume_fork"
    assert {row["path"] for row in audit["config_diffs"]} == {"reward.min_active_pursuers", "reward.coverage_ce_min_active_pursuers"}


def test_p1_local_max_is_cf3_plus_max_pool_only() -> None:
    from tools.run_continuous_ctde_training import _make_trainer
    cf3 = resolve_ladder_config(CF3)
    p1 = resolve_ladder_config(P1)
    assert (p1["actor"]["hidden_dim"], p1["actor"]["num_heads"], p1["actor"]["num_layers"]) == (256, 8, 4)
    assert p1["actor"]["context_pooling"] == "mean_max"
    trainer = _make_trainer(p1, "cpu")
    assert trainer.actor.context_pooling == "mean_max"
    assert trainer.actor.policy[0].in_features == 3 * 256
    for config in (cf3, p1):
        config.pop("run_name", None)
        config.pop("seed", None)
        config.pop("experiment_metadata", None)
    p1["actor"].pop("context_pooling")
    assert p1 == cf3


def test_cf2_to_p1_and_cf3_extension_commands_are_strict() -> None:
    from tools.supervise_cf2_stop100_then_p1_maxpool import p1_command
    from tools.supervise_cf3_extend200 import continuation_command

    smoke = p1_command(smoke=True)
    formal = p1_command(smoke=False)
    assert smoke[smoke.index("--total-steps") + 1] == "32"
    assert formal[formal.index("--total-steps") + 1] == "100000"
    assert formal[formal.index("--device") + 1] == "cuda:0"
    assert "--resume-checkpoint" not in formal
    assert "--resume-replay" not in formal

    cf3 = continuation_command()
    assert cf3[cf3.index("--resume-step") + 1] == "100000"
    assert cf3[cf3.index("--total-steps") + 1] == "200000"
    assert cf3[cf3.index("--device") + 1] == "cuda:1"


def test_p1_gate_extension_remains_on_gpu1_after_swap() -> None:
    from tools.supervise_p1_maxpool_extension import continuation_command

    command = continuation_command()
    assert command[command.index("--device") + 1] == "cuda:1"


def test_p1_100k_gate_requires_sustained_positive_geometry() -> None:
    from tools.supervise_p1_maxpool_extension import evaluate_gate

    weak = [
        {"step": step, "normal_capture_count": 0, "max_num_in_ring": 1,
         "fraction_steps_2plus_in_ring": 0.0, "fraction_steps_3plus_in_ring": 0.0}
        for step in range(76000, 100001, 1000)
    ]
    cf3 = copy.deepcopy(weak)
    for step in (78000, 80000, 82000, 84000):
        row = next(row for row in cf3 if row["step"] == step)
        row["fraction_steps_2plus_in_ring"] = 0.001
    for step in (80000, 84000):
        row = next(row for row in cf3 if row["step"] == step)
        row["max_num_in_ring"] = 3
        row["fraction_steps_3plus_in_ring"] = 0.001

    assert evaluate_gate(weak, cf3)["passed"] is False

    repeated_two = copy.deepcopy(weak)
    for step in (80000, 90000, 100000):
        next(row for row in repeated_two if row["step"] == step)["fraction_steps_2plus_in_ring"] = 0.001
    # Geometry without a normal capture is now explicitly insufficient.
    assert evaluate_gate(repeated_two, cf3)["passed"] is False

    capture = copy.deepcopy(weak)
    capture[-1]["normal_capture_count"] = 1
    capture[-1]["distinct_normal_capture_episodes"] = 1
    # One capture without matched repeated geometry remains a failure.
    assert evaluate_gate(capture, cf3)["passed"] is False

    matched = copy.deepcopy(capture)
    for step in (78000, 80000, 82000):
        next(row for row in matched if row["step"] == step)["fraction_steps_2plus_in_ring"] = 0.001
    for step in (80000, 82000):
        row = next(row for row in matched if row["step"] == step)
        row["max_num_in_ring"] = 3
        row["fraction_steps_3plus_in_ring"] = 0.001
    assert evaluate_gate(matched, cf3)["passed"] is True

    multiple_capture = copy.deepcopy(weak)
    for row in multiple_capture[-2:]:
        row["normal_capture_count"] = 1
        row["distinct_normal_capture_episodes"] = 1
    assert evaluate_gate(multiple_capture, cf3)["passed"] is True


def test_pc0_transition_semantics_forbid_cf3_replay_inheritance() -> None:
    from tools.prepare_pc0_actor_warmstart_bundle import transition_semantics_audit

    audit = transition_semantics_audit(CF3, PC0)
    assert audit["source_capture_transition_terminal"] is True
    assert audit["target_capture_transition_terminal"] is False
    assert audit["target_post_capture_window_steps"] == 300
    assert audit["terminal_semantics_changed"] is True
    assert audit["bootstrap_target_semantics_changed"] is True
    assert audit["old_replay_inheritance_allowed"] is False
    assert audit["required_initialization"].startswith("actor_only_warmstart")


def test_pc0_is_cf3_plus_post_capture_window_only() -> None:
    cf3 = resolve_ladder_config(CF3)
    pc0 = resolve_ladder_config(PC0)
    assert scene_config(cf3, "capture")["voradj"]["capture_episode_ends_on_capture"] is True
    assert scene_config(pc0, "capture")["voradj"]["capture_episode_ends_on_capture"] is False
    assert scene_config(pc0, "capture")["reward"]["post_capture_coverage_window_steps"] == 300
    for config in (cf3, pc0):
        config.pop("run_name", None)
        config.pop("seed", None)
        config.pop("device", None)
        config.pop("experiment_metadata", None)
    pc0["reward"]["post_capture_coverage_window_steps"] = cf3["reward"]["post_capture_coverage_window_steps"]
    pc0["tasks"]["capture"].pop("reward", None)
    pc0["tasks"]["capture"]["voradj"] = copy.deepcopy(cf3["tasks"]["capture"]["voradj"])
    assert pc0 == cf3


def test_50k_gpu_swap_commands_keep_line_state_separate() -> None:
    from tools.supervise_cf3_p1_gpu_swap_50k import LINES, continuation_command

    cf3 = continuation_command(LINES["cf3"])
    p1 = continuation_command(LINES["p1"])
    assert cf3[cf3.index("--device") + 1] == "cuda:0"
    assert cf3[cf3.index("--total-steps") + 1] == "200000"
    assert p1[p1.index("--device") + 1] == "cuda:1"
    assert p1[p1.index("--total-steps") + 1] == "100000"
    for name, command in (("cf3", cf3), ("p1", p1)):
        checkpoint = command[command.index("--resume-checkpoint") + 1]
        replay = command[command.index("--resume-replay") + 1]
        assert f"/{LINES[name]['tag']}/" in checkpoint
        assert f"/{LINES[name]['tag']}/" in replay
        assert "resume_frozen_gpu_swap_step_000050000" in checkpoint
        assert command[command.index("--resume-step") + 1] == "50000"
    assert LINES["p1"]["tag"] not in cf3[cf3.index("--resume-checkpoint") + 1]
    assert LINES["cf3"]["tag"] not in p1[p1.index("--resume-checkpoint") + 1]


def test_cf3_stable_gate_needs_three_normal_and_not_mainly_stationary() -> None:
    from tools.supervise_cf3_stable_gate_pc0 import gate_summary

    rows = [
        {"step": 50000 + 1000 * index, "distinct_normal_capture_episodes": normal,
         "distinct_stationary_capture_episodes": stationary,
         "fraction_steps_2plus_in_ring": 0.01, "fraction_steps_3plus_in_ring": 0.001,
         "max_2plus_ring_hold_steps": 4, "max_3plus_ring_hold_steps": 2}
        for index, (normal, stationary) in enumerate(((1, 0), (1, 0), (0, 1)), start=1)
    ]
    assert gate_summary(rows)["training_gate_passed"] is False
    rows.append({"step": 54000, "distinct_normal_capture_episodes": 1,
                 "distinct_stationary_capture_episodes": 0,
                 "fraction_steps_2plus_in_ring": 0.0, "fraction_steps_3plus_in_ring": 0.0})
    summary = gate_summary(rows)
    assert summary["distinct_normal_capture_episodes"] == 3
    assert summary["distinct_stationary_capture_episodes"] == 1
    assert summary["training_gate_passed"] is True
