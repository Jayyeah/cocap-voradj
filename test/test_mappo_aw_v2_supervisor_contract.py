from __future__ import annotations

import json
from pathlib import Path

from tools import supervise_mappo_aw_v2_20260831 as supervisor


def test_aw_v2_contract_constants_and_independent_paths() -> None:
    assert supervisor.GPU_INDEX == 1
    assert supervisor.TOTAL_STEPS == 400_000
    assert supervisor.CHECKPOINT_INTERVAL == 25_000
    assert supervisor.EVAL_EPISODES == 20
    assert supervisor.MIN_FREE_GPU_MIB == 26_000
    assert supervisor.ACTIVE_MIN_FREE_GPU_MIB == 2_000
    assert supervisor.CLIP_STREAK_LIMIT == 5
    assert supervisor.SATURATION_STREAK_LIMIT == 5
    assert supervisor.PRE_TANH_STREAK_LIMIT == 5
    assert supervisor.VALUE_STREAK_LIMIT == 3
    assert "mappo_aw_v2_20260831" in str(supervisor.CONFIG_ROOT)
    assert "2026-08-31_mappo_aw_v2" in str(supervisor.ARTIFACT_ROOT)
    assert "mappo_aw_v2" in supervisor.SESSION_PREFIX


def test_health_warns_and_pauses_on_each_sustained_continuous_policy_failure() -> None:
    assert supervisor.health([{"approx_kl_max": 0.06}])["warn"] is True
    assert supervisor.health([{"clip_fraction": 0.31}])["warn"] is True
    assert supervisor.health([{"post_tanh_saturation_ratio": 0.51}])["warn"] is True

    assert supervisor.health([{"approx_kl_max": 0.11}] * 2)["critical"] is False
    kl = supervisor.health([{"approx_kl_max": 0.11}] * 3)
    assert kl["critical"] is True
    assert kl["critical_kl_streak"] == 3

    clip = supervisor.health([{"clip_fraction": 0.31}] * 5)
    assert clip["critical"] is True
    assert clip["critical_clip_streak"] == 5

    saturation = supervisor.health([{"post_tanh_saturation_ratio": 0.96}] * 5)
    assert saturation["critical"] is True
    assert saturation["critical_saturation_streak"] == 5

    pre_tanh = supervisor.health([{"pre_tanh_mean_a": 5.01}] * 5)
    assert pre_tanh["critical"] is True
    assert pre_tanh["critical_pre_tanh_mean_streak"] == 5

    value = supervisor.health([{"value_loss": 1001.0}] * 3)
    assert value["critical"] is True
    assert value["critical_value_streak"] == 3
    value_grad = supervisor.health([{"value_grad_norm": 100.1}] * 3)
    assert value_grad["critical"] is True


def test_health_checks_nested_finite_values() -> None:
    assert supervisor.health([{"ppo": {"entropy": 1.0, "value_loss": 2.0}}])["finite"] is True
    result = supervisor.health([{"ppo": {"entropy": float("nan")}}])
    assert result["finite"] is False
    assert result["critical"] is True
    assert "nonfinite_metric" in result["critical_reasons"]


def test_gpu_foreign_process_audit_never_reclassifies_other_commands() -> None:
    own = {
        "pid": 101,
        "owner": "yjq",
        "cmdline": "python tools/run_small_step_ac_migration.py --config configs/experiments/mappo_aw_v2_20260831/seed1.yaml",
        "used_memory_mib": 1000,
    }
    foreign = {
        "pid": 202,
        "owner": "xrq",
        "cmdline": "python -m limo_adapter.server_omnivla_full --port 8890",
        "used_memory_mib": 17029,
    }
    assert supervisor.is_aw_v2_process(own) is True
    assert supervisor.is_aw_v2_process(foreign) is False
    assert supervisor.foreign_gpu_processes([own, foreign]) == [foreign]
    assert supervisor.aw_v2_gpu_processes([own, foreign]) == [own]


def test_resource_gate_waits_for_vram_and_foreign_compute_process() -> None:
    foreign = {"pid": 202, "owner": "xrq", "cmdline": "foreign", "used_memory_mib": 17029}
    snapshot = {
        "gpu": {"memory_free_mib": 25_999},
        "disk_free_gib": 50.0,
        "compute_process_query_ok": True,
        "foreign_gpu_processes": [foreign],
    }
    blockers = supervisor.resource_blockers(snapshot)
    assert "free_vram_below_26g" in blockers
    assert "foreign_gpu_compute_process" in blockers
    shared_blockers = supervisor.resource_blockers(snapshot, block_foreign_processes=False)
    assert "free_vram_below_26g" in shared_blockers
    assert "foreign_gpu_compute_process" not in shared_blockers
    assert supervisor.gpu_free_mib(snapshot) == 25_999


def test_flock_prevents_a_second_supervisor_instance(tmp_path: Path) -> None:
    path = tmp_path / "supervisor" / "gpu1.lock"
    first = supervisor.acquire_lock(path)
    assert first is not None
    try:
        assert supervisor.acquire_lock(path) is None
    finally:
        import fcntl

        fcntl.flock(first.fileno(), fcntl.LOCK_UN)
        first.close()


def _write_eval(run: Path, step: int, capture: float, collision: float = 0.2) -> None:
    (run / "evaluations").mkdir(parents=True, exist_ok=True)
    (run / "evaluations" / f"step_{step:09d}.json").write_text(
        json.dumps(
            {
                "step": step,
                "deterministic": {
                    "capture": {
                        "capture_rate": capture,
                        "normal_capture_rate": capture,
                        "stationary_capture_rate": 0.0,
                        "collision_rate": collision,
                        "visited_2plus_ring_rate": 0.3,
                        "visited_3plus_ring_rate": 0.1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_final_gate_has_three_explicit_routes(tmp_path: Path, monkeypatch) -> None:
    seeds = []
    for index, capture in enumerate((0.20, 0.15, 0.25), start=1):
        run = tmp_path / f"seed{index}"
        _write_eval(run, 400_000, capture)
        (run / "learning_metrics.jsonl").write_text(
            json.dumps({"step": 400_000, "approx_kl_max": 0.01, "clip_fraction": 0.05, "value_loss": 1.0}) + "\n",
            encoding="utf-8",
        )
        seeds.append({"index": index, "run": run})
    monkeypatch.setattr(supervisor, "SEEDS", tuple(seeds))
    assert supervisor.final_gate()["decision"] == "CONTINUOUS_AC_ROUTE_ESTABLISHED"

    for item in seeds:
        _write_eval(item["run"], 400_000, 0.0)
    assert supervisor.final_gate()["decision"] == "HEALTHY_DISCRETE_TO_CONTINUOUS_GAP"

    seeds[0]["run"].joinpath("learning_metrics.jsonl").write_text(
        json.dumps({"approx_kl_max": 0.11}) + "\n" * 3,
        encoding="utf-8",
    )
    # A nonfinite row is a direct implementation-health failure and routes to
    # the third branch independently of the evaluation result.
    seeds[1]["run"].joinpath("learning_metrics.jsonl").write_text(
        json.dumps({"value_loss": float("inf")}) + "\n",
        encoding="utf-8",
    )
    assert supervisor.final_gate()["decision"] == "UNHEALTHY_CONTINUOUS_POLICY_PPO_IMPLEMENTATION"
