from tools import evaluate_vxy_stage2_cross_retention_20260903 as audit


def test_stage2_contract_is_fail_closed() -> None:
    config = {
        "env": {
            "num_pursuers": 8,
            "num_evaders": 2,
            "num_obstacles": 2,
            "action_mode": "vxy9",
            "collision_semantics": "legacy_end_step",
        },
        "action": {"mode": "discrete_desired_velocity_2d_body"},
        "iqn": {"action_quantile_samples": 32},
    }
    assert all(audit.validate_stage2_contract(config).values())
    config["env"]["collision_semantics"] = "synchronized_swept_v1"
    try:
        audit.validate_stage2_contract(config)
    except ValueError as error:
        assert "legacy_collision" in str(error)
    else:
        raise AssertionError("collision contract drift was accepted")


def test_ring_timing_helpers() -> None:
    values = [0, 2, 2, 1, 3, 3, 3, 0]
    assert audit.first_step(values, 2) == 2
    assert audit.first_step(values, 3) == 5
    assert audit.max_hold(values, 2) == 3
    assert audit.max_hold(values, 3) == 3


def test_paired_comparison_keeps_seed_alignment() -> None:
    baseline = [
        {"seed": 10, "captured": False, "normal_capture": False, "stationary_capture": False, "collision": True, "length": 100},
        {"seed": 11, "captured": True, "normal_capture": True, "stationary_capture": False, "collision": False, "length": 50},
    ]
    candidate = [
        {"seed": 11, "captured": True, "normal_capture": True, "stationary_capture": False, "collision": False, "length": 40},
        {"seed": 10, "captured": True, "normal_capture": True, "stationary_capture": False, "collision": False, "length": 80},
    ]
    result = audit.paired_comparison(baseline, candidate, "capture")
    assert result["seeds"] == [10, 11]
    assert result["binary"]["captured"]["candidate_only"] == 1
    assert result["binary"]["captured"]["candidate_minus_baseline"] == 0.5
    assert result["continuous"]["length"]["candidate_minus_baseline_mean"] == -15.0
