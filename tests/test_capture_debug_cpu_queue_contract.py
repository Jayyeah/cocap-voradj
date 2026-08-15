from __future__ import annotations

import subprocess
from pathlib import Path

from tools import run_capture_debug_cpu_queue as queue


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_cpu_queue_stage_order_and_isolation() -> None:
    stages = queue.planned_stages()
    assert [stage.name for stage in stages] == [
        "preflight_cpu_tests",
        "a1_legacy_t0_20",
        "a1_fixed_t0_20",
        "a2_fixed_t0_t025_t1_20",
        "a3_iqn_fixed_20",
    ]
    for stage in stages:
        queue.command_invariants(stage.command)
        assert not any(str(token).lower().startswith("cuda:") for token in stage.command)
    assert queue.cpu_environment()["CUDA_VISIBLE_DEVICES"] == ""


def test_a1_and_a2_commands_are_canonical_paired_probes() -> None:
    legacy = queue.a1_command(fixed=False)
    fixed = queue.a1_command(fixed=True)
    a2 = queue.a2_command()
    assert _value(legacy, "--device") == _value(fixed, "--device") == "cpu"
    assert _value(legacy, "--episodes") == _value(fixed, "--episodes") == "20"
    assert _value(legacy, "--collision-semantics") == "legacy_end_step"
    assert _value(fixed, "--collision-semantics") == "synchronized_swept_v1"
    assert legacy[legacy.index("--temperatures") + 1 :] == [
        "0", "--collision-semantics", "legacy_end_step",
    ]
    assert a2[a2.index("--temperatures") + 1 :] == [
        "0", "0.25", "1", "--collision-semantics", "synchronized_swept_v1",
    ]
    assert Path(_value(legacy, "--output")) == queue.A1_LEGACY
    assert Path(_value(fixed, "--output")) == queue.A1_FIXED


def test_a3_conditional_confirmation_and_a4_contract() -> None:
    low = {
        "kind": "iqn_corrected_cf3_capture_eval", "device": "cpu",
        "checkpoint_hash_verified": True, "checkpoint_sha256": queue.IQN_SHA256,
        "collision_semantics": "synchronized_swept_v1",
        "summary": {"episodes": 20, "normal_capture_count": 3},
        "records": [{}] * 20,
    }
    high = {**low, "summary": {"episodes": 20, "normal_capture_count": 4}}
    assert queue.a3_needs_confirmation(low) is False
    assert queue.a3_needs_confirmation(high) is True

    legacy = queue.a4_command(fixed=False)
    fixed = queue.a4_command(fixed=True)
    assert _value(legacy, "--episodes") == "300"
    assert _value(fixed, "--episodes") == "100"
    for command in (legacy, fixed):
        assert _value(command, "--temperature") == "1"
        assert _value(command, "--tail-steps") == "50"
        assert _value(command, "--max-cases-per-category") == "5"
        assert _value(command, "--device") == "cpu"


def test_detached_launcher_is_valid_shell_and_cpu_only() -> None:
    launcher = queue.ROOT / "tools/launch_capture_debug_cpu_queue.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=" in text
    assert "run_capture_debug_cpu_queue.py" in text
    assert "run_continuous_ctde_training.py" not in text
    assert "--device cuda" not in text.lower()
