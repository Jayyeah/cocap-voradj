from __future__ import annotations

from tools.run_crms_baseline_regression import CORE_FIELDS


def test_regression_compares_success_and_timing_contract_fields() -> None:
    assert "capture_success_bool" in CORE_FIELDS
    assert "coverage_success_bool" in CORE_FIELDS
    assert "collision_event" in CORE_FIELDS
    assert "steps" in CORE_FIELDS
