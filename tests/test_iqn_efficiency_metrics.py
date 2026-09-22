from __future__ import annotations

import pytest

from tools.iqn_token_matched_20260919 import efficiency_timing_summary


def test_efficiency_summary_excludes_failures_and_reports_censoring() -> None:
    rows = [
        {"success": True, "steps": 10, "collision": False, "boundary": False},
        {"success": True, "steps": 20, "collision": False, "boundary": False},
        {"success": False, "steps": None, "collision": True, "boundary": False},
        {"success": False, "steps": None, "collision": False, "boundary": False},
    ]

    result = efficiency_timing_summary(rows, success_key="success", value_key="steps")

    assert result["episodes"] == 4
    assert result["success_n"] == 2
    assert result["failure_n"] == 1
    assert result["censored_n"] == 1
    assert result["unsuccessful_n"] == 2
    assert result["collision_n"] == 1
    assert result["mean"] == pytest.approx(15.0)
    assert result["median"] == pytest.approx(15.0)
    assert result["p90"] == pytest.approx(19.0)


def test_efficiency_summary_keeps_success_count_when_time_is_missing() -> None:
    rows = [
        {"success": True, "steps": 12, "collision": False, "boundary": False},
        {"success": True, "steps": None, "collision": False, "boundary": False},
    ]

    result = efficiency_timing_summary(rows, success_key="success", value_key="steps")

    assert result["success_n"] == 2
    assert result["missing_time_n"] == 1
    assert result["n"] == 1
    assert result["mean"] == pytest.approx(12.0)

