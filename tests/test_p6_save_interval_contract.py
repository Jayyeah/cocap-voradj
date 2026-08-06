"""Contract tests for the P6 checkpoint/replay persistence interval."""
from __future__ import annotations

from tools.run_continuous_p6_screening import DEFAULT_SAVE_INTERVAL, _persistence_due


def test_default_save_interval_is_25000() -> None:
    assert DEFAULT_SAVE_INTERVAL == 25000


def test_persistence_due_at_step1_and_interval_multiples() -> None:
    assert _persistence_due(1, 25000) is True
    assert _persistence_due(24999, 25000) is False
    assert _persistence_due(25000, 25000) is True
    assert _persistence_due(25001, 25000) is False
    assert _persistence_due(50000, 25000) is True
    assert _persistence_due(75000, 25000) is True
    assert _persistence_due(100000, 25000) is True


def test_persistence_due_ignores_invalid_interval_except_step1() -> None:
    assert _persistence_due(1, 0) is True
    assert _persistence_due(25000, 0) is False
    assert _persistence_due(25000, -1) is False


def test_persistence_due_supports_custom_interval() -> None:
    assert _persistence_due(100, 100) is True
    assert _persistence_due(200, 100) is True
    assert _persistence_due(150, 100) is False
