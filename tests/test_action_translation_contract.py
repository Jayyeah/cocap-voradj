from __future__ import annotations

import numpy as np

from cocap_voradj.training.continuous.legacy_action_translation import (
    LegacyIQNActionToWorldAccelerationAdapter,
)


def test_translated_action_is_finite_world_acceleration_and_l2_bounded() -> None:
    adapter = LegacyIQNActionToWorldAccelerationAdapter(a_max=0.4)
    for action in ((-0.4, 0.0), (0.4, np.pi / 6), (0.0, 0.0)):
        result = adapter.translate(action, {"velocity": np.asarray([1.0, -0.5]), "theta": 0.3})
        assert result.rejected is False
        assert result.action.shape == (2,)
        assert np.all(np.isfinite(result.action))
        assert float(np.linalg.norm(result.action)) <= 0.4 + 1e-6
        assert np.isfinite(result.terminal_velocity_error)


def test_translated_action_rejects_nan_legacy_command() -> None:
    adapter = LegacyIQNActionToWorldAccelerationAdapter()
    result = adapter.translate([np.nan, 0.0], {"velocity": np.zeros(2), "theta": 0.0})
    assert result.rejected is True
