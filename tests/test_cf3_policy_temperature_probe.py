from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from tools.probe_cf3_policy_temperature import (
    apply_collision_semantics_override,
    epsilon_stream,
    parse_temperature_modes,
    sample_temperature_actions,
    summarize,
)


class _Actor:
    config = SimpleNamespace(a_max=0.4, w_max=0.5)

    def distribution(self, obs):
        batch = obs["self"].shape[0]
        loc = torch.tensor([[0.2, -0.3]], dtype=torch.float32).expand(batch, -1)
        scale = torch.tensor([[0.5, 0.25]], dtype=torch.float32).expand(batch, -1)
        return Normal(loc, scale), torch.log(scale)


class _Adapter:
    def validate_with_diagnostics(self, action):
        return np.asarray(action, dtype=np.float32), SimpleNamespace(action_rejected=False)


def _trainer():
    return SimpleNamespace(actor=_Actor(), device=torch.device("cpu"))


def test_temperature_contract_scales_latent_noise_before_tanh() -> None:
    obs = {"self": np.zeros((2, 1), dtype=np.float32)}
    epsilon = np.asarray([[1.0, -2.0], [-1.0, 2.0]], dtype=np.float32)
    deterministic, _ = sample_temperature_actions(
        _trainer(), obs, 2, _Adapter(), 0.0, epsilon,
    )
    reduced, _ = sample_temperature_actions(
        _trainer(), obs, 2, _Adapter(), 0.25, epsilon,
    )
    normal, _ = sample_temperature_actions(
        _trainer(), obs, 2, _Adapter(), 1.0, epsilon,
    )
    expected_det = np.tanh(np.asarray([[0.2, -0.3], [0.2, -0.3]])) * np.asarray([0.4, 0.5])
    expected_reduced = np.tanh(
        np.asarray([[0.2, -0.3], [0.2, -0.3]])
        + 0.25 * np.asarray([[0.5, 0.25], [0.5, 0.25]]) * epsilon
    ) * np.asarray([0.4, 0.5])
    expected_normal = np.tanh(
        np.asarray([[0.2, -0.3], [0.2, -0.3]])
        + np.asarray([[0.5, 0.25], [0.5, 0.25]]) * epsilon
    ) * np.asarray([0.4, 0.5])
    np.testing.assert_allclose(deterministic, expected_det, atol=1e-7)
    np.testing.assert_allclose(reduced, expected_reduced, atol=1e-7)
    np.testing.assert_allclose(normal, expected_normal, atol=1e-7)


def test_epsilon_stream_is_fixed_shape_and_reproducible() -> None:
    left = epsilon_stream(123, 12)
    right = epsilon_stream(123, 12)
    for _ in range(3):
        a = next(left)
        b = next(right)
        assert a.shape == (12, 2)
        np.testing.assert_array_equal(a, b)


def test_temperature_modes_and_invalid_values() -> None:
    modes = parse_temperature_modes([0.0, 0.25, 1.0])
    assert [(item.name, item.temperature) for item in modes] == [
        ("deterministic_t0", 0.0),
        ("reduced_t0p25", 0.25),
        ("stochastic_t1", 1.0),
    ]
    with pytest.raises(ValueError, match="non-negative"):
        parse_temperature_modes([-0.1])
    with pytest.raises(ValueError, match="unique"):
        parse_temperature_modes([0.0, 0.0])


def test_collision_semantics_override_is_copy_on_write() -> None:
    source = {"env": {"width": 120.0}}
    untouched = apply_collision_semantics_override(source, None)
    fixed = apply_collision_semantics_override(source, "synchronized_swept_v1")
    assert source == {"env": {"width": 120.0}}
    assert untouched == source
    assert fixed["env"]["collision_semantics"] == "synchronized_swept_v1"
    with pytest.raises(ValueError, match="unsupported collision semantics"):
        apply_collision_semantics_override(source, "unknown")


def test_summary_keeps_normal_stationary_ring_hold_and_collision_separate() -> None:
    rows = [
        {
            "normal_capture": True, "stationary_capture": False, "captured": True,
            "collision": False, "visited_2plus_ring": True, "visited_3plus_ring": True,
            "fraction_steps_2plus_in_ring": 0.2, "fraction_steps_3plus_in_ring": 0.1,
            "max_2plus_ring_hold_steps": 7, "max_3plus_ring_hold_steps": 3,
            "length": 100, "action_rejected_rate": 0.0,
        },
        {
            "normal_capture": False, "stationary_capture": True, "captured": True,
            "collision": True, "visited_2plus_ring": True, "visited_3plus_ring": False,
            "fraction_steps_2plus_in_ring": 0.1, "fraction_steps_3plus_in_ring": 0.0,
            "max_2plus_ring_hold_steps": 4, "max_3plus_ring_hold_steps": 0,
            "length": 80, "action_rejected_rate": 0.0,
        },
    ]
    result = summarize(rows)
    assert result["normal_capture_count"] == 1
    assert result["stationary_capture_count"] == 1
    assert result["capture_count"] == 2
    assert result["collision_count"] == 1
    assert result["episodes_with_3plus_ring"] == 1
    assert result["max_3plus_ring_hold_steps"] == 3
