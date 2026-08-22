from dataclasses import replace
import math

import numpy as np

import maadpg_reproduction.evaluation_contract as contract
from maadpg_reproduction.config import (
    TargetPolicyConfig,
    default_environment_config,
)


class ZeroActor:
    def select_actions(self, observation):
        assert np.asarray(observation).shape == (3, 26)
        return np.zeros((3, 2), dtype=np.float32)


class FixedRingEnvironment(contract._HoldAuditEnvironment):
    def reset(self, *, seed=None, options=None):
        del options
        target = np.array([1.0, 1.0])
        angles = np.arange(3) * 2.0 * math.pi / 3.0
        pursuers = target + 0.10 * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        return super().reset(
            seed=seed,
            options={
                "pursuer_positions": pursuers,
                "target_position": target,
                "obstacles": [],
            },
        )


def _stationary_config():
    base = default_environment_config()
    return replace(
        base,
        target_policy=TargetPolicyConfig(policy_id="stationary_target_v1"),
        horizon_steps=20,
    )


def test_full_deterministic_contract_reports_instant_and_ten_step_hold(monkeypatch):
    monkeypatch.setattr(contract, "_HoldAuditEnvironment", FixedRingEnvironment)
    summary = contract.evaluate_policy(
        ZeroActor(), _stationary_config(), [11, 12], mode="deterministic"
    )
    assert summary.paper_terminal_success_rate == 1.0
    assert summary.strict_eq18_success_rate == 1.0
    assert summary.instantaneous_success_rate == 1.0
    assert summary.hold_10_success_rate == 1.0
    assert summary.collision_rate == 0.0
    assert summary.target_in_hull_rate == 1.0
    assert summary.radial_band_rate == 1.0
    assert summary.angular_gap_rate == 1.0
    assert summary.separation_feasibility_rate == 1.0
    assert summary.obstacle_feasibility_rate == 1.0
    assert summary.boundary_feasibility_rate == 1.0
    assert summary.action_saturation_fraction == 0.0
    assert summary.mean_pursuer_speed == 0.0
    assert all(item.capture_step == 1 for item in summary.episodes)


def test_stochastic_and_guidance_diagnostics_are_explicit_modes(monkeypatch):
    monkeypatch.setattr(contract, "_HoldAuditEnvironment", FixedRingEnvironment)
    stochastic = contract.evaluate_policy(
        ZeroActor(), _stationary_config(), [21], mode="stochastic"
    )
    guidance = contract.evaluate_policy(
        ZeroActor(), _stationary_config(), [21], mode="guidance"
    )
    assert stochastic.mode == "stochastic"
    assert stochastic.mean_action_norm > 0.0
    assert stochastic.guidance_adoption_fraction is None
    assert guidance.mode == "guidance"
    assert guidance.guidance_adoption_fraction is not None
    assert 0.0 <= guidance.guidance_adoption_fraction <= 1.0
