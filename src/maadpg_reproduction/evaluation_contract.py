"""Full preregistered MAADPG evaluation contract and diagnostic modes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np

from .config import EnvironmentConfig
from .dynamics import canonical_normalized_action
from .env import MAADPGPursuitEnv
from .evaluation import wilson_interval
from .exploration import IndependentOUNoise, OUExplorationConfig
from .gate import AdaptiveDifferenceGate
from .maddpg import MADDPGLearner
from .pfm import potential_field_actions
from .success import CaptureEvaluation


EvaluationMode = Literal["deterministic", "stochastic", "guidance"]


class _HoldAuditEnvironment(MAADPGPursuitEnv):
    """Allow diagnostic continuation only after strict success termination."""

    def step(self, actions):
        result = super().step(actions)
        if result[4]["capture"].success and result[2]:
            self._episode_done = False
        return result


@dataclass(frozen=True)
class ContractEvaluationEpisode:
    scenario_seed: int
    mode: EvaluationMode
    paper_terminal_success: bool
    strict_eq18_success: bool
    instantaneous_success: bool
    hold_10_success: bool
    capture_step: int | None
    terminated: bool
    truncated: bool
    steps: int
    target_in_hull: bool
    radial_band_ok: bool
    angular_gap_ok: bool
    boundary_ok: bool
    separation_ok: bool
    obstacle_clearance_ok: bool
    collision_boundary: bool
    collision_obstacle: bool
    collision_teammate: bool
    collision_target_hard: bool
    collision_provenance: str | None
    hold_audit_collision: bool
    mean_action_norm: float
    maximum_action_norm: float
    action_saturation_fraction: float
    mean_pursuer_speed: float
    maximum_pursuer_speed: float
    guidance_adoption_fraction: float | None
    final_maximum_target_distance: float
    final_minimum_obstacle_clearance: float


@dataclass(frozen=True)
class ContractEvaluationSummary:
    mode: EvaluationMode
    episodes: tuple[ContractEvaluationEpisode, ...]
    paper_terminal_success_rate: float
    strict_eq18_success_rate: float
    instantaneous_success_rate: float
    hold_10_success_rate: float
    strict_wilson_95_low: float
    strict_wilson_95_high: float
    collision_rate: float
    truncation_rate: float
    conditional_mean_steps_to_capture: float | None
    target_in_hull_rate: float
    radial_band_rate: float
    angular_gap_rate: float
    separation_feasibility_rate: float
    obstacle_feasibility_rate: float
    boundary_feasibility_rate: float
    mean_action_norm: float
    action_saturation_fraction: float
    mean_pursuer_speed: float
    guidance_adoption_fraction: float | None
    paper_terminal_definition: str


def _evaluation_noise(seed: int, horizon: int) -> IndependentOUNoise:
    derived = int(
        np.random.SeedSequence([seed, 0x4556414C]).generate_state(
            1, dtype=np.uint64
        )[0]
    )
    config = OUExplorationConfig(
        sigma_start=0.05,
        sigma_end=0.05,
        decay_steps=max(1, horizon),
    )
    return IndependentOUNoise(config, seed=derived)


def _paper_terminal(capture: CaptureEvaluation, info: dict, config: EnvironmentConfig) -> bool:
    return bool(
        capture.target_in_hull
        and info["reward_terms"]["target_fan_area"]
        <= config.reward.stage_s4 + config.capture.tolerance
        and float(np.max(capture.target_distances))
        <= config.reward.stage_d_cap + config.capture.tolerance
    )


def _collision_provenance(collision: dict) -> str | None:
    order = ("boundary", "obstacle", "teammate", "target_hard")
    active = [name for name in order if bool(np.any(collision[name]))]
    return "+".join(active) if active else None


def evaluate_policy(
    learner: MADDPGLearner,
    environment_config: EnvironmentConfig,
    scenario_seeds: Iterable[int],
    *,
    mode: EvaluationMode = "deterministic",
) -> ContractEvaluationSummary:
    if mode not in ("deterministic", "stochastic", "guidance"):
        raise ValueError(f"unknown evaluation mode: {mode}")
    episodes: list[ContractEvaluationEpisode] = []
    for seed_value in scenario_seeds:
        seed = int(seed_value)
        env = _HoldAuditEnvironment(config=environment_config)
        observation, _ = env.reset(seed=seed)
        noise = _evaluation_noise(seed, environment_config.horizon_steps)
        gate = AdaptiveDifferenceGate()
        action_norms: list[float] = []
        speeds: list[float] = []
        gate_adoptions = 0
        gate_comparisons = 0
        paper_success = False
        strict_success = False
        hold_success = False
        hold_collision = False
        capture_step: int | None = None
        primary_terminated = primary_truncated = False
        primary_collision: dict | None = None
        report_capture: CaptureEvaluation | None = None
        final_info: dict | None = None

        def policy_step(current_observation, evaluation_step):
            nonlocal gate_adoptions, gate_comparisons
            actor_action = learner.select_actions(current_observation)
            if mode == "stochastic":
                perturbation = noise.sample(evaluation_step)
                actor_action = np.stack(
                    [
                        canonical_normalized_action(
                            actor_action[index] + perturbation[index]
                        )
                        for index in range(3)
                    ]
                )
            if mode == "guidance":
                state = env.state
                guidance = potential_field_actions(
                    state.pursuer_positions,
                    state.target_position,
                    env.obstacle_pairs,
                )
                gated = gate.step_environment(env, actor_action, guidance.actions)
                gate_adoptions += int(np.sum(gated.decision.adopted_pfm))
                gate_comparisons += 3
                return (
                    gated.observation,
                    gated.terminated,
                    gated.truncated,
                    gated.info,
                )
            next_observation, _, terminated, truncated, info = env.step(actor_action)
            return next_observation, terminated, truncated, info

        evaluation_step = 0
        while True:
            observation, terminated, truncated, info = policy_step(
                observation, evaluation_step
            )
            evaluation_step += 1
            executed = np.asarray(info["executed_action"], dtype=np.float64)
            action_norms.extend(np.linalg.norm(executed, axis=1).tolist())
            speeds.extend(
                np.linalg.norm(env.state.pursuer_velocities, axis=1).tolist()
            )
            capture = info["capture"]
            paper_success = paper_success or _paper_terminal(
                capture, info, environment_config
            )
            final_info = info
            report_capture = capture
            if capture.success:
                strict_success = True
                capture_step = int(info["step_count"])
                primary_terminated = True
                primary_truncated = False
                primary_collision = info["collision"]
                hold_success = True
                for _ in range(9):
                    observation, hold_term, hold_trunc, hold_info = policy_step(
                        observation, evaluation_step
                    )
                    evaluation_step += 1
                    hold_executed = np.asarray(
                        hold_info["executed_action"], dtype=np.float64
                    )
                    action_norms.extend(
                        np.linalg.norm(hold_executed, axis=1).tolist()
                    )
                    speeds.extend(
                        np.linalg.norm(
                            env.state.pursuer_velocities, axis=1
                        ).tolist()
                    )
                    if not hold_info["capture"].success:
                        hold_success = False
                    if bool(np.any(hold_info["collision"]["by_agent"])):
                        hold_collision = True
                        hold_success = False
                        break
                    if hold_trunc:
                        hold_success = False
                        break
                    if hold_term and not hold_info["capture"].success:
                        hold_success = False
                        break
                break
            if terminated or truncated:
                primary_terminated = bool(terminated)
                primary_truncated = bool(truncated)
                primary_collision = info["collision"]
                break
        if final_info is None or report_capture is None or primary_collision is None:
            raise AssertionError("evaluation did not produce a terminal record")
        collision_source = _collision_provenance(primary_collision)
        episodes.append(
            ContractEvaluationEpisode(
                scenario_seed=seed,
                mode=mode,
                paper_terminal_success=paper_success,
                strict_eq18_success=strict_success,
                instantaneous_success=strict_success,
                hold_10_success=hold_success,
                capture_step=capture_step,
                terminated=primary_terminated,
                truncated=primary_truncated,
                steps=capture_step or int(final_info["step_count"]),
                target_in_hull=report_capture.target_in_hull,
                radial_band_ok=report_capture.radial_band_ok,
                angular_gap_ok=report_capture.angular_gap_ok,
                boundary_ok=report_capture.boundary_ok,
                separation_ok=report_capture.separation_ok,
                obstacle_clearance_ok=report_capture.obstacle_clearance_ok,
                collision_boundary=bool(
                    np.any(primary_collision["boundary"])
                ),
                collision_obstacle=bool(
                    np.any(primary_collision["obstacle"])
                ),
                collision_teammate=bool(
                    np.any(primary_collision["teammate"])
                ),
                collision_target_hard=bool(
                    np.any(primary_collision["target_hard"])
                ),
                collision_provenance=collision_source,
                hold_audit_collision=hold_collision,
                mean_action_norm=float(np.mean(action_norms)),
                maximum_action_norm=float(np.max(action_norms)),
                action_saturation_fraction=float(
                    np.mean(np.asarray(action_norms) >= 0.99)
                ),
                mean_pursuer_speed=float(np.mean(speeds)),
                maximum_pursuer_speed=float(np.max(speeds)),
                guidance_adoption_fraction=(
                    gate_adoptions / gate_comparisons
                    if gate_comparisons
                    else None
                ),
                final_maximum_target_distance=float(
                    np.max(report_capture.target_distances)
                ),
                final_minimum_obstacle_clearance=float(
                    report_capture.minimum_obstacle_clearance
                ),
            )
        )
    if not episodes:
        raise ValueError("evaluation requires at least one scenario seed")
    total = len(episodes)
    strict_count = sum(item.strict_eq18_success for item in episodes)
    low, high = wilson_interval(strict_count, total)
    capture_steps = [
        item.capture_step for item in episodes if item.capture_step is not None
    ]
    collision_count = sum(item.collision_provenance is not None for item in episodes)
    guidance_values = [
        item.guidance_adoption_fraction
        for item in episodes
        if item.guidance_adoption_fraction is not None
    ]
    return ContractEvaluationSummary(
        mode=mode,
        episodes=tuple(episodes),
        paper_terminal_success_rate=sum(
            item.paper_terminal_success for item in episodes
        )
        / total,
        strict_eq18_success_rate=strict_count / total,
        instantaneous_success_rate=sum(
            item.instantaneous_success for item in episodes
        )
        / total,
        hold_10_success_rate=sum(item.hold_10_success for item in episodes)
        / total,
        strict_wilson_95_low=low,
        strict_wilson_95_high=high,
        collision_rate=collision_count / total,
        truncation_rate=sum(item.truncated for item in episodes) / total,
        conditional_mean_steps_to_capture=(
            float(np.mean(capture_steps)) if capture_steps else None
        ),
        target_in_hull_rate=sum(item.target_in_hull for item in episodes) / total,
        radial_band_rate=sum(item.radial_band_ok for item in episodes) / total,
        angular_gap_rate=sum(item.angular_gap_ok for item in episodes) / total,
        separation_feasibility_rate=sum(
            item.separation_ok for item in episodes
        )
        / total,
        obstacle_feasibility_rate=sum(
            item.obstacle_clearance_ok for item in episodes
        )
        / total,
        boundary_feasibility_rate=sum(item.boundary_ok for item in episodes)
        / total,
        mean_action_norm=float(np.mean([item.mean_action_norm for item in episodes])),
        action_saturation_fraction=float(
            np.mean([item.action_saturation_fraction for item in episodes])
        ),
        mean_pursuer_speed=float(
            np.mean([item.mean_pursuer_speed for item in episodes])
        ),
        guidance_adoption_fraction=(
            float(np.mean(guidance_values)) if guidance_values else None
        ),
        paper_terminal_definition=(
            "assumed Eq.33 diagnostic: target_in_hull and target_fan_area<=S4 "
            "and max_target_distance<=d_cap"
        ),
    )
