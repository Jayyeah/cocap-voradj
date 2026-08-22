"""Matched-budget trainer with exact mid-episode runtime state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import random
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import torch

from .config import EnvironmentConfig, default_environment_config
from .dynamics import canonical_normalized_action
from .env import MAADPGPursuitEnv
from .exploration import IndependentOUNoise, OUExplorationConfig
from .gate import AdaptiveDifferenceGate, GateDecision
from .guidance_config import AdaptiveGateConfig, PotentialFieldConfig
from .maddpg import MADDPGLearner, MADDPGUpdateDiagnostics
from .pfm import potential_field_actions
from .replay import JointReplayBuffer
from .training_config import MADDPGConfig, default_maddpg_config


FloatArray = NDArray[np.float64]
Variant = Literal["maddpg", "maadpg"]


@dataclass(frozen=True)
class TrainerConfig:
    variant: Variant
    training_seed: int
    environment: EnvironmentConfig = field(default_factory=default_environment_config)
    maddpg: MADDPGConfig = field(default_factory=default_maddpg_config)
    exploration: OUExplorationConfig = field(default_factory=OUExplorationConfig)
    potential_field: PotentialFieldConfig = field(default_factory=PotentialFieldConfig)
    adaptive_gate: AdaptiveGateConfig = field(default_factory=AdaptiveGateConfig)

    def validate(self) -> None:
        if self.variant not in ("maddpg", "maadpg"):
            raise ValueError(f"unknown training variant {self.variant}")
        self.environment.validate()
        self.maddpg.validate()
        self.exploration.validate()
        self.potential_field.validate()
        self.adaptive_gate.validate()
        if (
            self.maddpg.agent_count != self.environment.pursuer_count
            or self.maddpg.observation_dim
            != self.environment.pursuer_observation_dim
        ):
            raise ValueError("learner and environment shapes disagree")


@dataclass
class TrainingRuntime:
    environment_steps: int
    gradient_steps: int
    episodes_completed: int
    current_scenario_seed: int
    current_observation: FloatArray
    current_episode_return: FloatArray
    current_episode_length: int
    current_episode_gate_adoptions: NDArray[np.int64]
    gate_adoptions_total: NDArray[np.int64]
    gate_comparisons_total: int


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    scenario_seed: int
    environment_step: int
    length: int
    returns: tuple[float, float, float]
    terminated: bool
    truncated: bool
    success: bool
    collision_boundary: bool
    collision_obstacle: bool
    collision_teammate: bool
    collision_target_hard: bool
    gate_adoptions: tuple[int, int, int]


@dataclass(frozen=True)
class TrainerStep:
    environment_step: int
    gradient_step: int
    reward: tuple[float, float, float]
    executed_action: FloatArray
    terminated: bool
    truncated: bool
    replay_size: int
    updates: tuple[MADDPGUpdateDiagnostics, ...]
    gate_decision: GateDecision | None
    episode: EpisodeRecord | None


def _derived_seed(training_seed: int, stream: int) -> int:
    sequence = np.random.SeedSequence([int(training_seed), int(stream), 0x4D414144])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


class MAADPGTrainer:
    def __init__(
        self,
        config: TrainerConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self.env = MAADPGPursuitEnv(config.environment)
        self.learner = MADDPGLearner(
            config.maddpg,
            device=self.device,
            initialization_seed=_derived_seed(config.training_seed, 1),
        )
        self.replay = JointReplayBuffer(
            config.maddpg.replay_capacity,
            agent_count=config.maddpg.agent_count,
            observation_dim=config.maddpg.observation_dim,
            action_dim=config.maddpg.action_dim,
            seed=_derived_seed(config.training_seed, 2),
        )
        self.exploration = IndependentOUNoise(
            config.exploration, seed=_derived_seed(config.training_seed, 3)
        )
        self.gate = AdaptiveDifferenceGate(config.adaptive_gate)
        scenario_seed = self.episode_scenario_seed(0)
        observation, _ = self.env.reset(seed=scenario_seed)
        self.runtime = TrainingRuntime(
            environment_steps=0,
            gradient_steps=0,
            episodes_completed=0,
            current_scenario_seed=scenario_seed,
            current_observation=observation,
            current_episode_return=np.zeros(3, dtype=np.float64),
            current_episode_length=0,
            current_episode_gate_adoptions=np.zeros(3, dtype=np.int64),
            gate_adoptions_total=np.zeros(3, dtype=np.int64),
            gate_comparisons_total=0,
        )

    def episode_scenario_seed(self, episode_index: int) -> int:
        return _derived_seed(self.config.training_seed, 10_000 + episode_index)

    def _exploratory_actor_actions(self) -> FloatArray:
        deterministic = self.learner.select_actions(self.runtime.current_observation)
        noise = self.exploration.sample(self.runtime.environment_steps)
        return np.stack(
            [
                canonical_normalized_action(deterministic[index] + noise[index])
                for index in range(3)
            ]
        )

    def step_once(self) -> TrainerStep:
        observation = self.runtime.current_observation.copy()
        actor_actions = self._exploratory_actor_actions()
        gate_decision: GateDecision | None = None
        if self.config.variant == "maadpg":
            state = self.env.state
            if state is None:
                raise AssertionError("trainer environment has no state")
            guidance = potential_field_actions(
                state.pursuer_positions,
                state.target_position,
                self.env.obstacle_pairs,
                self.config.potential_field,
            )
            gated = self.gate.step_environment(
                self.env, actor_actions, guidance.actions
            )
            next_observation = gated.observation
            reward = gated.reward
            terminated = gated.terminated
            truncated = gated.truncated
            info = gated.info
            gate_decision = gated.decision
            adopted = gate_decision.adopted_pfm.astype(np.int64)
            self.runtime.current_episode_gate_adoptions += adopted
            self.runtime.gate_adoptions_total += adopted
            self.runtime.gate_comparisons_total += 3
        else:
            next_observation, reward, terminated, truncated, info = self.env.step(
                actor_actions
            )
        executed_action = np.asarray(info["executed_action"], dtype=np.float64)
        self.replay.add(
            observation,
            executed_action,
            reward,
            next_observation,
            terminated=terminated,
            truncated=truncated,
        )
        self.runtime.environment_steps += 1
        self.runtime.current_episode_length += 1
        self.runtime.current_episode_return += reward
        self.runtime.current_observation = next_observation

        updates: list[MADDPGUpdateDiagnostics] = []
        if (
            len(self.replay) >= self.config.maddpg.replay_warmup
            and self.runtime.environment_steps
            % self.config.maddpg.update_every_environment_steps
            == 0
        ):
            for _ in range(self.config.maddpg.gradient_steps_per_update):
                batch = self.replay.sample(
                    self.config.maddpg.batch_size, device=self.device
                )
                updates.append(self.learner.update(batch))
                self.runtime.gradient_steps += 1

        episode_record = None
        if terminated or truncated:
            collision = info["collision"]
            episode_record = EpisodeRecord(
                episode_index=self.runtime.episodes_completed,
                scenario_seed=self.runtime.current_scenario_seed,
                environment_step=self.runtime.environment_steps,
                length=self.runtime.current_episode_length,
                returns=tuple(float(value) for value in self.runtime.current_episode_return),
                terminated=bool(terminated),
                truncated=bool(truncated),
                success=bool(info["capture"].success),
                collision_boundary=bool(np.any(collision["boundary"])),
                collision_obstacle=bool(np.any(collision["obstacle"])),
                collision_teammate=bool(np.any(collision["teammate"])),
                collision_target_hard=bool(np.any(collision["target_hard"])),
                gate_adoptions=tuple(
                    int(value)
                    for value in self.runtime.current_episode_gate_adoptions
                ),
            )
            self.runtime.episodes_completed += 1
            next_seed = self.episode_scenario_seed(self.runtime.episodes_completed)
            next_observation, _ = self.env.reset(seed=next_seed)
            self.exploration.reset_episode()
            self.runtime.current_scenario_seed = next_seed
            self.runtime.current_observation = next_observation
            self.runtime.current_episode_return.fill(0.0)
            self.runtime.current_episode_length = 0
            self.runtime.current_episode_gate_adoptions.fill(0)

        self._validate_runtime_consistency(check_observation=False)
        return TrainerStep(
            environment_step=self.runtime.environment_steps,
            gradient_step=self.runtime.gradient_steps,
            reward=tuple(float(value) for value in reward),
            executed_action=executed_action.copy(),
            terminated=bool(terminated),
            truncated=bool(truncated),
            replay_size=len(self.replay),
            updates=tuple(updates),
            gate_decision=gate_decision,
            episode=episode_record,
        )

    def _validate_runtime_consistency(
        self, *, check_observation: bool = True
    ) -> None:
        state = self.env.state
        if state is None:
            raise RuntimeError("trainer environment state is missing")
        expected_size = min(self.runtime.environment_steps, self.replay.capacity)
        expected_cursor = self.runtime.environment_steps % self.replay.capacity
        checks = {
            "runtime_vs_replay_insertions": (
                self.runtime.environment_steps, self.replay.total_insertions
            ),
            "runtime_vs_learner_updates": (
                self.runtime.gradient_steps, self.learner.update_count
            ),
            "runtime_vs_environment_episode_step": (
                self.runtime.current_episode_length, state.step_count
            ),
            "replay_size": (self.replay.size, expected_size),
            "replay_cursor": (self.replay.cursor, expected_cursor),
        }
        mismatches = {
            name: values for name, values in checks.items() if values[0] != values[1]
        }
        if mismatches:
            raise RuntimeError(
                f"trainer/replay/runtime consistency failure: {mismatches}"
            )
        if check_observation and not np.array_equal(
            self.runtime.current_observation, self.env.observation()
        ):
            raise RuntimeError(
                "runtime observation differs from environment observation"
            )

    @staticmethod
    def _global_rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy_legacy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_global_rng_state(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy_legacy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])

    def state_dict(self, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self._validate_runtime_consistency()
        return {
            "checkpoint_version": 1,
            "checkpoint_kind": "full_runtime",
            "contains_replay": True,
            "trainer_config": self.config,
            "runtime": deepcopy(self.runtime),
            "environment": self.env.snapshot(),
            "learner": self.learner.state_dict(),
            "replay": self.replay.state_dict(),
            "exploration": self.exploration.state_dict(),
            "global_rng": self._global_rng_state(),
            "metadata": deepcopy(metadata or {}),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("checkpoint_version") != 1:
            raise ValueError("unsupported trainer checkpoint version")
        if state["trainer_config"] != self.config:
            raise ValueError("trainer checkpoint configuration mismatch")
        self.learner.load_state_dict(state["learner"])
        self.replay.load_state_dict(state["replay"])
        self.env.restore(state["environment"])
        self.exploration.load_state_dict(state["exploration"])
        self.runtime = deepcopy(state["runtime"])
        self._validate_runtime_consistency()
        self._restore_global_rng_state(state["global_rng"])

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> "MAADPGTrainer":
        trainer = cls(state["trainer_config"], device=device)
        trainer.load_state_dict(state)
        return trainer
