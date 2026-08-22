"""Exact one-step adaptive difference gate following Eqs. 21-22."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dynamics import canonical_normalized_action
from .guidance_config import AdaptiveGateConfig


FloatArray = NDArray[np.float64]


class CounterfactualEnvironment(Protocol):
    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...

    def step(
        self, actions: ArrayLike
    ) -> tuple[FloatArray, FloatArray, bool, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class GateDecision:
    actor_actions: FloatArray
    pfm_actions: FloatArray
    adopted_actions: FloatArray
    actor_rewards: FloatArray
    pfm_rewards: FloatArray
    delta_ratios: FloatArray
    delta_percent: FloatArray
    adopted_pfm: NDArray[np.bool_]
    actor_terminated: bool
    actor_truncated: bool
    pfm_terminated: NDArray[np.bool_]
    pfm_truncated: NDArray[np.bool_]


@dataclass(frozen=True)
class GatedEnvironmentStep:
    decision: GateDecision
    observation: FloatArray
    reward: FloatArray
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class AdaptiveDifferenceGate:
    def __init__(self, config: AdaptiveGateConfig | None = None) -> None:
        self.config = config or AdaptiveGateConfig()
        self.config.validate()

    @staticmethod
    def _actions(value: ArrayLike, *, name: str) -> FloatArray:
        actions = np.asarray(value, dtype=np.float64)
        if actions.shape != (3, 2) or not np.all(np.isfinite(actions)):
            raise ValueError(f"{name} must be finite with shape (3,2)")
        return np.stack([canonical_normalized_action(action) for action in actions])

    def evaluate(
        self,
        env: CounterfactualEnvironment,
        actor_actions: ArrayLike,
        pfm_actions: ArrayLike,
        *,
        candidate_order: Sequence[int] = (0, 1, 2),
    ) -> GateDecision:
        order = tuple(int(index) for index in candidate_order)
        if sorted(order) != [0, 1, 2]:
            raise ValueError("candidate_order must be a permutation of fixed slots 0,1,2")
        actor = self._actions(actor_actions, name="actor_actions")
        pfm = self._actions(pfm_actions, name="pfm_actions")
        original = env.snapshot()
        pfm_rewards = np.empty(3, dtype=np.float64)
        pfm_terminated = np.empty(3, dtype=bool)
        pfm_truncated = np.empty(3, dtype=bool)
        try:
            _, actor_reward, actor_term, actor_trunc, _ = env.step(actor)
            actor_rewards = np.asarray(actor_reward, dtype=np.float64)
            if actor_rewards.shape != (3,) or not np.all(np.isfinite(actor_rewards)):
                raise ValueError("counterfactual actor reward must be finite shape (3,)")
            for index in order:
                env.restore(original)
                candidate = actor.copy()
                candidate[index] = pfm[index]
                _, reward, terminated, truncated, _ = env.step(candidate)
                reward_vector = np.asarray(reward, dtype=np.float64)
                if reward_vector.shape != (3,) or not np.all(np.isfinite(reward_vector)):
                    raise ValueError("counterfactual PFM reward must be finite shape (3,)")
                pfm_rewards[index] = reward_vector[index]
                pfm_terminated[index] = terminated
                pfm_truncated[index] = truncated
        finally:
            env.restore(original)
        denominator = np.abs(pfm_rewards) + self.config.denominator_lambda
        delta = (pfm_rewards - actor_rewards) / denominator
        adopted = delta >= self.config.beta_ratio
        actions = actor.copy()
        actions[adopted] = pfm[adopted]
        return GateDecision(
            actor_actions=actor,
            pfm_actions=pfm,
            adopted_actions=actions,
            actor_rewards=actor_rewards,
            pfm_rewards=pfm_rewards,
            delta_ratios=delta,
            delta_percent=100.0 * delta,
            adopted_pfm=adopted,
            actor_terminated=bool(actor_term),
            actor_truncated=bool(actor_trunc),
            pfm_terminated=pfm_terminated,
            pfm_truncated=pfm_truncated,
        )

    def step_environment(
        self,
        env: CounterfactualEnvironment,
        actor_actions: ArrayLike,
        pfm_actions: ArrayLike,
        *,
        candidate_order: Sequence[int] = (0, 1, 2),
    ) -> GatedEnvironmentStep:
        decision = self.evaluate(
            env,
            actor_actions,
            pfm_actions,
            candidate_order=candidate_order,
        )
        observation, reward, terminated, truncated, info = env.step(
            decision.adopted_actions
        )
        executed = info.get("executed_action")
        if executed is not None and not np.allclose(
            np.asarray(executed, dtype=np.float64),
            decision.adopted_actions,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError("real environment action differs from gate adoption")
        return GatedEnvironmentStep(
            decision=decision,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
