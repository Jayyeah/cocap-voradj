#!/usr/bin/env python3
"""Formal Pure-Coverage runner for the exact categorical AW9 SAC contract."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from cocap_voradj.models.shared_local_ac import (
    SharedLocalACNetworkConfig,
    assert_runtime_aw9,
    parameter_count,
)
from cocap_voradj.training.discrete_sac import (
    DiscreteSACConfig,
    DiscreteSACTrainer,
    atomic_torch_save,
)
from cocap_voradj.training.replay import stack_obs
from cocap_voradj.training.shared_local_ac import (
    ReplayWarmupError,
    SharedLocalReplay,
    Transition,
    clone_obs,
    state_hash,
    tensor_obs,
)
from cocap_voradj.training.trainer import load_config
from tools.run_shared_local_ac_capability import epsilon_value, make_env


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class DiscreteSACCoverageRunner:
    """One shared local categorical actor and two vector-valued AW9 critics."""

    def __init__(self, config: dict[str, Any], config_path: Path, *, smoke: bool, output_root: Optional[Path], resume_path: Optional[Path]):
        self.config = copy.deepcopy(config)
        self.config_path = config_path.resolve()
        self.smoke = bool(smoke)
        self.seed = int(self.config.get("seed", 2026092202))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = str(self.config.get("device", "cuda:0"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device.startswith("cuda"):
            # Engineering-only numerical guard for this CUDA/PyTorch build.
            # The SAC equations, optimizer values, observations, and reward are unchanged.
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.mha.set_fastpath_enabled(False)
        self.output_root = Path(output_root or self.config.get("output_root", "runs"))
        self.run_name = str(self.config.get("run_name", self.config_path.stem))
        self.run_dir = self.output_root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ac = self.config.get("ac", {}) or {}
        self.network_config = SharedLocalACNetworkConfig(**dict(ac.get("network", {}) or {}))
        self.trainer_config = DiscreteSACConfig(
            gamma=float(ac.get("gamma", 0.99)),
            tau=float(ac.get("tau", 0.005)),
            actor_lr=float(ac.get("actor_lr", 3e-4)),
            critic_lr=float(ac.get("critic_lr", 3e-4)),
            alpha_lr=float(ac.get("alpha_lr", 3e-4)),
            alpha_init=float(ac.get("alpha_init", 0.2)),
            target_entropy=float(ac["target_entropy"]) if ac.get("target_entropy") is not None else None,
            adam_eps=float(ac.get("adam_eps", 1e-8)),
            max_grad_norm=float(ac["max_grad_norm"]) if ac.get("max_grad_norm") is not None else None,
        )
        self.trainer = DiscreteSACTrainer(self.network_config, self.trainer_config, self.device)
        self.initial_hashes = {
            "actor": state_hash(self.trainer.actor),
            "critic1": state_hash(self.trainer.critic1),
            "critic2": state_hash(self.trainer.critic2),
        }
        self.env = make_env(self.config, "voradj_coverage", self.seed + 47)
        self.observations = self.env.reset()
        assert_runtime_aw9(self.env)
        self.aw9_live = [(float(a), float(w)) for a, w in self.env.pursuers[0].action_list]
        self.replay = SharedLocalReplay(int(ac.get("replay_capacity", 1_000_000)))
        self.replay_rng = np.random.RandomState(self.seed + 902)
        self.global_step = 0
        self.episode_id = 0
        self.episode_count = 0
        self.action_counts: Counter[int] = Counter()
        self.epsilon_values: list[float] = []
        self.telemetry: deque[dict[str, Any]] = deque(maxlen=256)
        self.episode_tail: deque[dict[str, Any]] = deque(maxlen=50)
        self.resume_info: dict[str, Any] = {"resumed": False, "replay_policy": "scratch"}
        if resume_path is not None:
            self._load_resume(Path(resume_path))
        if self.smoke:
            self._apply_smoke_overrides()

    def _contract(self) -> dict[str, Any]:
        return {
            "schema": "ac-a2-discrete-sac-pure-coverage-v1",
            "task": "voradj_coverage",
            "action_size": 9,
            "action_grid": self.aw9_live,
            "observation_contract": "existing_shared_local_aw9",
            "reward_contract": "existing_final_pure_coverage",
            "target": "sum_pi_min_twin_q_minus_alpha_log_pi",
            "no_gumbel": True,
            "no_continuous_conversion": True,
            "training_line": "parallel_capability_experiment",
        }

    def _load_resume(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.trainer.load_payload(payload, self._contract(), load_optimizer=True)
        self.global_step = int(payload.get("global_step", 0))
        self.episode_count = int(payload.get("episode", 0))
        if payload.get("python_rng") is not None:
            random.setstate(payload["python_rng"])
        if payload.get("numpy_rng") is not None:
            np.random.set_state(payload["numpy_rng"])
        if payload.get("torch_rng") is not None:
            torch.set_rng_state(payload["torch_rng"].cpu())
        self.resume_info = {"resumed": True, "replay_policy": "rewarm_without_replay", "replay_size": 0}

    def _apply_smoke_overrides(self) -> None:
        smoke = (self.config.get("ac", {}) or {}).get("smoke", {}) or {}
        self.smoke_steps = int(smoke.get("steps", 64))
        self.smoke_batch_size = int(smoke.get("batch_size", 8))
        self.smoke_min_replay = int(smoke.get("min_replay_size", 8))
        self.env.episode_max_length = int(smoke.get("episode_max_length", 24))

    def _record_episode(self) -> None:
        record = dict(self.env.episode_record(task="coverage"))
        record["episode"] = self.episode_count
        self.episode_tail.append(record)
        self.episode_count += 1
        self.episode_id += 1
        self.observations = self.env.reset()

    def _collect_step(self) -> None:
        old_obs = self.observations
        active = [index for index, observation in enumerate(old_obs) if observation is not None]
        actions: list[Optional[int]] = [None] * len(old_obs)
        epsilon = epsilon_value(self.config, self.global_step)
        self.epsilon_values.append(epsilon)
        pending: dict[int, tuple[int, float, float]] = {}
        if active:
            batch = stack_obs([old_obs[index] for index in active], self.device)
            with torch.no_grad():
                sampled, actor_prob, behavior_prob = self.trainer.actor.sample_behavior(batch, epsilon)
            for row, index in enumerate(active):
                action = int(sampled[row].item())
                actions[index] = action
                self.action_counts[action] += 1
                pending[index] = (action, float(actor_prob[row].item()), float(behavior_prob[row].item()))
        result = self.env.step(actions, [])
        next_obs = result.observations
        for index in active:
            obs = old_obs[index]
            action, actor_probability, behavior_probability = pending[index]
            info = result.infos[index] if index < len(result.infos) else {}
            metadata = dict(info.get("replay_metadata", {}) or {})
            terminated = bool(info.get("terminated", False))
            truncated = bool(info.get("truncated", False))
            done = bool(result.dones[index])
            nxt = next_obs[index] if next_obs[index] is not None else {key: np.zeros_like(value) for key, value in obs.items()}
            self.replay.add(Transition(
                obs=clone_obs(obs), action=action, reward=float(result.rewards[index]), next_obs=clone_obs(nxt),
                done=done, terminated=terminated, truncated=truncated,
                phase=str(metadata.get("phase", "pure_coverage")), replay_class="recovery_pure",
                behavior_probability=behavior_probability, epsilon=epsilon, actor_probability=actor_probability,
                episode_id=self.episode_id, agent_id=index, metadata=metadata,
            ))
        self.observations = next_obs
        self.global_step += 1
        self._maybe_update()
        if any(result.dones):
            self._record_episode()

    def _maybe_update(self) -> None:
        ac = self.config.get("ac", {}) or {}
        train_freq = int(ac.get("train_freq", 4))
        min_replay = self.smoke_min_replay if self.smoke else int(ac.get("min_replay_size", 3000))
        batch_size = self.smoke_batch_size if self.smoke else int(ac.get("batch_size", 128))
        if self.global_step % train_freq != 0 or len(self.replay) < max(min_replay, batch_size):
            return
        try:
            transitions = self.replay.sample_uniform(batch_size, self.replay_rng)
        except ReplayWarmupError:
            return
        batch = {
            "obs": tensor_obs([item.obs for item in transitions], torch.device(self.device)),
            "actions": torch.as_tensor([item.action for item in transitions], dtype=torch.long, device=self.device),
            "rewards": torch.as_tensor([item.reward for item in transitions], dtype=torch.float32, device=self.device),
            "next_obs": tensor_obs([item.next_obs for item in transitions], torch.device(self.device)),
            "terminated": torch.as_tensor([item.terminated for item in transitions], dtype=torch.bool, device=self.device),
        }
        stats = self.trainer.update(batch)
        if float(stats.get("finite", 0.0)) != 1.0:
            raise FloatingPointError(f"A2 non-finite update at step {self.global_step}: {stats}")
        stats.update({"global_step": self.global_step, "replay_size": len(self.replay)})
        self.telemetry.append(stats)

    @torch.no_grad()
    def _eval_episode(self, seed: int) -> dict[str, Any]:
        env = make_env(self.config, "voradj_coverage", seed)
        obs = env.reset()
        assert_runtime_aw9(env)
        total = 0.0
        for _ in range(int(env.episode_max_length)):
            active = [index for index, item in enumerate(obs) if item is not None]
            actions: list[Optional[int]] = [None] * len(obs)
            if active:
                batch = stack_obs([obs[index] for index in active], self.device)
                values = self.trainer.actor.deterministic_action(batch).cpu().tolist()
                for index, action in zip(active, values):
                    actions[index] = int(action)
            result = env.step(actions, [])
            total += float(np.sum(result.rewards))
            obs = result.observations
            if any(result.dones):
                break
        record = dict(env.episode_record(task="coverage"))
        record["return_sum"] = total
        record["mode"] = "argmax"
        return record

    def _evaluate(self, step: int) -> dict[str, Any]:
        episodes = 1 if self.smoke else int(((self.config.get("ac", {}) or {}).get("evaluation", {}) or {}).get("episodes", 20))
        records = [self._eval_episode(self.seed + 5000 + index + int(step) * 1000) for index in range(episodes)]
        successes = [record for record in records if bool(record.get("coverage_strict_success", record.get("ce_success", False)))]
        times = [float(record.get("time_to_ce")) for record in successes if record.get("time_to_ce") is not None]
        def stats(values: list[float]) -> dict[str, Any]:
            if not values:
                return {"mean": None, "median": None, "p90": None, "success_n": 0}
            array = np.asarray(values, dtype=float)
            return {"mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.percentile(array, 90)), "success_n": int(array.size)}
        return {
            "schema": "ac-a2-pure-coverage-eval-v1",
            "step": int(step),
            "episodes": len(records),
            "strict_ce_success": int(len(successes)),
            "strict_ce_rate": float(len(successes) / max(len(records), 1)),
            "collision_n": int(sum(bool(record.get("collision", record.get("collision_event", False))) for record in records)),
            "ce_rms_mean": float(np.mean([float(record.get("ce_rms", record.get("coverage_ce_center_rms", 0.0))) for record in records])),
            "ce_max_mean": float(np.mean([float(record.get("ce_max", record.get("coverage_ce_center_max", 0.0))) for record in records])),
            "area_cv_mean": float(np.mean([float(record.get("area_cv", record.get("coverage_strict_area_cv", 0.0))) for record in records])),
            "time_to_strict_ce_seconds": stats(times),
            "records": records,
        }

    def _checkpoint(self, step: int) -> None:
        model = self.run_dir / "checkpoints" / f"model_step_{int(step):09d}.pt"
        atomic_torch_save(self.trainer.checkpoint_payload(contract=self._contract(), global_step=step, episode=self.episode_count, include_runtime=False), model)
        resume = self.run_dir / "resume_latest" / "full_resume.pt"
        atomic_torch_save(self.trainer.checkpoint_payload(contract=self._contract(), global_step=step, episode=self.episode_count, include_runtime=True), resume)

    def _write_progress(self, step: int, evaluation: dict[str, Any]) -> None:
        write_json(self.run_dir / "progress.json", {
            "task_id": "A2", "status": "RUNNING", "step": int(step), "next_formal": 25000 if step < 25000 else 50000 if step < 50000 else 75000 if step < 75000 else 100000 if step < 100000 else None,
            "device": self.device, "run_dir": self.run_dir, "replay_size": len(self.replay), "episode_count": self.episode_count,
            "alpha": float(self.trainer.alpha.detach()), "telemetry_tail": list(self.telemetry)[-20:], "evaluation": evaluation,
        })

    def run(self) -> dict[str, Any]:
        total = int(self.smoke_steps if self.smoke else self.config.get("total_timesteps", 100000))
        milestones = {0, 25000, 50000, 75000, 100000}
        if not self.smoke:
            initial = self._evaluate(0)
            self._checkpoint(0)
            self._write_progress(0, initial)
        for _ in range(total):
            self._collect_step()
            if not self.smoke and self.global_step in milestones:
                evaluation = self._evaluate(self.global_step)
                self._checkpoint(self.global_step)
                self._write_progress(self.global_step, evaluation)
        evaluation = self._evaluate(self.global_step)
        action_counts = {str(index): int(self.action_counts.get(index, 0)) for index in range(9)}
        counts = np.asarray(list(action_counts.values()), dtype=float)
        probabilities = counts / max(float(counts.sum()), 1.0)
        action_entropy = float(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum())
        final_hashes = {
            "actor": state_hash(self.trainer.actor),
            "critic1": state_hash(self.trainer.critic1),
            "critic2": state_hash(self.trainer.critic2),
        }
        result = {
            "schema": "ac-a2-discrete-sac-pure-coverage-run-v1",
            "task_id": "A2", "config_path": self.config_path, "run_name": self.run_name,
            "device": self.device, "smoke": self.smoke, "steps": self.global_step, "episodes": self.episode_count,
            "formal_network_parameter_counts": {"actor": parameter_count(self.trainer.actor), "critic1": parameter_count(self.trainer.critic1), "critic2": parameter_count(self.trainer.critic2)},
            "initial_hashes": self.initial_hashes, "final_hashes": final_hashes,
            "aw9_live": self.aw9_live, "action_counts": action_counts, "action_entropy": action_entropy,
            "replay_size": len(self.replay), "alpha": float(self.trainer.alpha.detach()),
            "telemetry_tail": list(self.telemetry)[-20:], "episode_tail": list(self.episode_tail)[-10:],
            "all_telemetry_finite": all(float(item.get("finite", 0.0)) == 1.0 for item in self.telemetry),
            "deterministic_eval": evaluation, "contract": self._contract(), "resume_info": self.resume_info,
        }
        write_json(self.run_dir / ("smoke_report.json" if self.smoke else "report.json"), result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument("--output-root")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-path")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    if args.device:
        config["device"] = args.device
    if args.run_name:
        config["run_name"] = args.run_name
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    result = DiscreteSACCoverageRunner(
        config, config_path, smoke=args.smoke,
        output_root=Path(args.output_root) if args.output_root else None,
        resume_path=Path(args.resume_path) if args.resume_path else None,
    ).run()
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
