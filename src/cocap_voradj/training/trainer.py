
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from cocap_voradj.config import ConfigManager
from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.base import CoCapEnv
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.training.replay import ReplayBuffer, concat_replay_batches, stack_obs


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_scalar_step_schedule(
    raw_schedule: Any,
    *,
    default_value: float,
    field_name: str,
) -> List[Tuple[int, float]]:
    if not raw_schedule:
        return [(0, float(default_value))]
    if not isinstance(raw_schedule, list):
        raise ValueError(f"{field_name}_schedule must be a list")
    parsed: List[Tuple[int, float]] = []
    for item in raw_schedule:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}_schedule entries must be mappings")
        step = int(item.get("step", 0))
        if step < 0:
            raise ValueError(f"{field_name}_schedule step must be non-negative")
        raw_value = item.get("value", item.get(field_name, item.get("weight")))
        if raw_value is None:
            raise ValueError(
                f"{field_name}_schedule entries need value, weight, or {field_name}"
            )
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{field_name}_schedule values must be finite")
        parsed.append((step, value))
    parsed.sort(key=lambda item: item[0])
    steps = [step for step, _ in parsed]
    if len(steps) != len(set(steps)):
        raise ValueError(f"{field_name}_schedule contains duplicate steps")
    return parsed


def scalar_step_schedule_value(
    schedule: List[Tuple[int, float]],
    *,
    global_step: int,
    default_value: float,
) -> float:
    value = float(default_value)
    for step, candidate in schedule:
        if int(global_step) < step:
            break
        value = float(candidate)
    return value


def set_global_config(config: Dict[str, Any]) -> None:
    manager = ConfigManager.get_instance()
    manager._config = copy.deepcopy(config)


def zero_like_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.zeros_like(v) for k, v in obs.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def safe_json_dumps(payload: Any, **kwargs: Any) -> str:
    return json.dumps(_json_safe(payload), allow_nan=False, **kwargs)


FULL_RESUME_SCHEMA = "cocap_iqn_full_resume_v1"


def _resume_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the immutable training contract used to reject unsafe resumes."""

    contract = copy.deepcopy(config)
    for key in ("device", "output_root", "run_name", "total_timesteps"):
        contract.pop(key, None)
    checkpointing = contract.get("checkpointing")
    if isinstance(checkpointing, dict):
        checkpointing.pop("resume_path", None)
        checkpointing.pop("full_resume_path", None)
    return contract


def _resume_contract_hash(config: Dict[str, Any]) -> str:
    encoded = safe_json_dumps(_resume_contract(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CoCapTrainer:
    _FULL_RESUME_STATIC_FIELDS = frozenset({
        "config",
        "project_root",
        "device",
        "run_dir",
        "ckpt_dir",
        "episode_log",
        "metric_log",
        "model",
        "target_model",
        "optimizer",
        "total_timesteps",
        "full_resume_enabled",
        "resume_path",
        "full_resume_path",
    })

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.project_root = Path(__file__).resolve().parents[3]
        self.seed = int(self.config.get("seed", 20260629))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.train_mode = self.config.get("train_mode", "coverage")
        if self.train_mode not in {"coverage", "encirclement", "m1new", "voradj", "voradj_mixed_coverage"}:
            raise ValueError("train_mode must be coverage, encirclement, m1new, voradj, or voradj_mixed_coverage")

        run_root = Path(self.config.get("output_root", "runs"))
        run_name = self.config.get("run_name", f"{self.train_mode}_{time.strftime('%Y%m%d_%H%M%S')}")
        self.run_dir = run_root / run_name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "effective_config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        self.episode_log = (self.run_dir / "episodes.jsonl").open("a", encoding="utf-8")
        self.metric_log = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")

        self.total_timesteps = int(self.config.get("total_timesteps", 20000))
        checkpointing_cfg = self.config.get("checkpointing", {}) or {}
        self.full_resume_enabled = bool(checkpointing_cfg.get("full_resume", False))
        raw_resume_path = str(checkpointing_cfg.get("resume_path", "")).strip()
        self.resume_path = self._resolve_checkpoint_path(raw_resume_path) if raw_resume_path else None
        raw_full_resume_path = str(checkpointing_cfg.get("full_resume_path", "")).strip()
        self.full_resume_path = (
            self._resolve_checkpoint_path(raw_full_resume_path)
            if raw_full_resume_path
            else self.ckpt_dir / "resume_latest.pt"
        )
        iqn_cfg = self.config.get("iqn", {})
        self.batch_size = int(iqn_cfg.get("batch_size", 64))
        self.gamma = float(
            (self.config.get("discount", {}) or {}).get(
                "gamma",
                iqn_cfg.get("gamma", 0.99),
            )
        )
        self.min_replay_size = int(iqn_cfg.get("min_replay_size", 1000))
        self.train_freq = int(iqn_cfg.get("train_freq", 4))
        self.target_update_freq = int(iqn_cfg.get("target_update_freq", 10000))
        self.checkpoint_freq = int(iqn_cfg.get("checkpoint_freq", 25000))
        self.log_freq_steps = int(iqn_cfg.get("log_freq_steps", 1000))
        self.epsilon_start = float(iqn_cfg.get("epsilon_start", 0.6))
        self.epsilon_final = float(iqn_cfg.get("epsilon_final", 0.05))
        self.epsilon_decay_steps = int(iqn_cfg.get("epsilon_decay_steps", max(self.total_timesteps // 2, 1)))
        self.grad_clip = float(iqn_cfg.get("gradient_clip_norm", iqn_cfg.get("gradient_clip", 10.0)))
        self.training_quantile_samples = max(1, int(iqn_cfg.get("training_quantile_samples", iqn_cfg.get("train_quantiles", iqn_cfg.get("num_quantiles", 16)))))
        self.target_quantile_samples = max(1, int(iqn_cfg.get("target_quantiles", self.training_quantile_samples)))
        self.huber_kappa = max(float(iqn_cfg.get("huber_kappa", 1.0)), 1e-6)
        self.update_rule = str(iqn_cfg.get("update_rule", "distributional_iqn")).strip().lower()
        if self.update_rule not in {"distributional_iqn", "mean_dqn"}:
            raise ValueError("iqn.update_rule must be 'distributional_iqn' or 'mean_dqn'")
        self.replay_done_mode = str(iqn_cfg.get("replay_done_mode", "terminal_state")).strip().lower()
        base_learning_rate = float(iqn_cfg.get("learning_rate", 1e-4))
        schedule_cfg = iqn_cfg.get("learning_rate_schedule", []) or []
        self.learning_rate_schedule = sorted(
            [
                (
                    int(item.get("step", 0)),
                    float(item.get("learning_rate", item.get("lr", base_learning_rate))),
                )
                for item in schedule_cfg
            ],
            key=lambda item: item[0],
        )
        if not self.learning_rate_schedule:
            self.learning_rate_schedule = [(0, base_learning_rate)]
        self.current_learning_rate = base_learning_rate
        reward_cfg = self.config.get("reward", {}) or {}
        self.base_coverage_ce_speed_weight = float(
            reward_cfg.get("coverage_ce_speed_weight", 0.05)
        )
        self.coverage_ce_speed_weight_schedule = parse_scalar_step_schedule(
            reward_cfg.get("coverage_ce_speed_weight_schedule", []),
            default_value=self.base_coverage_ce_speed_weight,
            field_name="coverage_ce_speed_weight",
        )
        self.current_coverage_ce_speed_weight = self.base_coverage_ce_speed_weight
        self.update_steps = 0
        self.global_step = 0
        self.episode_idx = 0
        self.loss_ema: Optional[float] = None

        model_cfg = CoCapNetConfig(
            hidden_dim=int(iqn_cfg.get("hidden_dim", 128)),
            num_heads=int(iqn_cfg.get("num_heads", 4)),
            num_layers=int(iqn_cfg.get("num_layers", 2)),
            action_size=int(iqn_cfg.get("action_size", 9)),
            self_feature_dim=int(iqn_cfg.get("self_feature_dim", 7)),
            max_pursuers=int(self.config.get("perception", {}).get("max_pursuer_num", 12)),
            max_evaders=int(self.config.get("perception", {}).get("max_evader_num", 8)),
            max_obstacles=int(self.config.get("perception", {}).get("max_obstacle_num", 5)),
            num_quantiles=int(iqn_cfg.get("num_quantiles", 16)),
            num_cosine_features=int(iqn_cfg.get("num_cosine_features", 32)),
            architecture=str(iqn_cfg.get("architecture", "dual_head")),
            pursuing_embed_dim=int(iqn_cfg.get("pursuing_embed_dim", 8)),
        )
        self.model = CoCapIQN(model_cfg).to(self.device)
        self.target_model = CoCapIQN(model_cfg).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=base_learning_rate, eps=float(iqn_cfg.get("adam_eps", 1e-8)))
        self.multitask_initialization = (
            (self.config.get("multitask_training", {}) or {}).get("initialization", {}) or {}
        )
        pretrained = self.config.get("pretrained", {}) or {}
        if self.train_mode == "m1new" and self.multitask_initialization:
            if pretrained.get("path"):
                raise ValueError("m1new multitask initialization cannot be combined with pretrained.path")
            self._load_multitask_specialists(self.multitask_initialization)
        elif pretrained.get("path"):
            self._load_pretrained(pretrained)

        capacity = int(iqn_cfg.get("replay_capacity", 1000000))
        if self.train_mode == "m1new":
            m1_cfg = self.config.get("m1", {}) or {}
            self.task_order = list(m1_cfg.get("task_order", ["encirclement", "coverage"]))
            self.replays = {"coverage": ReplayBuffer(capacity), "encirclement": ReplayBuffer(capacity)}
            self.cross_init = {"coverage": deque(maxlen=int(m1_cfg.get("cross_init_capacity", 500))),
                               "encirclement": deque(maxlen=int(m1_cfg.get("cross_init_capacity", 500)))}
            self.step_balanced = bool(m1_cfg.get("step_balanced_task_selection", False))
            self.task_total_steps: Dict[str, int] = {"coverage": 0, "encirclement": 0}
            rebalance = m1_cfg.get("update_rebalance", {}) or {}
            self.update_rebalance_enabled = bool(rebalance.get("enabled", False))
            self.update_rebalance_strategy = str(rebalance.get("strategy", "adaptive_interleaved"))
            if self.update_rebalance_strategy != "adaptive_interleaved":
                raise ValueError("m1.update_rebalance.strategy must be 'adaptive_interleaved'")
            self.update_rebalance_threshold = int(rebalance.get("threshold", 50))
            self.update_rebalance_min_replay_size = int(rebalance.get("min_replay_size", self.min_replay_size))
            self.update_rebalance_toggle = 0
        elif self.train_mode == "voradj_mixed_coverage":
            self.task_order = ["voradj", "voradj_coverage"]
            default_counts = {
                "pursuing": self.batch_size // 2,
                "pre_capture_cover": self.batch_size // 4,
                "post_capture_cover": self.batch_size - self.batch_size // 2 - self.batch_size // 4,
            }
            configured_counts = (self.config.get("voradj", {}) or {}).get("replay_batch_counts", {}) or {}
            self.voradj_replay_classes = list(configured_counts) if configured_counts else list(default_counts)
            allowed_classes = {
                "pursuing",
                "pre_capture_cover",
                "post_capture_cover",
                "post_capture_real",
                "recovery_pure",
            }
            if not set(self.voradj_replay_classes).issubset(allowed_classes):
                raise ValueError("voradj.replay_batch_counts contains an unknown replay class")
            if not {"pursuing", "pre_capture_cover"}.issubset(self.voradj_replay_classes):
                raise ValueError("voradj replay classes must include pursuing and pre_capture_cover")
            post_classes = set(self.voradj_replay_classes) & {
                "post_capture_cover",
                "post_capture_real",
                "recovery_pure",
            }
            if not post_classes:
                raise ValueError("voradj replay classes must include a post-capture coverage class")
            self.replays = {name: ReplayBuffer(capacity) for name in self.voradj_replay_classes}
            self.voradj_replay_batch_counts = {
                name: int(configured_counts.get(name, default_counts.get(name, 0)))
                for name in self.voradj_replay_classes
            }
            if sum(self.voradj_replay_batch_counts.values()) != self.batch_size:
                raise ValueError("voradj.replay_batch_counts must sum to iqn.batch_size")
            if any(count <= 0 for count in self.voradj_replay_batch_counts.values()):
                raise ValueError("voradj.replay_batch_counts values must be positive")
            recovery_cfg = (self.config.get("voradj", {}) or {}).get("recovery", {}) or {}
            self.recovery_init_pool = deque(maxlen=int(recovery_cfg.get("capture_state_pool_capacity", 1000)))
            self.recovery_from_capture_ratio = float(recovery_cfg.get("captured_state_ratio", 0.5))
            self.recovery_map_random_ratio_within_non_capture = float(
                recovery_cfg.get("map_random_ratio_within_non_capture", recovery_cfg.get("ordinary_map_random_ratio_within_non_capture", 0.0))
            )
            self.recovery_map_random_ratio_within_non_capture = min(max(self.recovery_map_random_ratio_within_non_capture, 0.0), 1.0)
            self.real_post_min_replay_size = int(
                (self.config.get("voradj", {}) or {}).get("real_post_min_replay_size", 256)
            )
            self.last_reset_source: Dict[str, str] = {}
            self.cross_init = {"coverage": deque(maxlen=1), "encirclement": deque(maxlen=1)}
            self.update_rebalance_enabled = False
            self.update_rebalance_strategy = "adaptive_interleaved"
            self.update_rebalance_threshold = 0
            self.update_rebalance_min_replay_size = self.min_replay_size
            self.update_rebalance_toggle = 0
            self.last_batch_task_counts: Dict[str, int] = {}
            self.last_batch_phase_counts: Dict[str, int] = {}
            self.last_batch_scene_counts: Dict[str, int] = {}
            self.last_batch_reset_source_counts: Dict[str, int] = {}
            self.last_batch_buffer_counts: Dict[str, int] = {}
            self.last_batch_reward_stats: Dict[str, Any] = {}
            self.last_batch_role_switch_count = 0
        elif self.train_mode == "voradj":
            task = "voradj"
            self.task_order = [task]
            self.replays = {task: ReplayBuffer(capacity)}
            self.cross_init = {"coverage": deque(maxlen=1), "encirclement": deque(maxlen=1)}
            self.update_rebalance_enabled = False
            self.update_rebalance_strategy = "adaptive_interleaved"
            self.update_rebalance_threshold = 0
            self.update_rebalance_min_replay_size = self.min_replay_size
            self.update_rebalance_toggle = 0
            self.last_batch_task_counts: Dict[str, int] = {}
            self.last_batch_phase_counts: Dict[str, int] = {}
            self.last_batch_scene_counts: Dict[str, int] = {}
            self.last_batch_reset_source_counts: Dict[str, int] = {}
        else:
            task = "coverage" if self.train_mode == "coverage" else "encirclement"
            self.task_order = [task]
            self.replays = {task: ReplayBuffer(capacity)}
            self.cross_init = {"coverage": deque(maxlen=1), "encirclement": deque(maxlen=1)}
            self.update_rebalance_enabled = False
            self.update_rebalance_strategy = "adaptive_interleaved"
            self.update_rebalance_threshold = 0
            self.update_rebalance_min_replay_size = self.min_replay_size
            self.update_rebalance_toggle = 0
        self.update_counts: Dict[str, int] = {task: 0 for task in self.replays}
        self.last_update_task: Optional[str] = None
        self.envs = {task: self._make_env(task) for task in self.task_order}
        self._set_coverage_ce_control_weights()
        self.apf_agents: Dict[str, List[ApfAgent]] = defaultdict(list)
        recent_tasks = set(self.replays) | set(self.task_order) | {"coverage", "encirclement", "voradj"}
        self.recent_window_size = int(self.config.get("recent_window", 100))
        self.recent = {task: deque(maxlen=self.recent_window_size) for task in recent_tasks}
        self.recent_diagnostic_fields = [
            "success",
            "captured",
            "fully_capture",
            "collision",
            "coverage_loose",
            "coverage_strict",
            "coverage_cv015",
            "soft_oob",
            "post_capture_coverage",
            "post_capture_window_expired",
            "pure_coverage",
            "coverage_geometric",
            "coverage_settled",
            "coverage_settle_timeout",
            "episode_end_stationary",
            "post_capture_stationary",
            "post_capture_area_ok",
            "post_capture_center_ok",
            "post_capture_inside_ok",
            "post_capture_cv015",
        ]
        self.recent_numeric_diagnostic_fields = [
            "length",
            "active_pursuers",
            "oob_pursuer_steps",
            "reward_per_transition",
            "coverage_strict_area_cv",
            "coverage_strict_center_ok_ratio",
            "coverage_strict_inside_ratio",
            "episode_end_mean_speed",
            "episode_end_max_speed",
        ]
        self.recent_diagnostics = {
            task: {
                field: deque(maxlen=self.recent_window_size)
                for field in self.recent_diagnostic_fields + self.recent_numeric_diagnostic_fields
            }
            for task in recent_tasks
        }
        if self.resume_path is not None:
            self._load_full_resume(self.resume_path)
        else:
            self._save_checkpoint("init.pt")

    def _resolve_checkpoint_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _load_pretrained(self, pretrained: Dict[str, Any]) -> None:
        path = self._resolve_checkpoint_path(str(pretrained["path"]))
        payload = torch.load(path, map_location=self.device)
        source = payload.get("state_dict", payload)
        target = self.model.state_dict()
        mode = str(pretrained.get("compatibility_mode", "shape_compatible"))
        adapted: List[str] = []
        loaded: List[str] = []
        skipped: List[str] = []

        source_for_load = dict(source)
        if mode == "voradj_v1_to_v2":
            env_cfg = self.config.get("env", {}) or {}
            distance_scale = float(np.hypot(
                float(env_cfg.get("x_boundary_right", env_cfg.get("width", 1.0))) - float(env_cfg.get("x_boundary_left", 0.0)),
                float(env_cfg.get("y_boundary_top", env_cfg.get("height", 1.0))) - float(env_cfg.get("y_boundary_bottom", 0.0)),
            ))
            distance_scale = max(distance_scale, 1e-6)

            self_key = "encoders.self.0.weight"
            if self_key in source and self_key in target and source[self_key].shape[1] == 8 and target[self_key].shape[1] == 9:
                old = source[self_key]
                new = torch.zeros_like(target[self_key])
                new[:, 0:2] = old[:, 0:2]
                new[:, 2] = old[:, 2] * distance_scale
                # New boundary-vector and OOB columns intentionally start at
                # zero because the old world-axis boundary semantics are not
                # transferable to robot-frame geometry.
                new[:, 6] = old[:, 5] * distance_scale
                new[:, 7] = old[:, 6] * distance_scale
                new[:, 8] = old[:, 7]
                source_for_load[self_key] = new
                adapted.append(self_key)

            scaled_columns = {
                "encoders.pursuers.0.weight": (0, 1, 4),
                "encoders.evaders.0.weight": (0, 1, 4),
                "encoders.obstacles.0.weight": (0, 1, 2, 3),
            }
            for key, columns in scaled_columns.items():
                if key in source and key in target and source[key].shape == target[key].shape:
                    value = source[key].clone()
                    value[:, list(columns)] *= distance_scale
                    source_for_load[key] = value
                    adapted.append(key)
        elif mode != "shape_compatible":
            raise ValueError(f"unsupported pretrained.compatibility_mode: {mode}")

        for key, value in source_for_load.items():
            if key in target and target[key].shape == value.shape:
                target[key] = value
                loaded.append(key)
            else:
                skipped.append(key)
        self.model.load_state_dict(target)
        self.target_model.load_state_dict(self.model.state_dict())
        report = {
            "path": str(path),
            "compatibility_mode": mode,
            "loaded_count": len(loaded),
            "adapted_keys": sorted(adapted),
            "skipped_keys": sorted(skipped),
        }
        (self.run_dir / "pretrained_load.json").write_text(
            safe_json_dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_multitask_specialists(self, initialization: Dict[str, Any]) -> None:
        sources = {
            "coverage": str(initialization.get("coverage_checkpoint", "")),
            "encirclement": str(initialization.get("encirclement_checkpoint", "")),
        }
        backbone_source = str(initialization.get("backbone_source", "coverage")).strip().lower()
        if backbone_source not in {"coverage", "encirclement"}:
            raise ValueError("multitask initialization.backbone_source must be coverage or encirclement")

        shared_prefixes = ("encoders.", "type_embedding.", "transformer.", "cos_embedding.", "pis")
        prefixes = {
            "coverage": (
                (shared_prefixes if backbone_source == "coverage" else ())
                + ("coverage_fusion.", "coverage_head.")
            ),
            "encirclement": (
                (shared_prefixes if backbone_source == "encirclement" else ())
                + (
                    "encirclement_fusion.",
                    "target_query.",
                    "target_key.",
                    "target_value.",
                    "encirclement_head.",
                )
            ),
        }

        own = self.model.state_dict()
        loaded_by_task: Dict[str, List[str]] = {}
        skipped_by_task: Dict[str, List[str]] = {}
        for task in ("coverage", "encirclement"):
            raw_path = sources[task]
            if not raw_path:
                continue
            path = self._resolve_checkpoint_path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"CoCap dual-task {task} checkpoint not found: {path}")
            payload = torch.load(path, map_location=self.device)
            state = payload.get("state_dict", payload)
            selected = {
                key: value
                for key, value in state.items()
                if key.startswith(prefixes[task])
            }
            compatible = {
                key: value
                for key, value in selected.items()
                if key in own and own[key].shape == value.shape
            }
            expected = {key for key in own if key.startswith(prefixes[task])}
            missing = sorted(expected - set(compatible))
            skipped = sorted(set(selected) - set(compatible))
            if missing or skipped:
                raise ValueError(
                    f"CoCap dual-task {task} checkpoint is not fully compatible: "
                    f"missing={missing}, skipped={skipped}"
                )
            for key, value in compatible.items():
                own[key] = value
            loaded_by_task[task] = sorted(compatible)
            skipped_by_task[task] = skipped

        self.model.load_state_dict(own)
        self.target_model.load_state_dict(self.model.state_dict())
        report = {
            "mode": f"{backbone_source}_backbone_plus_dual_native_heads",
            "backbone_source": backbone_source,
            "sources": sources,
            "loaded_counts": {task: len(keys) for task, keys in loaded_by_task.items()},
            "loaded_keys": loaded_by_task,
            "skipped_keys": skipped_by_task,
        }
        (self.run_dir / "multitask_pretrained_load.json").write_text(
            safe_json_dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _task_config(self, task: str) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.config)
        task_overrides = (self.config.get("tasks", {}) or {}).get(task, {})
        cfg = deep_update(cfg, task_overrides)
        cfg["env"] = copy.deepcopy(cfg.get("env", {}))
        cfg["env"]["num_evaders"] = int(cfg["env"].get("num_evaders", 0 if task in {"coverage", "voradj_coverage"} else 1))
        if task in {"coverage", "voradj_coverage"}:
            cfg["env"]["num_evaders"] = 0
        return cfg

    def _make_env(self, task: str) -> CoCapEnv:
        cfg = self._task_config(task)
        set_global_config(cfg)
        if task in {"voradj", "voradj_coverage"}:
            return VorAdjEnv(cfg, seed=self.seed + (43 if task == "voradj" else 47))
        return CoCapEnv(cfg, task=task, seed=self.seed + (17 if task == "coverage" else 31))

    def epsilon(self) -> float:
        frac = min(1.0, self.global_step / max(self.epsilon_decay_steps, 1))
        return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

    def _set_coverage_ce_control_weights(self) -> float:
        value = scalar_step_schedule_value(
            self.coverage_ce_speed_weight_schedule,
            global_step=self.global_step,
            default_value=self.base_coverage_ce_speed_weight,
        )
        for env in self.envs.values():
            if isinstance(env, VorAdjEnv):
                env.reward_cfg["coverage_ce_speed_weight"] = float(value)
        self.current_coverage_ce_speed_weight = float(value)
        return self.current_coverage_ce_speed_weight

    def _set_learning_rate(self) -> float:
        learning_rate = self.learning_rate_schedule[0][1]
        for step, candidate in self.learning_rate_schedule:
            if self.global_step < step:
                break
            learning_rate = candidate
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.current_learning_rate = float(learning_rate)
        return self.current_learning_rate

    def _reset_task(self, task: str) -> List[Optional[Dict[str, np.ndarray]]]:
        env = self.envs[task]
        set_global_config(env.config)
        initial_positions = None
        initial_active = None
        reset_source = "environment_default"
        if self.train_mode == "m1new":
            source = "encirclement" if task == "coverage" else "coverage"
            ratio = float(self.config.get("m1", {}).get("cross_init_ratio", 0.3))
            if self.cross_init[source] and random.random() < ratio:
                initial = random.choice(list(self.cross_init[source]))
                if isinstance(initial, dict):
                    initial_positions = initial.get("positions")
                    initial_active = initial.get("active_mask")
                else:
                    initial_positions = initial
                reset_source = f"cross_init_{source}"
        elif self.train_mode == "voradj_mixed_coverage" and task == "voradj_coverage":
            if self.recovery_init_pool and random.random() < self.recovery_from_capture_ratio:
                initial = random.choice(list(self.recovery_init_pool))
                initial_positions = initial.get("positions")
                initial_active = initial.get("active_mask")
                reset_source = "capture_snapshot"
            elif random.random() < self.recovery_map_random_ratio_within_non_capture:
                reset_source = "ordinary_map_random"
            else:
                reset_source = "synthetic_cluster"
        if self.train_mode == "voradj_mixed_coverage":
            self.last_reset_source[task] = reset_source
        original_spawn_mode = None
        if self.train_mode == "voradj_mixed_coverage" and task == "voradj_coverage" and reset_source == "ordinary_map_random":
            original_spawn_mode = env.env_cfg.get("pursuer_spawn_mode", "map_random")
            env.env_cfg["pursuer_spawn_mode"] = "map_random"
        try:
            obs = env.reset(initial_pursuer_positions=initial_positions, initial_pursuer_active=initial_active)
        finally:
            if original_spawn_mode is not None:
                env.env_cfg["pursuer_spawn_mode"] = original_spawn_mode
        self.apf_agents[task] = [ApfAgent(evader.a, evader.w) for evader in env.evaders]
        return obs

    def _evader_actions(self, task: str) -> List[Optional[int]]:
        env = self.envs[task]
        if not env.evaders:
            return []
        set_global_config(env.config)
        observations = env.get_evader_observations_for_apf()
        if hasattr(env, "configure_evader_apf_agents"):
            env.configure_evader_apf_agents(self.apf_agents[task])
        actions = []
        for i, obs in enumerate(observations):
            if obs is None:
                actions.append(None)
            else:
                if i >= len(self.apf_agents[task]):
                    self.apf_agents[task].append(ApfAgent(env.evaders[i].a, env.evaders[i].w))
                actions.append(self.apf_agents[task][i].act(obs))
        return actions

    def _select_actions(self, task: str, obs_list: List[Optional[Dict[str, np.ndarray]]]) -> Tuple[List[Optional[int]], List[int]]:
        active = [i for i, obs in enumerate(obs_list) if obs is not None]
        actions: List[Optional[int]] = [None] * len(obs_list)
        if not active:
            return actions, active
        batch = stack_obs([obs_list[i] for i in active], self.device)
        with torch.no_grad():
            a = self.model.act(batch, mode=task, epsilon=self.epsilon()).detach().cpu().numpy().tolist()
        for idx, action in zip(active, a):
            actions[idx] = int(action)
        return actions, active

    def _voradj_replay_class(self, metadata: Dict[str, Any]) -> str:
        phase = str(metadata.get("phase", "pre_capture"))
        task_label = str(metadata.get("task_label", "coverage"))
        if phase == "post_capture" and "post_capture_real" in self.replays:
            return "post_capture_real"
        if phase == "pure_coverage" and "recovery_pure" in self.replays:
            return "recovery_pure"
        if phase in {"post_capture", "pure_coverage"}:
            return "post_capture_cover"
        if task_label == "capture":
            return "pursuing"
        return "pre_capture_cover"

    def _replay_size_for_task(self, task: str) -> int:
        if self.train_mode == "voradj_mixed_coverage":
            return int(sum(len(self.replays[name]) for name in self.voradj_replay_classes))
        return len(self.replays[task])

    def _select_update_task(self, current_task: str) -> str:
        if self.train_mode != "m1new" or not self.update_rebalance_enabled:
            return current_task
        if self.global_step % self.train_freq != 0:
            return current_task
        cov_updates = self.update_counts.get("coverage", 0)
        enc_updates = self.update_counts.get("encirclement", 0)
        gap = cov_updates - enc_updates
        if abs(gap) <= self.update_rebalance_threshold:
            return current_task
        lagging = "coverage" if gap < 0 else "encirclement"
        if current_task == lagging:
            return current_task
        replay = self.replays.get(lagging)
        required = max(self.min_replay_size, self.batch_size, self.update_rebalance_min_replay_size)
        if replay is None or len(replay) < required:
            return current_task
        self.update_rebalance_toggle += 1
        if self.update_rebalance_toggle % 2 == 0:
            return lagging
        return current_task

    def _update(self, task: str) -> Optional[float]:
        update_name = task
        loss_mode = task
        if self.global_step % self.train_freq != 0:
            return None
        if self.train_mode == "voradj_mixed_coverage":
            sample_counts = dict(self.voradj_replay_batch_counts)
            if (
                "post_capture_real" in sample_counts
                and "recovery_pure" in sample_counts
                and len(self.replays["post_capture_real"]) < self.real_post_min_replay_size
            ):
                sample_counts["recovery_pure"] += sample_counts["post_capture_real"]
                sample_counts["post_capture_real"] = 0
            required_by_class = {}
            for name, count in sample_counts.items():
                if count <= 0:
                    continue
                minimum = self.real_post_min_replay_size if name == "post_capture_real" else self.min_replay_size
                required_by_class[name] = max(minimum, count)
            if any(len(self.replays[name]) < required for name, required in required_by_class.items()):
                return None
            batches = [
                self.replays[name].sample(count, self.device)
                for name, count in sample_counts.items()
                if count > 0
            ]
            batch = concat_replay_batches(batches)
            self.last_batch_task_counts = dict(batch.get("batch_task_counts", {}))
            self.last_batch_phase_counts = dict(batch.get("batch_phase_counts", {}))
            self.last_batch_scene_counts = dict(batch.get("batch_scene_counts", {}))
            self.last_batch_reset_source_counts = dict(batch.get("batch_reset_source_counts", {}))
            self.last_batch_buffer_counts = dict(batch.get("batch_buffer_counts", {}))
            self.last_batch_reward_stats = dict(batch.get("batch_reward_stats", {}))
            self.last_batch_role_switch_count = int(batch.get("batch_role_switch_count", 0))
            update_name = "voradj_mixed_coverage"
            loss_mode = "voradj"
        else:
            replay = self.replays[task]
            if len(replay) < max(self.min_replay_size, self.batch_size):
                return None
            if self.train_mode == "voradj":
                # Direct random sampling avoids an O(replay_size) metadata scan on
                # every update. Task/phase counts are still reported from the
                # sampled batch for monitoring.
                batch = replay.sample(self.batch_size, self.device)
                self.last_batch_task_counts = dict(batch.get("batch_task_counts", {}))
                self.last_batch_phase_counts = dict(batch.get("batch_phase_counts", {}))
                self.last_batch_scene_counts = dict(batch.get("batch_scene_counts", {}))
                self.last_batch_reset_source_counts = dict(batch.get("batch_reset_source_counts", {}))
                self.last_batch_buffer_counts = dict(batch.get("batch_buffer_counts", {}))
                self.last_batch_reward_stats = dict(batch.get("batch_reward_stats", {}))
                self.last_batch_role_switch_count = int(batch.get("batch_role_switch_count", 0))
            else:
                batch = replay.sample(self.batch_size, self.device)
        self._set_learning_rate()
        if self.update_rule == "distributional_iqn":
            loss = self._distributional_iqn_td_loss(batch, loss_mode)
        else:
            loss = self._mean_dqn_td_loss(batch, loss_mode)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.update_steps += 1
        self.update_counts[update_name] = self.update_counts.get(update_name, 0) + 1
        self.last_update_task = update_name
        if self.global_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        value = float(loss.item())
        self.loss_ema = value if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * value
        return value

    def _mean_dqn_td_loss(self, batch: Dict[str, torch.Tensor], task: str) -> torch.Tensor:
        pred = self.model(batch["obs"], num_tau=self.training_quantile_samples, mode=task)["q_values"].mean(dim=1)
        q_taken = pred.gather(1, batch["actions"].view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(batch["next_obs"], num_tau=self.target_quantile_samples, mode=task)["q_values"].mean(dim=1).max(dim=1).values
            target = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * next_q
        return F.smooth_l1_loss(q_taken, target)

    def _distributional_iqn_td_loss(self, batch: Dict[str, torch.Tensor], task: str) -> torch.Tensor:
        batch_size = int(batch["actions"].shape[0])
        n_tau = self.training_quantile_samples
        tau_target = torch.rand(batch_size, n_tau, device=self.device)
        with torch.no_grad():
            target_out = self.target_model(batch["next_obs"], num_tau=n_tau, mode=task, tau=tau_target)
            q_targets_next = target_out["q_values"].detach().max(2)[0].unsqueeze(1)
            q_targets = (
                batch["rewards"].view(batch_size, 1, 1)
                + self.gamma
                * (1.0 - batch["dones"].view(batch_size, 1, 1))
                * q_targets_next
            )

        tau = torch.rand(batch_size, n_tau, device=self.device)
        out = self.model(batch["obs"], num_tau=n_tau, mode=task, tau=tau)
        q_values = out["q_values"]
        actions = batch["actions"].unsqueeze(-1).unsqueeze(-1).expand(-1, n_tau, 1)
        q_expected = q_values.gather(-1, actions)
        td_error = q_targets - q_expected
        huber = torch.where(
            td_error.abs() <= self.huber_kappa,
            0.5 * td_error.pow(2),
            self.huber_kappa * (td_error.abs() - 0.5 * self.huber_kappa),
        )
        sampled_taus = out.get("taus", out.get("tau"))
        if sampled_taus is None:
            sampled_taus = tau.unsqueeze(-1)
        quantile_weight = (sampled_taus - (td_error.detach() < 0).float()).abs()
        return (quantile_weight * huber / self.huber_kappa).sum(dim=1).mean(dim=1).mean()

    @staticmethod
    def _deque_mean(values: deque) -> float:
        return float(np.mean(values)) if values else 0.0

    def _recent_diagnostic_summary(self, task: str) -> Dict[str, float]:
        diagnostics = self.recent_diagnostics.get(task, {})
        if not diagnostics:
            return {}
        return {
            "recent_success_rate": self._deque_mean(diagnostics["success"]),
            "recent_capture_rate": self._deque_mean(diagnostics["captured"]),
            "recent_fully_capture_rate": self._deque_mean(diagnostics["fully_capture"]),
            "recent_collision_rate": self._deque_mean(diagnostics["collision"]),
            "recent_coverage_loose_rate": self._deque_mean(diagnostics["coverage_loose"]),
            "recent_coverage_strict_rate": self._deque_mean(diagnostics["coverage_strict"]),
            "recent_coverage_cv015_rate": self._deque_mean(diagnostics["coverage_cv015"]),
            "recent_soft_oob_rate": self._deque_mean(diagnostics["soft_oob"]),
            "recent_post_capture_coverage_rate": self._deque_mean(diagnostics["post_capture_coverage"]),
            "recent_post_capture_window_expired_rate": self._deque_mean(diagnostics["post_capture_window_expired"]),
            "recent_pure_coverage_rate": self._deque_mean(diagnostics["pure_coverage"]),
            "recent_coverage_geometric_rate": self._deque_mean(diagnostics["coverage_geometric"]),
            "recent_coverage_settled_rate": self._deque_mean(diagnostics["coverage_settled"]),
            "recent_coverage_settle_timeout_rate": self._deque_mean(diagnostics["coverage_settle_timeout"]),
            "recent_episode_end_stationary_rate": self._deque_mean(diagnostics["episode_end_stationary"]),
            "recent_post_capture_stationary_rate": self._deque_mean(diagnostics["post_capture_stationary"]),
            "recent_post_capture_area_ok_rate": self._deque_mean(diagnostics["post_capture_area_ok"]),
            "recent_post_capture_center_ok_rate": self._deque_mean(diagnostics["post_capture_center_ok"]),
            "recent_post_capture_inside_ok_rate": self._deque_mean(diagnostics["post_capture_inside_ok"]),
            "recent_post_capture_cv015_rate": self._deque_mean(diagnostics["post_capture_cv015"]),
            "recent_avg_length": self._deque_mean(diagnostics["length"]),
            "recent_avg_active_pursuers": self._deque_mean(diagnostics["active_pursuers"]),
            "recent_avg_oob_pursuer_steps": self._deque_mean(diagnostics["oob_pursuer_steps"]),
            "recent_avg_reward_per_transition": self._deque_mean(diagnostics["reward_per_transition"]),
            "recent_avg_coverage_strict_area_cv": self._deque_mean(diagnostics["coverage_strict_area_cv"]),
            "recent_avg_coverage_strict_center_ok_ratio": self._deque_mean(diagnostics["coverage_strict_center_ok_ratio"]),
            "recent_avg_coverage_strict_inside_ratio": self._deque_mean(diagnostics["coverage_strict_inside_ratio"]),
            "recent_avg_episode_end_mean_speed": self._deque_mean(diagnostics["episode_end_mean_speed"]),
            "recent_avg_episode_end_max_speed": self._deque_mean(diagnostics["episode_end_max_speed"]),
        }

    def _done_episode(self, task: str, env: CoCapEnv, record: Dict[str, Any]) -> None:
        if self.train_mode == "m1new":
            state = {
                "positions": record.get("pursuer_all_positions", record.get("pursuer_positions", [])),
                "active_mask": record.get("pursuer_active_mask", [True] * len(record.get("pursuer_positions", []))),
            }
            if task == "coverage" and record.get("coverage_training_success"):
                self.cross_init["coverage"].append(state)
            if task == "encirclement" and record.get("fully_capture"):
                self.cross_init["encirclement"].append(state)
        elif self.train_mode == "voradj_mixed_coverage" and task == "voradj":
            capture_snapshot = record.get("capture_snapshot")
            if isinstance(capture_snapshot, dict):
                self.recovery_init_pool.append({
                    "step": int(capture_snapshot.get("step", 0)),
                    "positions": capture_snapshot.get("positions", []),
                    "active_mask": capture_snapshot.get("active_mask", []),
                })
        if task == "coverage":
            metric_success = record.get("coverage_training_success")
        elif task in {"voradj", "voradj_coverage"}:
            metric_success = record.get("episode_success")
        else:
            metric_success = record.get("fully_capture")
        self.recent[task].append(1.0 if metric_success else 0.0)
        voradj_metrics = record.get("voradj_metrics", {}) or {}
        post_capture_episode = bool(record.get("post_capture_episode", False))
        if task in self.recent_diagnostics:
            diagnostics = self.recent_diagnostics[task]
            bool_values = {
                "success": bool(metric_success),
                "captured": bool(record.get("captured", False)),
                "fully_capture": bool(record.get("fully_capture", False)),
                "collision": bool(record.get("collision_event", False)),
                "coverage_loose": bool(record.get("coverage_loose_success", False)),
                "coverage_strict": bool(record.get("coverage_strict_success", False)),
                "coverage_cv015": bool(record.get("coverage_cv015_success", record.get("coverage_cv_loose_success", False))),
                "soft_oob": bool(record.get("soft_boundary_out_of_bounds_event", False)),
                "post_capture_coverage": bool(voradj_metrics.get("post_capture_coverage_success", False)),
                "post_capture_window_expired": bool(voradj_metrics.get("post_capture_window_expired", False)),
                "pure_coverage": bool(voradj_metrics.get("pure_coverage_success", False)),
                "coverage_geometric": bool(record.get("coverage_geometric_success", False)),
                "coverage_settled": bool(record.get("coverage_settled_success", False)),
                "coverage_settle_timeout": bool(record.get("coverage_settle_timeout", False)),
                "episode_end_stationary": bool(record.get("episode_end_stationary", False)),
                "post_capture_stationary": bool(record.get("post_capture_stationary_success", False)),
                "post_capture_area_ok": bool(post_capture_episode and record.get("coverage_strict_area_ok", False)),
                "post_capture_center_ok": bool(post_capture_episode and record.get("coverage_strict_center_ok", False)),
                "post_capture_inside_ok": bool(post_capture_episode and record.get("coverage_strict_inside_ok", False)),
                "post_capture_cv015": bool(post_capture_episode and record.get("coverage_cv015_success", record.get("coverage_cv_loose_success", False))),
            }
            for key, value in bool_values.items():
                diagnostics[key].append(1.0 if value else 0.0)
            transitions = max(int(record.get("episode_transition_count", 0)), 1)
            diagnostics["length"].append(float(record.get("length", 0)))
            diagnostics["active_pursuers"].append(float(record.get("active_pursuers", 0)))
            diagnostics["oob_pursuer_steps"].append(float(record.get("soft_boundary_out_of_bounds_pursuer_steps", 0)))
            diagnostics["reward_per_transition"].append(float(record.get("episode_reward_sum", 0.0)) / transitions)
            diagnostics["coverage_strict_area_cv"].append(float(record.get("coverage_strict_area_cv", 0.0)))
            diagnostics["coverage_strict_center_ok_ratio"].append(float(record.get("coverage_strict_center_ok_ratio", 0.0)))
            diagnostics["coverage_strict_inside_ratio"].append(float(record.get("coverage_strict_inside_ratio", 0.0)))
            diagnostics["episode_end_mean_speed"].append(float(record.get("episode_end_mean_speed", 0.0)))
            diagnostics["episode_end_max_speed"].append(float(record.get("episode_end_max_speed", 0.0)))
        recent_summary = self._recent_diagnostic_summary(task)
        payload = {
            "episode": self.episode_idx,
            "global_step": self.global_step,
            "epsilon": self.epsilon(),
            "loss_ema": self.loss_ema,
            "replay_size": self._replay_size_for_task(task),
            "recent_success": float(np.mean(self.recent[task])) if self.recent[task] else 0.0,
            **recent_summary,
            "cross_init_coverage": len(self.cross_init["coverage"]),
            "cross_init_encirclement": len(self.cross_init["encirclement"]),
            **record,
        }
        if isinstance(voradj_metrics, dict):
            payload.update({
                "post_capture_coverage_success": bool(voradj_metrics.get("post_capture_coverage_success", False)),
                "post_capture_window_expired": bool(voradj_metrics.get("post_capture_window_expired", False)),
                "pure_coverage_success": bool(voradj_metrics.get("pure_coverage_success", False)),
                "capture_terminal_done": bool(voradj_metrics.get("capture_terminal_done", False)),
                "capture_agent_count": int(voradj_metrics.get("capture_agent_count", 0)),
                "coverage_agent_count": int(voradj_metrics.get("coverage_agent_count", 0)),
                "enemy_neighbor_ratio": float(voradj_metrics.get("enemy_neighbor_ratio", 0.0)),
                "support_candidate_count": int(voradj_metrics.get("support_candidate_count", 0)),
                "support_reward_blend_enabled": bool(voradj_metrics.get("support_reward_blend_enabled", False)),
                "support_reward_blend_active_count": int(voradj_metrics.get("support_reward_blend_active_count", 0)),
                "support_reward_blend_active_ratio": float(voradj_metrics.get("support_reward_blend_active_ratio", 0.0)),
                "support_reward_capture_weight": float(voradj_metrics.get("support_reward_capture_weight", 0.0)),
                "support_reward_coverage_weight": float(voradj_metrics.get("support_reward_coverage_weight", 0.0)),
                "reward_support_blend_capture_sum": float(voradj_metrics.get("reward_support_blend_capture_sum", 0.0)),
                "reward_support_blend_coverage_sum": float(voradj_metrics.get("reward_support_blend_coverage_sum", 0.0)),
            })
        if self.train_mode == "voradj_mixed_coverage":
            payload.update({
                f"replay_size_{name}": len(self.replays[name])
                for name in self.voradj_replay_classes
            })
            payload["recovery_init_pool_size"] = len(self.recovery_init_pool)
        if hasattr(self, "task_total_steps"):
            cov_s = self.task_total_steps.get("coverage", 0)
            enc_s = self.task_total_steps.get("encirclement", 0)
            payload["task_total_steps_coverage"] = cov_s
            payload["task_total_steps_encirclement"] = enc_s
            payload["task_step_ratio_cov_enc"] = round(cov_s / max(enc_s, 1), 3)
        if self.train_mode == "m1new":
            cov_u = self.update_counts.get("coverage", 0)
            enc_u = self.update_counts.get("encirclement", 0)
            payload["updates_coverage"] = cov_u
            payload["updates_encirclement"] = enc_u
            payload["update_ratio_cov_enc"] = round(cov_u / max(enc_u, 1), 3)
            payload["update_rebalance_enabled"] = bool(self.update_rebalance_enabled)
        self.episode_log.write(safe_json_dumps(payload, ensure_ascii=False) + "\n")
        self.episode_log.flush()
        self.episode_idx += 1
        print(safe_json_dumps({k: payload[k] for k in [
            "episode",
            "global_step",
            "task",
            "length",
            "episode_success",
            "recent_success",
            "recent_capture_rate",
            "recent_coverage_strict_rate",
            "recent_collision_rate",
            "recent_soft_oob_rate",
            "recent_post_capture_stationary_rate",
            "recent_post_capture_area_ok_rate",
            "recent_post_capture_center_ok_rate",
            "coverage_loose_success",
            "coverage_strict_success",
            "captured",
            "fully_capture",
            "post_capture_coverage_success",
            "post_capture_stationary_success",
            "coverage_strict_area_ok",
            "coverage_strict_center_ok",
            "coverage_strict_area_cv",
            "coverage_strict_center_ok_ratio",
            "pure_coverage_success",
            "capture_terminal_done",
            "collision_event",
        ] if k in payload}, ensure_ascii=False), flush=True)

    def _full_resume_runtime_state(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self._FULL_RESUME_STATIC_FIELDS
        }

    def _save_full_resume(self) -> Path:
        if not getattr(self, "_train_runtime_ready", False):
            raise RuntimeError("full resume cannot be saved before training runtime initialization")
        path = self.full_resume_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        payload = {
            "schema": FULL_RESUME_SCHEMA,
            "contract_hash": _resume_contract_hash(self.config),
            "contract": _resume_contract(self.config),
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "runtime": self._full_resume_runtime_state(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def _load_full_resume(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"IQN full-resume checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema") != FULL_RESUME_SCHEMA:
            raise ValueError("IQN full-resume checkpoint schema mismatch")
        expected_hash = _resume_contract_hash(self.config)
        if payload.get("contract_hash") != expected_hash:
            raise ValueError("IQN full-resume training contract mismatch; refusing unsafe resume")
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("_train_runtime_ready", False):
            raise ValueError("IQN full-resume checkpoint is missing initialized runtime state")
        self.model.load_state_dict(payload["model"])
        self.target_model.load_state_dict(payload["target_model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        for key, value in runtime.items():
            if key in self._FULL_RESUME_STATIC_FIELDS:
                continue
            setattr(self, key, value)
        if self.global_step > self.total_timesteps:
            raise ValueError(
                f"resume step {self.global_step} exceeds configured total_timesteps {self.total_timesteps}"
            )
        rng = payload.get("rng", {})
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        set_global_config(self.envs[self.current_task].config)
        (self.run_dir / "resume_loaded.json").write_text(
            safe_json_dumps(
                {
                    "schema": FULL_RESUME_SCHEMA,
                    "path": str(path),
                    "global_step": int(self.global_step),
                    "episode": int(self.episode_idx),
                    "contract_hash": expected_hash,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _save_checkpoint(self, name: str) -> Path:
        path = self.ckpt_dir / name
        self.model.save(
            str(path),
            extra={
                "global_step": self.global_step,
                "episode": self.episode_idx,
                "train_mode": self.train_mode,
                "coverage_ce_speed_weight": self.current_coverage_ce_speed_weight,
            },
        )
        latest = self.ckpt_dir / "latest.pt"
        self.model.save(
            str(latest),
            extra={
                "global_step": self.global_step,
                "episode": self.episode_idx,
                "train_mode": self.train_mode,
                "coverage_ce_speed_weight": self.current_coverage_ce_speed_weight,
            },
        )
        if self.full_resume_enabled and getattr(self, "_train_runtime_ready", False):
            self._save_full_resume()
        return path

    def train(self) -> Path:
        if getattr(self, "_train_runtime_ready", False):
            task_cursor = int(self.task_cursor)
            task = str(self.current_task)
            obs_list = self.current_observations
            episode_done = bool(self.current_episode_done)
            episode_reward = float(self.current_episode_reward)
            episode_reward_components = dict(self.current_episode_reward_components)
            episode_transition_count = int(self.current_episode_transition_count)
        else:
            task_cursor = 0
            task = self.task_order[task_cursor % len(self.task_order)]
            obs_list = self._reset_task(task)
            episode_done = False
            episode_reward = 0.0
            episode_reward_components = {
                "reward_coverage": 0.0,
                "reward_capture": 0.0,
                "reward_safety": 0.0,
                "reward_terminal": 0.0,
                "reward_speed_repeat": 0.0,
                "reward_early_speed": 0.0,
                "reward_deceleration": 0.0,
                "reward_settled_terminal": 0.0,
                "reward_motion_penalty": 0.0,
                "reward_ce_center": 0.0,
                "reward_ce_control": 0.0,
                "reward_ce_pbrs": 0.0,
                "reward_ce_terminal_correction": 0.0,
                "reward_support_blend_capture": 0.0,
                "reward_support_blend_coverage": 0.0,
            }
            episode_transition_count = 0
            self._train_runtime_ready = True
        while self.global_step < self.total_timesteps:
            env = self.envs[task]
            self._set_coverage_ce_control_weights()
            actions, active = self._select_actions(task, obs_list)
            prev_obs = {i: obs_list[i] for i in active}
            evader_actions = self._evader_actions(task)
            result = env.step(actions, evader_actions)
            step_done = bool(all(result.dones))
            for i in active:
                next_obs = result.observations[i]
                if next_obs is None:
                    next_obs = zero_like_obs(prev_obs[i])
                transition_done = bool(result.dones[i])
                info_state = result.infos[i].get("state") if i < len(result.infos) and isinstance(result.infos[i], dict) else None
                if self.replay_done_mode == "terminal_state" and info_state == "all targets captured":
                    transition_done = False
                metadata = {}
                if i < len(result.infos) and isinstance(result.infos[i], dict):
                    metadata = result.infos[i].get("replay_metadata", {}) or {}
                metadata = dict(metadata)
                if self.train_mode in {"voradj", "voradj_mixed_coverage"}:
                    metadata.setdefault("scene", task)
                if self.train_mode == "voradj_mixed_coverage":
                    metadata.setdefault(
                        "recovery_reset_source",
                        self.last_reset_source.get(task, "unknown"),
                    )
                replay_key = task
                if self.train_mode == "voradj_mixed_coverage":
                    replay_key = self._voradj_replay_class(metadata)
                    metadata["buffer_class"] = replay_key
                self.replays[replay_key].add(prev_obs[i], int(actions[i]), float(result.rewards[i]), next_obs, transition_done, metadata=metadata)
                episode_reward += float(result.rewards[i])
                episode_transition_count += 1
                for key in episode_reward_components:
                    episode_reward_components[key] += float(metadata.get(key, 0.0))
            self.global_step += 1
            update_task = self._select_update_task(task)
            loss = self._update(update_task)
            if self.global_step % self.log_freq_steps == 0:
                logged_update_task = self.last_update_task if loss is not None else update_task
                metric_payload = {
                    "global_step": self.global_step,
                    "task": task,
                    "update_task": logged_update_task,
                    "epsilon": self.epsilon(),
                    "learning_rate": self.current_learning_rate,
                    "coverage_ce_speed_weight": self.current_coverage_ce_speed_weight,
                    "loss": loss,
                    "loss_ema": self.loss_ema,
                    "replay_size": self._replay_size_for_task(task),
                }
                metric_payload.update(self._recent_diagnostic_summary(task))
                if self.train_mode == "m1new":
                    cov_u = self.update_counts.get("coverage", 0)
                    enc_u = self.update_counts.get("encirclement", 0)
                    metric_payload.update({
                        "updates_coverage": cov_u,
                        "updates_encirclement": enc_u,
                        "update_ratio_cov_enc": round(cov_u / max(enc_u, 1), 3),
                        "replay_size_coverage": len(self.replays["coverage"]),
                        "replay_size_encirclement": len(self.replays["encirclement"]),
                        "update_rebalance_enabled": bool(self.update_rebalance_enabled),
                    })
                if self.train_mode in {"voradj", "voradj_mixed_coverage"}:
                    metric_payload.update({
                        "batch_task_counts": getattr(self, "last_batch_task_counts", {}),
                        "batch_phase_counts": getattr(self, "last_batch_phase_counts", {}),
                        "batch_scene_counts": getattr(self, "last_batch_scene_counts", {}),
                        "batch_reset_source_counts": getattr(self, "last_batch_reset_source_counts", {}),
                        "batch_buffer_counts": getattr(self, "last_batch_buffer_counts", {}),
                        "batch_role_switch_count": getattr(self, "last_batch_role_switch_count", 0),
                        "batch_reward_stats": getattr(self, "last_batch_reward_stats", {}),
                    })
                if self.train_mode == "voradj_mixed_coverage":
                    metric_payload.update({
                        f"replay_size_{name}": len(self.replays[name])
                        for name in self.voradj_replay_classes
                    })
                    metric_payload.update({
                        "recovery_init_pool_size": len(self.recovery_init_pool),
                        "recovery_from_capture_ratio": float(self.recovery_from_capture_ratio),
                        "recovery_map_random_ratio_within_non_capture": float(self.recovery_map_random_ratio_within_non_capture),
                        "updates_voradj_mixed_coverage": self.update_counts.get("voradj_mixed_coverage", 0),
                    })
                self.metric_log.write(safe_json_dumps(metric_payload, ensure_ascii=False) + "\n")
                self.metric_log.flush()
            obs_list = result.observations
            episode_done = step_done
            if episode_done:
                record = env.episode_record(task=task)
                if self.train_mode == "voradj_mixed_coverage":
                    record["recovery_reset_source"] = self.last_reset_source.get(task, "unknown")
                record["episode_reward_sum"] = episode_reward
                record["episode_transition_count"] = episode_transition_count
                record["episode_reward_component_sums"] = dict(episode_reward_components)
                record["episode_reward_component_means"] = {
                    key: value / max(episode_transition_count, 1)
                    for key, value in episode_reward_components.items()
                }
                # Track total steps per task for balanced buffer
                episode_steps = int(env.episode_step)
                if hasattr(self, "task_total_steps"):
                    self.task_total_steps[task] = self.task_total_steps.get(task, 0) + episode_steps
                self._done_episode(task, env, record)
                # Select next task: round-robin or step-balanced
                if hasattr(self, "step_balanced") and self.step_balanced:
                    # Pick the task with fewer total steps
                    cov_steps = self.task_total_steps.get("coverage", 0)
                    enc_steps = self.task_total_steps.get("encirclement", 0)
                    task = "coverage" if cov_steps <= enc_steps else "encirclement"
                else:
                    task_cursor += 1
                    task = self.task_order[task_cursor % len(self.task_order)]
                obs_list = self._reset_task(task)
                episode_reward = 0.0
                episode_reward_components = {key: 0.0 for key in episode_reward_components}
                episode_transition_count = 0
                episode_done = False
            self.task_cursor = int(task_cursor)
            self.current_task = str(task)
            self.current_observations = obs_list
            self.current_episode_done = bool(episode_done)
            self.current_episode_reward = float(episode_reward)
            self.current_episode_reward_components = dict(episode_reward_components)
            self.current_episode_transition_count = int(episode_transition_count)
            if self.global_step % self.checkpoint_freq == 0:
                self._save_checkpoint(f"step_{self.global_step}.pt")
        self.task_cursor = int(task_cursor)
        self.current_task = str(task)
        self.current_observations = obs_list
        self.current_episode_done = bool(episode_done)
        self.current_episode_reward = float(episode_reward)
        self.current_episode_reward_components = dict(episode_reward_components)
        self.current_episode_transition_count = int(episode_transition_count)
        final = self._save_checkpoint(f"final_step_{self.global_step}.pt")
        self.episode_log.close()
        self.metric_log.close()
        return final


def load_config(path: str, _stack: Optional[Tuple[Path, ...]] = None) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    stack = tuple(_stack or ())
    if config_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Cyclic config extends chain: {chain}")
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    parent = loaded.pop("extends", None)
    if parent is None:
        return loaded
    parent_path = Path(str(parent)).expanduser()
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    base = load_config(str(parent_path), _stack=(*stack, config_path))
    return deep_update(base, loaded)


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = int(args.total_timesteps)
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.device is not None:
        cfg["device"] = args.device
    if args.resume_path is not None:
        cfg.setdefault("checkpointing", {})["full_resume"] = True
        cfg["checkpointing"]["resume_path"] = args.resume_path
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="CoCap Voronoi-adjacency IQN trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-path", default=None)
    args = parser.parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    trainer = CoCapTrainer(cfg)
    final = trainer.train()
    print(f"final_checkpoint={final}")


if __name__ == "__main__":
    main()
