#!/usr/bin/env python3
"""Required preflight for the scratch IQN ROLE-token control."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from cocap_voradj.training.trainer import CoCapTrainer, deep_update, load_config, set_global_config
from tools.evaluate_iqn_role_token_formal_20260918 import evaluate_checkpoint


Z_CONFIG = ROOT / "configs/experiments/iqn_zstate_b3_20260918/z_state.yaml"


def _digest_state(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _same_state(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> bool:
    return all(torch.equal(first[key].detach().cpu(), second[key].detach().cpu()) for key in first)


def _build_model_config(config: Mapping[str, Any]) -> CoCapNetConfig:
    iqn = config.get("iqn", {}) or {}
    return CoCapNetConfig(
        hidden_dim=int(iqn.get("hidden_dim", 128)),
        num_heads=int(iqn.get("num_heads", 4)),
        num_layers=int(iqn.get("num_layers", 2)),
        action_size=int(iqn.get("action_size", 9)),
        self_feature_dim=int(iqn.get("self_feature_dim", 7)),
        max_pursuers=int(config.get("perception", {}).get("max_pursuer_num", 12)),
        max_evaders=int(config.get("perception", {}).get("max_evader_num", 8)),
        max_obstacles=int(config.get("perception", {}).get("max_obstacle_num", 5)),
        num_quantiles=int(iqn.get("num_quantiles", 16)),
        num_cosine_features=int(iqn.get("num_cosine_features", 32)),
        architecture=str(iqn.get("architecture", "dual_head")),
        pursuing_embed_dim=int(iqn.get("pursuing_embed_dim", 8)),
        pursuer_feature_dim=int(iqn.get("pursuer_feature_dim", 7)),
        include_is_pursuing=bool(iqn.get("include_is_pursuing", True)),
        include_z_state=bool(iqn.get("include_z_state", False)),
        pursuing_late_fusion=bool(iqn.get("pursuing_late_fusion", True)),
    )


def _resolve_scene(config: Mapping[str, Any], scene: str) -> dict[str, Any]:
    task = "voradj_coverage" if scene == "coverage" else "voradj"
    cfg = deep_update(dict(config), config.get("tasks", {}).get(task, {}))
    cfg.setdefault("env", {})["num_evaders"] = 0 if scene == "coverage" else 1
    return cfg


def _observation_contract_check(role_config: Mapping[str, Any], z_config: Mapping[str, Any]) -> dict[str, Any]:
    if role_config.get("reward") != z_config.get("reward"):
        raise AssertionError("ROLE/Z reward config drift")
    if role_config.get("env") != z_config.get("env"):
        raise AssertionError("ROLE/Z environment/termination config drift")
    if role_config.get("voradj") != z_config.get("voradj"):
        raise AssertionError("ROLE/Z topology/reward-support config drift")
    role_per = role_config.get("perception", {}) or {}
    role_iqn = role_config.get("iqn", {}) or {}
    if role_per.get("include_is_pursuing") is not True or role_per.get("include_z_state") is not False:
        raise AssertionError("ROLE perception contract is not binary role-token")
    if role_per.get("friend_ordering_mode") != "physical_only":
        raise AssertionError("ROLE friend ordering is not physical-only")
    if role_iqn.get("include_is_pursuing") is not True or role_iqn.get("include_z_state") is not False:
        raise AssertionError("ROLE IQN contract is not binary role-token")
    if int(role_iqn.get("pursuing_embed_dim", -1)) != 0 or role_iqn.get("pursuing_late_fusion"):
        raise AssertionError("ROLE late-fusion branch remains configured")
    if (role_config.get("pretrained") or {}).get("path"):
        raise AssertionError("ROLE scratch config contains a pretrained path")
    return {
        "reward_exact_match": True,
        "env_exact_match": True,
        "voradj_exact_match": True,
        "role_self_token": True,
        "role_friend_token": True,
        "friend_ordering": "physical_only",
        "pursuing_late_fusion": False,
        "pursuing_embed_dim": 0,
        "scratch_pretrained_path": None,
    }


def _observation_shape_and_order_check(role_config: Mapping[str, Any], z_config: Mapping[str, Any]) -> dict[str, Any]:
    shape_rows = []
    for scene in ("voradj", "voradj_coverage"):
        role = _resolve_scene(role_config, scene)
        z = _resolve_scene(z_config, scene)
        set_global_config(role)
        role_env = VorAdjEnv(copy.deepcopy(role), seed=2026091801)
        role_obs = list(role_env.reset())
        set_global_config(z)
        z_env = VorAdjEnv(copy.deepcopy(z), seed=2026091801)
        z_obs = list(z_env.reset())
        for index, (role_item, z_item) in enumerate(zip(role_obs, z_obs)):
            if role_item is None or z_item is None:
                if role_item is not None or z_item is not None:
                    raise AssertionError(f"ROLE/Z active-mask mismatch in {scene} agent {index}")
                continue
            if role_item["self"].shape != (9,) or role_item["pursuers"].shape != (8, 7):
                raise AssertionError("ROLE token observation shape mismatch")
            if z_item["self"].shape != (9,) or z_item["pursuers"].shape != (8, 7):
                raise AssertionError("Z token observation shape mismatch")
            if not np.array_equal(role_item["types"], z_item["types"]):
                raise AssertionError("ROLE/Z token type ordering mismatch")
            if not np.array_equal(role_item["masks"], z_item["masks"]):
                raise AssertionError("ROLE/Z token mask ordering mismatch")
            if not np.allclose(role_item["self"][:8], z_item["self"][:8], atol=0.0, rtol=0.0):
                raise AssertionError("ROLE/Z self physical features differ")
            if not np.allclose(role_item["pursuers"][:, :6], z_item["pursuers"][:, :6], atol=0.0, rtol=0.0):
                raise AssertionError("ROLE/Z friend physical ordering differs")
            if not np.all(np.isin(role_item["self"][-1], [0.0, 1.0])):
                raise AssertionError("self is_pursuing token is not binary")
            if not np.all(np.isin(role_item["pursuers"][:, -1], [0.0, 1.0])):
                raise AssertionError("friend is_pursuing token is not binary")
            shape_rows.append({"scene": scene, "agent": index, "self": list(role_item["self"].shape), "friends": list(role_item["pursuers"].shape)})
    return {"role_z_physical_order_exact": True, "rows": shape_rows}


def run_preflight(config_path: Path, output: Path, device: str, steps: int) -> dict[str, Any]:
    role_config = load_config(str(config_path))
    z_config = load_config(str(Z_CONFIG))
    contract = _observation_contract_check(role_config, z_config)
    shapes = _observation_shape_and_order_check(role_config, z_config)
    model = CoCapIQN(_build_model_config(role_config)).to(device)
    model.eval()
    if model.pursuing_embed is not None or model.config.pursuing_late_fusion:
        raise AssertionError("fresh ROLE model contains a late-fusion branch")
    if model.single_action_feature[0].in_features != model.config.hidden_dim:
        raise AssertionError("ROLE single-action path has an unexpected role shortcut width")
    scratch = {
        "pretrained_path_absent": not bool((role_config.get("pretrained") or {}).get("path")),
        "model_has_no_late_fusion": True,
        "initial_state_digest": _digest_state(model.state_dict()),
        "nonzero_parameter_count": int(sum(int(torch.count_nonzero(value)) for value in model.state_dict().values() if value.is_floating_point())),
    }

    preflight_root = output.parent / "preflight_iqn_role_token_20260918"
    run_dir = preflight_root / "run"
    resume_path = run_dir / "checkpoints" / "resume_latest.pt"
    preflight_config = copy.deepcopy(role_config)
    preflight_config.update({
        "device": device,
        "train_mode": "voradj",
        "total_timesteps": int(steps),
        "output_root": str(preflight_root),
        "run_name": "run",
    })
    preflight_config.setdefault("iqn", {}).update({
        "min_replay_size": 8,
        "batch_size": 8,
        "train_freq": 1,
        "target_update_freq": 1,
        "checkpoint_freq": max(8, int(steps) // 2),
        "log_freq_steps": 1,
    })
    preflight_config["checkpointing"] = {
        "full_resume": True,
        "full_resume_path": str(resume_path),
    }
    trainer = CoCapTrainer(preflight_config)
    initial_state = copy.deepcopy(trainer.model.state_dict())
    final_checkpoint = trainer.train()
    final_state = copy.deepcopy(trainer.model.state_dict())
    if trainer.update_steps <= 0:
        raise AssertionError("preflight completed without a real IQN optimizer update")
    if _same_state(initial_state, final_state):
        raise AssertionError("preflight optimizer update did not change parameters")
    if trainer.loss_ema is None or not math.isfinite(float(trainer.loss_ema)):
        raise AssertionError("preflight loss is not finite")
    replay = trainer.replays["voradj"]
    if len(replay) < 8:
        raise AssertionError("preflight replay insertion is incomplete")
    batch = replay.sample(8, device)
    if batch["actions"].shape[0] != 8 or not torch.isfinite(batch["rewards"]).all():
        raise AssertionError("preflight replay sample is invalid")
    if not _same_state(trainer.model.state_dict(), trainer.target_model.state_dict()):
        raise AssertionError("preflight target-network update did not run")
    if not resume_path.is_file():
        raise AssertionError("preflight full-resume checkpoint was not saved")
    loaded = CoCapIQN.load(str(final_checkpoint), device=device).eval()
    if not _same_state(final_state, loaded.state_dict()):
        raise AssertionError("checkpoint save/load changed model state")
    resume_config = copy.deepcopy(preflight_config)
    resume_config["checkpointing"]["resume_path"] = str(resume_path)
    resumed = CoCapTrainer(resume_config)
    if int(resumed.global_step) != int(trainer.global_step):
        raise AssertionError("full-resume global step mismatch")
    if not _same_state(final_state, resumed.model.state_dict()):
        raise AssertionError("full-resume model state mismatch")

    eval_path = output.parent / "preflight_eval_smoke.json"
    evaluation = evaluate_checkpoint(config_path.resolve(), Path(final_checkpoint).resolve(), eval_path, episodes=1, seed_base=2026092801, device=device)
    if evaluation.get("status") != "complete":
        raise AssertionError("formal evaluator smoke did not complete")
    gpu = {"device": device, "cuda_available": bool(torch.cuda.is_available())}
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        gpu.update({
            "device_name": torch.cuda.get_device_name(torch.device(device)),
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(torch.device(device))),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(torch.device(device))),
        })
    metrics_path = run_dir / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    result = {
        "schema": "iqn-role-token-scratch-preflight-v1",
        "status": "PASS",
        "config": str(config_path.resolve()),
        "z_config": str(Z_CONFIG.resolve()),
        "contract": contract,
        "observation": shapes,
        "scratch": scratch,
        "real_env_rollout": {"steps": int(trainer.global_step), "episodes": int(trainer.episode_idx)},
        "optimization": {
            "updates": int(trainer.update_steps),
            "loss_ema": float(trainer.loss_ema),
            "loss_finite": True,
            "parameters_changed": True,
            "replay_size": int(len(replay)),
            "replay_sample_rows": int(batch["actions"].shape[0]),
            "target_network_updated": True,
        },
        "checkpoint_resume": {
            "checkpoint": str(Path(final_checkpoint).resolve()),
            "checkpoint_load_exact": True,
            "full_resume": str(resume_path.resolve()),
            "full_resume_exact": True,
            "resume_step": int(resumed.global_step),
        },
        "evaluator_smoke": {"path": str(eval_path.resolve()), "status": evaluation["status"]},
        "gpu": gpu,
        "last_metrics": metrics[-1] if metrics else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    if args.steps < 16:
        raise ValueError("--steps must be at least 16 for replay/update/checkpoint smoke")
    result = run_preflight(args.config.resolve(), args.out.resolve(), args.device, args.steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
