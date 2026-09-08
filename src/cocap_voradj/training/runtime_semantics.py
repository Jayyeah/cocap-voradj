"""Live environment/policy preflight. Config values alone are not evidence."""
from __future__ import annotations
import math
import numpy as np
import torch


def assert_config_namespaces(config):
    bad = [k for k in config.get("reward", {}) if k.startswith(("support_reward_", "vct_ls_support_reward_"))]
    if bad:
        raise ValueError(f"ignored reward namespace keys {bad}; use voradj.support_reward_* instead")
    for task in config.get("tasks", {}).values():
        if isinstance(task, dict):
            assert_config_namespaces(task)


def runtime_facts(env, actor=None):
    assert_config_namespaces(env.config)
    cw, vw = env._vct_ls_support_reward_weights()
    p = env.pursuers[0]
    facts = {"topology": env._perception_topology_version(), "enemy_token_rule": "surface_radius" if env._vct_ls_enabled() else "voronoi_adjacency_not_radius", "global_enemy_flag": bool(env.per_cfg.get("global_evader_visibility", False)), "support_capture_weight": cw, "support_coverage_weight": vw, "support_blend_enabled": env._vct_ls_support_reward_blend_enabled(), "action_mode": env.action_mode, "decision_dt": float(p.dt*p.N), "physics_dt": float(p.dt), "collision_semantics": env.env_cfg.get("collision_semantics", "legacy_end_step"), "capture_radius": float(env.env_cfg.get("capture_distance", env.reward_cfg.get("r_e", 8))), "capture_k": int(env.reward_cfg.get("k_required", 3)), "capture_reward_mode": env._capture_reward_mode(), "pursuers": len(env.pursuers), "evaders": len(env.evaders), "enemy_radius": env._vct_ls_sensing_radius("enemy"), "yaw_held_for_vxy": bool(getattr(env, "desired_velocity_action", False)), "a_longitudinal_max": float(max(p.a)), "omega_max": float(max(p.w)), "v_max": float(p.max_speed), "drag": float(p.coefficient_water_resistance), "servo_acceleration_limit": float(env.config.get("action", {}).get("servo_acceleration_limit", env.config.get("action", {}).get("a_max", .4)))}
    facts["adapter_a_max"] = getattr(env.action_adapter, "a_max", None)
    facts["adapter_w_max"] = getattr(env.action_adapter, "w_max", None)
    if actor is not None:
        facts["actor_training"] = bool(actor.training)
        facts["active_stochastic_modules"] = [name for name,m in actor.named_modules() if m.training and ((isinstance(m, torch.nn.Dropout) and m.p > 0) or (isinstance(m, torch.nn.MultiheadAttention) and m.dropout > 0))]
        facts["backbone_trainable_parameters"] = sum(p.numel() for p in actor.encoder.parameters() if p.requires_grad)
    return facts


def assert_runtime(env, actor=None, expected=None, categorical=False):
    facts = runtime_facts(env, actor)
    expected = expected or env.config.get("runtime_semantic_assertions", {})
    for key, value in expected.items():
        if key not in facts:
            raise ValueError(f"Unknown runtime assertion: {key}")
        actual = facts[key]
        equal = math.isclose(actual, value, rel_tol=0, abs_tol=1e-8) if isinstance(value, float) and isinstance(actual,(int,float)) else actual == value
        if not equal:
            raise ValueError(f"runtime assertion {key}: expected {value!r}, got {actual!r}")
    if categorical and (facts.get("actor_training") or facts.get("active_stochastic_modules")):
        raise ValueError("categorical PPO requires deterministic eval-mode forward")
    if categorical:
        facts["zero_update_policy_probe"] = categorical_zero_update_probe(actor, env.get_observations())
        grid = actor.action_grid.detach().cpu().numpy()
        expected_grid = np.asarray(env.pursuers[0].action_list)
        if env.continuous_aw_action and not np.allclose(grid, expected_grid, atol=1e-6):
            raise ValueError("categorical grid differs from executed AW action centers")
    return facts


def assert_only_changed(before, after, allowed):
    changed = {k for k in set(before)|set(after) if before.get(k) != after.get(k)}
    allowed = set(allowed)
    if changed != allowed:
        raise ValueError(f"runtime changed fields {sorted(changed)} != declared {sorted(allowed)} (NOOPs also fail)")
    return sorted(changed)


def initial_state_fingerprint(env):
    import hashlib, json
    payload = {"pursuers": [[p.x,p.y,p.theta,p.speed,p.r] for p in env.pursuers], "evaders": [[e.x,e.y,e.theta,e.speed,e.r] for e in env.evaders], "obstacles": [[o.x,o.y,o.r] for o in env.obstacles]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assert_paired_records(first, second):
    if len(first) != len(second):
        raise ValueError("paired record counts differ")
    for a,b in zip(first, second):
        for key in ("seed", "initial_state_fingerprint"):
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f"paired evaluation {key} mismatch or absent")


@torch.no_grad()
def categorical_zero_update_probe(actor, observations):
    from cocap_voradj.training.small_step_ac import tensor_tree
    if actor.training:
        raise ValueError("zero-update probe requires eval-mode forward")
    live=[o for o in observations if o is not None]
    device=next(actor.parameters()).device
    batch=tensor_tree({k:np.stack([o[k] for o in live]) for k in live[0]},device)
    first=actor.distribution(batch);second=actor.distribution(batch)
    diff=second.logits-first.logits
    if not torch.isfinite(diff).all() or float(diff.abs().max())>1e-4:
        raise ValueError("zero-update forward/log-prob mismatch")
    ratio=diff.exp()
    return {"max_log_prob_error":float(diff.abs().max()),"max_ratio_error":float((ratio-1).abs().max()),"all_actions_finite":True,"rows":len(live),"mean_entropy":float(first.entropy().mean()),"mean_top_probability":float(first.probs.max(-1).values.mean()),"forward_mode":"eval", "policy_updates":0}
