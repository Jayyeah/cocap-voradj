"""Offline diagnosis for the existing 25k CTDE MASAC checkpoint/replay/eval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cocap_voradj.training.continuous.formal_config import resolve_formal_config
from cocap_voradj.training.continuous.joint_replay import JointReplayBuffer
from tools.run_continuous_ctde_training import _make_trainer


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"
ART = ROOT / "artifacts/2026-08-06_ctde_contract/ctde_25k"


def _percentiles(values, qs=(5, 25, 50, 75, 95)):
    if len(values) == 0:
        return {q: 0.0 for q in qs}
    return {q: float(np.percentile(np.asarray(values, dtype=float), q)) for q in qs}


def _min_distance(transition) -> float:
    g = transition.global_state
    width = 120.0
    height = 120.0
    pursuer_positions = np.asarray(g["pursuers"][:4, :4, :2], dtype=float).reshape(-1, 2)
    evader_positions = np.asarray(g["evaders"][:, :2], dtype=float)
    evader_positions = evader_positions[np.any(np.abs(evader_positions) > 1e-9, axis=1)]
    if len(evader_positions) == 0:
        return float("inf")
    distances = []
    for p in pursuer_positions:
        p_xy = np.asarray([p[0] * width, p[1] * height])
        for e in evader_positions:
            e_xy = np.asarray([e[0] * width, e[1] * height])
            distances.append(float(np.linalg.norm(p_xy - e_xy)))
    return float(min(distances)) if distances else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(ART / "ctde_25k_report.json"))
    parser.add_argument("--replay", default=str(ART / "ctde_25k_replay.pkl"))
    parser.add_argument("--checkpoint", default=str(ART / "ctde_25k_step25000.pt"))
    parser.add_argument("--eval", default=str(ART / "ctde_25k_eval20.json"))
    parser.add_argument("--out", default=str(ART / "ctde_25k_offline_diagnosis.md"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    replay = JointReplayBuffer.load(args.replay, report["manifest"])
    config = resolve_formal_config(FORMAL)
    trainer = _make_trainer(config, args.device)
    ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.load_checkpoint(args.checkpoint, ckpt["contract"])
    eval_data = json.loads(Path(args.eval).read_text(encoding="utf-8"))

    scene_rewards = {scene: [] for scene in ("capture", "pure_ce", "mixed_crms")}
    scene_action_norms = {scene: [] for scene in ("capture", "pure_ce", "mixed_crms")}
    collision_rewards = []
    normal_rewards = []
    collision_action_norms = []
    normal_action_norms = []
    min_distances = {"capture": [], "mixed_crms": []}
    discovery_rewards = []
    discovery_action_norms = []
    no_discovery_rewards = []
    no_discovery_action_norms = []
    near_zero_actions = 0
    radial_saturated = 0
    action_count = 0

    for tid in replay.transition_ids:
        tr = replay.get(tid)
        scene = tr.scene
        action_norm = float(np.linalg.norm(tr.actions[:4], axis=1).mean())
        reward_mean = float(np.mean(tr.rewards[:4]))
        scene_rewards[scene].append(reward_mean)
        scene_action_norms[scene].append(action_norm)
        action_count += 1
        if action_norm <= 0.02:
            near_zero_actions += 1
        if action_norm >= 0.95 * config["action"]["a_max"]:
            radial_saturated += 1
        collision = "collision" in tr.metadata["event_ids"]
        if collision:
            collision_rewards.append(reward_mean)
            collision_action_norms.append(action_norm)
        else:
            normal_rewards.append(reward_mean)
            normal_action_norms.append(action_norm)
        if scene in ("capture", "mixed_crms"):
            d = _min_distance(tr)
            if np.isfinite(d):
                min_distances[scene].append(d)
        if scene == "capture":
            if "discovery" in tr.metadata["event_ids"]:
                discovery_rewards.append(reward_mean)
                discovery_action_norms.append(action_norm)
            else:
                no_discovery_rewards.append(reward_mean)
                no_discovery_action_norms.append(action_norm)

    # Per-scene Q/TD samples through the loaded critic.
    scene_q_td = {}
    for scene in ("capture", "pure_ce", "mixed_crms"):
        ids = [tid for tid in replay.transition_ids if replay.get(tid).scene == scene][:64]
        if not ids:
            continue
        batch = replay.batch_from_ids(ids, device=trainer.device)
        active = batch["active_mask"].bool()
        rewards = batch["rewards"].to(trainer.device)
        bootstrap = (~batch["terminated"].to(trainer.device).bool()).float()
        with torch.no_grad():
            next_actions, next_log_prob, _ = trainer._actor_sample(batch["next_local_obs"])
            target_q = torch.minimum(
                trainer.target_critic1(batch["next_global_state"], next_actions),
                trainer.target_critic2(batch["next_global_state"], next_actions),
            ) - trainer.alpha.detach() * next_log_prob
            target = rewards + trainer.config.gamma * bootstrap * target_q
            q1 = trainer.critic1(batch["global_state"], batch["actions"])
            q2 = trainer.critic2(batch["global_state"], batch["actions"])
        q1m = q1[active].detach().cpu().numpy()
        q2m = q2[active].detach().cpu().numpy()
        targetm = target[active].detach().cpu().numpy()
        tdm = (q1[active] - target[active]).abs().detach().cpu().numpy()
        scene_q_td[scene] = {
            "q1": _percentiles(q1m),
            "q2": _percentiles(q2m),
            "target_q": _percentiles(targetm),
            "td_error": _percentiles(tdm),
            "td_mean": float(np.mean(tdm)),
        }

    lines = [
        "# CTDE 25k Offline Diagnosis",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- replay: `{args.replay}`",
        f"- eval: `{args.eval}`",
        f"- report: `{args.report}`",
        "",
        "## 1. Collision vs reward / TD tail",
        "",
        f"- collision transitions: {len(collision_rewards)}",
        f"- collision reward mean: {float(np.mean(collision_rewards)) if collision_rewards else 0:.4f}, "
        f"p95: {_percentiles(collision_rewards, (95,))[95]:.4f}",
        f"- non-collision reward mean: {float(np.mean(normal_rewards)) if normal_rewards else 0:.4f}",
        f"- collision action norm mean: {float(np.mean(collision_action_norms)) if collision_action_norms else 0:.4f}",
        f"- non-collision action norm mean: {float(np.mean(normal_action_norms)) if normal_action_norms else 0:.4f}",
        "",
        "> 25k 训练日志只保留 update tail；per-transition Q/TD 的碰撞长尾需在下一轮 metrics.jsonl 中直接落盘。",
        "",
        "## 2. Critic gradient by scene/role",
        "",
        "- 当前 checkpoint 未按 scene/role 保存梯度，只能给出 scene 级 Q/TD 样本。",
        "| scene | Q1 p50 | Q2 p50 | target p50 | TD mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene, value in scene_q_td.items():
        lines.append(
            f"| {scene} | {value['q1'][50]:.3f} | {value['q2'][50]:.3f} | "
            f"{value['target_q'][50]:.3f} | {value['td_mean']:.3f} |"
        )
    lines += [
        "",
        "## 3. Capture min-distance",
        "",
        "- replay 中没有 episode id，只能按 transition 汇总；min-distance 使用 central global_state 重建。",
    ]
    for scene in ("capture", "mixed_crms"):
        d = min_distances[scene]
        lines.append(
            f"- {scene}: n={len(d)}, median={np.median(d) if d else float('nan'):.2f}, "
            f"p25={_percentiles(d, (25,))[25]:.2f}, p75={_percentiles(d, (75,))[75]:.2f}"
        )
    lines += [
        "",
        "> 需要 episode 级 initial/final/min/AUC；当前 eval/replay 缺少 episode id，下一轮诊断实验必须写入。",
        "",
        "## 4. Discovery before/after",
        "",
        f"- discovery event transitions: {len(discovery_rewards)}",
        f"- discovery reward mean: {float(np.mean(discovery_rewards)) if discovery_rewards else 0:.4f}",
        f"- no-discovery reward mean: {float(np.mean(no_discovery_rewards)) if no_discovery_rewards else 0:.4f}",
        f"- discovery action norm mean: {float(np.mean(discovery_action_norms)) if discovery_action_norms else 0:.4f}",
        f"- no-discovery action norm mean: {float(np.mean(no_discovery_action_norms)) if no_discovery_action_norms else 0:.4f}",
        "",
        "## 5. Pure CE energy/CV",
        "",
        "- 25k eval：pure_ce success=0/20、collision_rate=0.55、CE strict/CV<0.15=0/20。",
        "- replay 未保存每步 energy/CV；下一轮 runner 需在 metrics.jsonl 中输出 episode 级 energy/CV 曲线。",
        "",
        "## 6. Actor action distribution",
        "",
        f"- action_count={action_count}",
        f"- near-zero (<=0.02) rate={near_zero_actions / max(action_count, 1):.4f}",
        f"- radial saturation (>=0.95*a_max) rate={radial_saturated / max(action_count, 1):.4f}",
        f"- action norm mean={float(np.mean([v for values in scene_action_norms.values() for v in values])):.4f}",
        f"- action norm p95={_percentiles([v for values in scene_action_norms.values() for v in values], (95,))[95]:.4f}",
        "",
        "## 7. Primary blocker",
        "",
        "- 25k 无捕获、无 coverage 成功，碰撞 0.45--0.55；replay 角色池 pursuing/support 已存在，但缺少 post-capture。",
        "- 当前最可能首要阻塞点：capture 的 credit/几何信号不足以驱动持续逼近，且碰撞惩罚主导；其次缺少 episode 级几何指标，无法量化微小改善。",
        "",
        "## 8. Eval summary",
        "",
    ]
    for scene, value in eval_data.items():
        lines.append(
            f"- {scene}: capture={value['capture_rate']:.2f}, success={value['success_rate']:.2f}, "
            f"collision={value['collision_rate']:.2f}"
        )
    lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
