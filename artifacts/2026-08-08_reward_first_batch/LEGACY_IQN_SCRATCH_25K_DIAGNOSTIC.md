# LEGACY IQN SCRATCH 25K DIAGNOSTIC（2026-08-08）

## IQN scratch config

```text
IQN scratch config: configs/experiments/voradj_a3_apfnew_sqrtn_20260723/iqn_scratch_early_25k50k_20260808.yaml
  （原 a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml 的副本；仅改 checkpoint_freq 50000->25000 与 run_name）
checkpoint family: voradj_a3_apfnew_sqrtn_4v1_scratch_mix_2m_20260723_run1（2M 成功线，main）
commit: 当前工作树（配置来自 main 成功系列，未改算法字段）
seed: 2026071543（与成功线相同，便于对照早期轨迹）
map: 120x120，4p1e1obs
action: 离散 unicycle (a,w) 3x3 = [-0.4,0,0.4] x [-pi/6,0,pi/6]
reward: voradj_mixed_coverage 原合同（timestep -1 / collision -160 / goal 120）
spawn: map_random，min sep 15
perception: 原 IQN VCT-LS robot-frame local observation
episode: 3000
batch: 128；train_freq: 4；replay: 1M（semantic/physical 四缓冲）
epsilon: 0.6 -> 0.05 over 1M（本 25k 内 epsilon≈0.586-0.589，与原 2M 线早期一致）
```

全部算法字段 `[IQN-ALIGN]` 未改；本轮仅为 25k/50k 早训参照（连续 50k 轨迹，step_25000.pt 为 25k checkpoint）。

## 1. IQN scratch 25k capture 是否非零

- 训练期（≤25k）：voradj(capture) 36 集，capture=0；`recent_capture_rate`=0.0。
- 25k checkpoint 行为评估（20 集、1000 步、确定性）：capture=0/20。
- 结论：**capture 仍为零**。

## 2. 与当前 MASAC Stage4 的行为对照（25k）

| 指标 | IQN 25k | MASAC 4B seed1 25k | MASAC 4B seed2 25k |
|---|---:|---:|---:|
| capture | 0/20 | 0/20 | 0/20 |
| collision | 100% | 5% | 100% |
| d1_min（20 集均值） | 2.60 | ~14（eval20 min-dist 14.45） | ~15-17 |
| fraction_closing | 0.70 | 0.475（n=2 冒烟） | 未测 |
| abs_bearing_error | 0.90 | 0.70（n=2 冒烟） | 未测 |
| 任意步单机进入 8m | 10.8% steps | ~0 | ~0 |
| 2+ 机同时进 ring(8-10.5m) | 0.05% steps | ~0 | ~0 |
| 训练期进入 capture 半径（capture_agent_count>0） | 36/36 集（max 2 机） | 未记录（term≈coll） | 未记录 |

## 3. IQN 25k 是否已形成 pursuit 行为

**部分形成，方向是“单机激进追击”**：动作几乎全为全加速+全转向（action 0/6/7/8 占 99%+），closing=0.70，单机频繁进入 8m（10.8% 步），训练期 100% 集次进入 capture 半径（最多 2 机）。但**没有多机包围**（2+ in ring 仅 0.05%），collision 失控（100%），因此 capture=0。

## 4. 连续指标是否正向

- 正向：fraction_closing 0.70、d1_min 2.60、单机 ring 访问 10.8% 步、训练期全部集次进入 8m。
- 负向/伪信号：d1_progress 均值 -25.6（多数集因全体碰撞死亡而提前结束，d_final 用 99 填充，该值不可直接解读）；bearing error 0.90 与 turn_correct 0.52 并不优于 random（0.526 冒烟），说明转向“方向正确率”并未学到。
- 结论：IQN 25k 的 pursuit 是“猛冲+转向”而非“有效包围”，接近信号真实存在但多机协调与碰撞控制未建立。

## 5. replay / reward capture-positive transitions

- 训练期 capture 事件：0。
- 25k 内 voradj 集 reward：mean per transition -2.77；episode_reward_sum mean -2415；reward_safety_mean≈-120（碰撞惩罚占主导），capture/terminal=0。
- 结论：IQN 25k 也没有体验 capture 正反馈；它的接近行为主要来自探索+稠密 reward 的“冲撞”解，而非 capture 成功驱动。

## 6. IQN 25k reward density vs MASAC

- IQN：-2.77/transition（碰撞惩罚密集拖低）。
- MASAC 4B seed1：active-agent reward mean -0.29（碰撞少，ring 信号弱）。
- 两者 reward density 都很低且无 capture 正反馈；IQN 略“更差”（因为碰撞惩罚发生频率高），但 IQN 靠激进探索获得了单机接近信号。

## 7. 离散动作分布

- 20 集统计：a=-0.4: 9180 次；a=+0.4: 9748 次；a=0: 3 次（几乎不用零加速）。
- omega=±π/6 占 99%+；omega=0 仅 1841 次（且集中在 a=+0.4）。
- 结论：25k IQN 策略 ≈ “全油门 + 全转向”双极振荡，无精细控制。

## 8. 对 ladder 的含义（Situation B 倾向）

- IQN 25k capture=0，但 closing/ring 访问已有明显正向 → 支持 Situation B 解读：**25k 早期，capture=0 不能代表“完全没学”，连续行为指标必须先看**。
- 同时 IQN 25k 的接近是“冲撞式”的，MASAC 4B 则是“游荡/规避式”——两者都不是可晋级的 pursuit。
- 若 MASAC A 批出现 closing/bearing/ring 改善但 capture=0，不应直接判死，需对照本条 IQN 参照评估其“有效接近”质量（包围 vs 冲撞）。

## 9. 文件与复现

- 评估结果：`artifacts/2026-08-08_reward_first_batch/iqn_scratch_25k_behavior.json`
- 工具：`tools/evaluate_iqn_scratch_early.py`
- 25k checkpoint：`runs/iqn_scratch_early_25k50k_20260808/checkpoints/step_25000.pt`
- 50k 诊断：待训练完成后输出 `LEGACY_IQN_SCRATCH_50K_DIAGNOSTIC.md`。
