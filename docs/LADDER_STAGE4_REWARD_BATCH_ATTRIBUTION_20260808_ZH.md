# Stage4B reward 第一批消融归因与后续备案（2026-08-08）

> 目的：把 A1/A2/A3 的归因逻辑、下一轮探索批归因、以及“超出预期”的决策树固化为可执行规则，避免后续窗口/Agent 丢失上下文。
> 判定铁律：一切晋级以 eval20 为准（4ep 诊断仅作提示，上次出现过 12.92 假象）；训练 capture 事件（replay ≥50 奖励）与 Q 趋势作为佐证，不作为 gate 本身。

## 1. 本轮消融设计（已运行）

Base = Stage4B 原配置（moving evader / 0 obstacle），seed1=2026080801，25k。

| 线 | 相对 Base 的改动 | 隔离通道 | 对照 |
|---|---|---|---|
| A1 | omega_ring_ms 2→4；ring_ms_progress_clip 3→6 | reward 幅度/clip | vs 4B 原两 seed |
| A2 | A1 + omega_approach 0→3（距离进度项叠加 ring，含代码改动） | “旧 capture 吸引项” | vs A1（增量） |
| A3 | A1 + warmup 5000→10000 | 探索期长度 | vs A1（增量） |

归因结论规则：
- A1−原配置：回答“reward 幅度是否为瓶颈”。
- A2−A1：回答“approach 项是否提供额外接近梯度”。
- A3−A1：回答“warmup 是否补充探索/数据多样性”。
- A2 vs A3：同 base 的不同机制；若均有效，A3 改动更小（仅 warmup 字段），A2 改动更大（字段+代码）→ 按“更优且改动更小”优先。

指标矩阵（每线 25k 后回填）：
1. eval20：capture rate、avg min-distance、collision rate；
2. 训练 replay：capture_events（≥50 奖励条数）、collision_events、ce_success；
3. 训练轨迹：Q1 趋势（前 15k vs 后 10k）、窗口 capture/collision 分布（前期 vs 后期）、speed/action_norm/log_std 末期；
4. 必要时 1000 步重评估（排除 400 cap 掩盖慢接近）。

## 2. 下一轮探索批（E 批）归因设计

触发：reward 批“无明显积极”（所有线 eval20 capture=0 且 min-dist ≥ 13.59 且训练 capture_events=0；或仅弱信号不足 gate）。
Base：reward 批中最积极/最少差者（Q 趋势最平、capture_events 最多）；若全无信号取 A1。
方向已获用户预授权（2026-08-08：“若无明显积极，则再做一轮以探索为改变的”）。

| 线 | 相对 Base 改动 | 隔离通道 |
|---|---|---|
| E1 | target_entropy -2→-4 | 持续探索熵 |
| E2 | alpha_init 0.2→0.5 | 早期探索权重 |
| E3 | warmup 增量（若 Base=A3 则跳过，用 E1/E2） | 探索期长度 |
| E4（机制批，若探索批仍无信号） | update_every 4→2 | 学习密度（非探索；单独归因） |

归因规则：
- E1 vs E2：若 E1 有效 → 持续熵是关键；若 E2 有效 → 早期探索是关键；两者都有效且接近 → 取改动更小/更稳者（以 5k 冒烟稳定性与 25k eval20 为准）。
- E3 仅当 reward 批显示 warmup 有微弱正向（A3 略好于 A1）时优先。
- 验证节奏：5k 冒烟（Q 停止单调坍缩、capture_events>0、log_std 不提前收缩）→ 25k seed1 → eval20 → 积极则 seed2。

## 3. 超出预期的归因备案（决策树，发现即执行，无需再次询问方向）

- Case 1 全部积极：选更优且改动更小（默认 A1）；若 A2 显著更优（capture ≥2× 或 min-dist 低 ≥2 且 collision 可控）选 A2；A3 仅当 A1 不达标。选定后：4B seed2 → 复制到 4C 两 seed → 通过后 4D 两 seed。
- Case 2 仅 A2 积极：approach 项为必要增量 → 微调 omega_approach 权重（2.0/4.0 单变量）验证稳定性，不叠加其他。
- Case 3 仅 A3 积极：warmup 为主因 → E 批 Base=A3（E1/E2 增量），不放缓。
- Case 4 仅 A1 积极：reward 幅度为主因 → 直接 seed2，探索批暂停。
- Case 5 全部无信号：→ E1/E2/E3 并行（Base=A1）→ 若 E 批仍无信号 → E4 机制批（需用户确认 update ratio 冻结项）→ 50k 续训最佳线（确认无信号）→ snapshot/teacher 失败触发评估 → 合同层重审（oracle 已证明可学，概率低，除非发现 eval/训练分布差异）。
- Case 6 碰撞失控（collision>50%）即使有 capture：判不积极；识别“冲撞型坍缩”（term≈coll）；下轮在该线做探索侧缓解（warmup/entropy/alpha，已批准方向），不新增变量。
- Case 7 训练有 capture 但 eval 无：→ 1000 步重评估 + 检查 eval/train 分布差异（seed/spawn/确定性）→ 属数据链问题先修再判。
- Case 8 单线中途异常：只重跑该线，其他线归因不受影响；异常记录台账。

## 4. 详细 TODO（本批完成前）

- [ ] 三线 25k seed1 运行中（A1 cuda:0 / A2 cuda:1 / A3 cuda:0），约 10 分钟一次低频检查
- [ ] 每线 10k 初检：Q 趋势、capture 窗口事件、speed/action/log_std（早停信号）
- [ ] 25k 报告 → 监督自动 analyze + eval20；人工复核 eval20（非 4ep）
- [ ] 回填 §1 指标矩阵，判定 Case
- [ ] 按 Case 执行：积极 → seed2 + 择优 → 4C 复制两 seed → 4D；无积极 → E1/E2/E3 并行
- [ ] 每线完成输出【Stage4B A_X 完成说明】；台账实时更新
- [ ] 探索批（若触发）：5k 冒烟 → 25k seed1 → eval20 → seed2
