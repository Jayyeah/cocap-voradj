# CoCap 双线推进规划：Critic Audit + IQN Clean-up / Emergent Local Swarm

> 日期：2026-09-17  
> Canonical planning doc：后续若提及“**双线规划** / **Critic Audit** / **IQN Clean-up roadmap** / **local emergent CoCap**”，优先读取本文件最新版本；如与聊天摘要冲突，以仓库最新代码、runtime artifact 与本文件后续修订为准。

---

## 0. 当前站位与总原则

### 0.1 当前事实锚点

- `experiment/density-normalized-sensing-v2-20260915` 已完成四条 NormSense/MAPPO 目标线并 STOP_FOR_MASTER_REVIEW。
- Final NormSense Pure-Capture baseline 到 250k 仅有弱 capture 信号，collision 仍高；short-horizon Conservative300k 未改善。
- Full-Mix Original 与 CaptureDownweight05 到 100k 均未形成有效 joint capability，静态 reward downweight 未解决根因。
- `experiment/td3-local-aw-stage1-20260916` 的 Stage-1 local TD3 capability audit 基本失败：
  - pure coverage scratch collapse 到 no-op 等价行为；
  - pure capture scratch 无稳定 capture；
  - IQN-warm actor 在 RL 前本来具备很强能力，但 TD3 更新后不能稳定保留；
  - capture warm 只出现短暂 ring/capture 信号后再次退化。
- 已有证据反复说明：**Actor 表达能力/是否存在高性能策略不是主要问题；当前更值得诊断的是 critic/value estimation、bootstrapping 与 policy-improvement 是否在已有好策略附近给出正确方向。**

### 0.2 项目后续拆成两条并行线

| 线 | 定位 | 最终目的 | 止损原则 |
|---|---|---|---|
| **A. Critic / Actor-Critic Audit** | 有边界的根因诊断线 | 判断 local/central critic、bootstrap、policy improvement 谁是真瓶颈；只在证据明确时再进入 central critic / MATD3 | 做完统一 frozen-actor audit 后必须收口；没有明确赢家则停止继续扩 Actor-Critic 算法树 |
| **B. IQN Clean-up / Emergent Local Swarm** | **论文主方法线** | 把 Final IQN 清理成真正 local、homogeneous、shared-policy、无显式角色分配的 CoCap | 每阶段只改一个概念；优先 distillation / warm-start / matched ablation，避免反复 2M scratch |

### 0.3 最高优先级目标

不是继续寻找“最新、更强的通用 RL 算法”，而是尽快形成：

1. **理论上说得清楚**的局部信息/局部交互机制；
2. **仿真上稳定**的 Capture + Coverage + lifecycle；
3. **实验/鲁棒性上完整**的验证闭环；
4. 若 Critic Audit 真找到明确机制，再把 Actor-Critic 作为可选增强，而不是卡住论文主线。

---

# A 线：Critic / Actor-Critic Audit

## A0 — 固定高性能 Actor 与统一 trajectory bank

### 目标

完全移除：

- exploration scarcity；
- actor drift；
- online state-distribution drift；
- 不同算法各自采样导致的 dataset confound。

把问题缩成：

> **在同一个高性能策略产生的数据上，不同 critic 输入/目标机制能否正确解释 return，并区分成功/碰撞/恢复等关键状态？**

### 实施

冻结一个高性能 policy 作为 data generator：

- 优先 Final IQN / 已验证 BC Actor；
- data collection 期间 Actor 完全不更新。

构造统一 matched trajectory bank，必须覆盖并标注：

- pure coverage；
- ordinary pursuit；
- ring2；
- ring3 / near-capture；
- normal capture；
- stationary capture（若出现，单列）；
- collision；
- capture terminal transition；
- early recovery；
- late recovery；
- pure coverage steady-state。

稀有样本必须主动 oversample，尤其：

- normal capture；
- ring3 near-success；
- collision；
- capture→early-recovery。

第一阶段直接保存 empirical discounted Monte-Carlo return：

\[
G_t=\sum_{k=0}^{T-t}\gamma^k r_{t+k}
\]

不要先用 bootstrap target。

### 输出

一套冻结、版本化 dataset，后续 TD3/SAC/MADDPG/MAPPO critic 全部共用。

### Gate

- success/failure 都有足够样本；
- phase/role/event coverage 可核验；
- dataset hash 固定；
- 不允许不同 critic 自己重新采各自数据。

---

## A1 — Critic Information / Identifiability Audit

### 目标

先回答：**critic 的输入信息和函数形式本身，是否足以拟合已有 policy 的 return？**

### 候选 critic

| ID | 形式 | 输入 | 主要问题 |
|---|---|---|---|
| A1-LQ | TD3/SAC-style Local Q | \(o_i,a_i\) | 单体局部信息是否足够预测 return |
| A1-NQ | Neighbor Q（可选但优先级较高） | focal + local/VorAdj neighbors + neighbor actions | 是否只缺局部协同信息，而非全局 central state |
| A1-CQ | MADDPG-style Central Q | train-time global/set state + joint actions | teammate action information 是否真正必要 |
| A1-V | MAPPO-style V | 当前 central state/context，无 action | state-value formulation / phase aliasing 是否更难校准 |

### 公平性要求

- backbone 容量尽量 matched；
- 相同 train/heldout split；
- 相同 optimizer budget；
- 相同 MC target；
- 不允许 central critic 同时偷换成巨大 GNN/Transformer，而 local critic 只用小 MLP。

### 第一轮只做监督回归

Q critic：

\[
Q_\theta(x_t,a_t)\rightarrow G_t^{MC}
\]

V critic：

\[
V_\phi(x_t)\rightarrow G_t^{MC}
\]

### 统一评价

- heldout RMSE / MAE；
- explained variance；
- calibration；
- phase-conditioned error；
- early-recovery error；
- ring3 / near-capture error；
- success vs collision 条件误差。

Q critic 额外检查：

\[
Q(s,a_{good})>Q(s,a_{bad})
\]

以及 action-ranking correlation。

### 预期结论分流

| 结果 | 解释 | 后续 |
|---|---|---|
| Local Q 已很好 | local information 够；Stage-1 TD3 更像 bootstrap / policy-improvement 问题 | 不需要 central critic |
| Local 差，Neighbor/Central 明显好 | teammate observation/action 是真实缺失信息 | 允许后续 MATD3 / central critic 小规模验证 |
| Q 都好、MAPPO V 明显差 | MAPPO state-value / phase aliasing 更可疑 | 聚焦 V/GAE，不泛化成所有 critic 问题 |
| 所有 critic 都差 | observation/history 仍非充分 Markov 表示，或 return 条件方差太大 | 查 state/history/belief，不继续堆算法 |

---

## A2 — Native Bootstrap Audit

### 目标

把“representation/input 不够”与“Bellman/GAE bootstrap 本身不稳定”分开。

只有 A1 中 MC-supervised 能拟合好的 critic 才进入 A2。

### 原生 target

TD3：

\[
y=r+\gamma\min(Q'_1,Q'_2)
\]

SAC：

\[
y=r+\gamma[\min Q'-\alpha\log\pi(a'|s')]
\]

MADDPG：

\[
y=r+\gamma Q'(s',\mathbf a')
\]

MAPPO：

\[
V(s),\quad GAE
\]

### 核心观察

不是只看最终 RMSE，而是看：

> **一个原本能拟合 MC 的 critic，一进入自己的 TD/GAE target 后是否开始失真？**

重点记录：

- Q/V drift；
- target variance；
- ring3/capture rare-event bias；
- success/collision ordering 是否翻转；
- early-recovery 是否再次成为最大误差区。

### Gate

若 A1 PASS、A2 FAIL：

\[
\boxed{\text{bootstrapping / target dynamics 成为主嫌疑}}
\]

---

## A3 — Frozen-Actor Critic Pretrain

### 目标

只测试“在固定好 policy 附近，critic 能否稳定学会当前策略分布”。

### 实施

1. 初始化 strong actor；
2. actor 完全冻结；
3. critic 用当前 policy distribution 训练；
4. 持续检查 MC calibration、success/collision ordering、rare ring3 state prediction；
5. 达标前禁止解冻 actor。

### 成功条件

- heldout calibration 稳定；
- rare capture 不被系统性低估；
- collision/near-success 能区分；
- 不再出现明显 Q/V drift。

---

## A4 — Conservative Online Policy Improvement

### 目标

只有 best critic 进入；验证 critic 健康后，是否能**在不毁掉强 Actor 的前提下**继续 improvement。

### 建议形式

不要再次裸 TD3/MADDPG 放开 actor。

可采用：

\[
L_{actor}=-\lambda_Q Q(s,\pi(s))+\lambda_{BC}L_{BC}
\]

或 KL/action-distance-to-teacher constraint。

### 实施原则

- 初始 teacher/BC constraint 较强；
- actor LR 小；
- critic 已校准后才解冻；
- matched seeds；
- 先短窗口再扩大；
- 允许轻微性能波动，但不能再次 catastrophic collapse。

### 成功目标

- 原有 capture/coverage 能力基本保持；
- policy performance 能恢复或提升；
- critic action ranking 与真实 rollout 方向一致。

---

## A5 — Actor-Critic 线最终判决

只允许三个结局：

### A-PASS-LOCAL

Local Q 健康，问题主要在 bootstrap / policy improvement。

- 不做 central critic；
- 只保留最小必要的 conservative improvement 方法。

### A-PASS-CENTRAL

Local 明显失败，而 Neighbor/Central 明确成功。

才允许主比较：

\[
\text{Local TD3 twin-Q}
\quad vs\quad
\text{Centralized MATD3 twin-Q}
\]

MADDPG single-central-Q 作为次级 baseline。

### A-STOP

若没有清晰赢家、或 online 仍持续破坏高性能 Actor：

> **停止 Actor-Critic 主线，不再继续 SAC/MATD3/GNN critic 架构树。**

论文主线继续使用 IQN。

---

# B 线：IQN Clean-up / Emergent Local Swarm

## B0 — 冻结 Final IQN 基准

### 目标

建立不可移动 reference，避免 cleanup 后不知道性能损失来自哪里。

### 记录

至少固定：

- 4v1 / 8v2 / 12v3；
- capture；
- coverage；
- mixed lifecycle；
- post-capture recovery；
- collision；
- mission/task time；
- 当前 observation contract；
- 当前 reward contract；
- 当前 backbone / `is_pursuing` path；
- 当前 global/local friendly sensing semantics。

---

## B1 — 删除 `is_pursuing` shortcut

### 目标

验证当前人工 role bit 是否真正必要。

最终希望 policy 不再直接读取：

- `is_pursuing`；
- explicit capture/support/coverage role one-hot；
- global phase。

### 第一实验

使用 Final IQN teacher：

\[
\text{Final IQN teacher}
\rightarrow
\text{student without is\_pursuing}
\]

其他 observation 尽量完全一致。

优先：

- distillation / BC；
- 不直接从 scratch 2M。

### Gate

若 student 能基本保留：

- capture；
- coverage；
- post-capture recovery；
- full lifecycle；

则 `is_pursuing` 可以删除。

若显著掉性能，不立即恢复 bit，而先定位它实际上代理了什么真实信息：

- target memory；
- neighbor target evidence；
- pursuing-neighbor geometry；
- phase/history。

---

## B2 — Local Friendly Topology + Local / Truncated Voronoi

### 目标

移除 global-friendly Voronoi 依赖，让 coverage 真正建立在局部通信拓扑上。

### 明确三个尺度

\[
R_s=\text{enemy/obstacle onboard sensing radius}
\]

\[
R_c=\text{friendly communication radius}
\]

\[
R_v=\text{local Voronoi computation radius}
\]

不要假设：

\[
R_c=2R_s
\]

而是利用 local Voronoi 的充分条件：

\[
\boxed{R_c\ge2R_v}
\]

对 agent \(i\) 而言，距离 \(>2R_v\) 的友军不可能影响 \(B(p_i,R_v)\) 内的 Voronoi ownership。

### 实施

- 单独 geometry calibration \(R_c/\sqrt{A/N}\)；
- friend token 只来自 \(\|p_j-p_i\|\le R_c\)；
- enemy/obstacle 仍使用 onboard \(R_s\)；
- local/truncated Voronoi 只依赖通信可见 friends。

### 无邻居 fallback

禁止：

\[
neighbor=\varnothing\Rightarrow centroid=self
\]

第一版候选：

- last valid local Voronoi centroid；
- 若从未有有效邻居，则 map/reference frontier anchor；
- reconnect 后恢复 normal local Voronoi。

### Gate

- pure coverage 能稳定恢复；
- connectivity 不因局部化大面积断裂；
- 不出现 self-centroid freeze；
- global friend information 真正从 policy/runtime contract 中退出。

---

## B3 — `is_pursuing` → Continuous Local Activation `z_i`

### 目标

用局部感知 + 邻居传播 + 时间衰减，替代离散人工角色标签。

定义直接 target evidence：

\[
d_i^t=
\begin{cases}
1,& i\text{ 直接观测到 enemy}\\
0,& otherwise
\end{cases}
\]

同步更新：

\[
\boxed{
z_i^{t+1}=\max\left(d_i^t,\lambda z_i^t,\eta\max_{j\in\mathcal N_i^t}z_j^t\right)
}
\]

其中：

\[
0<\lambda,\eta<1
\]

### 语义

- \(z\approx0\)：coverage-like；
- 中间 \(z\)：support-like；
- \(z\approx1\)：direct pursuit-like。

这些都只是**分析标签**，不是 policy 的显式 role state。

### 更新时序

禁止同一 step 内：

\[
z_i^t\leftarrow z_j^t\leftarrow z_k^t
\]

形成 algebraic loop。

必须采用：

\[
z_{t+1}=F(d_t,z_t,\mathcal N_t)
\]

### Markov / Replay 合同

`z_i` 必须成为真实 observation/state 组成部分，并写入 replay：

\[
(o_t,z_t,a_t,r_t,o_{t+1},z_{t+1},done)
\]

这样 augmented state：

\[
(s_t,z_t)
\]

仍保持合法 Markov transition。

禁止 reward 使用隐藏 `z_i` 而 policy/critic 看不到它。

修改以下任意项后默认 fresh replay：

- \(\eta\)；
- \(\lambda\)；
- propagation rule；
- `z`-dependent reward semantics。

---

## B4 — 先只替换 observation semantics，不改 reward

### 目标

把 information representation 与 reward redesign 解耦。

第一版：

\[
is\_pursuing\rightarrow z_i
\]

但 reward 仍保持当前已验证过的 capture/support/coverage 逻辑。

Evaluator 可继续事后分类 role，但 role 不进入 policy。

### 实施

- teacher distillation / BC 优先；
- matched rollout；
- 小规模 fine-tuning；
- 不重新设计 reward。

### Gate

证明 continuous local activation 能替代人工 role shortcut 而不明显损失 Final IQN lifecycle 能力。

---

## B5 — `z_i`-Conditioned Reward

### 前提

只有 B4 PASS 才进入。

### 第一候选

\[
\boxed{
r_i=z_i r_{capture,i}+(1-z_i)r_{coverage,i}+r_{safe,i}
}
\]

使 support 不再是第三套离散 reward contract，而是两个任务目标之间的连续中间状态。

### 重要风险

如果：

- capture reward 倾向向目标收缩；
- coverage reward 倾向向 Voronoi center 展开；

则中间 \(z\) 可能产生互相抵消或错误的 action preference。

### 因此先做离线 reward/action-preference audit

对典型 support states 分别检查：

- \(r_{capture}\) preferred actions；
- \(r_{coverage}\) preferred actions；
- blended reward preferred actions。

### 若简单线性 blend 不合理

改为更结构化：

\[
r_i=r_{safe}+z_i r_{target-response}+r_{local-space}
\]

让 local-space / spacing 成为始终存在的 swarm objective，而不是两个完全相反奖励的线性硬混合。

### Gate

- support-like states 产生合理接近/补位行为；
- coverage 不被过度吸走；
- capture 能力不明显下降；
- replay contract 清晰、fresh replay 管理正确。

---

## B6 — Local Inhibition / Capture Demand（条件式）

### 进入条件

只有出现：

\[
ring2/ring3\uparrow
\quad\text{但}\quad
collision\approx100\%
\]

或多人持续冗余涌向同一目标时才实现。

### 目标

在保持 local/emergent 的前提下，形成：

\[
\text{local attraction}+\text{local inhibition}
\]

避免所有 UAV 都无限向目标聚集。

### 候选局部信息

- target radial distance；
- local angular gap；
- nearby pursuer count；
- relative speed；
- target-side occupancy。

可以定义局部需求：

\[
D_i^j
\]

并令：

\[
z_{i,eff}=z_iD_i
\]

当 local ring 已基本闭合：

\[
D_i\downarrow
\]

额外 attraction 自动变弱。

### 禁止

- central assignment；
- explicit target capacity；
- 高层 recruiter；
- 显式 role scheduler。

---

## B7 — Full Lifecycle Consolidation

### 最终 policy 输入目标

尽量收敛为：

\[
\{self, local\ friends, local\ enemies, local\ obstacles, z_i\}
\]

不包含：

- `is_pursuing`；
- role one-hot；
- global friend set；
- global enemy coordinates；
- explicit phase。

### 4v1 首先证明完整 lifecycle

\[
coverage\rightarrow detection\rightarrow support/pursuit\rightarrow capture\rightarrow release\rightarrow coverage
\]

### 再做 8v2 / 12v3

验证：

- temporary subgroup formation；
- concurrent target response；
- 未受威胁 UAV 保持 coverage；
- capture 后资源自然回流；
- repeated arrivals。

### 目标

使“capture/support/coverage”只成为对宏观 emergent behavior 的事后解释，而不是 policy 内部显式状态机。

---

## B8 — Robustness / Paper Experiments

主方法稳定后再做，优先级：

1. N / target count scaling；
2. sensing range；
3. communication range；
4. target appearance/removal；
5. repeated arrivals；
6. obstacle layouts；
7. sensing noise；
8. communication delay/loss；
9. stronger/reactive evader。

若条件允许，再补最小真实系统实验或高真实性 Level-3 simulation（delay/noise/dynamics perturbation）。

不要再把主要算力用来比较大量通用 RL algorithms。

---

# 两条线的交汇规则

## 若 A 线找到 central critic 明显更好

可将：

\[
\text{B 线 local-emergent observation/reward}
+
\text{centralized critic training}
\]

组合，但 execution 仍保持 local homogeneous actor。

## 若 A 线只证明 local Q 健康

B 线继续使用 IQN/local critic 即可，不做 centralization。

## 若 A 线无明确结果

直接 A-STOP，不阻塞 B 线。

---

# 当前实时 TODO（2026-09-17）

## P0 — 立即并行启动

### A 线

- [ ] A0：冻结 high-performance actor；
- [ ] 建立统一 trajectory bank；
- [ ] 明确 train/heldout split 与 dataset hash；
- [ ] 确保 normal capture / ring3 / collision / early recovery 足够样本；
- [ ] 保存 empirical MC return 与 event/phase metadata。

### B 线

- [ ] B0：冻结 Final IQN canonical baseline；
- [ ] B1：实现/准备删除 `is_pursuing` 的 student；
- [ ] 优先做 distillation / BC，不从 scratch；
- [ ] matched rollout 验证 `is_pursuing` 是否真是必要 shortcut。

## P1 — P0完成后并行

### A 线

- [ ] A1-LQ：Local Q MC regression；
- [ ] A1-CQ：Central Q MC regression；
- [ ] A1-V：MAPPO-style V MC regression；
- [ ] 条件允许加入 A1-NQ Neighbor Q；
- [ ] 统一 heldout calibration / phase error / success-vs-collision ordering。

### B 线

- [ ] B2：定义 \(R_s,R_c,R_v\)；
- [ ] geometry calibration `R_c >= 2 R_v`；
- [ ] 实现 local/truncated Voronoi；
- [ ] 实现 no-neighbor fallback；
- [ ] 验证 pure coverage locality 与 connectivity。

## P2 — 根据 Gate 推进

### A 线

- [ ] 只有 A1 PASS 的 critic 进入 A2 native bootstrap；
- [ ] 判定 representation vs bootstrap；
- [ ] 决定 A-PASS-LOCAL / A-PASS-CENTRAL / A-STOP。

### B 线

- [ ] B3：实现 local continuous activation `z_i`；
- [ ] 明确同步传播与 replay schema；
- [ ] B4：先只替换 observation role semantics，不改 reward；
- [ ] distillation + small fine-tuning 验证。

## P3 — 方法收口

- [ ] 若 B4 PASS，再进入 B5 `z`-conditioned reward；
- [ ] 只有出现过度聚集/高 collision 才进入 B6 local inhibition；
- [ ] B7 完整 lifecycle consolidation；
- [ ] B8 robustness / paper experiment matrix。

---

# 明确禁止的低价值扩张

在没有新证据前，禁止自动开展：

- 继续扫 TD3/SAC/MAPPO 超参；
- 再加更多 critic heads/width/depth；
- 因 Stage-1 失败直接上 MATD3；
- 无 Gate 地引入 GNN/GAT central critic；
- 同时改 sensing + reward + action space + topology；
- 每个 cleanup 都从 scratch 训练到数百万 step；
- 把 `z_i` 做成正在在线变化的 learned gate 后仍复用旧 replay；
- 把 local swarm 主线改造成显式 task allocator / recruiter。

---

# 最终论文方法目标图

```text
Local enemy / obstacle sensing
          +
Local friendly communication
          ↓
  Local / truncated Voronoi
          +
Local target evidence z_i
          ↓
  One shared homogeneous IQN
          ↓
     Individual action
          ↓
────────────────────────────
Emergent swarm behavior:
Coverage
  → Detection
  → Support propagation
  → Temporary encirclement
  → Capture
  → Signal decay / release
  → Coverage recovery
```

### 方法叙事原则

最终不把以下内容当显式控制状态机：

- capture agent；
- support agent；
- coverage agent。

它们只作为对局部规则产生的宏观行为的分析标签。

最终希望得到：

> **一个 homogeneous / fungible UAV swarm，在有限局部感知与局部通信下，通过共享策略、局部空间几何与可衰减 target evidence，自发完成 persistent coverage → temporary cooperative capture → release/recovery，而无需显式角色分配。**

---

# 规划维护规则

1. 后续每完成一个主要 Gate，只更新本文件相关阶段的状态，不重写历史结论。
2. 新增实验前先说明它属于 A/B 哪一阶段、回答哪一个因果问题。
3. 一次实验尽量只改变一个核心机制；若是 bundle intervention，必须预声明。
4. B 线是论文主线，不等待 A 线成功。
5. A 线必须有止损；若没有明确机制收益，停止扩展 Actor-Critic。
6. 所有 runtime/code/artifact 与本规划冲突时，以最新真实 runtime/artifact 为准，并回写本文件。
