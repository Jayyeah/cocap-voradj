# CoCap 连续动作 MARL 架构重构选型与实验计划

> 日期：2026-08-04  
> 状态：历史设计决策稿；2026-08-05 ax/ay 代码修订已落地，训练需按修订后的新合同重新开始
> 目标线：在当前 CR-MS + CE + VCT-LS 有限视域主线之上，将追捕者的 9 离散动作与 IQN 重构为二维连续加速度控制和 off-policy CTDE 多智能体 Actor–Critic

> **2026-08-05 实现修订（当前唯一合同）：** 连续 Actor 直接输出机体系 `a_x,a_y`，径向 squash 保证 `||a||<=a_max`；`v_max` 不再由策略侧 feasible velocity layer 处理，而由环境动力学统一限制。旧 velocity 公式、配置和 P6 结果保留为设计/故障审计，不可直接续训。完整原因、优劣和迁移规则见 `docs/CONTINUOUS_ACTION_AXAY_REFACTOR_20260805_ZH.md`。
> 动态训练状态不在本文维护；当前六条 500k、AW bridge、实际结果复盘规程和条件 TODO 以 `docs/CONTINUOUS_ACTION_MARL_EXPERIMENT_TRACKER_20260804_ZH.md` 为准。
> 项目内建议简称：**CoCap-PSA-MASAC**（Parameter-Shared Attentive MASAC）。这是本项目实现名，不是某篇论文中可直接照搬的固定算法名。

## 0. 结论先行

本轮架构首选为：

> **参数共享的局部 Transformer 随机 Actor + 训练期特权实体集合双 Critic + 联合经验回放的 attention-MASAC。**

具体含义如下。

1. 所有同构 pursuer 共用一个 Actor。Actor 执行时只读取当前 VCT-LS 有限视域局部 token，保留当前多层 Transformer、mask、目标注意力、summary attention 和 `is_pursuing` 表征；只移除 IQN quantile/9-Q 头，换成二维连续随机策略头。
2. 训练时建立两个相互独立的 centralized soft-Q Critic。Actor 与 Critic 可以复用同一种 entity-token/attention **代码骨架**，但输入合同、归一化器和 learned parameters 完全独立；两个 Critic 之间也不共享 learned encoder。Critic 读取全局实体集合、全体 pursuer 的联合连续动作和 active mask，并通过 masked attention 输出每个 active pursuer 的 $Q_i$。执行阶段完全丢弃 Critic，仍是分散执行。
3. 算法骨干采用 SAC：off-policy replay、重参数化随机策略、twin Q、target critics、Polyak soft update 和自动温度；它比 MADDPG/MATD3 更适合有限视域下“搜索—发现—追随—围捕”的多模态探索。
4. **当前实现改为显式加速度动作：** 对环境暴露的动作定义为机体系 `a_cmd=(a_x,a_y)`。Actor 直接约束 `||a_cmd||<=a_max`；环境旋转、积分并负责 `v_max` 速度上限。下方旧速度投影公式仅保留作历史设计对照，不是当前代码合同：

   \[
   \mathbf u_{i,t}^{B}\in\mathbb B_2(1),
   \qquad
   \left\|\mathbf v_{i,t+1}^{cmd}\right\|_2\le v_{\max},
   \qquad
   \left\|\mathbf v_{i,t+1}^{cmd}-\mathbf v_{i,t}\right\|_2
   \le a_{\max}\Delta T
   \]

   因而最终 $v^{cmd}$ 本身已经满足 $v_{\max}$ 与可调 $a_{\max}$，不是把任意远端速度交给环境再裁剪。环境只校验合同，并在 10 个 $0.05\,\mathrm{s}$ 子步内执行一条满足约束的线性速度参考；现有一次 RL 决策仍持续 $0.5\,\mathrm{s}$。

   **当前 ax/ay 公式覆盖：**

   \[
   \mathbf a_{i,t}^{B}=a_{\max}\tanh(\|\mathbf z\|_2)\frac{\mathbf z}{\max(\|\mathbf z\|_2,\epsilon)},\qquad
   \|\mathbf a_{i,t}^{B}\|_2\le a_{\max}
   \]

   \[
   \mathbf v_{t+1}=\operatorname{clip}_{v_{\max}}\left(\mathbf v_t+R(\mathrm{yaw})\mathbf a_{i,t}^{B}\Delta T\right)
   \]

   当前 SAC 的 log-prob、critic 输入和 joint replay `actions` 全部指向 \(\mathbf a^B\)；环境的 \(\operatorname{clip}_{v_{\max}}\) 只表示物理速度边界，并记录 `speed_limited`。
5. $v_{\max}=3.0\,\mathrm{m/s}$ 保持不变。$a_{\max}=0.4$ 并不偏大；作为二维总加速度上限时反而很可能过小。首轮应扫描

   \[
   a_{\max}\in\{0.4,\ 0.8,\ 1.6\}\ \mathrm{m/s^2}
   \]

   其中 $0.4$ 对应旧纵向推力，$0.8$ 约对应旧模型满速制动时的净减速度，$1.6$ 约对应旧模型满速最大转弯时的速度向量加速度包络。建议把 $0.8$ 作为首个工程候选，但最终值必须由动力学 screening 决定。
6. 当前 CR-MS 的 ring-MS capture、support 的 $0.5/0.5$ approach-only/CE、VCT-LS 局部感知、CE reward/success 和 4v1→8v2→12v3 课程先不改。旧 $64/16/32/16$ replay 的“避免稀有语义饿死”意图保留，但不把逐-agent IQN 的四类固定计数硬搬进 joint CTDE；新方案改为统一 joint ring、所有 active agent 的 masked loss，以及可调的 regime/event 分层采样。
7. 传统 MADDPG 只作为经典对照；**Set-Attention MATD3** 是第一备选。如果 MASAC 长期在 CE 末态抖动且温度调节无法修复，再切换确定性 Actor，而不是一开始就放弃最大熵探索。

最重要的工程边界是：**不能只把 IQN 最后一层从 9 改成 2。** 前端 token encoder 可以保留，但 action API、动力学、joint replay、Actor/Critic、更新损失、checkpoint、screening 和可视化工具都必须同步支持连续动作；否则得到的既不是正确的 SAC，也不是 CTDE。

---

## 1. 当前主线与代码现状

### 1.1 需要继承的项目能力

本重构线应从当前最强 CR-MS 语义出发，而不是退回旧 mix：

- direct detector 使用 compact、velocity-weighted ring importance mean-shift capture reward；旧 approach/MS/front 项在 direct 分支关闭；
- VCT-LS 使用 pursuer-only Voronoi 通信拓扑，以及半径 $20\,\mathrm m$ 的敌人/障碍局部感知；
- one-hop support 仍保持 coverage 角色，只接收

  \[
  r_i^{support}
  =0.5r_i^{approach-only}+0.5r_i^{CE}
  \]

  且 attraction target 只来自直接看到敌人的通信邻居，不把全局敌人坐标偷渡进 Actor；
- coverage 使用 CE centroid-energy、PBRS、strict CE success；$CV<0.15$ 仍只作为并列诊断，而不是替代 CE strict；
- 当前逐-agent IQN 使用 $64/16/32/16$ 四类 batch，它是本次 replay 重构必须保留的**对照信息**，但不是 MASAC 的固定算法合同；

- 课程仍以 4v1 scratch 为第一阶段，随后 8v2、12v3；各阶段开启独立 capture、pure coverage 和 mix screening，并在阶段最优模型上自动做正式 20-rollout/10-GIF。

### 1.2 当前网络实际上是什么

当前实现不是 CTDE，而是 **parameter-sharing independent IQN**：

- 所有 active pursuer 的局部观测被堆成 batch，由同一个 `CoCapIQN` 分别推理；
- 每个 pursuer 只根据自己的局部 token 选一个整数动作；
- replay 逐 agent 保存

  ```text
  (obs_i, int(action_i), reward_i, next_obs_i, done_i, metadata_i)
  ```

- Q 网络看不到同步的全局状态和其他 pursuer 的联合动作。

当前 CR-MS effective config 的主要网络尺寸为：

```text
architecture          voradj_single_head
hidden_dim            256
Transformer layers    4
attention heads       8
batch size            128
training quantiles    8
discrete actions      9
```

单机局部观测为：

```text
self       [9]
pursuers   [8, 7]
evaders    [8, 7]
obstacles  [5, 5]
masks      [22]
types      [22]
```

其中 `self[9]` 是：

```text
[vx_robot, vy_robot,
 nearest_obstacle / map_diag,
 boundary_vec_x_robot / map_diag,
 boundary_vec_y_robot / map_diag,
 out_of_bounds,
 CE_centroid_dx_robot / map_diag * sqrt(N),
 CE_centroid_dy_robot / map_diag * sqrt(N),
 is_pursuing]
```

12v3 中虽然有 12 个 pursuer，但每个 Actor 最多读取 8 个友军 token；这属于有限通信/有限输入语义，不应被 centralized Critic 的全局信息改变。

当前 Transformer 前端及 IQN 头可分别在下列位置核对：

- [`src/cocap_voradj/models/iqn.py`](../src/cocap_voradj/models/iqn.py)：entity MLP、type embedding、Transformer、target/summary attention、quantile embedding 和 9-Q head；
- [`src/cocap_voradj/envs/voronoi_adjacency.py`](../src/cocap_voradj/envs/voronoi_adjacency.py)：VCT-LS 感知、动态 capture/coverage 标签与 token 打包；
- [`src/cocap_voradj/training/replay.py`](../src/cocap_voradj/training/replay.py)：当前逐 agent 整数动作 replay；
- [`src/cocap_voradj/training/trainer.py`](../src/cocap_voradj/training/trainer.py)：epsilon-greedy、IQN gather/max loss 和四类分层采样。

### 1.3 当前动作和动力学

旧追捕者动作是两个离散集合的笛卡尔积：

\[
a\in\{-0.4,0,0.4\}\ \mathrm{m/s^2},
\qquad
\omega\in\left\{-\frac{\pi}{6},0,\frac{\pi}{6}\right\}\ \mathrm{rad/s}
\]

因此共有 9 个整数动作。旧动力学近似为：

\[
\dot s=a-ks,
\qquad
k=\frac{0.4}{3}=0.1333
\]

\[
\mathbf v=s
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix},
\qquad
\dot\theta=\omega
\]

底层子步和决策周期为：

\[
\delta t=0.05\ \mathrm s,
\qquad
N_{sub}=10,
\qquad
\Delta T=N_{sub}\delta t=0.5\ \mathrm s
\]

相关实现见 [`robot.py`](../src/cocap_voradj/dynamics/robot.py) 与 [`pursuer.py`](../src/cocap_voradj/dynamics/pursuer.py)。Evader 目前仍由 APF 输出离散 $(a,\omega)$ action index；本轮只连续化 pursuer，不应把 evader APF 一起重写。

---

## 2. 架构选型

### 2.1 选型标准

本项目不是普通的单智能体连续控制。主架构必须同时满足：

1. 二维连续动作与硬 $v_{\max},a_{\max}$ 约束；
2. 同构多智能体参数共享、CTDE、局部执行；
3. 4/8/12 pursuer 和 agent 中途失活时的 mask/集合处理；
4. off-policy 长期 replay，能持续复用稀缺的发现、capture、post-capture 和 recovery transition；
5. 有限视域下足够的探索能力；
6. 不破坏当前 CR-MS/CE/VCT-LS reward 和课程，能逐项归因；
7. 能复用现有 token Transformer，而不是重新引入固定长度拼接。

### 2.2 候选对比

| 候选 | 连续动作 | CTDE | Off-policy/replay | 变规模 | 有限视域探索 | 本项目定位 |
|---|---:|---:|---:|---:|---:|---|
| MADDPG | 是 | 是 | 是 | 原始 concat 很弱 | 确定性 + 外加噪声 | 经典对照，不做主线 |
| TD3 | 是 | 单智能体原版否 | 是 | 不固有支持 | 确定性 + 平滑噪声 | local baseline；MARL 需改成 MATD3 |
| Set-Attention MATD3 | 是 | 是 | 是 | 好 | 中等 | 第一备选 |
| SAC | 是 | 原版否 | 是 | 取决于编码 | 随机最大熵 | 学习骨干，不单独作为最终 MARL |
| Attention-MASAC | 是 | 是 | 是 | 好 | 好 | **主线** |
| MAPPO | 是 | 是 | 否 | 取决于编码 | 随机 | 强 on-policy 对照，但不符合长期 replay 目标 |
| FACMAC | 是 | 是 | 是 | 较好 | 确定性 | 大规模 credit assignment 失败时的后备 |

### 2.3 为什么不是 MADDPG

MADDPG 的 centralized critic/decentralized actor 是正确方向，但原始结构通常把所有 agent state/action 展平拼接。这会带来：

- 4v1、8v2、12v3 输入维度不同；
- agent 顺序影响 Q；
- agent 失活需要吸收态或固定槽位；
- 确定性 Actor 的搜索主要依靠人为加噪，在局部观测造成的多峰决策中更容易早期坍缩；
- 单 Critic 更容易把 function approximation error 反馈给 Actor。

MADDPG 仍值得保留为小规模经典 baseline，但不值得让新的长期架构继承这些结构性限制。

### 2.4 为什么不是直接使用 TD3/MATD3

TD3 的 twin Q、delayed actor update、target smoothing 能有效减轻连续 Actor–Critic 的过估计；Set-Attention MATD3 因此是很强的第一备选。但它仍是确定性策略：

- local sensing 下“向左搜索或向右搜索”“跟随不同 pursuing 邻居”等动作可能都合理；
- capture 和 CE 都存在多个近似等价空间解；
- 单一确定性输出加各向同性噪声，不等价于学习状态相关的探索分布。

因此先用 SAC 的随机策略；若后续证据显示 CE 末态熵噪声长期无法收敛，再用同一 encoder、joint replay 和 attention critic 接口切 MATD3，转换成本较低。

### 2.5 为什么不是 MAPPO

MAPPO 能做连续 CTDE，也是一条非常可靠的 cooperative MARL 对照线。但它是 on-policy：旧 rollout 很快失效，不能长期反复利用当前任务中稀缺的 capture/recovery 样本。当前项目已经投入大量工程在分阶段经验池、capture snapshot recovery 和 screening 上，主线改用 MAPPO 会舍弃最有价值的样本复用能力。

### 2.6 为什么选 attention-MASAC

选中的实现综合了三个成熟方向：

- SAC 提供连续、off-policy、最大熵随机 Actor、twin Q 和自动温度；
- MADDPG/MAAC 提供 CTDE 和按 agent 查询的 centralized critic；
- Set Transformer/GAT/现有 token Transformer 提供 mask、置换等变和变规模实体聚合。

它尤其契合当前任务的三个特点：

1. **有限视域与重新发现。** Actor 不能看到全局敌人，但 Critic 可在训练时用真实全局状态降低估值方差；执行时不存在信息泄漏。
2. **direct/support 动态角色。** 使用 focal-agent $Q_i$ 和原始 $r_i$，可保留 direct ring-MS 与 support $0.5/0.5$ reward 差异；不需要一开始把所有奖励压成 team sum。
3. **变规模。** 参数共享 Actor 和 masked entity/action critic 不依赖固定 agent 拼接；损失按 active 数取均值，避免 12v3 的梯度和熵权重天然变成 4v1 的三倍。

---

## 3. 新连续动作的正式定义

### 3.1 外部速度动作与策略内部变量必须分开命名

首版对环境暴露的连续动作仍是二维期望地速：

\[
\mathbf a_{i,t}^{cmd}
=
\frac{\mathbf v_{i,t+1}^{cmd,B}}{v_{\max}}
\in\mathbb R^2
\]

其中 $\mathbf v_{i,t+1}^{cmd,B}$ 是**下一个决策边界可达到的机体系速度**。但 Actor 随机头不直接生成一个任意的绝对速度；它先生成单位圆盘内的策略变量：

\[
\mathbf u_{i,t}^{B}
=
\begin{bmatrix}
u_{x,i,t}^{B}\\
u_{y,i,t}^{B}
\end{bmatrix},
\qquad
\left\|\mathbf u_{i,t}^{B}\right\|_2<1
\]

$\mathbf u^B$ 表示本决策周期内归一化的**可行速度改变量**。完整部署策略是：

\[
o_{i,t}
\xrightarrow{\text{Actor}}
\mathbf u_{i,t}^{B}
\xrightarrow{\text{FeasibilityLayer}(\mathbf v_t,v_{\max},a_{\max})}
\mathbf v_{i,t+1}^{cmd,B}
\]

因此从端到端接口看，策略最终给出的仍是 $(v_x^{cmd},v_y^{cmd})$；引入 $\mathbf u$ 只是为了让随机分布、熵和 $a_{\max}$ 可行域都定义清楚。

这里 $B$ 是**生成该观测时的机器人/传感器坐标系**。选择机体系而非世界系的原因来自当前 observation contract：局部几何、速度、当前 CR-MS boundary 和 CE centroid 都已旋转进机器人坐标系，self token 中没有绝对 $\sin\theta,\cos\theta$。如果 Actor 在同一局部观测下被要求直接输出世界系动作，就会出现 observation-action aliasing：相同输入可能对应不同世界方向，策略不可辨识。

可行速度层完成后只旋转一次：

\[
\mathbf v_{i,t+1}^{cmd,W}
=R(\theta_{i,t})\mathbf v_{i,t+1}^{cmd,B}
\]

该世界系命令在本次 $0.5\,\mathrm s$ 决策窗口内固定。不能在每个子步按正在变化的 $\theta$ 反复旋转，否则同一 action 会在执行中自行改变世界方向。

若后续确实要让 Actor 输出 world-frame $(v_x,v_y)$，必须至少给 self token 加入 $\sin\theta,\cos\theta$，并单独做旋转泛化回归；它不是首版最小改动。

### 3.2 速度圆盘，而不是逐轴方框

`max v=3.0` 应表示欧氏地速上限：

\[
\left\|\mathbf v_i\right\|_2\le v_{\max}=3.0\ \mathrm{m/s}
\]

不能写成：

\[
v_x=v_{\max}\tanh z_x,
\qquad
v_y=v_{\max}\tanh z_y
\]

因为对角动作会达到：

\[
\sqrt{v_x^2+v_y^2}=\sqrt2v_{\max}=4.24\ \mathrm{m/s}
\]

也不应在逐轴 tanh 后做 hard norm clip，却继续把它伪装成可逆的 SAC action distribution；clip 是多对一映射，物理 action 的 log-probability 不再能用普通 change-of-variables 直接计算。3.4 的 speed projection 属于另一种有意设计：熵明确留在径向变量 $\mathbf u$ 上，投影只是状态相关的可行速度层，并单独监控投影频率，不把 $\log\pi(\mathbf u)$ 冒充为 $\log\pi(\mathbf v^{cmd})$。

### 3.3 推荐的径向 squashed Gaussian：公式、几何和数值实现

设当前局部 Transformer 输出为 $h_i$。Actor 随机头给出二维对角高斯参数：

\[
\boldsymbol\mu_i=f_\mu(h_i),
\qquad
\log\boldsymbol\sigma_i
=\operatorname{clip}
\left(f_\sigma(h_i),\ell_{\min},\ell_{\max}\right)
\]

标准差由 $\boldsymbol\sigma_i=\exp(\log\boldsymbol\sigma_i)$ 得到。通过重参数化采样：

\[
\mathbf z_i
=\boldsymbol\mu_i
+\boldsymbol\sigma_i\odot\boldsymbol\epsilon_i,
\qquad
\boldsymbol\epsilon_i\sim\mathcal N(\mathbf0,I)
\]

其中 $\mathbf z_i\in\mathbb R^2$ 没有边界。写成极坐标形式：

\[
r_i=\left\|\mathbf z_i\right\|_2,
\qquad
\mathbf q_i=\frac{\mathbf z_i}{r_i}\qquad(r_i>0)
\]

径向 squash 只压缩半径、不改变方向：

\[
\mathbf u_i^B
=
\operatorname{rsquash}(\mathbf z_i)
=
\frac{\tanh r_i}{r_i}\mathbf z_i
=
\tanh(r_i)\mathbf q_i
\]

当 $r_i=0$ 时使用连续极限 $\mathbf u_i^B=\mathbf0$。它有两个直接性质：

\[
\left\|\mathbf u_i^B\right\|_2
=\tanh r_i<1
\]

\[
\frac{\mathbf u_i^B}{\|\mathbf u_i^B\|_2}
=
\frac{\mathbf z_i}{\|\mathbf z_i\|_2}
\qquad(r_i>0)
\]

即高斯样本的方向完整保留，而原始半径 $r\in[0,\infty)$ 被单调压到单位圆盘半径 $\tanh r\in[0,1)$。这与逐轴 tanh 的方形动作域不同。其逆变换为：

\[
\mathbf z
=
\frac{\operatorname{atanh}\|\mathbf u\|_2}
{\|\mathbf u\|_2}\mathbf u
\]

所以除连续定义的原点外，它是 $\mathbb R^2$ 到开单位圆盘的一一映射。

二维 Jacobian 有一个径向特征值和一个切向特征值：

\[
\lambda_r
=
\frac{\mathrm d\tanh r}{\mathrm dr}
=
\operatorname{sech}^2r
\]

\[
\lambda_{\perp}
=
\frac{\tanh r}{r}
\]

因此：

\[
\left|
\det\frac{\partial\mathbf u}{\partial\mathbf z}
\right|
=
\operatorname{sech}^2r
\frac{\tanh r}{r}
\]

变换后策略密度为：

\[
\log\pi_\theta(\mathbf u\mid o)
=
\log\mathcal N
\left(\mathbf z;\boldsymbol\mu,\operatorname{diag}(\boldsymbol\sigma^2)\right)
-
\log\left(\operatorname{sech}^2r\right)
-
\log\left(\frac{\tanh r}{r}\right)
\]

实现不能在 $r\approx0$ 时直接相除。小半径可使用：

\[
\frac{\tanh r}{r}
=
1-\frac{r^2}{3}+\frac{2r^4}{15}+O(r^6)
\]

\[
\log|J|
=
-\frac{4}{3}r^2+\frac{11}{45}r^4+O(r^6)
\]

大半径的第一项可稳定写为：

\[
\log\left(\operatorname{sech}^2r\right)
=
2\left[\log2-r-\operatorname{softplus}(-2r)\right]
\]

确定性评估使用：

\[
\mathbf u_i^{det}
=
\operatorname{rsquash}(\boldsymbol\mu_i)
\]

准确说它是“latent mean 经 squash 后的确定性动作”，不一定等于变换后动作分布的数学均值：

\[
\operatorname{rsquash}\left(\mathbb E[\mathbf z]\right)
\ne
\mathbb E[\operatorname{rsquash}(\mathbf z)]
\]

本项目的 SAC 熵正则定义在 $\mathbf u$ 空间；下一节的可行速度层再把它确定性映射为最终 $v^{cmd}$。必须测试 $r\rightarrow0$、大 $r$ 和动作边界附近的 log-prob 与梯度均为有限值。

如果径向分布实现阻塞 smoke，可临时把归一化增量限制为每轴 $u_x,u_y\in[-1/\sqrt2,1/\sqrt2]$ 的标准 tanh Gaussian，再进入同一可行速度层；它严格落在单位圆盘内，但损失轴向最大增量，只能作为短期诊断，不作为最终动作合同。

### 3.4 $a_{\max}$ 在策略侧生成可行命令，环境只校验

采纳本轮建议：不再让 Actor 给出任意远端 $v^{cmd}$，然后依靠仿真环境的最大加速度裁剪。定义一个决策周期允许的最大速度改变量：

\[
A=a_{\max}\Delta T,
\qquad
\Delta T=N_{sub}\delta t=0.5\ \mathrm s
\]

先把当前世界系速度旋转到生成 action 的决策时刻机体系：

\[
\mathbf v_{i,t}^{B}
=
R(-\theta_{i,t})\mathbf v_{i,t}^{W}
\]

Actor 的单位圆盘变量给出一个自由速度增量：

\[
\Delta\mathbf v_{i,t}^{free,B}
=
A\mathbf u_{i,t}^{B}
\]

\[
\mathbf v_{i,t+1}^{pre,B}
=
\mathbf v_{i,t}^{B}
+\Delta\mathbf v_{i,t}^{free,B}
\]

最后由 Actor 后、环境前的可行速度层投影到速度圆盘：

\[
\mathbf v_{i,t+1}^{cmd,B}
=
\Pi_{\mathcal B_v}
\left(\mathbf v_{i,t+1}^{pre,B}\right)
\]

\[
\Pi_{\mathcal B_v}(\mathbf x)
=
\frac{\mathbf x}
{\max\left(1,\|\mathbf x\|_2/v_{\max}\right)},
\qquad
\mathcal B_v=\{\mathbf v:\|\mathbf v\|_2\le v_{\max}\}
\]

该映射同时保证两个硬约束。速度上限由投影定义直接成立。对加速度上限，由于到闭凸集的欧氏投影是非扩张映射，且 $\mathbf v_t^B\in\mathcal B_v$：

\[
\begin{aligned}
\left\|
\mathbf v_{t+1}^{cmd,B}-\mathbf v_t^B
\right\|_2
&=
\left\|
\Pi_{\mathcal B_v}(\mathbf v_t^B+A\mathbf u_t^B)
-\Pi_{\mathcal B_v}(\mathbf v_t^B)
\right\|_2\\
&\le
A\|\mathbf u_t^B\|_2\\
&\le
a_{\max}\Delta T
\end{aligned}
\]

所以最终命令本身已经满足：

\[
\|\mathbf v_{t+1}^{cmd}\|_2\le v_{\max},
\qquad
\frac{\|\mathbf v_{t+1}^{cmd}-\mathbf v_t\|_2}{\Delta T}
\le a_{\max}
\]

环境收到可行命令后，不再静默 clip。它只执行断言，并在子步中采用线性速度参考：

\[
\mathbf v_{t,k}^{W}
=
\mathbf v_t^W
+\frac{k}{N_{sub}}
\left(
\mathbf v_{t+1}^{cmd,W}-\mathbf v_t^W
\right),
\qquad
k=0,\ldots,N_{sub}
\]

每个子步因此具有同一个净平面加速度：

\[
\mathbf a_{t,k}^{W}
=
\frac{\mathbf v_{t,k+1}^{W}-\mathbf v_{t,k}^{W}}{\delta t}
=
\frac{\mathbf v_{t+1}^{cmd,W}-\mathbf v_t^W}{\Delta T},
\qquad
\|\mathbf a_{t,k}^{W}\|_2\le a_{\max}
\]

位置用梯形积分：

\[
\mathbf p_{t,k+1}
=
\mathbf p_{t,k}
+\frac{\mathbf v_{t,k}^{W}+\mathbf v_{t,k+1}^{W}}{2}\delta t
\]

环境断言应允许一个数值容差 $\epsilon_c$；超限应抛错或记录 contract violation，不能静默修正。碰撞响应或未来外扰可以使实际末速度偏离命令，但那属于环境转移，不能把实际速度偷换为策略 action。

这里 $a_{\max}$ 是 action adapter 的显式可调参数。若一个训练 run 内固定，它只写入 config、checkpoint 和 replay manifest；若同一模型跨 episode 随机化 $a_{\max}$，则必须把：

\[
\rho_a
=
\frac{a_{\max}\Delta T}{v_{\max}}
\]

作为可部署的 self/context feature 输入 Actor 和 Critic，否则相同 observation/action 会对应不同转移规律。

SAC 的熵定义在一一映射的 $\mathbf u$ 空间，Critic 和 replay 使用最终归一化命令：

\[
\mathbf a_t^{cmd}
=
\frac{\mathbf v_{t+1}^{cmd,B}}{v_{\max}}
\]

Actor 目标实际是：

\[
J_\pi
=
\mathbb E
\left[
\alpha\log\pi_\theta(\mathbf u_t\mid o_t)
-Q\left(S_t,g_{S_t}(\mathbf u_t)\right)
\right]
\]

其中 $g_{S_t}$ 是上述可行速度层。速度圆盘投影在边界处是多对一映射，因此首版不伪造 $\log\pi(\mathbf v^{cmd}\mid o)$ 的 Jacobian。`raw_z` 和历史 $\mathbf u$ 不需要写入正式 replay；在线记录 `speed_projection_rate` 与投影前后差值即可。若投影长期频繁，说明策略在速度边界浪费 latent entropy，应调整 $a_{\max}$、目标熵或进一步研究无投影的可行域参数化。

若 action 是高层期望速度，不应再机械叠加旧模型的 $-k\mathbf v$，否则恒定命令会产生稳态偏置。若未来需要空气阻力，应在低层控制器中显式补偿，并重新标定净平面 $a_{\max}$。

### 3.5 为什么 $0.4$ 很可能不是“过大”而是“过小”

若没有策略侧可行速度层，最坏从 $(3,0)$ 瞬时切换到 $(-3,0)$，一个决策周期内的等效加速度为：

\[
a_{eff}
=\frac{6}{0.5}
=12\ \mathrm{m/s^2}
\]

所以“直接设置速度”一定过激；但这不能推出硬约束 $a_{\max}=0.4$ 过大。

旧系统的 $0.4$ 只是纵向控制输入。满速最大转弯时还有：

\[
a_\perp
=v_{\max}\omega_{\max}
=3\frac{\pi}{6}
\approx1.57\ \mathrm{m/s^2}
\]

旧系统在满速、最大制动输入时的瞬时净减速度约为：

\[
a_{brake,old}
=0.4+kv_{\max}
=0.4+0.4
=0.8\ \mathrm{m/s^2}
\]

用无量纲 reachable-velocity ratio 判断新约束强度：

\[
\rho_a
=\frac{a_{\max}\Delta T}{v_{\max}}
\]

首轮三个候选为：

| $a_{\max}$ | $\rho_a$ | 理想制动距离 $d=v_{\max}^2/(2a_{\max})$ | 物理含义 |
|---:|---:|---:|---|
| $0.4$ | $0.0667$ | $11.25\,\mathrm m$ | 旧纵向推力锚点，二维转向/停车可能过慢 |
| $0.8$ | $0.1333$ | $5.625\,\mathrm m$ | 旧满速制动锚点，首个工程候选 |
| $1.6$ | $0.2667$ | $2.8125\,\mathrm m$ | 接近旧满速横向转弯包络 |

三个候选都远低于“一步全速反向”所需的 $12\,\mathrm{m/s^2}$，并不属于无约束瞬移。最终选择不能只看 capture rate，还要同时看 speed-projection rate、可达速度改变量、过冲、碰撞、CE stationary、路径长度和 jerk。

### 3.6 yaw、速度方向与有限感知必须分别定义

#### 3.6.1 当前 VCT-LS 不是扇形 FOV

当前 CR-MS effective config 为：

```text
perception.angle          = 2π
enemy_sensing_radius     = 20 m
obstacle_sensing_radius  = 20 m
```

因此当前“有限视域”的准确含义是**有限半径下的全向局部感知**。enemy/obstacle token 的可见性主要由 surface clearance 是否落入感知半径决定；yaw 目前不会决定实体是否可见。它只影响两件事：

1. 世界系几何与速度如何旋入 Actor 的局部 token；
2. 机体系速度命令如何旋回世界系。

所以“yaw 不更新会让敌人离开固定扇形 FOV”不适用于当前配置。只有未来设置 $\phi_{FOV}<2\pi$ 时，才应同时满足距离和夹角条件：

\[
d_{ij}^{surface}\le R_{sense}
\]

\[
\left|
\operatorname{wrap}
\left(
\operatorname{atan2}(y_j-y_i,x_j-x_i)-\theta_i
\right)
\right|
\le\frac{\phi_{FOV}}{2}
\]

#### 3.6.2 三个方向不能混为一个变量

定义机体/传感器 yaw、实际地速方向和命令方向：

\[
\theta_t
=
\text{body/sensor yaw}
\]

\[
\psi_t^v
=
\operatorname{atan2}(v_{y,t}^W,v_{x,t}^W)
\]

\[
\psi_t^{cmd}
=
\operatorname{atan2}(v_{y,t+1}^{cmd,W},v_{x,t+1}^{cmd,W})
\]

机头与航迹的偏差为：

\[
\beta_t
=
\operatorname{wrap}(\psi_t^v-\theta_t)
\]

对二维全向 UAV，$\beta_t\ne0$ 是合法侧飞状态，不应强制假设速度必须沿机头方向。命令与当前航迹方向的偏差为：

\[
e_{\psi,t}
=
\operatorname{wrap}(\psi_t^{cmd}-\psi_t^v)
\]

由于 $a_{\max}$，一次决策内部真实速度沿线性参考从 $\mathbf v_t$ 过渡到 $\mathbf v_{t+1}^{cmd}$，因此速度幅值与方向相对 command 存在正常滞后。无碰撞、无外扰时，决策边界末端达到 command；周期内有：

\[
\mathbf e_{v,t,k}
=
\mathbf v_{t+1}^{cmd}-\mathbf v_{t,k}
=
\left(1-\frac{k}{N_{sub}}\right)
\left(\mathbf v_{t+1}^{cmd}-\mathbf v_t\right)
\]

在低速时，即使 $\|\mathbf e_v\|$ 很小，角度误差也可能很大，所以监控应同时报告向量误差、speed error 和只在速度超过阈值时定义的 direction error。

#### 3.6.3 零速没有速度方向

当 $\|\mathbf v_t\|_2=0$ 时，$\operatorname{atan2}(0,0)$ 没有物理定义。不能把它硬编码为零角度，否则每次停车都会把局部坐标系跳回世界 $x$ 轴。若未来 yaw 需要跟踪航迹，应使用迟滞：

\[
v_{enter}>v_{exit}>0
\]

- $\|\mathbf v\|\ge v_{enter}$：速度方向有效；
- $\|\mathbf v\|\le v_{exit}$：速度方向失效并保持上一 yaw；
- 中间区间保持上一次有效/无效状态。

这样可避免低速数值噪声让所有局部 token 共同旋转抖动。

#### 3.6.4 为什么不把 body frame 直接定义为瞬时速度方向

若每时刻强制：

\[
\theta_t=\psi_t^v
\]

那么它实际是 course frame，而不是独立 body/sensor frame，会带来：

1. self velocity 几乎总变成 $[\|\mathbf v\|,0]$，网络失去侧滑/横向速度信息；
2. 零速方向不存在，必须依赖历史 fallback；历史若未入状态会引入隐藏状态；
3. 低速微小噪声可造成接近 $180^\circ$ 的 frame 翻转，使全部友军、敌人、障碍、CE 与 boundary token 同时跳变；
4. 速度向量受 $a_{\max}$ 约束，不代表低速时速度**角度**受约束；极小 $\Delta\mathbf v$ 就可大幅改变方向；
5. action 在一个由 action 自己将改变的 frame 中表达，形成不必要的坐标循环依赖；
6. 若未来使用扇形 FOV，相当于允许传感器借低速方向噪声瞬时转向，绕过 yaw-rate 约束；
7. 有风/外流时地速方向本来就不等于机体朝向，用地速定义 body 会让感知朝向随外流漂移；
8. 世界状态只有微小变化时局部表示却大幅旋转，会增大 replay 的 TD target 方差，CE 停稳阶段尤其敏感。

#### 3.6.5 首版与未来扇形 FOV 的推荐合同

为隔离“离散 IQN → 连续 MASAC”这一变量，当前全向有限半径感知的首版建议：

```yaml
yaw:
  mode: hold
  init: aligned_world_axis  # P0 smoke：yaw=0
```

即 episode 内保持初始化 yaw。P0 smoke 先让所有 agent 的 yaw=0，使 body frame 与世界坐标重合，降低连续动作首轮的变量数量；这不是最终旋转泛化方案。二维全向 UAV 仍可以保持朝向并沿任意平面方向移动。连续线稳定后，再加入共同旋转初始化或随机 yaw，做旋转鲁棒性测试。此时 `theta` 明确表示局部坐标基，不再被误读为速度方向。

注意：当前源码在未显式传入 theta 时仍默认随机初始化 yaw；`init: aligned_world_axis` 是连续动作线待实现的配置/初始化分支，不改变旧 `unicycle_discrete` 线。

未来启用 $\phi_{FOV}<2\pi$ 的扇形感知后，再比较 bounded command-heading tracker：

\[
\psi_k^{ref}
=
\operatorname{atan2}(v_{y,k}^{cmd,W},v_{x,k}^{cmd,W})
\]

\[
\theta_{k+1}
=
\operatorname{wrap}
\left[
\theta_k
+\operatorname{clip}
\left(
\operatorname{wrap}(\psi_k^{ref}-\theta_k),
-\omega_{\max}\delta t,
\omega_{\max}\delta t
\right)
\right]
\]

当 command speed 低于阈值时保持 yaw。若实验表明移动方向与观察方向必须独立控制，再将动作扩为 $(v_x^B,v_y^B,\omega_{yaw})$，而不是用瞬时地速方向隐式操纵传感器。

### 3.7 Boundary 的当前坐标语义与推荐升级

当前 strongest CR-MS effective config 已明确使用：

```text
boundary_feature_mode = nearest_vector_robot_oob
```

代码先在世界系求最近矩形边界点，再通过 `_robot_frame(..., is_vector=True)` 旋入机体系。因此当前 self token 的 `boundary_vec_x_robot/boundary_vec_y_robot` **不是世界轴值**，并且另有 `out_of_bounds`。用户观察到的世界系 $x/y$ 语义对应旧/默认 `axis_signed` 分支：它把 world-x 最近竖墙距离和 world-y 最近横墙距离直接写入 self；旧 IQN 在固定轴地图和固定训练分布上可以拟合，并不说明该表示具有好的旋转语义。

当前 local nearest-vector 比旧世界轴表示更合理，但仍有三个缺点：

- 只暴露最近的一面墙，丢失角点处另一面墙以及远端墙信息；
- 最近墙在 medial axis 上 hard switch。方形中心附近可能从“指向左墙”跳到“指向右墙”，完全等距时还受固定列表顺序 tie-break 影响；
- 对未来非矩形边界、内外双区和“出区后再进入”扩展不自然。

长期推荐把矩形边界表示为 4 个 half-space boundary tokens。令每面墙满足：

\[
(\mathbf n_j^W)^T\mathbf x\le b_j
\]

其中 $\mathbf n_j^W$ 是外法向。对 pursuer $i$ 定义连续的 signed clearance 与局部法向：

\[
c_{ij}
=
b_j-(\mathbf n_j^W)^T\mathbf p_i^W
\]

\[
\mathbf n_{ij}^B
=
R(-\theta_i)\mathbf n_j^W
\]

$c_{ij}>0$ 表示在该墙内侧，$c_{ij}<0$ 表示越过该墙。每面墙的 token 可取：

\[
\mathbf b_{ij}
=
\left[
\operatorname{clip}\left(\frac{c_{ij}}{D},-c_{max},c_{max}\right),
n_{ij,x}^B,
n_{ij,y}^B,
\mathbb I(c_{ij}<0)
\right]
\]

四面墙作为无固定顺序的 token set 进入 attention，同时在 self 中保留总标志：

\[
o_i^{OOB}
=
\mathbb I\left(\min_j c_{ij}<0\right)
\]

这种表示在角点同时暴露两面墙，每个 clearance 都连续，没有 hard argmin 跳变，并可通过增加 boundary/region type 扩展到多边形、内区和外区。`out_of_bounds` 必须保留，正好作为未来“允许外出—再进入”的接口；届时再增加 `region_id`、outside duration 或 re-entry target 等真正因果状态，不应现在预填未来信息。

本线初版冻结当前 self token：继续使用 `nearest_vector_robot_oob`，不改变 boundary 维度、坐标语义和 out-of-bound flag。四墙 token 只作为后续独立消融方案。不要在同一次 smoke 中同时改变动作、Critic 和 observation shape。

### 3.8 保留双动作模式

环境必须通过显式配置分流：

```yaml
pursuer:
  action_mode: acceleration_2d_body
  max_speed: 3.0
  max_acceleration: 0.8

continuous_action:
  action_constraint: policy_acceleration_disk
  env_constraint_mode: assert_only

yaw:
  mode: hold
```

旧线继续使用：

```yaml
pursuer:
  action_mode: unicycle_discrete
```

旧 checkpoint、旧 rollout、旧 GIF 和旧 reward tests 必须仍走原分支，不能通过全局替换 `Robot.update_state` 让旧实验的动力学悄然变化。

---

## 4. CoCap-PSA-MASAC 网络与学习目标

### 4.1 整体结构

```text
执行期 local obs_i
        │
        ▼
共享 LocalEntityTokenEncoder_ψ（保留当前 Actor Transformer 前端）
        │ h_i
        ▼
共享 stochastic Actor head → μ_i, logσ_i → u_i∈unit disk
        │
        ▼
可行速度层(v_i, vmax, amax) → feasible v_cmd,i
        │
        └────────────── decentralized execution

训练期 privileged global entity set + joint feasible commands + masks
        ├── GlobalEntityTokenEncoder_φ1 → Central Critic Q_φ1 → [Q_1^1,...,Q_1^N]
        └── GlobalEntityTokenEncoder_φ2 → Central Critic Q_φ2 → [Q_2^1,...,Q_2^N]
```

Actor 与 Critic 的“Transformer”不是同一个网络。二者可复用 token-attention 的实现类和无参数几何工具，但 Actor 处理有限视域、单机自中心的 local schema；Critic 处理训练期 privileged、世界系一致的 global schema，normalizer 和 learned parameters 独立。$Q_{\phi_1}$ 与 $Q_{\phi_2}$ 也从输入 encoder 到输出 head 全部独立，避免 twin-Q 误差因共享 trunk 而高度相关。

“focal pursuer $i$”不是 direct pursuer、leader 或固定编号，而是**当前所查询个体回报 $Q_i$ 的任意 active pursuer**。同一条 joint transition 中，每个 active pursuer 都依次作为 focal；一次向量化前向即可同时输出所有 $Q_i$。

### 4.2 可直接继承的 Actor 前端

首版新增独立的 `LocalEntityTokenEncoder`，按当前 `CoCapIQN` Transformer 前端的功能和参数命名复制/映射，但**不修改现有 `CoCapIQN` 的实现路径**。这样可以先验证连续线、权重迁移和旧线零回归；只有连续线跑通且两边都有完整回归测试后，才考虑把共同部分抽成共享实现。可保留并选择性加载：

```text
encoders.*
type_embedding.*
transformer.*
target_query/key/value.*
summary_role_embedding.*
summary_attention.*
summary_fusion.*
pursuing_embed.*
single_action_feature.*
```

删除或不迁移：

```text
cos_embedding.*
quantile embedding / tau sampling
single_head 的 9-Q 输出层
IQN quantile Huber loss
Q gather / next-action max
epsilon-greedy
hard target IQN copy
```

即：

\[
h_i=E_\psi(o_i)\in\mathbb R^{256}
\]

\[
(\boldsymbol\mu_i,\log\boldsymbol\sigma_i)
=A_\theta(h_i)
\]

初期使用一个共享 Actor head，并继续把 `is_pursuing` 融入 feature；不要同时拆 capture/coverage 双 Actor，否则 role switch、动作连续化和网络改造无法分别归因。

旧 CR-MS checkpoint 只能 shape-compatible 初始化 encoder；Actor head 和两个 Critics 必须新建。还要特别注意：旧 unicycle 策略的机体系横向速度长期接近零，`self[1]=v_y^B` 对旧 encoder 训练不足，因此“encoder 成功加载”不代表新横向控制已经迁移。实验必须比较 scratch 与 encoder transfer。

### 4.3 Actor dropout 处理

当前 Transformer 与 summary attention 的 dropout 在源码中硬编码为 `0.1`，YAML `dropout` 未实际消费；当前 target IQN 也没有明确切到 `eval()`。对 SAC 来说，Actor 已经通过 $\epsilon$ 产生有定义的随机性，额外 dropout 会让 $\log\pi$ 漏掉一层随机变量；target critic 的 dropout 则会增加 Bellman target 噪声。

建议：

- 把 dropout 变为显式配置；
- 第一版连续 Actor 和两个 Critics 均用 `dropout=0.0`；
- deterministic evaluation 强制 `eval()`；
- 若后续要恢复 dropout，单独做消融，不与动作重构同时解释。

这保留的是 token 处理结构和可迁移参数，而不是保留一个当前未被配置正确控制的随机细节。

### 4.4 Centralized twin critics：输入、focal query 与扩规模

#### 4.4.1 与 Actor 同构还是异构

答案是：**建模范式同类，数据语义异构，参数完全独立。** Actor 和 Critic 都可使用 type embedding、masked attention 和 entity set，但不能把 Actor 已截断、已旋转、仅局部可见的 token 当作完整 central state。训练期为每个 Critic 单独构造全局实体 token：

```text
pursuer token:
  normalized world position/velocity, sin(yaw), cos(yaw),
  active/collision flag, direct/support/coverage role,
  final feasible command a_cmd or v_cmd,
  optional causal controller state

evader token:
  normalized world position/velocity, active flag

obstacle token:
  normalized position/radius

boundary token:
  global wall normal, signed clearance/half-space parameters

context token:
  map scale/bounds, active target count, team size,
  only deployable causal task/dynamics parameters
```

每个 pursuer 的 final feasible command embedding 必须绑定到对应 pursuer token；若 Critic 主要几何采用世界系，可用已知 yaw 将 body command 可微旋转到世界系再融合。两个 Critic 使用独立的 raw-feature normalizer、token MLP、attention encoder、pool/query 和 output head：

\[
\mathbf q^{(k)}
=Q_{\phi_k}(S,\mathbf A^{cmd},M)
\in\mathbb R^{N_{max}},
\qquad
k\in\{1,2\}
\]

其中 active mask $M$ 使 padded/失活 agent 不参与 attention 和 loss。每个 pursuer 的 contextual token 作为自己的 focal query，得到：

\[
Q_{\phi_k}^{i}(S,\mathbf A)
=
\mathbb E\left[G_i\mid S,\mathbf A\right]
\]

同一个 $S,\mathbf A$ 对所有 focal $i$ 相同，区别是 query 与该 agent 的 $r_i$ Bellman target。direct、support、pre-capture coverage、post-capture coverage 都可以成为 focal；role 只是当前状态特征，不改变“focal”的定义。这样具有：

- 对 pursuer 排列的等变性；
- 对其他实体排列的不变/等变性；
- 4/8/12 规模共享参数；
- direct/support 个体 reward 的保真 credit assignment。

#### 4.4.2 是否复用各 pursuer 的 self token

需要区分三种“复用”：

1. **可复用无参数特征函数。** 坐标变换、速度归一化、role/OOB 定义和 mask 工具可以共用，避免语义漂移；
2. **可复用 raw 物理语义。** Critic pursuer token 当然可以包含 body/world velocity、role、OOB 等 self 中已有的物理量，但还必须补全 world position、heading、全局实体与 joint feasible commands；
3. **首版不复用 Actor learned $h_i$。** 当前 $h_i$ 已被每台机各自坐标系、最多 8 友军截断和局部可见性上下文化。把所有 $h_i$ 拼起来既不是一致 global state，又会重复实体；若不 detach，critic loss 还会污染 Actor encoder；即使 detach，也会形成随 Actor 更新而漂移的 Critic 输入。

因此首版 Critic 使用新的 global encoder。后续可将 $\operatorname{stop\_gradient}(h_i)$ 经独立 projection 作为**辅助分支**消融，测试它是否提供局部决策摘要，但不能替代 global raw tokens，也不能让两个 Critic 共享这一 learned trunk。

#### 4.4.3 对更大规模的适应性

集合编码使参数量不依赖 pursuer 顺序或实际 active 数，但 dense self-attention 对总 token 数 $T$ 的成本仍为：

\[
\text{time/memory}=O(T^2)
\]

当前 4v1、8v2、12v3 的全局实体数很小，full attention 是合理首版。扩规模必须同时做到：

- 不使用固定 agent-ID embedding；使用 active/entity mask；
- reward、entropy、Q loss 和 actor loss按 active 数做 masked mean；
- 4/8/12 混合规模训练、agent dropout 和 cross-size matrix，而不只验证 shape；
- batch 按 token budget 或规模 bucket 组织，减少 padding 浪费；
- size/context 只能表达真正因果量，不能让网络依赖训练规模捷径。

当 $N$ 进一步增大到 full attention 成为瓶颈时，再依次比较 KNN/sparse graph attention、focal-query cross-attention、Set Transformer inducing tokens 或分层 team pooling。结构能接收不同 $N$ 只表示“可计算”，不能据此宣称任意规模的零样本性能泛化。

### 4.5 为什么主线使用 per-agent focal Q

当前 reward 是逐 agent、逐角色生成的。若直接使用：

\[
r_{team}=\sum_i r_i
\]

则 12v3 的 reward 尺度天然大于 4v1；若使用简单均值，又可能把 direct ring-MS 与 support approach/CE 的局部 credit 混平。因此首版保持：

\[
G_i=\sum_{l=0}^{\infty}\gamma^l r_{i,t+l}
\]

并让 centralized critic 预测：

\[
Q_i(S,\mathbf A^{cmd})
=\mathbb E[G_i\mid S,\mathbf A^{cmd}]
\]

Actor 仍因 $Q_i$ 读取全体 action 而学习协调；只是监督目标保持现有 reward 语义。后续若发现局部奖励导致自私解，可再消融：

\[
r_i^{mix}
=(1-\lambda)r_i
+\lambda\frac{1}{N_t}\sum_j r_j
\]

但 $\lambda>0$ 不应和首个架构 smoke 同时加入。

### 4.6 Soft Bellman target

对 active focal agent $i$，从当前共享 Actor 采样下一时刻联合动作：

\[
\mathbf u_j'
\sim\pi_\theta(\cdot\mid o_j'),
\qquad
j\in\mathcal A_{t+1}
\]

再通过下一状态对应的可行速度层得到 final command：

\[
\mathbf a_j^{cmd\prime}
=
g_{S',j}(\mathbf u_j'),
\qquad
\mathbf A^{cmd\prime}
=
\{\mathbf a_j^{cmd\prime}\}_{j\in\mathcal A_{t+1}}
\]

首版 per-agent soft target 定义为：

\[
y_i
=r_i
+\gamma b_i
\left[
\min_{k\in\{1,2\}}
Q_{\bar\phi_k}^{i}(S',\mathbf A^{cmd\prime})
-\alpha\log\pi_\theta(\mathbf u_i'\mid o_i')
\right]
\]

其中 $b_i$ 是显式 bootstrap mask，不再把所有 `done` 混为一个布尔量。Critic loss 为：

\[
J_Q
=
\frac{1}{\sum_b\sum_i M_{b,i}}
\sum_b\sum_i M_{b,i}
\sum_{k=1}^{2}
\left(
Q_{\phi_k}^{i}(S_b,\mathbf A_b^{cmd})-y_{b,i}
\right)^2
\]

所有 agent 求平均而不是求和，避免规模变化改变有效 critic learning rate。

### 4.7 Shared Actor loss

对每个 focal agent $i$，联合动作中只有其自身 action 对当前 focal loss 反向传播，其他 agent action 使用 `stop_gradient`：

\[
\mathbf A_\theta^{cmd,(i)}
=
\left(
\operatorname{sg}(g_{S,1}(\mathbf u_1)),\ldots,
g_{S,i}(\mathbf u_i),\ldots,
\operatorname{sg}(g_{S,N}(\mathbf u_N))
\right)
\]

\[
J_\pi
=
\frac{1}{\sum_b\sum_i M_{b,i}}
\sum_b\sum_i M_{b,i}
\left[
\alpha\log\pi_\theta(\mathbf u_{b,i}\mid o_{b,i})
-
\min_k Q_{\phi_k}^{i}
\left(S_b,\mathbf A_{\theta,b}^{cmd,(i)}\right)
\right]
\]

因为 Actor 参数共享，对所有 focal $i$ 的平均梯度最终都会更新同一个 $\theta$。对其他 action 截断梯度可避免一个 focal Q 通过所有共享 action 重复产生 $N^2$ 级耦合梯度。

### 4.8 自动温度

初始使用一个共享温度：

\[
\alpha=\exp(\log\alpha)>0
\]

\[
J_\alpha
=
-\frac{1}{\sum_b\sum_i M_{b,i}}
\sum_b\sum_i M_{b,i}
\log\alpha
\left[
\operatorname{sg}
\left(
\log\pi_\theta(\mathbf u_{b,i}\mid o_{b,i})
\right)
+\mathcal H_{target}
\right]
\]

二维归一化 action 的首个候选为：

\[
\mathcal H_{target}=-2
\]

必须按 active agent 数取均值。不要把 joint entropy 直接求和，否则 12v3 的熵项约为 4v1 的三倍。若 capture 正常而 CE strict 因末态抖动受损，依次尝试：

1. 检查 deterministic eval 是否本来已稳定；
2. 降低 coverage/recovery 样本的 target entropy 或做后期 anneal；
3. 使用 role/当前可观测 control-regime-conditioned $\alpha$；
4. 切换 Set-Attention MATD3。

不要在首版直接关闭熵，因为这会同时削弱有限视域搜索。

### 4.9 IQN 的分布式价值如何处理

首版使用 scalar twin soft-Q，把“连续动作 + CTDE + joint replay”先跑通。不要同时实现连续 quantile critic，否则失败后无法区分来自动作、SAC、CTDE、分布式 target 还是 quantile 回归。

基础线 solid 后，如确有证据表明 return 多峰/尾部风险信息有价值，可用 TQC 或 distributional SAC 恢复 quantile critic。它是第二阶段延续，不是第一次重构的阻塞项。

---

## 5. Joint replay 与训练数据合同

### 5.1 为什么当前逐 agent replay 不够

Centralized critic 必须在同一个时间点看到：

\[
(S_t,\mathbf A_t^{cmd},\mathbf r_t,S_{t+1})
\]

当前 replay 把每个 agent transition 独立打散，无法可靠重建同步的其他 agent action，也无法区分 padding、失活和下一时刻 active set。把整数 action 改成 float `[2]` 只能得到 parameter-sharing local SAC，仍不是 CTDE。

### 5.2 精简后的 logical transition schema

原稿同时保存 `raw_z`、body/world command、executed velocity、executed acceleration 和 saturation，混合了“Bellman 训练合同”与“调试日志”。正式 replay 不需要如此多种 $v$。每个 environment decision 只保存一次 joint transition，逻辑 schema 建议为：

```text
transition_id

actor_obs_t                  ragged/padded [N_max, ...]
critic_state_t               global entity-set tensors
agent_mask_t                 [N_max] bool

action_cmd_body_norm         [N_max, 2] float32
                              # final feasible v_cmd^B / v_max
reward_total                 [N_max] float32
reward_basis                 [N_max, K] optional fixed tensor
                              # 仅在计划 reward relabel 时保存

actor_obs_tp1
critic_state_tp1
agent_mask_tp1

terminated                   [N_max] bool
truncated                    scalar bool
bootstrap_mask / discount    [N_max] float32

active_target_mask           [E_max] bool
role_mask                    [N_max] direct/support/coverage
origin                       online_mix/post_capture/recovery_pure
event_bits                   discovery/final_capture/collision/CE_success/...
team_size / target_count / obstacle_count
```

其中唯一需要持久化的动作是：

\[
\mathbf a_{i,t}^{cmd}
=
\frac{\mathbf v_{i,t+1}^{cmd,B}}{v_{\max}}
\]

它是 Actor + 可行速度层最终交给环境的命令，也是 Critic 的 action。其余字段处理如下：

- `raw_z` 和历史 behavior log-prob：标准 off-policy SAC Bellman 更新不需要，不持久化；
- 策略内部 $\mathbf u$：Actor update 时由当前策略重新采样，历史值不需要；
- `command_velocity_world`：可由 body command 与当时 yaw 唯一恢复；
- `executed_velocity`：已经存在于 $S_{t+1}$ / next observation；
- `executed_acceleration`：无碰撞时由相邻速度和 $\Delta T$ 推导，有碰撞时结果同样体现在 next state；
- `speed_projection_active`、command error、jerk：以在线聚合指标或低频 debug trace 保存，不占每条正式 replay；
- `reward_basis`：只有确实要跨 reward 版本重算时才保存，而且使用固定维 tensor，不保存每条 Python dict。

逻辑 schema 中的 current/next 都列出，是为了定义 Bellman tuple；物理存储可让同一 episode 内 $S_{t+1}$ 引用下一条的 $S_t$，只在 terminal/chunk 边界补存，从而减少重复。

### 5.3 统一 joint ring，而不是硬还原 $64/16/32/16$

新的 CTDE 推荐：

```text
一个 joint-transition ring store
若干轻量 semantic/event index（只存 transition_id）
每条 sampled transition 对所有 active pursuer 计算 masked loss
```

一次 Critic 前向输出 $[Q_1,\ldots,Q_N]$；同一 pre-capture transition 中的 direct、support 和普通 coverage pursuer 同时参加训练：

\[
L_Q
=
\frac{
\sum_{b,i}M_{b,i}(Q_i-y_i)^2
}{
\sum_{b,i}M_{b,i}
}
\]

不再为每条 joint transition 只抽一个 `(transition_id, focal_agent_id)`，也不再建立四个物理 joint buffers。否则既浪费 joint context，又会让 12-agent transition 因人工 focal 抽样被重复读取。

旧 $64/16/32/16$ 的价值在于提醒我们：长 pre-capture 轨迹可能淹没真实 post-capture 和 recovery coverage。但这四个数来自逐-agent IQN，转换到 joint transition 后不再等价：一条 transition 本身可以同时含 direct、support 和 coverage 三种角色。推荐使用可调混合采样：

\[
P_{sample}
=
\lambda_uP_{uniform}
+\lambda_rP_{regime}
+\lambda_eP_{event}
\]

首个 smoke 起点可用：

\[
(\lambda_u,\lambda_r,\lambda_e)
=
(0.70,0.20,0.10)
\]

- $P_{uniform}$：整个 joint ring 均匀采样，维持真实 visitation distribution；
- $P_{regime}$：在 `active_target` 与 `coverage_only` 间平衡，并为 coverage-only 设置例如 25% 的 batch 下限；
- $P_{event}$：从首次发现、final capture 前后窗口、collision、CE strict success/near-success 等稀有事件索引采样。

实现上，每条 transition 入 ring 时派生 `regime` 和 `event_bits`，并把 transition_id 写入轻量索引：

```text
discovery   敌人首次进入队伍可见集合，或 agent 首次从 coverage 切换到 capture
capture     当前步发生 loose/stationary capture；同时登记前后 K 步窗口
collision   pursuer/evader/obstacle collision 或 boundary breach
CE-success  CE strict/settle success latch 首次翻转；near-success 可独立建池
```

每个 batch slot 按 70/20/10 选择 sampler；例如 batch=64 时约为 45/13/6 个 source slots。coverage-only 最低比例是额外的硬 floor，若最终不足 16 条，就从 uniform slots 中替换为 coverage-only transition，因此最终 source 数可以偏离名义 70/20/10。event index 只存 ID，不复制数据。抽到一个 joint transition 后仍对所有 active pursuer 做 masked loss。事件池为空时回退到 uniform/regime，并记录 fallback 次数。

$0.70/0.20/0.10$ 只是 smoke 初值，不是新算法教条。旧计数粗略折算为 active-target/coverage-only 的 $80/48=62.5\%/37.5\%$，可作为分布监控对照，但因为采样单位已经变化，不能要求数值严格复现。后续依据 replay occupancy、各 regime TD error、样本利用率和 screening 调整。

常见 off-policy CTDE（如 joint-action centralized critic）通常保存同步 joint tuple，并对所有 agents 并行更新。多任务情况下：

- state/action/reward 语义相同：共享 replay，按 task/regime 做 balanced sampling；
- reward 或 dynamics 真正不同：使用显式 task-conditioned Actor/Critic，或拆 replay/critic；
- 辅助预测任务的数据对象不同：可以另建 auxiliary buffer，但这不能推出围捕 phase 也应拆四池。

首版不启用 TD-error PER。若后续启用，必须记录每条样本的实际 sampling probability，并用 importance weight 校正；否则 sampler 会悄然改变优化目标。

### 5.4 `phase` 是采样元数据，不应成为隐式控制 oracle

把字符串写入 replay metadata 不会自动让 Actor 或 Critic“感知 phase”；只有显式作为网络输入，或通过采样分布，才会影响训练。当前 Actor self token 有 `is_pursuing`，没有 global phase oracle。

从控制语义看，`post_capture` 与 `pure_coverage` 都满足：

\[
\text{active target count}=0
\]

并执行相同 CE coverage 目标，因此新线默认把它们合并为：

```text
control_regime = coverage_only
```

两者只保留 `origin`：

- `post_capture`：从真实围捕结束队形进入 coverage；
- `recovery_pure`：从 recovery snapshot 或纯 coverage reset 分布开始。

`origin` 用于采样、分布诊断和 ablation，不输入 Actor；Critic 也优先读取真实 active-target mask、实体状态和真正因果的 timer/latch，而不读取“这个 episode 曾经有敌人”这一历史捷径。

需要注意：当前代码中 pure/post 的 CE 几何目标基本相同，但 reset 来源、capture-boundary PBRS reset、post-capture window 和 episode termination 仍可能不同。因此新线应优先统一二者的 coverage reward/horizon 合同。若某个 timer 或任务上下文确实改变最优动作，则它必须是部署时可得的因果状态，并同时进入 Actor/Critic；只把 `phase` 给 Critic 会保留 Actor 侧的 observation-action aliasing。

当前代码还有一个正确语义需要保留：transition 的 phase 描述**动作被选择时**的状态，所以捕获最后敌人的那条 transition 仍标作 pre-capture；下一条才进入 coverage-only。

### 5.5 terminated、truncated 与 phase boundary

当前 trainer 把多种结束原因压成 `done`，并通过字符串 `state == "all targets captured"` 把 capture 后 transition 改为可 bootstrap。重构后应显式定义：

- capture 完成但进入 post-capture coverage：不是 terminal，$b_i=1$；
- 真正任务成功/失败或 agent 因 collision 永久失活：按合同设置 `terminated_i`；
- episode time limit：`truncated=True`，通常允许 bootstrap；
- agent 在下一时刻被 mask：明确是否需要 posthumous team credit。首版 per-agent reward 下建议对永久失活 focal agent 停止 bootstrap。

不得继续依赖日志字符串修补 Bellman target。

### 5.6 旧 replay 和旧 checkpoint 的可用范围

旧离散 transition 属于不同 action semantics 和不同动力学，不能直接混入新的 critic replay。可用方式只有：

1. shape-compatible 初始化 token encoder；
2. 用旧 IQN 生成固定数据，按“旧动作执行 $0.5\,\mathrm s$ 后的实际速度”构造 teacher target，做短期 behavior distillation；
3. 仅作为离线表征预训练数据，不参与新 MDP 的 Bellman loss。

即使做 distillation，也要与 scratch 并行比较，防止旧 unicycle 表征限制横向速度学习。

### 5.7 checkpoint 必须升级

新 checkpoint 至少保存：

```text
actor
critic_1 / critic_2
target_critic_1 / target_critic_2
log_alpha
actor_optimizer
critic_optimizer(s)
alpha_optimizer
global_step / episode / RNG state
observation normalization state
action contract and dynamics metadata
algorithm/version/config hash
```

checkpoint 与 replay shard 分开保存，不能在每个模型 checkpoint 中重复打包几 GB replay。自动恢复课程若不加载兼容 replay，必须明确 warm-up 行为，不能恢复后立刻用空池训练。

### 5.8 持久化 replay：收益、代价、容量与跨训练合同

#### 5.8.1 优势与劣势

| 方面 | 优势 | 劣势/风险 |
|---|---|---|
| 异常恢复 | 重启后无需重新 warm-up，可继续利用 rare capture/CE 样本 | snapshot、校验和恢复会增加暂停和 I/O |
| 课程衔接 | 4v1→8v2→12v3 可少量混入旧规模，降低遗忘 | 旧规模占比过高会让 4v1 长期主导新阶段 |
| 实验复用 | 同一兼容数据可比较新网络、初始化和 sampler | stale behavior 过多会拖慢适应；off-policy 不等于任意旧数据都安全 |
| 稀有事件 | final capture、真实 post-capture、CE success 不会因 ring 很快覆盖而丢失 | 过度保留成功事件会扭曲真实 visitation distribution |
| 可审计性 | 可复现 stage-best 所见数据分布，并做离线诊断/relabel | schema/version 迁移、损坏和对象序列化复杂 |
| 存储 | 紧凑 array/shard 可流式读写 | Python `deque + dict + ndarray` 的对象开销和碎片会很大 |

跨 run replay 还会引入类似 offline RL 的 action-distribution extrapolation：旧数据占比过高时，Critic 可能主要拟合当前 Actor 很少访问的动作。建议先采集一批新数据，再把跨 run 旧样本限制在 batch 的约 20%--30%，并依据验证结果衰减，而不是从第一个 update 起只训练旧池。

#### 5.8.2 当前规模的存储估算

现有每 agent、每时刻局部 observation 上限有：

\[
9+8\times7+8\times7+5\times5
=146
\]

仅 12-agent current/next float32 Actor observation 就占：

\[
146\times12\times2\times4
=14016\ \mathrm{bytes}
\approx13.7\ \mathrm{KiB}
\]

加上 global critic state、masks、action、reward 和 termination，一个未压缩但紧凑 array joint transition 预计约 $16$--$20\,\mathrm{KiB}$：

| joint capacity | 紧凑 float32 原始数据估算 |
|---:|---:|
| 100k | 约 1.6--2.0 GB |
| 250k | 约 4--5 GB |
| 1M | 约 16--20 GB |

若使用嵌套 Python objects，实际 RAM 可能是原始 tensor 字节数的 1.5--3 倍。若 observation 用经过误差验证的 float16、mask/enum 紧凑编码、type 隐式生成、按 active agent ragged 存储，并复用相邻 current/next state，可降到约 $8$--$12\,\mathrm{KiB}$/transition；此时 250k 约 2--3 GB，1M 仍约 8--12 GB。

因此不能把旧 `replay_capacity=1_000_000` 直接解释成 1M joint decisions。首版建议：

- 用 10k 条真实 transition 序列化实测后外推，而不是只按公式猜；
- RAM hot ring 从 200k--300k joint decisions 起步，并按可用内存调整；
- stage boundary 可选写独立 replay snapshot；
- 稀有事件保存 20k--50k 的 ID/index archive，不复制 observation；
- replay shard 与 25k/50k 模型 checkpoint 分离。

#### 5.8.3 能否跨训练使用

可以，但必须保存 raw observation/state，不保存旧 encoder embedding，并通过严格 compatibility manifest。以下内容原则上必须一致：

```text
action units / body-vs-world frame / vmax / decision interval
feasibility-layer and low-level dynamics semantics
collision response
observation/yaw/FOV/boundary schema and normalization
reward / termination / bootstrap semantics
global critic state and entity-mask schema
```

允许不同的是 behavior-policy checkpoint、Actor/Critic 网络结构、optimizer，以及 schema 支持范围内的 4/8/12 team size。以下情况有条件复用：

- reward 改变：只有保存足够 `reward_basis` 并经过版本化 relabel 才能用；
- $a_{\max}$ 改变：默认视为不同 action contract、分 shard。若低层转移对同一 final $v^{cmd}$ 完全相同，可只复用满足新约束 $\|v^{cmd}-v_t\|\le a_{\max}^{new}\Delta T$ 的旧 transition，但必须有显式过滤测试；
- boundary/observation 改版：必须有确定性 converter 与逐字段契约测试，不能靠缺省填零；
- dynamics、collision 或 termination 改变：不能直接做同一 Bellman MDP 的 replay。

每个 replay shard 建议保存：

```text
schema_version
code_commit
action_contract_hash
dynamics_hash
observation_hash
reward_hash
termination_hash
normalization_state
stage/team-size metadata
checksum
```

加载器默认 strict reject，不做“尽量加载”。持久化模式至少分为：`none`、同 run 精确恢复的 `resume`、只保留稀有事件窗口的 `event_archive`，以及明确用于离线研究的 `full_dataset`；主线先实现 `resume`，跨 run 复用作为独立消融。

---

## 6. 对现有 reward、评测与工具链的影响

### 6.1 CR-MS 与 support reward

ring-MS 本体主要依赖前后 pursuer/evader 位置与 evader 速度，support approach-only 依赖距离进度，因此对连续动力学原则上兼容。首版不改其权重和定义。

真正需要改的是动作诊断：当前 `_action_accel_turn` 会把 integer action 反解成 `(a,\omega)`；连续 action 若继续走该函数，异常分支可能静默返回 `(0,0)`，导致 motion penalty/日志看似正常但实际错误。新分支必须直接记录真实：

\[
\|\mathbf v\|_2,
\qquad
\|\mathbf a_{exec}\|_2,
\qquad
\|\mathbf j\|_2,
\qquad
\text{speed-projection rate / command tracking error}
\]

### 6.2 CE control cost

当前 CE control cost 使用标量 speed、纵向 acceleration、angular velocity。连续模式改为：

\[
c_u
=
\lambda_v
\left(
\frac{\|\mathbf v\|_2}{v_{\max}}
\right)^2
+
\lambda_a
\left(
\frac{\|\mathbf a_{exec}\|_2}{a_{\max}}
\right)^2
\]

首版 `yaw=hold` 没有 yaw action cost；未来启用 bounded tracker 时，如确需平滑传感器转动，再独立增加小权重。加速度项必须使用实际执行加速度，而不是可行速度层投影前的候选速度差。

当前 CR-MS 配置的 acceleration/angular 权重本来为 0，speed 权重在 200k 后仅为 0.0005，因此首轮可以保持相同 schedule，先验证中心能量和 strict success；不要趁重构同时大幅增强能耗惩罚。

### 6.3 stationary 与 strict success

连续模式必须始终维护：

```text
robot.velocity = [vx_world, vy_world]
robot.speed    = norm(robot.velocity)
robot.theta    = body/sensor yaw state
trajectory     = [x, y, theta, speed, vx, vy]
```

否则 CE strict、settle、episode-end speed、GIF 和 ring-MS velocity weighting 会读到陈旧字段。CE strict 仍排在 $CV\le0.15$ 诊断之前，不因架构改变 success semantics。

### 6.4 screening 与 GIF

以下工具当前都隐含 integer action，需要按 checkpoint metadata dispatch：

- `tools/rollout_voradj_visual.py`
- `tools/evaluate_checkpoints.py`
- `tools/evaluate_ce_coverage.py`
- `tools/batch_rollouts.py`
- `tools/batch_rollouts_parallel.py`

连续模型 deterministic evaluation 使用 $\operatorname{rsquash}(\mu)$ 后再经过可行速度层，不采样；它是 latent mean 的确定性变换，不严格等于动作分布期望。正式 20-rollout/10-GIF 仍沿用固定 seed 列表和现有视觉配置，尾迹默认关闭。screening 输出必须新增 `algorithm`, `action_mode`, `a_max`, `yaw_mode`, `actor_deterministic` 字段，防止新旧 checkpoint 混评。

---

## 7. 风险预测与验收办法

| 风险 | 预期症状 | 根因 | 首要检查/处置 |
|---|---|---|---|
| robot-frame obs 配 world-frame action | 同一几何在不同朝向表现完全不同 | observation-action aliasing | 首版 body-frame；做旋转等变单测 |
| 逐轴速度 clip | 对角速度超过 3 | 方框不是速度圆盘 | 径向 squash；检查每子步范数 |
| 瞬时写入速度 | 反向等效加速度约 12 | 没有策略侧可行速度层 | 先生成满足 $a_{\max}$ 的 final command；env assert-only |
| $a_{\max}=0.4$ 过小 | 每周期可达 $\Delta v$ 太小、追击转向/停车尺度不合适 | 把旧纵向输入误当旧总加速度 | 扫描 0.4/0.8/1.6 |
| $a_{\max}$ 过大 | 碰撞增加、动作近瞬移 | 约束太弱 | 制动距离、反向/90°测试、collision |
| yaw/速度方向混用 | 低速 frame 翻转、局部 token 抖动 | 用瞬时地速方向定义 body | 当前全向感知首版 `yaw=hold`；未来扇形 FOV 再 bounded tracker |
| collision tunneling | 高速穿过障碍/边界 | 完成 10 子步后才刷新碰撞 | 新模式逐子步、同步检测 |
| speed projection 频繁 | latent entropy 浪费在相同/近似 final command | 速度圆盘边界的多对一投影 | 记录 projection rate/delta；调 $a_{\max}$、熵或改进可行域映射 |
| nearest boundary 跳变 | 场地中轴附近连续速度方向突跳 | hard nearest-wall argmin/tie-break | 首版记录；基础线后消融 four-wall half-space tokens |
| CE 末态抖动 | deterministic 还行但训练/随机 rollout 不停 | SAC entropy 与 stationary 目标冲突 | 分角色 entropy/speed；后期 entropy 调整 |
| 非 Markov 控制滤波 | 同一 obs/action 得到不同转移 | 额外 controller 依赖未观测 history | 首版可行层只依赖当前 velocity；若加 jerk filter，把历史命令入 obs |
| 敌人离开感知半径后失忆 | reacquisition 失败 | 单帧 Actor 状态混叠 | 先保留现状；证据成立后加 GRU + sequence replay |
| phase oracle/aliasing | 训练 Q 很好、部署 Actor 决策冲突 | 只给 Critic 历史 phase，或 pure/post 实际合同不同 | 合并 coverage-only；因果上下文若必要须同时给 Actor |
| 对称策略坍缩 | 多机给出完全相同方向并拥挤 | 参数共享 + 对称局部输入 | 对称场景测试；相对几何/role，不先加固定 agent ID |
| reward/entropy 随 N 放大 | 8v2/12v3 Q 和梯度爆增 | 对 active agent 求和 | 所有 loss 按 mask mean |
| Critic 信息泄漏 | 本地 rollout 依赖全局敌人 | actor/critic API 混用 | 独立输入类型；测试 Actor 不可访问 global state |
| joint replay 内存爆增 | batch 128×12 OOM、吞吐骤降 | 容量仍按逐 agent 1M 配置 | 容量按 joint steps 核算，batch 先 32/64 |
| 旧 replay 误用 | Q 无法收敛或学到错误动力学 | 离散 $(a,\omega)$ 与 desired velocity 非同一 MDP | 旧数据只做 encoder/BC，不做 Bellman |
| target 随机 | target Q 方差异常 | Critic dropout/train mode | critic dropout 0；target eval/no grad |
| 旧线被改坏 | 原 checkpoint rollout 漂移 | 没有 action-mode 隔离 | 双模式；固定 seed bit/metric regression |

还需额外关注两点。

第一，当前 `ConfigManager` 是全局 singleton。同一进程若未来并行 vector env 使用不同规模/配置，可能互相覆盖。首版用进程隔离并行环境；要做同进程 vectorization 时再移除动力学对象对全局配置的隐式依赖。

第二，central critic 对 agent set 的结构兼容不等于性能自动跨规模。4v1、8v2、12v3 仍需课程和混合规模训练；mask 只是让模型“能算”，不是保证“会泛化”。

---

## 8. 可执行的重构实验 TODO

### P0：冻结可对照基线

目标：在任何代码改动前，为归因建立不可变参照。

- [ ] 记录当前 CR-MS 4v1、8v2、12v3 stage-best checkpoint、effective config、筛选 JSON 和正式 20-rollout/10-GIF 路径。
- [ ] 固定 capture、pure coverage、mix 三套 seed 列表。
- [ ] 固定主指标优先级：capture/mix capture、CE strict、collision、capture steps；$CV\le0.15$ 保留为次级诊断。
- [ ] 保存旧动作模式下固定 seed 的 trajectory/summary 哈希或数值快照。
- [ ] 明确新线独立 run/config/artifact namespace，不覆盖当前 solid 结果。

完成门槛：旧线基线可一条命令复现；后续任何环境改动都能自动发现回归。

### P1：双动作模式与动力学 contract

目标：只证明 $(v_x^B,v_y^B)$ 是一个正确、可控、可评测的环境动作，不训练 RL。

- [ ] 正式消费 `pursuer.action_mode`，保留 `unicycle_discrete`，新增 `velocity_2d_body`。
- [ ] 新增 `action_spec`：shape、dtype、normalized disk、物理速度和坐标系。
- [ ] 实现 Actor 后、env 前的可行速度层；final $v^{cmd}$ 同时满足 L2 speed/acceleration contract。
- [ ] 实现 decision-frame body→world 一次旋转；env 只 assert，并以 10 子步线性速度参考和梯形位置积分执行。
- [ ] 当前全向感知首版实现 `yaw.mode=hold`；另为未来 bounded tracker 写零速/迟滞契约测试。
- [ ] Pursuer 连续、evader APF 离散动作在同一个 `env.step` 共存。
- [ ] 新模式逐子步做边界/障碍/机器人碰撞检查，避免 tunneling；旧模式路径不动。
- [ ] 同步维护 `velocity/speed/theta/trajectory`。
- [ ] 改造 CE vector control cost 和连续 motion diagnostics。

必做单测：

- [ ] action shape `(2,)`，NaN/Inf/越界拒绝；
- [ ] 对角命令、随机命令每子步均满足 $\lVert\mathbf v\rVert_2\le3$；
- [ ] policy-side final command 满足 $\|v^{cmd}-v_t\|/\Delta T\le a_{\max}$，环境不发生静默修正；
- [ ] 每个线性参考子步满足 $\lVert\Delta\mathbf v\rVert_2/\delta t\le a_{\max}$；
- [ ] 零速起步、满速制动、满速反向、满速 90° 转向；
- [ ] body→world 旋转等变；
- [ ] `speed == norm(velocity)`；
- [ ] zero command 能停车；
- [ ] `yaw=hold` 全 episode 不漂移、零速不出现 NaN；未来 tracker 的 heading rate/迟滞另测；
- [ ] pursuer 连续 + evader 离散混合接口；
- [ ] 子步障碍/边界碰撞无穿透；
- [ ] 旧 CR-MS 固定 seed 结果与改动前一致。

### P2：$a_{\max}$ 动力学 screening

目标：在训练噪声进入前排除明显不合理的机动包络。

对：

\[
a_{\max}\in\{0.4,0.8,1.6\}\ \mathrm{m/s^2}
\]

用 scripted/oracle velocity controller 运行固定场景：

- [ ] CE centroid step response：到达时间、最大过冲、进入 $0.15/0.30\,\mathrm{m/s}$ 停稳阈值时间；
- [ ] 追击转向：90°、180° 命令切换；
- [ ] ring 接近：捕获半径附近制动和切向绕行；
- [ ] 障碍与边界：最短制动距离、碰撞率；
- [ ] 记录 speed-projection rate、projection delta、实际 acceleration、jerk、路径长度。

选择原则：拒绝每周期可达速度变化过小、无法在 CE/围捕尺度内转向或停车的候选，也拒绝碰撞显著增加、接近瞬时速度控制的候选；speed projection 长期过高同样不通过。预期 $0.8$ 或 $1.6$ 比 $0.4$ 更合理，但在数据出来前不写死。

### P3：新增兼容 TokenEncoder 与连续 Actor

目标：保留现有前端能力，同时清除 IQN 专属更新链。

- [ ] 新增 `LocalEntityTokenEncoder`，复制当前 `CoCapIQN` 前端的功能合同并提供旧 IQN encoder key mapping；首版不改 `CoCapIQN` 源路径。
- [ ] 用固定输入证明新 encoder 在加载映射权重后与旧 IQN 前端表示一致，并再次运行旧 IQN 固定 seed regression。
- [ ] 新增 radial-squashed Gaussian Actor、stable log-prob、$\operatorname{rsquash}(\mu)$ deterministic action 和可行速度层。
- [ ] Actor/encoder dropout 显式配置，连续首版为 0。
- [ ] 增加 actor-only checkpoint load/save 和旧 IQN encoder key mapping。
- [ ] 测试 4v1/8v2/12v3 shape、padding invariance、实体 permutation、旋转等变、边界 log-prob/gradient。
- [ ] 测试 Actor rollout 的类型接口不可能读取 global critic state。

建议并行准备两个初始化：

```text
A: all scratch
B: strongest CR-MS IQN token encoder init，Actor head scratch
```

不把旧 Q head、optimizer 或 replay 带入。

### P4：Joint replay 与 local shared SAC smoke

目标：先把连续 off-policy 数据链跑通，再引入 centralized critic 的复杂度。P1 已实现独立、确定性的 feasible velocity adapter；本阶段 Actor 只调用它，不再复制第二套约束逻辑。

- [ ] 实现一个 joint ring store + 只含 `transition_id` 的 regime/event indexes；ring 覆盖时原子清理索引。
- [ ] 按 5.2 精简合同只存 final feasible command、state/mask/reward/bootstrap 与固定维 metadata；在线 diagnostics 不塞入每条 replay。
- [ ] 实现 `critic_mode=local` 的 shared SAC 诊断模式，Critic 只看 $o_i,a_i^{cmd}$，但使用同一 joint replay schema。
- [ ] 每个 sampled joint transition 对全部 active agents 做 masked loss；实现可调 uniform/regime/event mixture，并验证 coverage/event floor。
- [ ] 以 10k 条真实数据测量 bytes/transition；实现 replay `resume` snapshot、strict manifest 和空池 warm-up。
- [ ] batch 先从 32 或 64 开始，实测 GPU memory、updates/s，再决定是否回到 128。
- [ ] checkpoint 保存 actor/critics/targets/alpha/optimizers/RNG/config contract。

smoke 顺序：

1. pure CE coverage 4-agent；
2. independent capture 4v1；
3. current mixed CR-MS 4v1。

每条先跑单 seed 10k/25k 数值 smoke，再做 100k–200k screening。通过条件不是追求正式性能，而是：无 NaN/Inf、loss/Q/alpha 有界、采样 mixture/floor 符合配置、动作不长期贴边或频繁 speed projection、确定性策略明显优于 random velocity，并能在至少一个单任务上形成学习趋势。

local SAC 只是隔离诊断，不是最终架构。若它失败，先修动作/数据链；不要用 centralized critic 掩盖基础 bug。

### P5：实现 CoCap-PSA-MASAC

目标：加入真正 CTDE 的 masked centralized twin critics。

- [ ] 构建 training-only global entity set，不改变 Actor local observation；不把历史 `phase/origin` 当 oracle 输入。
- [ ] 实现两个从 global encoder 到 head 都完全独立的 attention critics；final feasible joint command 绑定 pursuer token，不复用 Actor learned $h_i$。
- [ ] 一次前向输出所有 focal $Q_i$，padded/失活位置不参与 loss。
- [ ] 实现 soft target、focal actor loss、自动温度和 Polyak update。
- [ ] 所有 reward/Q/entropy/gradient aggregation 对 active mask 求均值。
- [ ] target critics 无梯度、无 dropout；deterministic evaluation 完全可复现。
- [ ] 记录 twin-Q gap、TD error、actor grad、alpha、entropy、speed projection，以及按 control-regime/role/origin 分组的统计。

契约测试：

- [ ] agent permutation 后 $Q_i$ 同步 permutation；
- [ ] padded agent/action 对有效 $Q_i$ 无影响；
- [ ] 4/8/12 joint transition 可混 batch；
- [ ] other-agent action 改变时 focal Q 可改变；
- [ ] Actor 更新时其他 agent action 分支 stop-gradient；
- [ ] critic target 与 bootstrap mask 对 capture→post、collision、timeout 语义正确；
- [ ] 一次完整 update 后 actor、twin critics、alpha 都有有限梯度与参数变化。

### P6：4v1 预筛与正式训练

先做低成本因子预筛，不直接把所有组合跑满 2M。

建议第一轮：

| 因子 | 候选 |
|---|---|
| algorithm | local SAC diagnostic / PSA-MASAC |
| $a_{\max}$ | P2 选出的前 2 个 |
| init | scratch / IQN encoder transfer |
| seed | 先 1 个工程 seed，晋级后至少 3 seeds |

- [ ] 25k 数值 smoke；
- [ ] 每 50k 或 100k 同步做 capture/coverage/mix screening；
- [ ] 200k–300k 时淘汰明显失败组合；
- [ ] 晋级组合按当前 4v1 正式预算训练至 2M；
- [ ] 选择历史最优 checkpoint，而不是只取 final；
- [ ] 最优 checkpoint 自动做正式 20-rollout/10-GIF，尾迹默认关闭。

晋级门槛应预注册成相对当前 IQN CR-MS 基线的 non-inferiority，而不是训练后改口径。建议初始容差：

- capture 与 mix-capture 不低于对应 IQN baseline 5 个百分点以上；
- CE strict 不低于 baseline 5 个百分点以上；
- collision 不高于 baseline 2 个百分点以上；
- 同时报告 capture steps、stationary time、path length 和 action smoothness。

若连续架构在任务成功率持平但路径/平滑明显改善，可保留；若只因新动力学变得更灵活而成功率上升，必须通过 P8 的动作公平性消融确认来源。

### P7：8v2、12v3 课程与混合规模训练

- [ ] 用 4v1 历史最优 Actor/Critics warm-start 8v2；Critic entity/action set 只扩 mask，不换固定 concat head。
- [ ] 8v2 沿用当前 700k 预算和每 100k capture/coverage/mix screening；以历史最优晋级。
- [ ] 12v3 沿用当前 700k 预算和每 100k screening。
- [ ] 每阶段最优自动正式 20-rollout/10-GIF。
- [ ] 增加 mixed-size replay 试验：训练 batch 中混合 4/8/12 transition，检查 loss/entropy 是否保持规模不变。
- [ ] 增加 agent failure/mask 测试：随机失活 1–2 个 pursuer，不重启网络。

变规模报告至少包含：

```text
seen size: 4, 8, 12
cross-size zero-shot matrix
curriculum fine-tuned matrix
agent-failure robustness
GPU memory / steps per second by N
```

### P8：必要的归因消融

至少做以下四组，避免把“动作更灵活”“算法更强”和“Critic 更强”混成一个结论：

1. **动作公平性**：同一新速度动力学下，离散速度方向/幅值集合 vs 连续速度；
2. **CTDE 贡献**：local shared SAC vs PSA-MASAC；
3. **随机 Actor 贡献**：Set-Attention MATD3 vs PSA-MASAC；
4. **迁移贡献**：scratch vs IQN encoder transfer。

可选后续：

- focal per-agent reward vs 小比例 team-mean reward；
- shared $\alpha$ vs coverage 低熵 schedule；
- scalar soft-Q vs TQC；
- single-frame Actor vs GRU sequence Actor；
- radial Gaussian vs normalizing-flow Actor；
- current nearest-boundary vector vs four-wall half-space boundary token set；
- monolithic/nonmonotonic FACMAC critic。

不要在主线 solid 前同时加入 GRU、flow、intrinsic curiosity、复杂 PER 和 TQC。

### P9：正式评测与文档闭环

- [ ] 三套 deterministic screening 全程开启，结果按 checkpoint 留历史表；
- [ ] 主排序优先 capture/mix capture 和 CE strict，$CV\le0.15$ 放其后；
- [ ] 每个大阶段最优自动跑 20-rollout/10-GIF；固定种子、路径和场景；
- [ ] GIF 默认关闭 faded trail，除非显式参数开启；
- [ ] 正式对比中同时放 IQN baseline、PSA-MASAC、Set-Attention MATD3；
- [ ] 报告三 seed 均值、标准差/置信区间，而不是只报最好 seed；
- [ ] 记录完整 config、代码版本、action contract、checkpoint hash 和 artifact manifest；
- [ ] 更新项目 handoff、方法文档和后续 TODO。

---

## 9. 建议的代码边界

建议新增而不是把旧 IQN 训练器改成大量条件分支：

```text
src/cocap_voradj/models/entity_token_encoder.py
src/cocap_voradj/models/continuous_actor.py
src/cocap_voradj/models/central_attention_critic.py
src/cocap_voradj/models/masac.py
src/cocap_voradj/training/joint_replay.py
src/cocap_voradj/training/masac_trainer.py
```

需要 action-mode gated 修改：

```text
src/cocap_voradj/dynamics/robot.py
src/cocap_voradj/dynamics/pursuer.py
src/cocap_voradj/envs/base.py
src/cocap_voradj/envs/voronoi_adjacency.py
src/cocap_voradj/envs/coverage_ce.py
```

需要 checkpoint dispatch 修改：

```text
tools/rollout_voradj_visual.py
tools/evaluate_checkpoints.py
tools/evaluate_ce_coverage.py
tools/batch_rollouts.py
tools/batch_rollouts_parallel.py
```

旧 `iqn.py`、旧 trainer 和旧 replay 在新线 solid 前保留，作为可运行 baseline。等新旧模式都有完整 regression 后，再考虑抽公共接口，不能先大规模删除旧路径。

建议配置骨架：

```yaml
algorithm:
  name: cocap_psa_masac
  critic_mode: central_attention_focal

pursuer:
  action_mode: acceleration_2d_body
  max_speed: 3.0
  max_acceleration: 0.8
  velocity_substeps: 10

continuous_action:
  action_constraint: policy_acceleration_disk
  env_constraint_mode: assert_only
  action_replay_semantics: body_acceleration

yaw:
  mode: hold

masac:
  action_dim: 2
  action_distribution: radial_squashed_gaussian_acceleration
  hidden_dim: 256
  actor_transformer_layers: 4
  actor_attention_heads: 8
  actor_dropout: 0.0
  critic_transformer_layers: 3
  critic_attention_heads: 8
  critic_dropout: 0.0
  gamma: 0.99
  tau: 0.005
  actor_lr: 0.0001
  critic_lr: 0.0001
  alpha_lr: 0.0001
  target_entropy: -2.0
  batch_size: 64
  warmup_joint_steps: 5000
  update_every_env_steps: 1
  gradient_steps: 1
  grad_clip: 10.0

replay:
  capacity_joint_steps: 250000
  sample_uniform_weight: 0.70
  sample_regime_weight: 0.20
  sample_event_weight: 0.10
  coverage_only_min_fraction: 0.25
  persistence_mode: resume
  strict_manifest: true
```

这些学习率和 batch 只是 smoke 起点，不是未经实验即可宣布的最终超参数。

---

## 10. 观测、训练信息与部署信息边界

必须建立自动化信息泄漏测试。

Actor 可使用：

- 当前 self token；
- VCT-LS 允许的友军 token；
- 直接有限半径内的 enemy token；
- 有限感知 obstacle token；
- CE local centroid/boundary feature；
- `is_pursuing` 与当前已有 one-hop support signal。

Actor 不可使用：

- 未被本机/协议观测的真实敌人位置；
- 全体 agent joint action；
- global capture phase oracle；
- centralized critic embedding；
- 训练期 ground-truth map tensor。

Critic 训练时可使用：

- 全局真实 pursuer/evader/obstacle state；
- 全体 final feasible command 与 active/target mask；
- 当前真正因果的 dynamics/task context，以及必要的 timer/termination state；
- 当前 reward components，仅作为监督/诊断，不应把未来信息输入当前 Q。

Critic 不应把 `post_capture`、`recovery_pure` 等历史来源标签当作 privileged shortcut。`origin/phase/event_bits` 默认只进入 sampler 和日志；若某个上下文确实改变最优动作，则必须先证明它部署可得，并同时给 Actor。

部署导出只包含 `EntityTokenEncoder + Actor head + action adapter metadata`；不打包 centralized critics。

---

## 11. 最终建议

本线的正确故事不是“把 IQN 换成另一个常见算法”，而是：

1. 用机体系二维期望速度让控制更贴近高层无人机速度接口；
2. 用 Actor 后的 $v_{\max}/a_{\max}$ 可行速度层生成物理可达命令，环境只校验并执行满足约束的子步参考；
3. 保留已经证明有效的局部 token Transformer、CR-MS、CE、VCT-LS support，以及“稀有语义不被饿死”的 replay 设计目标；具体实现更新为统一 joint ring 与可调 regime/event 分层；
4. 将离散分布式 Q-learning 替换为共享随机 Actor、集中式实体注意力 twin critics 和 joint replay，使有限视域、多机协同和变规模训练在同一个 CTDE 框架内成立；
5. 通过 local SAC → PSA-MASAC → 8v2/12v3 的分层实验，把动作合同、off-policy 数据链和 centralized credit assignment 分开验证。

主线推荐顺序为：

```text
动作/动力学 contract
→ TokenEncoder + radial Actor
→ joint replay + local SAC smoke
→ focal attention MASAC 4v1
→ 8v2 / 12v3 课程
→ MATD3 与动作公平性消融
→ TQC / memory / flow 等后续增强
```

在 P2 数据出来之前，不应声称 $a_{\max}=0.4$ 过大；从旧动力学包络、制动距离和决策周期看，更值得警惕的是它作为二维总加速度时过于保守。`0.8` 是合理起点，`0.4/0.8/1.6` 的短 screening 是必要实验，而不是可省略的调参细节。

---

## 12. 文献依据

### 12.1 本地文献库

本地综述入口：[`RL_PE_Encirclement_网络与动作空间梳理.md`](../../TERL/reference/baseline文献_2026-07-28/00_report/RL_PE_Encirclement_网络与动作空间梳理.md)。

重点文献：

1. **NAGC, 2025**：有限视域多 UAV 追捕；continuous SAC、twin Q、replay、GAT centralized critic。动作是 $[\omega,a]$，不是本项目的 $[v_x,v_y]$。见 `01_local_selected_baselines/2025_IEEE-CAA-JAS_limited-visual-field_multi-uav-pursuit_NAGC_Peng.pdf`，重点 pp.1352、1354–1359、1362–1363。
2. **ViPER, CoRL 2025**：共享图 Actor、训练期特权 attentive critic、off-policy SAC/replay 和 agent failure；动作本身仍为离散图节点。见 `02_new_arxiv_external/2025_CoRL_ViPER_visibility-based-pursuit-evasion_graph-attention-MARL_Wang.pdf`，重点 §4 与 Appendix B/C。
3. **CEL-MADDPG, 2024**：连续二维力/加速度、$v_{\max},a_{\max}$、课程与经验筛选；但固定 3-agent concat critic。见 `03_encirclement_focused/2024_ESWA_multi-uav-roundup_CEL-MADDPG_curriculum-experience-learning_Li.pdf`，重点 §3.1、§3.3–§3.4、§4.3。
4. **CBC-TP Net, 2023**：期望平面加速度、CTDE、变长 BiLSTM 与双 replay；支持“连续控制 + 变数量 + off-policy”的组合可行性。见 `01_local_selected_baselines/2023_IEEE-TNNLS_game-of-drones_multi-uav-pe-drl_Zhang.pdf`，重点 pp.7902–7905、7908。
5. **De Souza et al., 2021**：TD3、共享策略/经验、局部观测和课程，支持 MATD3 作为备选。见 `02_new_arxiv_external/2021_IEEE-RAL-ICRA_decentralized-pursuit_deep-rl_unicycle-drones_DeSouza.pdf`。
6. **GNN coverage, ICRA 2022**：二维速度控制与变规模共享图策略的结构证据，但属于 imitation learning 且无 $a_{\max}$。见 `01_local_selected_baselines/2022_ICRA_coverage-control_GNN-voronoi-lloyd_Gosrich.pdf`。
7. **MAPPO, NeurIPS 2022**：强 CTDE baseline，但 on-policy，且该文实验主体系不是本项目所需的长期 replay 连续围捕。见 `01_local_selected_baselines/2022_NeurIPS_MAPPO_cooperative-marl_Yu.pdf`。

文献库中名为 `2025_IROS_sensor-constrained_pursuit-evasion_decentralized-MARL_car-like.pdf` 的文件首页实际是 quantum physics-informed neural networks/PDE 论文，文件名与内容不符，**不得用作本次 PE/MARL 证据**。

### 12.2 一手公开资料

- [MADDPG：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- [TD3：Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html)
- [SAC：Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)
- [MAAC：Actor-Attention-Critic for Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v97/iqbal19a.html)
- [FACMAC：Factored Multi-Agent Centralised Policy Gradients](https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html)
- [MAPPO：The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)
- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)
- [Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
- [TQC：Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://proceedings.mlr.press/v119/kuznetsov20a.html)
