# CR-MS 4v1 方法浓缩版：从旧 Mix 到局部感知围捕—覆盖

更新时间：2026-08-03

## 1. 核心思路

CR-MS 面向带障碍的局部感知多机器人任务：4 个 pursuer 搜索并围捕 1 个 evader，随后恢复空间覆盖。相比旧 mix，新方法不再依赖多项 capture reward 和复合 coverage 判据的拼接，而是将问题拆为四层：

1. **VCT-LS**：用 pursuer-only Voronoi 图定义通信邻居，用局部传感器决定敌人与障碍是否可见；
2. **Ring-MS**：直接发现敌人的机器人追逐安全围捕环上的可占据目标，而不是追敌人中心；
3. **Support bridge**：未直接看见敌人、但邻居正在追捕的机器人获得轻量支援信号；
4. **CE**：搜索阶段与围捕后的 coverage 统一最小化 Voronoi 单元中心能量。

其总体逻辑是：**局部发现触发围捕，一阶通信触发支援，未参与追捕者维持覆盖，捕获后全体回到 CE coverage。**

## 2. 旧 Mix 与 CR-MS 的奖励对比

### 2.1 旧 Mix Capture

旧 capture dense reward 同时包含时间惩罚、直接吸引、旧 mean-shift 和独立前向奖励：

$$
r_{i,t}^{\mathrm{old\text{-}cap}}
=r_{\mathrm{time}}
+\omega_a\operatorname{clip}\left(d_{i,t}-d_{i,t+1},-c_d,c_d\right)
+\omega_m r_{i,t}^{\mathrm{old\text{-}MS}}
+\omega_f r_{i,t}^{\mathrm{front}}.
$$

其矛盾在于：approach 鼓励靠近敌人中心，MS 鼓励占据环形空位，front 又单独鼓励前置堵截，时间惩罚进一步推动短视接近。多个目标需要共同调权，且在局部感知下并非所有机器人都能计算这些项。

### 2.2 CR-MS 的角色化奖励

设机器人 $i$ 是否直接看见敌人为 $\delta_i$，是否为一阶支援者为 $s_i$。新方法使用

$$
r_{i,t}^{\mathrm{task}}
=
\begin{cases}
r_{i,t}^{\mathrm{Ring\text{-}MS}}, & \delta_i=1,\\
0.5r_{i,t}^{\mathrm{approach}}+0.5r_{i,t}^{\mathrm{CE}}, & \delta_i=0,\ s_i=1,\\
r_{i,t}^{\mathrm{CE}}, & \delta_i=0,\ s_i=0.
\end{cases}
$$

直接追捕者不再使用旧 approach、旧 MS、front 和每步时间惩罚；对应权重均为零。碰撞、安全、边界和 capture terminal 项继续保留。

## 3. VCT-LS：通信与感知解耦

在去除障碍膨胀区域后的自由空间 $\mathcal F$ 上，只使用 pursuer 构造 Voronoi 单元：

$$
\mathcal V_i
=\left\{x\in\mathcal F:\left\|x-p_i\right\|
\leq\left\|x-p_k\right\|,\ \forall k\right\}.
$$

两个单元共享边界时建立一阶通信边：

$$
(i,k)\in\mathcal E
\iff
\partial\mathcal V_i\cap\partial\mathcal V_k\neq\varnothing.
$$

evader 和 obstacle 不作为通信节点。敌人是否可见由 20 m 表面距离传感器决定：

$$
\delta_{ij}
=\mathbb I\!\left[
\left\|p_i-e_j\right\|-r_i-r_j\leq20
\right],
\qquad
\delta_i=\max_j\delta_{ij}.
$$

若机器人自己未发现敌人，但某个一阶邻居已发现，则

$$
s_i
=\mathbb I\!\left[
\delta_i=0
\ \land
\exists k\in\mathcal N_i:\delta_k=1
\right].
$$

支援机器人只能从 friend token 得知邻居的 `is_pursuing`，不会在 observation 中获得 enemy 坐标。K10 release delay 只平滑机器人自身直接感知的短暂丢失。

## 4. Ring-MS：从“追敌人”改为“追围捕位置”

### 4.1 紧凑围捕环

感知半径只负责触发奖励；真正的奖励候选环为

$$
R_{\mathrm{in}}
=d_{\mathrm{safe}}+m_{\mathrm{in}}+r_i+r_j
\approx6.74,
\qquad
R_{\mathrm{pref}}=8.0,
\qquad
R_{\mathrm{out}}=10.5.
$$

因此机器人直接看见 enemy 后，目标不是 enemy 中心，也不是 20 m 感知边界，而是约 8 m 的安全围捕环。

### 4.2 径向与速度方向权重

候选点 $c$ 的径向权重为

$$
w_r(c)
=\exp\left[
-\frac{\left(\left\|c-e_j\right\|-R_{\mathrm{pref}}\right)^2}
{2\sigma_r^2}
\right],
\qquad \sigma_r=2.0.
$$

速度门控和角度权重为

$$
g_v
=\operatorname{clip}\left(
\frac{\left\|v_j\right\|-0.30}{1.00-0.30},0,1
\right),
$$

$$
w_{\theta}(c)
=\operatorname{clip}\left(
1+0.25g_v
\frac{(c-e_j)^{\mathsf T}v_j}
{\left\|c-e_j\right\|\left\|v_j\right\|},
0.75,1.25
\right).
$$

敌人静止时 $g_v=0$，环上方向均匀；敌人运动时，前方略微增权、后方略微减权。候选点还需通过地图、障碍和队友占用过滤。记有效性掩码为 $w_{\mathrm{valid}}(c)$，则

$$
w(c)=w_r(c)w_{\theta}(c)w_{\mathrm{valid}}(c).
$$

### 4.3 围捕目标与进度奖励

先求候选点加权平均方向，再投影回偏好半径：

$$
\bar c=\frac{\sum_c w(c)c}{\sum_c w(c)},
\qquad
t_i=e_j+R_{\mathrm{pref}}
\frac{\bar c-e_j}{\left\|\bar c-e_j\right\|}.
$$

使用动作前状态固定目标 $t_i(t)$，奖励当前动作带来的距离进展：

$$
r_{i,t}^{\mathrm{Ring\text{-}MS}}
=2\operatorname{clip}\left(
\left\|p_i(t)-t_i(t)\right\|
-\left\|p_i(t+1)-t_i(t)\right\|,
-3,3
\right).
$$

这样，旧 mean-shift 的空位选择与 front reward 的前置意图被统一进一个可解释目标中，同时避免 approach 把机器人推向敌人中心。

## 5. Support Bridge：只给“靠近事件”的训练信用

支援机器人看不到 enemy，无法从 observation 重构完整围捕环，所以不直接使用 Ring-MS。其 capture 部分只保留距离吸引：

$$
r_{i,t}^{\mathrm{approach}}
=\operatorname{clip}\left(
\left\|p_i(t)-e_{j^*}(t)\right\|
-\left\|p_i(t+1)-e_{j^*}(t+1)\right\|,
-3,3
\right).
$$

$j^*$ 仅从正在追捕的一阶邻居直接看见的 evader 集合中选择。该位置只用于训练 reward，不加入支援机器人的 observation。其作用是让机器人先学会靠近追捕事件，提高进入直接感知域的概率；真正发现敌人后再切换到 Ring-MS。保留 0.5 CE 则防止支援行为完全退化为盲目聚集。

## 6. CE：搜索和捕获后的统一覆盖目标

对机器人 $i$ 的可达 Voronoi 单元，计算包含自身位置的连通分量质心 $c_i$；若质心不可达，则投影到最近可达网格点。归一化中心误差为

$$
d_i=\frac{\sqrt N}{D}\left\|p_i-c_i\right\|.
$$

CE reward 使用中心平方能量、轻量控制代价和 PBRS：

$$
r_{i,t}^{\mathrm{CE}}
=10\left[
-d_i^2(t+1)-J_i^{\mathrm{ctrl}}(t)
+\gamma\Phi_i(s_{t+1})-\Phi_i(s_t)
\right],
\qquad
\Phi_i(s)=-d_i^2.
$$

前 200k 步速度权重为零，之后仅使用 $\lambda_v=5\times10^{-4}$；加速度和角速度权重为零。

CE strict 的成功条件为

$$
E_{\mathrm{RMS}}
=\sqrt{\frac{1}{N}\sum_i d_i^2}\leq0.05,
\qquad
E_{\max}=\max_i d_i\leq0.10,
$$

并连续保持 30 步。面积均衡单独记录为

$$
\operatorname{CV}_A
=\frac{\operatorname{Std}(A_1,\ldots,A_N)}
{\operatorname{Mean}(A_1,\ldots,A_N)},
$$

其中 `area-CV≤0.15` 只是诊断指标，不进入 CE reward，也不等同于 CE strict。

## 7. 4v1 训练与结果

4v1 从零训练 2M steps。每个更新 batch 使用 pursuing、pre-capture、post-capture、recovery 四类样本：

$$
64:16:32:16.
$$

recovery 中 75% 初态来自真实 capture snapshot，使 CE 反复学习从围捕后的紧凑、偏置队形恢复覆盖。每 100k 进行固定 seed、固定 IQN 分位中点的 deterministic screening，最终从 20 个 checkpoint 中选择历史最优。

最终选中 `step_2000000`。正式 capture、coverage、mix 各 20 rollout 的结果为：

| 场景 | Capture | CE strict | Area-CV≤0.15 | Collision | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| capture | 100% | — | — | 0% | 77.15 |
| coverage | — | 100% | 100% | 0% | 132.95 |
| mix | 100% | 90% | 95% | 5% | 227.45 |

这说明当前组合方法已经在 4v1 中同时形成稳定围捕、CE 中心收敛和较高概率的面积均衡。由于成功模型同时包含 VCT-LS、Ring-MS、support bridge 和 CE，该结果证明的是**整体组合成立**；各组件的独立因果贡献仍需受控 ablation。

## 8. 后续课程

课程自动执行

$$
4\mathrm{v}1\ (2\mathrm{M})
\longrightarrow
8\mathrm{v}2\ (700\mathrm{k})
\longrightarrow
12\mathrm{v}3\ (700\mathrm{k}).
$$

4v1 最优 checkpoint 已注入 8v2。后续阶段保持相同方法语义，只扩大 pursuer、evader 和 obstacle 数量，并将 post-capture window 调整为 600/700 步。每阶段每 100k screening，训满后自动选择历史最优、运行 20-rollout/10-GIF，并把 8v2 最优自动注入 12v3。

后续重点验证局部发现事件在大规模友方图上的传播、多 evader 下的追捕分工，以及更复杂 capture snapshot 下 CE 的覆盖恢复能力。
