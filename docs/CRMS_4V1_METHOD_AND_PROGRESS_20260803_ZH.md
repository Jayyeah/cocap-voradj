# CR-MS 4v1：局部感知下的围捕—覆盖一体化方法与当前进展

更新时间：2026-08-03

## 1. 方法定位

CR-MS 4v1 研究的是一个带障碍的多机器人围捕—覆盖连续任务：4 个 pursuer 在局部感知条件下搜索 1 个 evader；部分机器人发现目标后完成协同围捕，随后全体重新形成对自由空间的有效覆盖。

这条线不是简单地把 CE、VCT-LS 和新 MS 三个模块并列叠加，而是把旧 mix 中混杂的三个问题重新拆开：

1. VCT-LS 规定信息边界：机器人能感知什么、能和哪些队友通信；
2. 新 CR-MS 规定直接发现目标的机器人应该向哪里运动；
3. support reward 解决尚未直接看见目标、但知道邻居正在追捕的机器人如何获得有效学习信号；
4. CE 规定搜索阶段及围捕完成后的覆盖目标，并给出独立于旧面积综合门槛的中心收敛定义。

因此，该项目的核心故事可以概括为：在不向策略泄露全局敌人位置的前提下，用友方 Voronoi 图组织局部通信，用速度加权的紧凑围捕环指导直接追捕者，用轻量的支援吸引信号传播“向事件靠拢”的训练信用，再用 centroid-energy 统一搜索和围捕后的覆盖行为。

## 2. 旧 mix 的结构与主要矛盾

旧 mix 的 capture dense reward 主要由时间惩罚、直接吸引、旧 mean-shift 和前向奖励组成：

$$
r_{i,t}^{\mathrm{old\text{-}cap}}
=r_{\mathrm{time}}
+\omega_{a}\operatorname{clip}\left(d_{i,t}-d_{i,t+1},-c_d,c_d\right)
+\omega_{m}r_{i,t}^{\mathrm{old\text{-}MS}}
+\omega_{f}r_{i,t}^{\mathrm{front}}.
$$

这套结构能够学习围捕，但存在三个耦合问题。

第一，直接吸引倾向于缩短 pursuer 与 evader 的中心距离，而真正的围捕要求机器人停留在一个安全且可闭合的环上。靠近目标与保持围捕半径并不完全一致。

第二，旧 mean-shift 主要解决空位方向，速度前置又由单独的 front reward 补充。二者再与 approach 和每步时间惩罚共同调权，奖励解释和超参数耦合都较重。

第三，旧 observation/邻接关系没有严格区分通信和感知。对于有限感知的分布式任务，enemy 不应该因为参与几何图构造而隐式成为全局通信节点；否则策略在训练和部署时的信息语义不够干净。

旧 coverage 侧也把面积均衡、单元中心、边界内比例和静止等多个条件组合在一起。它能描述“形状是否均匀”，但容易把目标函数、终止条件和运动约束混成一个复合判据。CR-MS 因而不再把旧 coverage 作为默认目标，而改用 CE 中心能量，同时保留面积 CV 作为独立诊断。

## 3. VCT-LS：先明确分布式信息边界

### 3.1 友方 Voronoi 通信图

设时刻 $t$ 的有效 pursuer 集合为 $\mathcal P_t$，去除障碍膨胀区域后的自由空间为 $\mathcal F$。机器人 $i$ 的友方 Voronoi 单元定义为

$$
\mathcal V_i(t)
=\left\{x\in\mathcal F:\left\|x-p_i(t)\right\|
\leq\left\|x-p_k(t)\right\|,\ \forall k\in\mathcal P_t\right\}.
$$

若两个友方单元共享边界，则建立一阶通信边：

$$
(i,k)\in\mathcal E_t
\iff
\partial\mathcal V_i(t)\cap\partial\mathcal V_k(t)\neq\varnothing.
$$

动态图 $\mathcal G_t=(\mathcal P_t,\mathcal E_t)$ 只包含 pursuer。evader 和 obstacle 不作为通信节点；障碍只用于构造自由空间和局部 obstacle token。

### 3.2 敌人与障碍的局部感知

当前敌人和障碍感知半径均为 20 m，并按表面距离判断。对 pursuer $i$ 和 evader $j$，直接可见条件为

$$
\delta_{ij}(t)
=\mathbb I\!\left[
\left\|p_i(t)-e_j(t)\right\|-r_i-r_j
\leq R_{\mathrm{sense}}
\right],
\qquad R_{\mathrm{sense}}=20.
$$

定义机器人是否直接发现任一敌人为

$$
\delta_i(t)=\max_j\delta_{ij}(t).
$$

直接发现者的 self token 中 `is_pursuing=1`，并能获得对应 enemy token。没有直接发现目标的机器人不会获得敌人坐标。

VCT-LS 允许 friend token 携带邻居的 `is_pursuing`。因此一阶支援状态为

$$
s_i(t)
=\mathbb I\!\left[
\delta_i(t)=0
\ \land\ 
\exists k\in\mathcal N_i(t):\delta_k(t)=1
\right],
$$

其中 $\mathcal N_i(t)$ 是友方 Voronoi 一阶邻居集合。支援机器人知道“邻居正在追捕”，但仍不知道 enemy 的坐标。直接感知状态使用 K10 release delay 平滑短暂丢失，减少任务标签在感知边界附近抖动；该机制不会把邻居的状态伪装成自己的直接感知。

这一步的理论意义是把通信图和感知域解耦：图决定信息从谁传来，传感器决定哪些外部实体能进入 observation。后续所有奖励均建立在这个信息边界之上。

## 4. CE：把覆盖写成中心能量最小化

### 4.1 障碍感知的可达 Voronoi 中心

CE 在自由空间 $\mathcal F$ 上计算 pursuer-only Voronoi 分割。若障碍把某一单元切成多个连通分量，只保留包含机器人当前位置的连通分量 $\mathcal C_i$。其原始质心为

$$
\bar c_i
=\frac{1}{\mu(\mathcal C_i)}
\int_{\mathcal C_i}x\,\mathrm d x.
$$

如果 $\bar c_i$ 位于障碍膨胀区或不属于该连通分量，则将它投影到最近的可达网格点：

$$
c_i
=\Pi_{\mathcal C_i}(\bar c_i).
$$

这样得到的不是理想化多边形中心，而是考虑障碍与可达性的局部目标。

### 4.2 中心误差、控制代价与 PBRS

设参与 coverage 的有效机器人数量为 $N$，地图对角线长度为 $D$。归一化中心误差为

$$
d_i(t)
=\frac{\sqrt{N}}{D}
\left\|p_i(t)-c_i(t)\right\|,
$$

中心能量为

$$
J_i^{\mathrm{center}}(t)=d_i^2(t).
$$

$\sqrt N$ 缩放使不同机器人规模下的误差量级更可比较。一般形式的控制代价为

$$
J_i^{\mathrm{ctrl}}(t)
=\lambda_v\left(\frac{v_i}{v_{\max}}\right)^2
+\lambda_a\left(\frac{a_i}{a_{\max}}\right)^2
+\lambda_{\omega}\left(\frac{\omega_i}{\omega_{\max}}\right)^2.
$$

当前 CR-MS 采用延迟的轻量速度代价：训练前 200k 步 $\lambda_v=0$，之后 $\lambda_v=5\times10^{-4}$；$\lambda_a=\lambda_{\omega}=0$。这样先学习到达中心，再轻量抑制无效持续运动，避免从训练初期就因能耗项过早静止。

CE 使用势函数

$$
\Phi_i(s_t)=-\kappa d_i^2(t),
$$

其中 $\kappa=1$。单步 coverage reward 为

$$
r_{i,t}^{\mathrm{CE}}
=\eta\left[
-d_i^2(t+1)
-J_i^{\mathrm{ctrl}}(t)
+\gamma\Phi_i(s_{t+1})
-\Phi_i(s_t)
\right],
$$

当前缩放系数为 $\eta=10$。PBRS 项给出中心能量下降的稠密学习信号，同时保持目标解释集中在中心误差和轻量控制成本上。阶段切换或全体终止时使用相应 terminal correction，避免势函数残值污染不同 phase。

### 4.3 CE strict 与面积 CV 是两套指标

CE 的正式成功定义只检查中心误差：

$$
E_{\mathrm{RMS}}(t)
=\sqrt{\frac{1}{N}\sum_{i=1}^{N}d_i^2(t)},
$$

$$
E_{\max}(t)=\max_i d_i(t).
$$

要求

$$
E_{\mathrm{RMS}}\leq0.05,
\qquad
E_{\max}\leq0.10,
$$

并连续保持 30 步。这就是文档和实验汇报中的 `CE strict`。

面积均衡另行记录为

$$
\operatorname{CV}_{A}
=\frac{\operatorname{Std}(A_1,\ldots,A_N)}
{\operatorname{Mean}(A_1,\ldots,A_N)},
$$

其中 $A_i$ 是机器人 $i$ 所拥有的自由 Voronoi 网格面积。`CV≤0.15` 是 loose area-CV 诊断，不进入 CE reward，也不替代 CE strict。因此“CE strict 100%，area-CV≤0.15 为 75%”并不矛盾：机器人可以全部靠近各自单元中心，但最终单元面积仍不完全均匀。

## 5. 新 CR-MS：速度重要性紧凑围捕环

### 5.1 从追敌人中心改为追可占据围捕位置

新 MS 只对直接发现 enemy 的机器人生效。感知半径负责回答“是否看见目标”，奖励环负责回答“看见后应该去哪里”，二者不使用同一个半径。

对 pursuer $i$ 和 evader $j$，围捕环内边界首先按表面安全距离定义：

$$
R_{\mathrm{in}}^{\mathrm{surface}}
=d_{\mathrm{safe}}+m_{\mathrm{in}},
$$

再转换为中心距离：

$$
R_{\mathrm{in}}
=R_{\mathrm{in}}^{\mathrm{surface}}+r_i+r_j.
$$

当前参数为 $d_{\mathrm{safe}}=4.0$、$m_{\mathrm{in}}=0.5$，对应中心内半径约 $6.74$ m；偏好中心半径为

$$
R_{\mathrm{pref}}=8.0,
$$

外半径为

$$
R_{\mathrm{out}}=10.5.
$$

因此 20 m 只负责直接感知触发，实际奖励候选集中在约 $6.74\sim10.5$ m 的紧凑区域。这样既继承旧 A3 在 8 m 左右的有效围捕尺度，又避免宽环面积效应把目标推向感知边界。

### 5.2 径向偏好和速度方向重要性

对候选点 $c$，径向权重为

$$
w_r(c)
=\exp\left[
-\frac{\left(\left\|c-e_j\right\|-R_{\mathrm{pref}}\right)^2}
{2\sigma_r^2}
\right],
\qquad \sigma_r=2.0.
$$

设 evader 速度为 $v_j$。速度方向门控为

$$
g_v
=\operatorname{clip}\left(
\frac{\left\|v_j\right\|-v_{\mathrm{static}}}
{v_{\mathrm{full}}-v_{\mathrm{static}}},
0,1
\right),
$$

其中 $v_{\mathrm{static}}=0.30$、$v_{\mathrm{full}}=1.00$。定义候选方向和速度方向之间的夹角余弦：

$$
\cos\theta(c)
=\frac{(c-e_j)^{\mathsf T}v_j}
{\left\|c-e_j\right\|\left\|v_j\right\|}.
$$

角度权重为

$$
w_{\theta}(c)
=\operatorname{clip}\left(
1+\alpha g_v\cos\theta(c),
w_{\min},w_{\max}
\right),
$$

当前 $\alpha=0.25$、$w_{\min}=0.75$、$w_{\max}=1.25$。evader 静止时 $g_v=0$，整个环方向均匀；evader 运动时，前方候选点略微增权，后方略微减权。它把旧 front reward 的意图吸收到环目标内部，但不会把所有机器人强行吸到正前方。

候选点还必须满足地图边界、障碍安全和队友占用约束。当前使用半径 4 m 的 Cartesian disk 占用过滤。记有效性掩码为 $w_{\mathrm{valid}}(c)\in\{0,1\}$，总权重为

$$
w(c)=w_r(c)w_{\theta}(c)w_{\mathrm{valid}}(c).
$$

### 5.3 目标点与进度奖励

先用所有有效候选点计算加权平均方向：

$$
\bar c
=\frac{\sum_{c}w(c)c}{\sum_{c}w(c)},
$$

$$
u
=\frac{\bar c-e_j}{\left\|\bar c-e_j\right\|}.
$$

再把目标投影回偏好半径：

$$
t_i=e_j+R_{\mathrm{pref}}u.
$$

如果该投影点不可用，则从候选集中选择兼顾权重和偏好半径的回退点。该两阶段构造避免加权均值落进围捕环中心、障碍或不可达区域。

CR-MS reward 使用动作前状态固定目标，再计算该动作带来的目标距离进展：

$$
r_{i,t}^{\mathrm{ring}}
=\operatorname{clip}\left(
\left\|p_i(t)-t_i(t)\right\|
-\left\|p_i(t+1)-t_i(t)\right\|,
-c_{\mathrm{MS}},c_{\mathrm{MS}}
\right),
$$

其中 $c_{\mathrm{MS}}=3.0$。直接追捕者的 dense capture reward 为

$$
r_{i,t}^{\mathrm{direct\text{-}cap}}
=\omega_{\mathrm{ring}}r_{i,t}^{\mathrm{ring}},
\qquad \omega_{\mathrm{ring}}=2.0.
$$

旧的 approach、old mean-shift、front 三项权重均置零，capture timestep penalty 也置为零；terminal capture、碰撞、安全和边界项继续保留。这样 capture shaping 的含义变成单一问题：当前动作是否让机器人接近一个安全、可占据且考虑 evader 运动方向的围捕位置。

## 6. Support bridge：奖励中给方向，策略输入中不泄露坐标

第一批 CR-MS 只给直接发现者 Ring-MS，未直接感知 enemy 的支援机器人仍拿纯 CE reward。局部感知下，支援机器人看不到围捕环，也无法从自身 observation 重构 Ring-MS 目标；直接给它完整 Ring-MS 会造成奖励目标与策略信息不匹配。

当前第二批采用更轻的桥接方式。若 $s_i(t)=1$，支援机器人仍保持 coverage task label，但 reward 改为

$$
r_{i,t}^{\mathrm{support}}
=\lambda_s r_{i,t}^{\mathrm{approach}}
+(1-\lambda_s)r_{i,t}^{\mathrm{CE}},
\qquad \lambda_s=0.5.
$$

其中 approach 只保留旧 capture mix 中最简单的吸引进度：

$$
r_{i,t}^{\mathrm{approach}}
=\operatorname{clip}\left(
\left\|p_i(t)-e_{j^*}(t)\right\|
-\left\|p_i(t+1)-e_{j^*}(t+1)\right\|,
-3,3
\right).
$$

$j^*$ 只从正在追捕的一阶邻居直接看见的 evader 集合中选择。该目标用于训练 reward 计算，不被添加到支援机器人的 observation；部署时策略仍只依赖自身局部 token 和 friend token。因此这是一种训练期信用分配信号，而不是推理期全局敌人坐标泄露。

这一设计的理论逻辑是：support agent 不需要立即知道围捕环的精确相位，它首先需要学会“沿通信事件传播方向靠近”，以提高自己进入直接感知域的概率；一旦它真正看见 enemy，任务标签切换为 capture，并由完整 Ring-MS 接管。与此同时保留一半 CE，使支援行为不会完全退化为盲目追逐，仍维持空间展开和队形质量。

三种角色的 reward 因而可以统一写为

$$
r_{i,t}^{\mathrm{task}}
=
\begin{cases}
r_{i,t}^{\mathrm{direct\text{-}cap}}, & \delta_i(t)=1,\\
0.5r_{i,t}^{\mathrm{approach}}+0.5r_{i,t}^{\mathrm{CE}}, & \delta_i(t)=0,\ s_i(t)=1,\\
r_{i,t}^{\mathrm{CE}}, & \delta_i(t)=0,\ s_i(t)=0.
\end{cases}
$$

## 7. 4v1 训练设计

当前 4v1 从零训练 2M environment steps。IQN 的主要训练设置为：

$$
\epsilon:0.60\rightarrow0.05\quad\text{within 500k steps},
$$

$$
\eta_{\mathrm{lr}}
=
\begin{cases}
10^{-4}, & t<200\mathrm{k},\\
3\times10^{-5}, & t\geq200\mathrm{k}.
\end{cases}
$$

每个更新 batch 固定抽取

$$
n_{\mathrm{pursuing}}:n_{\mathrm{pre\text{-}capture}}:
n_{\mathrm{post\text{-}capture}}:n_{\mathrm{recovery}}
=64:16:32:16.
$$

recovery pure coverage 的初态中，75% 来自真实 capture snapshot；剩余 25% 再等分为普通 map-random 和 synthetic cluster。该分布让 CE 不只学习理想化纯覆盖，还反复看到真实围捕结束时常见的紧凑、偏置队形。

4v1 的 post-capture coverage window 为 500 步，训练 episode 上限为 3000 步。每 100k 保存 checkpoint，并使用固定环境 seed、

$$
\tau_k=\frac{k+0.5}{32},\qquad k=0,1,\ldots,31
$$

的 IQN 固定分位中点进行 deterministic screening。阶段必须训满 2M，再从 20 个 checkpoint 中选择历史最优，避免用训练窗口的偶发高点晋级。

## 8. 4v1 当前结果

全程 screening 最终选中 `step_2000000`。该点在 10-rollout screening 中达到：

- capture success 100%，碰撞 0%，平均 86.8 步；
- pure coverage CE strict 100%，area-CV≤0.15 为 100%；
- mix capture 100%，mix CE strict 100%，mix area-CV≤0.15 为 100%，碰撞 0%。

随后完成 capture、coverage、mix 各 20 rollout 的正式确定性评估：

| 场景 | Capture | CE strict | Area-CV≤0.15 | Collision | 平均步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| capture | 100% | 不适用 | 不适用 | 0% | 77.15 |
| coverage | 不适用 | 100% | 100% | 0% | 132.95 |
| mix | 100% | 90% | 95% | 5% | 227.45 |

这里的 mix CE strict 90% 是纯中心条件，即 $E_{\mathrm{RMS}}\leq0.05$、$E_{\max}\leq0.10$ 连续保持 30 步；area-CV≤0.15 的 95% 是另一项面积均衡诊断。二者必须分开解释。

这组结果说明，当前 4v1 模型已经同时形成了可靠的直接围捕、围捕后中心收敛和较高概率的面积均衡，并且 capture 时间没有因移除每步时间惩罚而明显失控。与只学会 capture、coverage 失败的 scratch 对照相比，CR-MS 4v1 已可作为后续规模课程的 strong anchor。

但因当前成功模型同时包含 VCT-LS、Ring-MS、support approach 和 CE，正式结论应是“该组合方法在 4v1 成立”，而不能仅凭这一条 run 宣称每个组件的独立因果增益。若需要论文级组件归因，仍应补做去 support、旧 capture、新 MS 无速度权重等受控 ablation。

## 9. 相比旧 mix 的方法变化

| 层面 | 旧 mix | 当前 CR-MS |
| --- | --- | --- |
| 通信图 | 通信与外部实体几何关系耦合较多 | pursuer-only Voronoi 一阶通信图 |
| 敌人信息 | 局部/邻接语义不够严格分离 | enemy token 仅来自 20 m 表面距离局部感知 |
| 直接 capture | approach + old MS + front + time penalty | 单一速度重要性紧凑 Ring-MS，time penalty 为 0 |
| 围捕半径 | 多项奖励间接平衡 | 显式 $R_{\mathrm{pref}}=8$ m，紧环约 $6.74\sim10.5$ m |
| 运动方向 | 独立 front reward | 速度门控角度权重进入 Ring-MS 候选分布 |
| support | 邻居状态可能缺少直接信用 | 0.5 neighbor-visible approach + 0.5 CE，不泄露 observation 坐标 |
| coverage | 面积、中心、inside、静止等复合 shaping/判定 | CE 中心平方能量 + PBRS + 延迟轻速度代价 |
| coverage 成功 | 旧多项综合 strict | CE RMS/max 中心误差连续保持 30 步 |
| 面积均衡 | 常与主成功条件混合 | area-CV≤0.15 只作独立诊断和选点辅助 |

## 10. 后续 8v2 与 12v3 课程

4v1 最优 `step_2000000` 已由自动 supervisor 注入 8v2 stage2。后续课程为

$$
4\mathrm{v}1\ (2\mathrm{M})
\longrightarrow
8\mathrm{v}2\ (700\mathrm{k})
\longrightarrow
12\mathrm{v}3\ (700\mathrm{k}).
$$

8v2 和 12v3 均使用 shape-compatible warm start，保持 VCT-LS、Ring-MS、support blend 与 CE 的方法语义不变，只扩大 pursuer、evader 和 obstacle 数量。post-capture coverage window 分别增加到 600 和 700 步；CE 所需最小参与者数量分别为 8 和 12。

两个后续阶段均采用从 step 0 开始的 $3\times10^{-5}$ 学习率，并令 epsilon 在 420k 内从 0.28 衰减到 0.05。每 100k 继续执行 capture、coverage、mix deterministic screening；阶段训满后选择历史最优，并自动生成对应的 20-rollout/10-GIF 展示。8v2 selection 会自动注入 12v3，无需人工修改 checkpoint 路径。

规模课程主要验证三个问题：

1. pursuer-only Voronoi 通信图扩大后，局部发现事件能否通过一阶邻居形成有效支援链；
2. 多 evader 情况下，直接发现者的 Ring-MS 是否能形成正确的局部目标分工，而不是所有 pursuer 聚向同一目标；
3. 围捕结束后的真实 snapshot 更复杂时，CE 是否仍能恢复中心收敛和面积均衡。

如果 8v2/12v3 延续 4v1 的 capture 与 CE strict 表现，就可以把 CR-MS 描述为一套具有规模课程可迁移性的局部感知围捕—覆盖方法；如果性能下降，则应首先按“敌人发现与通信覆盖、支援信用、多目标分配、围捕后 CE 恢复”四层定位，而不是重新把旧 mix 的多个 dense reward 全部加回。

