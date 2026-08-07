# Capture 奖励重构分支：速度重要性围捕环 MS

更新时间：2026-07-31

本文件记录一条新的并行实验线，暂称 `CR-MS-V0`，中文可记为“速度重要性围捕环”。它基于 VCT-LS 的局部敌人感知与友方 Voronoi 通信拓扑，但研究对象不是通信拓扑本身，而是 capture 阶段奖励函数的整合与简化。

边界说明：

- 本线与 VCT-LS、CE-Coverage、ZoneDemo 并行推进，互不覆盖。
- 本线默认使用旧 mix/A3 任务场景，不进入内外区 ZoneDemo。
- 本线默认保留 VCT-LS 的局部敌人感知：只有直接感知到 enemy 的 agent 才拿 capture shaping。
- 本线暂不使用 support agent 的 capture 奖励。support agent 仍按 coverage 奖励学习，是否参与围捕交给策略自己学。
- coverage 侧后续默认使用 CE-Coverage，即 `centroid_energy_v0`；旧 coverage 只保留为必要时的隔离 ablation，不作为新配置默认。

## 当前旧 Capture 奖励

当前 VCT-LS/A3 capture dense reward 主要由四块组成：

```text
r_time = timestep_penalty
r_approach = omega_approach * clip(d_before - d_after, -c_d, c_d)
r_ms = omega_mean_shift * mean_shift_reward
r_front = omega_front * front_reward
```

当前常用数值大致为：

```text
timestep_penalty = -1.0
omega_approach = 1.0
omega_mean_shift = 2.0
omega_front = 0.5
c_d = 3.0
r_e = 8.0
d_safe = 4.0
mean_shift_inner_margin = 2.0
mean_shift_outer_margin = 2.0
mean_shift_sense_radius = 35.0
front_theta = pi / 3
front_radial_sigma = 4.0
```

中文解释：

- `timestep_penalty`：每步时间惩罚，希望 agent 更快完成围捕。
- `approach`：距离 enemy 变近就给奖励，变远就给惩罚。
- `mean_shift`：在 enemy 周围采样一个环形栅格，求一个加权平均目标点，鼓励 pursuer 向这个目标点移动。
- `front`：根据 enemy 速度方向，额外鼓励 pursuer 去 enemy 前方区域。

这些项能工作，但在 VCT-LS 的局部感知语义下有几个不够自然的地方。

1. 吸引项在局部感知下价值下降。看不到 enemy 时无法计算；看得到 enemy 时，单纯靠近 enemy 容易和安全距离、成环围捕目标冲突。
2. 旧 MS 环的半径主要由 `r_e=8` 和 `inner/outer_margin=2` 决定，采样范围约为中心距离 `6..10`，但它没有和 VCT-LS 的敌人感知半径 `20m` 对齐。
3. 旧 MS 本身对 enemy 速度方向基本无感，前向偏好由另一项 `front_reward` 单独补充，导致 `MS + front + approach` 三项需要互相调权重。
4. 当障碍或局部可见区域让候选点集变成非凸形状时，简单求均值可能得到一个不够合理的方向，尤其在更大障碍物实验中风险更明显。

## 新想法

新 MS 项把“围捕成环”和“根据 enemy 速度前置补位”合成一个目标。其核心不是直接追 enemy，而是在 enemy 周围定义一个可占据围捕环，并按 enemy 速度方向给环上不同位置分配轻微不同的重要性。

直观描述：

- enemy 周围有一个圆环。
- 直接感知到 enemy 本体后，才开始计算这个虚拟围捕环。
- 感知触发半径和奖励候选环半径分离：感知半径仍用于判断 enemy 是否可见，奖励环不必直接扩到感知边界。
- 圆环内圈对应安全警戒边界，防止 pursuer 过度贴近 enemy。
- 奖励环在期望围捕半径附近权重最高，向内外递减。
- enemy 运动方向的正前方权重略高，背后权重略低。
- enemy 速度很低时，认为它接近静止，圆环各方向权重回到统一。

这样一项可以替代旧的 `approach + mean_shift + front` 三项。

2026-07-31 修订：不建议第一版把奖励外圈直接设为 VCT-LS 的 enemy 感知半径。若内外环约为中心距离 `6.7..22.2`，即使使用均匀权重，面积项也会让有效候选更偏外侧；若再叠加障碍和占用过滤，目标方向容易变钝，围捕半径可能明显大于旧 A3 的 `8m`。因此首版把 `enemy_sensing_radius=20m` 只作为“是否看到 enemy 本体”的触发条件，而把 reward candidate ring 设成围绕 `R_pref_center=8.0` 的紧环。

## 距离定义

为避免半径语义混乱，本线建议明确区分中心距离和表面距离。

设：

```text
p_i = 第 i 个 pursuer 的中心位置
e_j = 第 j 个 evader 的中心位置
r_p = pursuer 几何半径
r_e = evader 几何半径
d_center = ||p_i - e_j||
d_surface = d_center - r_p - r_e
```

当前 VCT-LS 已启用：

```text
local_sensing_uses_surface_distance = true
enemy_sensing_radius = 20.0
```

也就是当 `d_surface <= 20.0` 时，agent 视为直接感知到 enemy。若 pursuer 和 evader 半径都约为 `1.118m`，那么刚感知到 enemy 本体时的中心距离约为：

```text
R_detect_center = 20.0 + 1.118 + 1.118 = 22.236
```

这个半径只用于触发 capture shaping，不直接作为奖励环外圈。也就是说：

```text
direct_enemy_visible_i = (||p_i - e_j|| - r_p - r_e <= enemy_sensing_radius)
```

只有 `direct_enemy_visible_i=true` 时，第 `i` 个 pursuer 才能获得新 MS 项。若 sensor circle 只是碰到了理论围捕环，但 enemy 本体还不可见，则从局部感知角度看，这片区域和普通区域没有区别，新 MS 项必须为 0。

奖励候选环建议单独定义：

```text
R_inner_surface = d_safe + inner_extra_margin
R_inner_center = R_inner_surface + r_p + r_e
R_reward_outer_center = R_pref_center + outer_margin
```

首版推荐：

```text
d_safe = 4.0
inner_extra_margin = 0.5
R_inner_center ≈ 6.736
R_pref_center = 8.0
outer_margin = 2.5
R_reward_outer_center = 10.5
```

这比 `6.7..22.2` 更接近旧 MS 的 `6..10`，但仍保留“看到 enemy 后构造虚拟围捕环”的新语义。如果实机 rollout 显示过于保守，可以把 `inner_extra_margin` 降到 `0.0`；如果仍有贴近/碰撞，可以升到 `1.0`。若要测试大感知环，只应作为单独 ablation，而不是首版默认。

## 偏好半径

围捕环很宽，不能让 agent 只被外圈吸住，也不能让它贴到内圈。建议保留一个偏好半径 `R_pref_center`，作为真正鼓励接近的环半径。

首版推荐：

```text
R_pref_center = 8.0
```

原因：

- 这是旧 A3 中 `r_e=8.0` 的稳定经验值。
- 它大于推荐内圈 `6.736`，不会贴到安全警戒边界。
- 它位于首版紧环 `6.736..10.5` 的内部，既保留旧围捕尺度，又允许目标点在局部可行扇区中前后调整。

待确认点：

- 当前 `capture_distance=8.0` 在代码中更接近中心距离语义，而不是表面距离语义。首版为尽量少改动，建议继续把 `R_pref_center=8.0` 当作中心距离目标。
- 若后续要严格统一表面距离，可改为 `R_pref_surface`，再加 `r_p+r_e` 转为中心距离。但这会改变旧 A3 的实际围捕尺度，建议单独 ablation。

## 速度方向重要性

设 enemy 速度为 `v_e`，速度方向为：

```text
h = v_e / ||v_e||
```

对环上候选点 `c`，从 enemy 指向候选点的单位方向为：

```text
u = (c - e_j) / ||c - e_j||
cos_theta = dot(u, h)
```

含义：

- `cos_theta = 1`：候选点在 enemy 运动正前方。
- `cos_theta = 0`：候选点在侧向。
- `cos_theta = -1`：候选点在 enemy 身后。

当 enemy 速度较小时，不应强调前后方向，因为速度方向可能只是噪声。建议使用速度门控：

```text
g_v = clip((||v_e|| - v_static) / (v_full - v_static), 0, 1)
```

首版推荐：

```text
v_static = 0.30
v_full = 1.00
```

即：

- `||v_e|| <= 0.30`：认为接近静止，方向权重全部回到 1。
- `0.30 < ||v_e|| < 1.00`：方向权重逐渐生效。
- `||v_e|| >= 1.00`：方向权重完全生效。

角度权重首版建议用最容易解释的线性形式：

```text
w_angle = clip(1 + alpha * g_v * cos_theta, w_min, w_max)
```

保守推荐：

```text
alpha = 0.25
w_min = 0.75
w_max = 1.25
```

几个具体场景：

```text
enemy 几乎静止，||v_e||=0.10:
g_v=0，所有方向 w_angle=1.00

enemy 中速，||v_e||=0.65:
g_v=(0.65-0.30)/(1.00-0.30)=0.50
正前方 cos=1，w_angle=1.125
侧方 cos=0，w_angle=1.000
背后 cos=-1，w_angle=0.875

enemy 高速，||v_e||=1.20:
g_v=1
正前方 cos=1，w_angle=1.25
侧方 cos=0，w_angle=1.00
背后 cos=-1，w_angle=0.75
```

这个高低差不大，目的是让 agent 更愿意补 enemy 前方，但仍保留完整成环压力，避免所有 pursuer 都挤到前方。

## 候选点权重

候选点来自 enemy 周围的可达环形区域。首版默认使用紧奖励环：

```text
R_inner_center <= ||c - e_j|| <= R_reward_outer_center
```

候选点需要过滤：

- 不在地图外。
- 不在障碍物膨胀区内。
- 不与其他 pursuer 的占据半径冲突。
- 若开启 obstacle-free Voronoi/CE 同款自由空间，可使用 `free_mask_projected` 作为可达性掩码。

每个候选点的总权重建议为：

```text
w(c) = w_radial(c) * w_angle(c) * w_valid(c)
```

其中 `w_valid` 是 0 或 1。径向权重是必要项，不是可选装饰。原因是环形面积随半径增大而增大，若把候选环做宽而不加径向偏好，外圈会天然占更多候选点，实际围捕尺度会被拉大。

首版径向权重建议保留旧 MS 的高斯式偏好：

```text
w_radial = exp(- (||c - e_j|| - R_pref_center)^2 / (2 * sigma_r^2))
```

首版推荐：

```text
sigma_r = 2.0
```

如果发现候选点太少或目标跳动过大，可用带底座的径向权重：

```text
w_radial = 0.20 + 0.80 * exp(- (||c - e_j|| - R_pref_center)^2 / (2 * sigma_r^2))
```

但首版建议先用纯高斯，减少一个额外超参。

不推荐首版使用“内环最高、外环最低”的单调递减权重。它确实会让围捕半径更小，可能提升 ZoneDemo 这类困难场景中的堵截强度，但会把 agent 系统性推向安全边界，和 `d_safe`、evader APF 躲避、collision penalty 产生更强冲突。如果要测，可作为 `CR-MS-radius-ablation`：

```text
compact_gaussian: R_inner≈6.7, R_pref=8.0, R_outer=10.5, sigma=2.0
wide_gaussian: R_inner≈6.7, R_pref=8.0, R_outer=22.2, sigma=2.0
inner_decay: R_inner≈6.7, R_outer=10.5, w_radial 从内向外递减
```

当前更推荐 `compact_gaussian`。它可以理解为：只要 agent 感知到 enemy 本体，即使自己还在围捕环外，也能计算 enemy 周围期望半径约 8m 的目标点；从远处看，它近似表现为“向 enemy 附近的安全围捕环靠近”，因此保留了旧吸引项的一部分效果，但不会鼓励贴到 enemy 中心。

### 与 Evader APF 半径的关系

当前 A3 APF 默认 `force_range=15.0`；ZoneDemo 暂定 B1 为 `force_exponent=1.5`、`velocity_k=1.0`、`dynamic_position_repulsion_requires_closing=false`，`force_range` 仍为 `15.0`。动态 pursuer 在 evader APF 中按半径 `0.8` 传入，公式内部还减去 `0.8` 安全量：

```text
d_obs = center_distance - 0.8 - 0.8
APF active when d_obs < force_range
```

因此 evader 对 pursuer 的动态排斥大约在：

```text
center_distance < 16.6
```

时开始生效。这个半径明显大于我们的期望围捕半径 `R_pref_center=8.0`。含义是：一旦 pursuer 真正进入 8m 围捕环，evader 已经处在 APF 躲避压力里；而如果把 reward 外圈扩到 22m，则有相当一段候选区域在 evader APF 还没有强烈响应的位置，capture shaping 会变得偏外、偏慢。由此也支持首版采用紧奖励环，而不是直接使用感知边界作为奖励外圈。

### 占用过滤与相位占用

当前旧 MS 使用：

```text
mean_shift_occupancy_radius = 4.0
```

它是在平面上用一个半径 4m 的圆盘排除候选点，而不是按 enemy 中心相位排除。这个设计在半径 `8m` 的围捕环上大约等价于屏蔽：

```text
half_angle ≈ asin(4 / 8) ≈ 30°
full blocked sector ≈ 60°
```

但如果围捕环半径变成 `14m`，同样 4m 只相当于约 `33°` 的总扇区；半径越大，占用在角度上的效果越弱。这也是不宜默认使用大外环的另一个原因。

后续若要解决“多个 pursuer 和 enemy 同侧绕圈”的问题，建议增加相位占用版本：

```text
phi_k = atan2(p_k.y - e_j.y, p_k.x - e_j.x)
phi_c = atan2(c.y - e_j.y, c.x - e_j.x)
occupied_phase = abs(wrap(phi_c - phi_k)) <= phase_occupancy_width
```

首版可选参数：

```text
phase_occupancy_width = 25°..35°
phase_occupancy_radial_margin = 4.0
```

只在其他 pursuer 距离 enemy 的半径接近围捕环时，才屏蔽对应相位。这样比固定 4m 圆盘更符合“围捕环上某个方向已经有人占据”的几何意义。

理性判断：单靠 MS 的占用过滤，有机会让后续同侧 pursuer 改变角速度去补另一侧，但不保证。它成立需要几个条件：agent 能看到 enemy，本地或通信 observation 能反映已有 pursuer 的方位，目标点随空缺扇区稳定偏移，离散角速度动作能产生足够探索，并且训练中有足够多同侧拥挤样本。当前 `perception_range=20m` 对 `R_pref=8m` 基本够用，因为围捕环直径约 16m；若奖励环扩大到 12m 以上，另一侧距离可能超过 20m，本地感知就不够稳定。若 rollout 仍出现长期同侧绕圈，应考虑额外加入轻量角度覆盖奖励或 phase assignment，而不是继续堆复杂 MS 权重。

## 目标点计算

直接对所有候选点求加权均值可能在非凸可行区域里产生奇怪目标。因此首版推荐一个更稳的两步法：

1. 用加权候选点只计算方向。

```text
m = sum_c w(c) * c / sum_c w(c)
dir = normalize(m - e_j)
```

2. 把目标投影回偏好半径：

```text
t = e_j + R_pref_center * dir
```

若 `t` 不可达，或者落入障碍物膨胀区，则回退到候选集中“靠近 `R_pref_center` 且权重较高”的可达点：

```text
t = argmax_c [w(c) - beta_r * abs(||c-e_j|| - R_pref_center)]
```

这样可以避免“均值点落入障碍/环洞/不可达区域”的问题，也能保持旧 MS 中“根据可用空位给方向”的优点。

## 新 Capture Shaping

新 MS 奖励仍采用进度差形式，便于与旧奖励尺度对齐。时间语义必须固定为“用动作前状态计算目标，用动作前后位置比较到这个固定目标的距离变化”：

```text
t_k = target(s_k)
r_ring_ms = clip(||p_i^k - t_k|| - ||p_i^(k+1) - t_k||, -c_ms, c_ms)
```

下一步再重新计算：

```text
t_(k+1) = target(s_(k+1))
r_next = clip(||p_i^(k+1) - t_(k+1)|| - ||p_i^(k+2) - t_(k+1)||, -c_ms, c_ms)
```

这点很重要，因为每一步 enemy 位置、enemy 速度、其他 pursuer 占用都会变化。如果当前 transition 中同时用 `after` 状态改目标，就会让同一个动作既改变位置又改变奖励靶点，奖励解释会变脏。

当前旧 `_mean_shift_reward()` 基本使用 `before_e` 和 `before_p[i]` 计算目标，再比较 `before_p[i]` 与 `after_p[i]` 到该目标的距离，方向是对的；但占用过滤里用了 `after_p[j]` 排除其他 pursuer 附近候选点。新实现应改为全部使用 pre-action snapshot，即候选点、enemy 速度、其他 pursuer 占用都来自 `s_k`。

首版推荐：

```text
c_ms = 3.0
omega_ring_ms = 2.0
```

完整 dense capture reward 变为：

```text
r_capture_dense = omega_ring_ms * r_ring_ms
```

并建议关闭旧三项：

```text
omega_approach = 0
omega_mean_shift = 0
omega_front = 0
capture_reward_mode = ring_importance_ms_v0
```

capture terminal、碰撞惩罚、安全惩罚、边界惩罚先保持不变。

## 是否去掉 Time Penalty

本线建议把 capture dense 分支中的 `-1/step` time penalty 去掉，原因是：

- 新 reward 只奖励朝合理围捕环目标的进度，时间惩罚容易让 agent 重新选择短视贴近 enemy。
- coverage 新 CE 方向也在减少拼接式时间奖励/惩罚，转向“目标误差下降 + 能耗”的更干净结构。
- capture 快慢可以通过独立指标评估，例如 capture steps、first-detection steps，而不一定写成每步惩罚。

但这不是无风险改动。去掉 time penalty 后，策略可能更慢、更保守，尤其在离散动作空间中可能少一些“赶紧动起来”的压力。因此推荐做成显式开关：

```text
capture_timestep_penalty = 0.0
```

并在实验指标里额外记录：

```text
avg_capture_steps
first_detection_steps
fully_capture_steps
timeout_without_detection_rate
```

若 capture 速度明显下降，可考虑轻量恢复，例如：

```text
capture_timestep_penalty = -0.1 或 -0.2
```

不建议直接回到 `-1.0`，否则它会重新变成一个很强的全局偏置。

## Support Agent 奖励口径

本分支暂不考虑 support 个体的 capture 奖励。

具体口径：

- 自己直接感知到 enemy：进入 capture label，拿 `ring_importance_ms_v0` capture reward。
- 自己没感知到 enemy，但一阶友邻感知到 enemy：仍保留 coverage label，只通过 friend token 的 `is_pursuing` 观察到邻居状态。
- 自己和邻居都没感知到 enemy：coverage。

这和当前已有的 `support_reward_blend` 对照线不同。`support_reward_blend` 是另一条 VCT-LS 对照线，用 `0.5 capture + 0.5 coverage` 直接给 support agent dense reward。本分支先不采用它，避免 capture 奖励重构和 support credit assignment 两个因素混在一起。

潜在风险：

- 纯 support coverage 奖励可能让一阶邻居支援变慢。
- 但如果本线能在没有 support capture shaping 的情况下成立，说明 VCT-LS 的 observation 与通信拓扑本身足够自然，泛化价值更高。
- 若失败集中表现为“一人发现敌人但邻居不补位”，再开 `CR-MS-V0 + support_blend` 作为二阶 ablation。

## Coverage 侧与 CE 线关系

2026-07-31 最新口径：CE-Coverage 已基本可作为后续 coverage 默认侧接入。需要保留两个层次的判定：

- strict CE：`E_rms<=0.05`、`E_max<=0.10`、连续 hold 30 步。这是 CE 的主成功定义。
- loose CV：`final Voronoi area CV<0.15`。这是 old-mix 接回时的均布诊断/晋级辅助指标，不进入奖励函数，也不替代 strict CE。

截至当前核心文档，关键证据是：

- CE13 在 8-agent pure coverage 上 20 rollout 达到 `coverage_success=1.00`，说明 CE 在 pure coverage 中已经是当前最干净稳定的 coverage 候选。
- CE14B old mix final 的 strict CE 成功率不高，但按 `CV<0.15` loose 复核，mix 与 pure coverage 均达到 `1.00`，final CV 约 `0.08`。这说明 old-mix 接回后的面积均布已明显可用，只是 strict center hold 仍需在课程里继续稳住。
- 因此后续新 capture 实验，包括 CR-MS，coverage 侧默认接 CE；同时输出 strict CE 与 loose CV 两套指标。

默认 CE 参数口径：

```text
coverage_objective_version = centroid_energy_v0
coverage_ce_reward_scale = 10.0
coverage_ce_pbrs_enabled = true
coverage_ce_pbrs_kappa = 1.0
coverage_ce_pbrs_reset_mode = phase_and_all_terminal
coverage_ce_speed_weight_schedule = 0.0 before 200k, 0.0005 from 200k
coverage_ce_acceleration_weight = 0.0
coverage_ce_angular_velocity_weight = 0.0
coverage_ce_success_rms_threshold = 0.05
coverage_ce_success_max_threshold = 0.10
coverage_ce_success_hold_steps = 30
coverage_cv_loose_area_cv_threshold = 0.15
```

旧 coverage ablation 仍可保留，但必须显式标注为 `coverage_objective_version=legacy`，用于区分“capture reward 本身效果”和“CE coverage 接回效果”。

## 可行性判断

整体可行，且方向合理。

合理之处：

- 局部感知下，`enemy_sensing_radius` 只负责触发“我看见 enemy 本体”，reward ring 负责定义“看见后该去哪里围捕”，两者分离后语义更干净。
- 内圈绑定 `d_safe`，让围捕奖励天然避开安全警戒区域。
- 紧奖励环围绕旧经验半径 `8.0`，比 `6.7..22.2` 大环更不容易漏敌，也更接近 A3 已验证过的围捕尺度。
- 速度方向权重把旧 `front_reward` 合并进 MS，不再额外叠一个方向奖励。
- 关闭纯吸引项后，可以减少追敌人中心、贴近碰撞、破坏围捕环的倾向。
- 用一个目标进度奖励替代三项拼接，后续调参更少，也更适合写进论文方法部分。

主要风险：

- 前向权重太强会让 pursuer 聚到 enemy 前方，围捕环变成堵截点。
- 外圈太大时，候选点数随面积增大，若没有径向偏好，目标会变钝、动作响应慢。
- 去掉 time penalty 可能降低 capture 速度，需要把 capture steps 纳入选点。
- 障碍物导致候选区域非凸时，均值目标仍可能不可达，因此必须保留可达性回退。
- support agent 不拿 capture reward 时，发现敌人的一阶邻居是否会主动补位，需要实验验证。

## 推荐首版参数

建议 `CR-MS-V0` 首版不要贪心，先做保守改动：

```text
capture_reward_mode: ring_importance_ms_v0
omega_ring_ms: 2.0
ring_ms_progress_clip: 3.0

ring_ms_trigger_mode: direct_enemy_body_visible
ring_ms_detection_surface_radius: enemy_sensing_radius
ring_ms_inner_surface_radius: d_safe + 0.5
ring_ms_preferred_center_radius: 8.0
ring_ms_outer_center_radius: 10.5
ring_ms_radial_sigma: 2.0
ring_ms_cell_size: 1.5
ring_ms_occupancy_mode: cartesian_disk
ring_ms_occupancy_radius: 4.0

ring_ms_velocity_static_threshold: 0.30
ring_ms_velocity_full_threshold: 1.00
ring_ms_angle_alpha: 0.25
ring_ms_angle_weight_min: 0.75
ring_ms_angle_weight_max: 1.25

omega_approach: 0.0
omega_mean_shift: 0.0
omega_front: 0.0
capture_timestep_penalty: 0.0

support_reward_blend_enabled: false
coverage_objective_version: centroid_energy_v0
coverage_ce_reward_scale: 10.0
coverage_ce_pbrs_enabled: true
coverage_ce_pbrs_reset_mode: phase_and_all_terminal
coverage_ce_speed_weight_schedule: 0.0 -> 0.0005 at 200k
coverage_ce_success_rms_threshold: 0.05
coverage_ce_success_max_threshold: 0.10
coverage_ce_success_hold_steps: 30
coverage_cv_loose_area_cv_threshold: 0.15
```

第一轮 ablation 只建议动三处：

```text
角度权重弱版: alpha=0.15, weight range 0.85..1.15
角度权重标准版: alpha=0.25, weight range 0.75..1.25
角度权重强版: alpha=0.35, weight range 0.65..1.35
```

半径 ablation 建议放在第二组，或与角度权重分开做：

```text
R0 compact default: inner≈6.7, pref=8.0, outer=10.5, sigma=2.0
R1 old-like narrow: inner≈6.7, pref=8.0, outer=10.0, sigma=2.0
R2 wide weighted: inner≈6.7, pref=8.0, outer≈22.2, sigma=2.0
R3 inner-decay diagnostic: inner≈6.7, outer=10.5, inner high -> outer low
```

占用 ablation 不建议首轮混入。如果同侧绕圈仍明显，再测试：

```text
ring_ms_occupancy_mode: phase_sector
phase_occupancy_width: 30deg
phase_occupancy_radial_margin: 4.0
```

不建议第一轮同时改变 support reward、preferred radius、time penalty 和 terminal reward。coverage 侧按 CE 默认接入；如果要做旧 coverage 对照，应单独开 legacy ablation，不混在主线里。

## 实现 TODO

1. 增加配置开关。

```text
capture_reward_mode: legacy | ring_importance_ms_v0
capture_timestep_penalty: null 或数值
```

`legacy` 必须保持当前行为，避免破坏 A3、VCT-LS、CE、ZoneDemo。

2. 在环境中新增 `_ring_importance_ms_reward()`。

它复用旧 `_mean_shift_reward()` 的候选点采样、障碍过滤、占据过滤，但新增：

- trigger 使用 VCT-LS enemy sensing surface radius 判断是否直接看见 enemy 本体。
- reward outer radius 默认使用紧环 `R_reward_outer_center=10.5`，不要默认等于 sensing radius。
- inner radius 使用 `d_safe + margin` 转中心距离。
- angle weight 使用 enemy velocity gate。
- target 采用“加权方向 + preferred radius 投影 + 不可达 fallback”。
- 当前 transition 的 target 只从 pre-action snapshot 计算，包括其他 pursuer 的占用过滤。
- 首版占用保持 `cartesian_disk`，相位占用作为后续 ablation。

3. 修改 capture reward 入口。

当 `capture_reward_mode=ring_importance_ms_v0`：

```text
只用 omega_ring_ms * ring_ms_reward
不加 approach
不加 old mean_shift
不加 front
capture timestep penalty 按 capture_timestep_penalty 读取，首版为 0
```

4. 写测试。

至少覆盖：

- 静止 enemy 时，各角度权重均为 1。
- 高速 enemy 时，正前方候选点权重大于侧方，侧方大于背后。
- 只有直接感知到 enemy 本体时，新 MS reward 才非零；只接触理论围捕环不能触发。
- detection center radius 等于 `enemy_sensing_radius + r_p + r_e`。
- reward outer center radius 首版等于配置的 `10.5`，不等于 detection center radius。
- inner center radius 等于 `d_safe + margin + r_p + r_e`。
- transition reward 使用 `target(s_k)` 比较 `p_i^k` 与 `p_i^(k+1)`，占用过滤不能使用 `after_p[j]`。
- 无候选点或目标不可达时不会报错，并返回 0 或 fallback 目标。
- `capture_reward_mode=legacy` 时旧配置行为不变。
- support reward blend 默认关闭。

5. 增加调试可视化。

建议只在 debug GIF 或单步 probe 中画：

- enemy 周围内外圈。
- preferred radius。
- 新 MS 目标点。
- enemy 速度方向箭头。

默认 rollout GIF 不强制画这些元素，避免污染常规可视化。

## 实验 TODO

建议顺序如下。

### CR0：无训练几何 sanity rollout

目的：先确认新 reward target 的几何行为没有明显反直觉，再开训练。

```text
checkpoint: 当前 VCT-LS 4v1 / 8v2 可用模型
rollout: capture 和 mix 少量 GIF
配置: compact default，只画 debug target，不训练
```

重点看：

```text
enemy 未被直接感知时，新 MS target 不出现，reward 为 0
enemy 刚进入感知半径时，target 落在约 8m 围捕环，而不是 20m 外圈
同侧多个 pursuer 时，cartesian occupancy 是否能把 target 推向空缺扇区
target 是否因使用 after_p occupancy 而发生不合理跳变
```

### CR1：4v1 warm-start 短线

目的：验证新 reward 是否破坏已有 VCT-LS 围捕能力。

```text
场景: 4v1 old mix / VCT-LS
起点: 当前 VCT-LS 4v1 plain 最优或最新 checkpoint
步长: 300k--500k
半径: compact default, inner≈6.7/pref=8.0/outer=10.5/sigma=2.0
coverage: centroid_energy_v0, 同步统计 strict CE 与 CV<0.15 loose
support_reward_blend: false
检测: 每 50k 或 100k，capture/mix 各 20 rollout
```

重点指标：

```text
capture_success_rate
fully_capture_rate
collision_rate
avg_capture_steps
first_detection_steps
direct_detector_count
miss_or_invisible_enemy_steps
```

合格信号：

```text
capture_success_rate >= 0.95
collision_rate <= 0.05
avg_capture_steps 不明显慢于 VCT-LS baseline
GIF 中没有明显贴身追逐或前方聚团
GIF 中没有大围捕半径导致的漏敌
```

若 CR1 失败，优先看失败类型：

```text
太慢: 轻量恢复 capture_timestep_penalty=-0.1 或调高 omega_ring_ms
太外: 降 outer 到 10.0 或 sigma 到 1.5
太贴/碰撞: inner_extra_margin 0.5 -> 1.0
同侧绕圈: 暂不调半径，先试 phase_occupancy ablation
```

### CR2：4v1 scratch

目的：验证新 reward 从零是否可学。

```text
场景: 4v1 old mix / VCT-LS
起点: scratch
步长: 1M--2M
coverage: centroid_energy_v0
检测: 300k 起每 100k，最后选历史 capture 最优
```

若 CR1 成立但 CR2 不成立，说明 reward 对 warm policy 友好，但探索/信用分配仍不足。

### CR3：8v2 warm-start 适应

目的：验证在多 enemy、多 pursuer 场景下，新 MS 是否仍能支撑局部感知围捕。

```text
场景: 8v2 old mix / VCT-LS
起点: 当前 VCT-LS 8v2 warm 或 A3 stage3 8v2 best
步长: 300k--500k
coverage: centroid_energy_v0
检测: capture/mix 各 20 rollout，mix GIF 500 step 截断
```

重点额外看：

```text
不同 enemy 是否都被发现
是否出现只围一个 enemy 漏另一个 enemy
friend topology 是否能把发现信号扩散到局部支援
多个 pursuer 是否仍和同一个 enemy 同侧绕圈
```

### CR3b：占用相位 ablation

触发条件：CR1/CR3 中 capture success 尚可，但 GIF 仍有明显“同侧绕圈、另一侧空着”的现象。

```text
只改 ring_ms_occupancy_mode: phase_sector
phase_occupancy_width: 30deg
phase_occupancy_radial_margin: 4.0
其余保持 CR1/CR3 best 配置
```

判定标准不是单看 reward，而是看：

```text
同侧绕圈样本是否减少
capture steps 是否下降
collision 是否不升高
fully_capture 是否更稳定
```

### CR4：旧 Coverage 隔离对照

触发条件：若 `CR-MS + CE` 结果难以判断退化来自 capture 还是 CE 接回，可开一条短对照。

```text
capture: ring_importance_ms_v0
coverage: legacy old coverage
任务: old mix
起点: 与 CR1 或 CR3 相同 checkpoint
步长: 100k--300k
```

该线只用于归因，不作为后续默认方向。

## 当前判断

这条 capture reward 重构线值得推进。它不是简单调权重，而是在 VCT-LS 局部感知假设下重新定义 capture dense reward 的几何含义：看见 enemy 后，不再“追中心 + 求均值 + 额外抢前方”，而是“进入一个安全、可感知、带轻微前向重要性的围捕环”。

首版最关键的是保持温和：感知半径只做 enemy 本体触发，奖励环采用紧环 `6.7..10.5`，preferred radius 继续使用旧 `8.0`，前向权重只给轻微偏置，support capture reward 暂时关闭，coverage 侧默认接 CE。这样如果结果变好，可以比较明确地归因于“更干净的 capture ring MS + 更干净的 CE coverage”组合；如果结果变差，则通过旧 coverage 隔离对照、半径 ablation、占用模式 ablation 分别拆解。

## Screening 自动检查（2026-07-31 补充）

scratch1m 线在 100k 步检查点出现后，由 tools/watch_crms_scratch_screening.sh 自动执行 capture/mix/coverage 各 10 个 rollouts 的 screening，输出到：

```text
artifacts/2026-07-31_cr_ms_ring_vctls_ce_4v1/screening/scratch1m/step_<N>/
```

每个 step 目录含 all_summaries.json（capture/mix/coverage 三场景汇总）。关注指标：

- capture_success_rate：capture 成功率
- coverage_success_rate：CE strict coverage 成功率
- coverage_cv015_rate / coverage_cv015_best_rate：loose CV<0.15 成功率
- collision_rate、avg_steps：碰撞率与平均步数

重复启动脚本会自动跳过已完成（DONE 标记）的 step，不覆盖已有结果。scratch1m 启动方式（2026-07-31 已启动）：

```bash
bash tools/watch_crms_scratch_screening.sh \
  configs/experiments/cr_ms_vctls_ce_20260731/cr_ms_vctls_ce_4v1_scratch1m.yaml \
  runs/crms_vctls_ce4v1_scratch1m_20260731_run1 \
  artifacts/2026-07-31_cr_ms_ring_vctls_ce_4v1/screening/scratch1m \
  2026073300 cuda:0
```

warm500k 线如需同样检查，将 STEPS 环境变量收窄到 500k：

```bash
STEPS="100000 200000 300000 400000 500000" bash tools/watch_crms_scratch_screening.sh \
  configs/experiments/cr_ms_vctls_ce_20260731/cr_ms_vctls_ce_4v1_warm500k.yaml \
  runs/crms_vctls_ce4v1_warm500k_from_vctls_plain2m_20260731_run1 \
  artifacts/2026-07-31_cr_ms_ring_vctls_ce_4v1/screening/warm500k \
  2026073400 cuda:0
```

screening 只在 checkpoint 出现时短时运行（约 7 分钟/次），GPU0 上不影响 CE/CR-MS 训练。

## 2026-08-02：CR-MS 首轮训练完成与正式评估归档（deepseek 执行）

### 训练完成情况

- warm500k：500k 训满（08-01 02:57），final_step_500000.pt。
- scratch1m：1M 训满（08-01 08:59），final_step_1000000.pt。

### 正式 20-rollout 结果

评估口径：capture/coverage/mix 各 20 rollout / 10 GIF，固定 IQN 分位中点，步数上限 capture 1000 / coverage 1200 / mix 2200。

| 线 | capture | mix capture | mix CE strict | mix CV<0.15 | coverage CE strict | coverage CV<0.15 | 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warm500k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| scratch1m | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.80 | 0.00 |

screening 阶段（100k-900k 各点 10-rollout）capture 基本为 0，与最终评估一致。

### 结论与归因

1. ring_importance_ms_v0 首版在 4v1 上未能产生稳定 capture（warm 与 scratch 均为 0）。
2. 同批对照显示非 support 线全部 capture=0：VCT-LS plain warm（从 capturefocus 2m warm）与 plain scratch 同样为 0。因此不能把失败单独归因于 ring reward——4v1 训练分布下缺少 support blend 梯度时，即使保留旧 approach/ms/front（VCT-LS plain 线）也学不出围捕。
3. 训练窗口 capture（warm 曾 0.19-0.28）与确定性独立评估差距大，后续以正式 20-rollout 为准。
4. 正向信号：scratch1m 后期 coverage CV<0.15 达 0.80，ring 线至少发展出部分 coverage 行为；warm 线 coverage 未保留。

## 未来方向

1. 第二轮 CR-MS 必须加 support blend 对照（CR-MS 配置 + 0.5 capture / 0.5 coverage blend），分离 ring reward 与 support 梯度两个因素。
2. 若 support 对照成立，再做 CR3b 相位占用、radius ablation 与 CR4 旧 coverage 隔离对照。
3. 若 support 对照仍失败，优先检查 ring 目标点计算与 pre-action snapshot 语义，再考虑调整 omega_ring_ms 与奖励环半径。


## 2026-08-02：第二批 CR-MS support-approach + 最新 CE（Codex 接手）

本节覆盖上面的旧“未来方向”第 1 条，记录用户补充的正式二批口径。

### 奖励与 observation

- direct detector：继续使用首批 `ring_importance_ms_v0`，MS 参数、局部感知触发、零 time penalty 与旧三项关闭状态不改。
- support agent：仍是 coverage label，只从 friend token 知道一阶邻居在 pursuing，不增加 enemy token。
- support target：仅使用 pursuing 一阶邻居直接感知到的 evader id。
- support capture 半项不再调用 direct ring-MS，改为旧 mix-capture 中单独的距离吸引进度项；不包含 timestep、旧 MS 或 front。
- 总 support dense reward 固定为 `0.5 * legacy_approach_only + 0.5 * CE`。

新增开关 `voradj.support_reward_capture_component_mode=approach_only`。默认值仍为 `capture_task`，旧 VCT-LS/support 配置不改变。

### Coverage 与课程

coverage 侧沿用 CE-OldMix-v1 最新设置：replay `64/16/32/16`、captured snapshot ratio `0.75`、200k 后 `lambda_v=0.0005`，post window 按 4v1/8v2/12v3 使用 `500/600/700`。

配置目录：

```text
configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/
```

- stage1：4v1 scratch 2M，100k 到 2M 全程每 100k screening，不提前停。
- stage2：晋级后 8v2 warm 700k。
- stage3：再次晋级后 12v3 warm 700k。

### 验证与启动

2026-08-02：

- support approach-only 方向、权重、局部 target 与 direct ring-MS 隔离测试已加入。
- 项目全量测试 `38 passed`，shell/py_compile/diff check 通过。
- 2k GPU smoke 自然完成并生成完整 checkpoint/config/metrics/episodes。
- 正式 stage1 已在 `crms_sa_ce_s1_train_gpu0_0802` 启动，使用 GPU0。
- 全程 watcher 已在 `crms_sa_ce_s1_screen_gpu1_0802` 启动，使用 GPU1 做短时独立评估。
- 详细接手状态与检查清单见 `docs/PROJECT_HANDOFF_20260802_ZH.md`。

### 首个 100k screening 与 stage-cap 自动展示

2026-08-02 16:56，第二批 stage1 的 100k 确定性独立 screening 完成：

- capture：成功率 0.80，collision 0.20，平均 435.9 步；
- mix：capture 0.80，collision 0.20，CV<0.15 为 0.50，CE strict 为 0；
- pure coverage：CV<0.15 为 1.00，CE strict 为 0，collision 为 0。

这是第二批相对首批 capture=0 的早期正信号，但 stage1 仍严格训满 2M。新增 finalize_screened_run.py 后台等待全部 20 个 100k screening 点，按 capture/mix-capture 优先、碰撞与 coverage 次优的规则选择历史最佳，并自动跑对应 capture/coverage/mix 各 20-rollout/10-GIF。展示根目录为 artifacts/2026-08-02_three_line_stage_best_20rollout10gif/。
