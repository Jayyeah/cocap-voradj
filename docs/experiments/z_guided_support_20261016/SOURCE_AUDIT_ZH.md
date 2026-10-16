# Z-Guided Support v0 实现前源码审计

基线：`f0c0dfca8a508233f30ddbb5158699400d92a00f`，独立 worktree `/home/yjq/rl/CoCap1/iqn-z-guided-support-20261016`。
事实源：Z05 training 分支、独立 evaluation 分支，以及主工作区两份 20261015 paper 方法资料（只读）。本文件于代码修改前建立。

| 函数 | 基线行为与本轮约束 |
|---|---|
| `_advance_z_state` | reset 布置目标后初始化；step 的 reward/capture 之后、返回 next observation 之前同步更新一次。读取不可变 previous Z。λ=η=.5，阈值 .1；捕获不清零。 |
| `_pack_agent_obs` | physical_only 排序键为 robot-frame 相对 x/y、友机速度 x/y、距离、角度、内部 ID；排序后截断 8，再生成 float32 token/mask。Z 不参与排序。保留这一代码并记录映射；奖励不能使用其余 Voronoi 邻居。 |
| `_support_pursuing_friend_ids` | 自身必须 effective coverage；友机必须 effective capture。只用一跳 Voronoi 邻接；K10 角色标签不等于 Z。 |
| `_task_reward_role` | inactive → capture → 有 capture 邻居的 support → coverage。非零 Z、二/三跳传播不自动进入 support。保留路由；统计有非零可见邻居 Z 却不是 support 的 active transition，另报排除 capture 后比例。 |
| `support_reward_blend_active` | support blend 开启、有 active target 且当前 reward role 为 support。保留门控。 |
| `support_capture_task_reward` | approach_only 使用邻居能看到的真实敌人坐标，按动作后距离挑选目标；混合权重固定 .5/.5。v0 仅替换此内部目标与混合权重。 |
| `VorAdjEnv.step` | before_p/e、effective labels、before_z 先保存；移动/碰撞；计算任务奖励；capture event 与移除；按边界消去加权 CE next potential；Z 更新；terminal 消项；next observations。 |

## PBRS 与有效性

本轮使用 `w_t=1-alpha_t` 乘完整的原始 CE transition reward，不改 CE/PBRS 公式。`ce_reward_scales` 必须记录同一个 w_t，capture phase boundary 与 terminal correction 都以该 w_t 消项，禁止固定使用 .5。动态加权 PBRS 本身一般不再具有固定 potential 的全程望远镜性质：相邻步权重变化会留下 `(w_t-w_{t+1}) gamma Phi(s_{t+1})` 项。不能称此新 blend 为策略不变的 PBRS；需离线统计该权重切换残差，不能为抵消它而引入本轮以外的奖励。

## 候选安全与信息边界

只接受动作前实际打包的有效 friend tokens 及内部索引映射。inactive 不应进入观测；非有限 token 拒绝成为候选，开启新模式时自身或动作锚点非有限直接失败。友机移动时固定动作前锚点。self 与 max Z 在 1e-6 内相等时 alpha=0，完整 CE，确有停止当前吸引的可能；这是用户指定主实验，需统计，不能自动改 friend-first。真实 enemy distance 仅进入诊断，不得进入新选择/奖励计算。

## 预登记 Gate 与运行预算

T0 每个基线 key 逐 transition 相等（仅允许新增日志）；T1 选择/传播/衰减/截断/失活；T2 固定历史策略动作轨迹重算与全账检查通过后才训练。warm-start 共同 Stage1 selected step_1300000 SHA256 d35b7984d8ca03fa7b536811794578ff402e9b8ad48589e3318f2cb1077d702b；fresh optimizer/replay。train seed 2026101601；25k/50k/100k checkpoints；初始 bounded launch 为 25k，再看 matched 50 episode/scene/mode 评价与资源决定续训。最大 warm-start 100k，扩至 200k 需另记原因。argmax 和 sample 分报，固定 seed 和初始场景。捕获下降 >.10 或碰撞上升 >.10 为暂停扩展的工程告警；不是统计非劣性检验。足够有效 support 才能通过可评价性 Gate。未通过不进 Stage2/3；scratch 不能用25k/50k无捕获判失败，原 Stage1 学习尺度2M。

资源初查：GPU0/1 A6000 无 compute PID，root free 约88GB；不抢占任务。历史 selected/checkpoint/evaluation 全部只读。Git push 仅在核实当前分支有效授权后执行，否则本地提交并登记待发布。
