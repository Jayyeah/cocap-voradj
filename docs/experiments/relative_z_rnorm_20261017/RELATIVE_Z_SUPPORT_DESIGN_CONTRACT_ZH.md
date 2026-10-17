# Relative-Z Support完整科学合同

本轮只有两项科学改动：非Capture机器人的Relative-Z Support资格，以及地图尺度L→真实世界边长R（含保持原物理成功条件的阈值换算）。旧实验归档为legacy-gated-zsupport-rnorm-diagonal，新Stage1必须全新scratch。旧checkpoint仅允许历史/消融分析。

令动作前真正进入policy observation的有效友机集合为N_i^obs(t)，包含mask有效且经过实际排序和token cap截断的友机；不使用隐藏友机、敌人位置、目标ID或全局分配器。z_{i,max}=max_{j in N_i^obs} z_j，无有效邻居时为0。原Capture判定保留并具有最高优先级；其余机器人仅在z_{i,max}>z_i+TIE_EPS时为Support，否则Coverage，TIE_EPS=1e-6。任何合法非零友机Z都可触发；不能要求Z=1、Z>=0.5、直接检测者或Capture邻居。

H_i={j in N_i^obs: z_j与z_{i,max}在既有1e-6近似同值合同内}。在H_i中选择最近的单个友机j*，按实际float32 policy token中的距离判定，距离精确同值沿用几何字段字典序，完全不可区分时才内部稳定ID破平。原token排列顺序不决定选择，不取位置平均。self在1e-6内与最大值同值时保持Coverage、alpha=0。

alpha=z_{j*}。固定动作前友机位置作为锚点：r_ally=clip(||p_i^t-p_j*^t||-||p_i^{t+1}-p_j*^t||,-c_s,c_s)。r_support=alpha*r_ally+(1-alpha)*r_disp。当前正式support_reward_approach_weight=1、clip=3、static_capture_scale=1；环境实际代码仅对alpha*r_ally应用一次capture scale，因此实际公式与上式相等。若配置scale非1必须显式披露并预检失败，不能产生双倍权重。

新Support奖励不受active_targets全局布尔值限制。在目标真实失活后，Z保留自然传播/衰减；0.5秒一步，峰值从1自然经历1→0.5→0.25→0.125→0，约2秒归零。不得强制清零。已终止Episode不追加transition；若真实Recovery仍持续，非Capture机器人依据残留Z短时路由，随后回Coverage。目标终端事件、历史Z、Support路由与整局终止是不同概念。原Capture release-delay及实际捕获时的清理过程保留。

动作前快照用于本transition的reward_role与task_label；动作后更新Z并真实打包下一policy observation后生成next_task_label/task_switched。Capture归pursuing；pre-capture的Support/Coverage仍归pre_capture_cover；post_capture_real、recovery_pure按原phase优先。四类Replay配额、采样和优化不改变，不新增Replay类。

必须保留：Z同步previous-state传播；lambda=eta=.5；hard-zero .1；原Capture几何、环形候选点/权重、IW encirclement构造；AW9；normal/stationary capture事件；Transformer-IQN网络及IQN优化目标；task episode轮换与recovery初始化；四类语义Replay；原课程规模。不得引入额外网络、Gate、显式目标ID或全局最优分配器。

内部验收矩阵：self0/friends.25,.25选最近.25；self.125/friends.25,.5选.5；self.25/同值Coverage；self.5/较低Coverage；self0/friend.125 Support；无有效友机Coverage；隐藏.5不能使用；多个敌情残留以实际可观测最大Z为准；目标刚失活仍按.5局部证据Support；Z归零后非Capture回Coverage；Capture优先。不得为测试人为加入Capture邻居。

审核必须由真正独立gpt-6.1-sol、xhigh执行，独立阅读代码并核验/执行测试。A-L逐项审核：旧Capture邻居门控；policy相对Z；active_targets；角色覆盖；reward/obs/metadata/replay一致性；失活Z语义；全部地图R；正式旧L路径；物理成功标准；NaN/尖峰/PBRS；完整三阶段；论文公式一致性。未APPROVE禁止正式smoke与长训，修改后重审。报告须记录真实模型、任务ID、commit、源码行/测试证据、阻断/非阻断项及APPROVE/REQUEST_CHANGES/REJECT。
