# 需要用户确认的科学选择

以下表格是第一阶段的选择记录。**最新授权：完成 E1 learner 闭环，合同冲突以成功 CoCap 为准。** 本轮沿用成功 V2 Z05 Stage1、categorical AW9、原 T0 PPO，只做 fresh-model CPU smoke；没有改 reward/observation 设计。用户已明确允许所需输入维度/容量绑定。[实际结果](BATCH02_E1_BRIDGE_REPORT_20261011_ZH.md) 优先于表格中的历史 E1 范围。长期训练的初始化、reward clock 和 recovery 采样分布仍需在具体实验中确定。

| 选择 | 建议 | 另一条可行路径及影响 |
|---|---|---|
| N2的成功observation参考 | 直接恢复 **V2成功Z05完整合同**：self9/friend7末列Z、physical_only排序、λ=η=.5、floor=.10；M2审计已恢复Z的传播/释放/lifecycle | N2先恢复 **B0成功roleful合同**，M2再切Z。B0成功证据是R20 sensing；roleful+NormSense V2组合应标尚未验证，不能借用失败MAPPO同形状就当成功合同 |
| E1范围与首个scene | 首次E1为offline完整provider bridge，4P1E1O成功Stage1 mixed/capture/coverage各有明确profile，返回原CoCap obs/reward但不接optimizer | 若要求E1后立即训练“只改physics”的ARM，必须先定义reward/obs反事实baseline；不自动提供混合native19/global敌人与CoCap物理的临时任务 |
| reward clock与初始化 | 首个新CoCap scratch run采用成功IQN新任务clock0→200k的speed_weight schedule；evalclock独立明示。E1验证不需要checkpoint | 如warm actor用T0/P1/N1权重，明确仅weights transfer、fresh V/Adam/ValueNorm/env/RNG，另确认CoCap clock从0还是继承计数；不得称跨结构full-state exact resume，或自动继承失败MAPPO固定2M offset |
| 目标动作接口 | 保留成功CoCap的categorical AW9；C0 continuous保持独立诊断 | 若选continuous作为CoCap目标，需显式登记MAPPO adaptation及单独对照；C0仅证明native TERL场景可学，不证明CoCap全任务 |
| PPO默认 | 沿用T0已登记target_kl=.02和其它PPO参数；P1-treatment=.01暂为描述性稳定性线索 | 若用.01作Batch02默认，需明确选择并保留相同env/obs/reward的对照；不能把单分叉末点较好视为普适最优 |
| PPO recovery训练分布 | 恢复成功IQN的task reset/recovery合同，保持capture snapshot pool及其已登记采样比例；PPO使用on-policy rollout | 若简化为只从普通mixed reset开始，须标MAPPO adaptation/未验证设计。IQN replay 64/16/32/16是off-policy批构成，不能直接当作新的PPO task概率 |

**E1生产实施和有限 PPO 检查已授权并完成；后续独立 ARM 与长期训练未在本轮启动。** E1 工程接入采用成功 Z05 全 observation，不增加 roleful+V2 混合变量；以后如提出偏离成功合同的变更再明示。初始化、训练 clock 和 recovery 训练分布留到后续训练设计，不要求一次全部决定，也不作为 E1 离线工作的额外前置。Batch02长期预算、多seed、额外任务变量均保持未授权。

当前无需决定新的reward系数、sensor k、collision模型、CE阈值、APF参数或任务成功定义：优先照成功IQN恢复。若后续提出为了MAPPO训练方便的变化，一律独立标 **MAPPO adaptation**；对尚无证据的新组合标 **尚未验证的新设计**。

现有BASE独立QA仍pending是已读取的事实，不提出更严格的新科学门槛。本轮工程检查不能代替独立验证，也不要求先解决全部历史实验失败才能推进无依赖的离线工作。
