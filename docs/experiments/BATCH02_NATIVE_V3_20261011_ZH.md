# V3 minimal CoCap-MAPPO integration

PROVISIONAL。ARM名V3，对应用户版本路线的V4最小mix，区别于Batch01旧V3和Batch02 Pure Coverage V1。

代码历史parent=7cb8bc0；组合N3实现28a3337与R3实现7cb8bc0，源树来自N2/R2长训8ce5593，但独立worktree不修改已有作业。

任务只使用成功IQN-CoCap Z05 Stage1/Forward-Final V2，resolved hash6955ecf2…：原4P1E1O mixed、小规模单目标、原spawn/physics/current/APF/collision/boundary/action/terminal/reward/observation。原mix内部捕获后completion逻辑随成功provider保留；不另启动lifecycle/repeated arrival/multi-target/coverage扩展，未加入snapshot recovery或scale curriculum。

V3 actor=成功IQN.features/voradj_single_feature，原TERL AW9 categorical head；central critic/PPO/Adam/GAE/ValueNorm和原action-free geometry完全沿用E1。R3六分类ledger只读原reward，终止不变。actor只看focal self/neighbor/target/obstacle tokens，critic为training-only集中信息。

相对TERL baseline：环境/reward/observation已由E1/N2迁移；当前新增actor representation和reward telemetry。相对成功IQN：task合同保持；quantile Q/off-policy replay/epsilon optimizer换为TERL on-policy PPO/central V/categorical policy，reward logging增加分解。不是最终论文CoCap-MAPPO，不宣布学习成功。

N3与V3在相同初始化/seed下优化语义等价，V3仅追加R3 telemetry；R3与R2亦为账本区别，新增训练不会构成新reward因果证据。依赖关系是代码组合，不是性能前置门禁。

pilot固定1024 CPU decisions，4次PPO×6minibatches；另16步continuation及16步fresh-process replay为恢复诊断。evaluator smoke只执行screen首seed/argmax+sample的原完整episode，不能用于学习判断。long各1M/25k，从fresh scratch重新开始，不resume pilot权重。

新增evaluator仅替换factory import：任务/metrics/seed domains/screen10每模式/selection20/final50/原termination与模式保持Batch02-A合同。固定评估reward clock0无training schedule callback，与成功IQN评价一致；训练原callback在200k切换原值0→0.0005。final不参与选best；保留initial/selected/last与所有25k checkpoint。

必要测试仅import/config、rollout/PPO、reward/action/terminal/obs合同、evaluator兼容、save/load。没有性能阈值、格式或额外QA门禁。此前一个独立E1 QA授权不扩用到新代码；新验证是工程证据，不宣称新的independent validation。
