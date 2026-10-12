# BATCH01 migration audit

日期：2026-10-11，Asia/Shanghai。六条线全部完成既有授权预算和最后一个 screen，训练进程均已退出。结果全部保留为 **PROVISIONAL_LONG**。本轮没有训练、追加预算、selection/final 评估或 checkpoint promotion。

## HEAD、branch、BASE

当前仓库主目录 `../cocap-voradj` 是 `ladder/implementation-20260807`，HEAD `bd635499a690d2d003ae0863a9283115a124a76c`，含大量历史未跟踪 artifacts。它不是 Batch01 的迁移 BASE。

中央管理工作树是 `ac-master-dag-20260921`，branch `ops/ac-master-dag-20260921`，本轮开始 HEAD `c0dbbd4b0a2ccb054bf4fa74e0429e77363f3862`。进入时已有三个运行状态 JSON 的未提交更新，保留并按实际进程/screen 刷新。

迁移 Core branch `experiment/terl-mappo-batch01-v3r2-20261009`，交付 HEAD `d6e201052f0eac319342c35cd4efd9171df3fd54`；科学 candidate `9cc2d47c532c76239a61a0f6a19c8603358c92bf`；canonical lock `384b9ba587b905b879a08f01fa1073ca470bdc0a836459067d193d97b03a84ab`；原 T0 parent `bb794ca8435f06b9fa5693c0fed98f567320decf`。

BASE 当前为 `CORE_V3_SELFTEST_PASS / QA_PENDING / BASE_FREEZE_BLOCKED`，`BATCH01_BASE_SHA=null`。已完成 CPU87、CUDA3 和复用的 A3 脚本探针属于原 Commander 自测；同 SHA/lock 的独立签字仍缺。最新独立 QA HEAD `c20b632841485cd6fa1de4a2d0ef0e3dce96364d` 审的是 V2，不能替代 V3r2 验证。此次代码审计和 CPU fixture 也不构成独立 QA。

## 六条线最终状态

下表的 normal/collision 是末点 screen 的 deterministic / sample。C0 deterministic 为连续动作 mean，其余为 argmax。末点每模式20局，两个模式共享物理 seed，不能当作40个独立初态。

| ARM | branch 后缀 / HEAD | step / update | normal | collision | 状态 |
|---|---|---:|---|---|---|
| P1-control | batch01-p1-control-long-v3r2-20261010 / `0e5b03b` | 1M / 3920 | 40% / 45% | 45% / 40% | PROVISIONAL_LONG_COMPLETE |
| P1-treatment | batch01-p1-treatment-long-v3r2-20261010 / `5b53c3f` | 1M / 3920 | 95% / 70% | 5% / 10% | PROVISIONAL_LONG_COMPLETE |
| R1 | batch01-r1-long-v3r2-20261010 / `3ce4a20` | 1M / 3920 | 10% / 50% | 90% / 45% | PROVISIONAL_LONG_COMPLETE |
| N1 | batch01-n1-long-v3r2-20261010 / `d9957a5` | 1M / 3920 | 75% / 75% | 25% / 20% | PROVISIONAL_LONG_COMPLETE |
| T1 | batch01-t1-long-v3r2-20261010 / `e963c72` | Stage2 100k / 392 | 85% / 55% | 15% / 40% | PROVISIONAL_LONG_COMPLETE |
| C0 | batch01-c0-long-v3r2-20261010 / `64a6579` | 1M / 3920 | 40% / 40% | 10% / 15% | PROVISIONAL_LONG_COMPLETE |

R1 1M screen 在 2026-10-11 00:41 已完成，原 00:17 快照仍标 pending。本轮读取原件后刷新 registry、状态报告、DAG 的 engineering 状态，并归档末点原件；没有重跑评估。完整 HEAD、run_id、路径、checkpoint hash、配置和 source hash 见[审计 JSON](../../artifacts/2026-10-11_cocap_mappo_batch02/batch01_audit.json)与[当前快照](../../artifacts/2026-10-09_terl_mappo_batch01/commander_v3r2/status_20261011/snapshot.json)。

## 实际代码 delta

先执行六 ARM 对 candidate 的 `git diff -- src configs tools`，再核对每个运行 manifest 的 source_changes 和实际文件 SHA；没有根据旧文档猜路径。六个末点 checkpoint、sidecar step/hash 均匹配，所有 best-observed checkpoint 仍在本机且 hash 匹配。共同 Core 文件没有被 ARM 修改。全部原始 diff 在[diffs](../../artifacts/2026-10-11_cocap_mappo_batch02/diffs/)。以下代码路径均相对于对应 ARM worktree。

| ARM | 真实代码/config | 科学变化 | 初始化与限制 |
|---|---|---|---|
| P1 | `configs/experiments/terl_mappo_batch01_20261009/p1-{control,treatment}_delta.json`；`src/terl_mappo/batch01/arms/p1.py`；`runner.py` | 唯一 PPO 参数差异 `target_kl: .02 → .01`；control=.02 | 同一个775k完整 actor/V/两个Adam/ValueNorm/env/APF/RNG/counters 配对分叉；不是以旧 T0 1M 作 control |
| R1 | `src/terl_mappo/batch01/arms/r1.py`；`r1_delta.json`；`runner.py`/`evaluator.py` 调用 reward hook | `r_shaped = r_native - r_native_distance + 2 clip(d_pre(goal_pre)-d_post(goal_pre), -3, 3)`；目标为 pre-state 最近活跃敌人，固定该步环目标 | scratch actor/V/Adam/ValueNorm；仍保留 TERL time/global/emergency/collision/goal，done/物理不变；不是 Full CoCap Reward |
| N1 | `src/terl_mappo/batch01/arms/n1.py`；`n1_delta.json` | actor 换为 `LegacyVorAdjFeatureBackbone` 256/8/4 + linear categorical9；原 TERL actor 特征结构被替换 | self4、friend5×7、enemy8×7、obstacle5×5、19 slots、mask/type/order 均为 native；role late fusion 采用常数0 fallback；critic=T0。没有 VorAdj/CE centroid/Z/额外敌方信息 |
| C0 | `src/terl_mappo/batch01/arms/c0.py`；`secondary_contracts.py`；`c0_delta.json` | categorical9 → tanh Gaussian physical AW；`a∈[-.4,.4]`、`w∈[-π/6,π/6]`；joint logprob 包含物理缩放/tanh Jacobian，保存 pre-tanh latent | `PhysicalActionList` 传递真实二维物理命令，无 AW9 量化；native physics/reward/critic/PPO 保留；log_std 初始-.5，clamp[-5,1]。连续 differential entropy 单独解释 |
| T1 | `src/terl_mappo/batch01/arms/t1.py`；`secondary_contracts.py`；`t1_delta.json` | Stage1 3P1E0O4C → Stage2 4P1E1O6C；实现但未训练 Stage3 7P2E2O8C；P/E spawn 最小间距10→13→15 | actor775k warm transfer；critic/Adam/ValueNorm/env/RNG/counters 全新。固定7 actor槽和7P/2E/2O/8C critic容量，current槽含 presence mask；不是跨规模 exact resume |

共用新增工程模块为 `arms/runner.py`、`evaluator.py`、`preflight.py`、`long_budget.py`、`long_preflight.py`；T1/C0 还有 `secondary_contracts.py`。这些模块负责声明 delta、source pin、checkpoint、预算和动作后端消费，不是额外 task 定义。完整文件和 hash 按 ARM 列在机器审计，不能用相同文件名推定相同字节。

PPO 不变项由运行 manifest 核对：gamma=.99、GAE λ=.95、clip=.2、epochs=3、minibatches=2、actor_lr=3e-5、critic_lr=1e-4、entropy_coef=.01、value_coef=1、max_grad_norm=.5、ValueNorm=true（beta=.99999、epsilon=1e-5）、rollout=256。只有 P1-treatment 改 target_kl；本轮不把它自动推广为 Batch02 默认。

R1 的环构造仍是独立诊断设计：inner surface4.5、preferred center8、outer center10.5、cell1.5、radial sigma2、角权重 alpha.25、速度门.3→1、phase width π/6。其按角度排除友机候选的方法不是成功 IQN 中 `cartesian_disk` occupancy 的完整原始实现。R2 必须恢复成功 IQN 的 `_ring_ms_target/_ring_importance_ms_reward`，不能把 R1 的环函数当作 CoCap 合同。

## best、last、training curve

best-observed 是 screen 描述性最大 normal，再最小 collision，再最早 step；不是 selection 选出的正式 best。常规 n=10、milestone n=20，反复观测有选择偏差。所有 best 与 last 路径/hash 已保存在审计 JSON；没有创建 selected receipt。

| ARM | deterministic best step / normal / collision | sample best step / normal / collision | 保持情况 |
|---|---|---|---|
| P1-control | 775k / 90% / 10% | 775k / 90% / 10% | 末点40/45%，配对继续训练期间明显退化 |
| P1-treatment | 850k / 100% / 0% | 950k / 100% / 0% | 末点95/70%，sample仍有回落；单状态分叉支持后期稳定性线索，未证明跨seed优越性 |
| R1 | 325k / 90% / 10% | 750k / 95% / 5% | 频繁大幅振荡；900k两模式 normal=0、collision=100%，925k回到10/70% normal，末点10/50%仍明显低于峰值 |
| N1 | 325k / 100% / 0% | 725k / 100% / 0% | 可学习但非稳定收敛，后期多次回落；末点75/75% |
| T1 | 0 / 85% / 15% | 25k / 70% / 20% | warm actor起点已强；100k没有证明迁移训练提高 deterministic，sample下降。100k是review窗口，不能称规模迁移彻底失败 |
| C0 | mean750k / 100% / 0% | 750k / 70% / 15% | 连续动作学到捕获，但高峰未保持；末点40/40%。不能把mean峰值等同sample能力或正式 CoCap 连续控制成功 |

![六线完整 screen 曲线](../../artifacts/2026-10-11_cocap_mappo_batch02/all_screen_curves.png)

training curve 同时保留 actor/value loss、entropy、KL、clip、gradient、ValueNorm相关原记录；[training CSV](../../artifacts/2026-10-11_cocap_mappo_batch02/training_curves.csv)与[screen CSV](../../artifacts/2026-10-11_cocap_mappo_batch02/screen_curves.csv)分开。P1只分析775k后的配对续训，其余含合法 V3r2 pilot和长线。严格资源恢复会重做未保存尾段：用于绘图的同step记录保留最后一条，原日志/回滚回执不删。

前/后100个记录窗口的平均 categorical entropy：P1-control 1.151→1.108；treatment 1.226→1.202；R1 2.114→1.251；N1 2.192→1.241；T1 1.161→1.371。对应 explained variance：control -.060→-.512、treatment -.152→-.245、R1 .304→.458、N1 .405→-.233、T1 .115→.136。critic指标和低entropy不等于任务成功，P1/N1也有价值拟合局限。C0使用 physical differential entropy，末次-2.553并非错误；末次基础Gaussian entropy=.601、tanh saturation=.00794，不能凭负entropy判不收敛或和categorical作横比。

## R1 最终 screen 与诊断

1M argmax：2/20 normal，18/20 collision；20/20见过ring2、19/20见过ring3，但仅2/20达到strict geometry。17局包含 P-P collision、1局包含 P-E collision。sample：10/20 normal、9/20 collision，其中8局 P-P、1局 P-E。接近敌人、聚集人数与完整捕获能力明显分离。

末点 argmax 成功局 capture time 平均215.5s（仅2局）；sample159.15s（仅10局）。其余删失/失败，不能根据幸存成功局的时间声称效率提高。

| R1 training 指标 | 初始100记录 | 最后100记录 |
|---|---:|---:|
| 平均 shaped reward / agent decision | -1.063 | 3.916 |
| 平均 raw native reward | -.033 | 7.134 |
| 保留的 native global 分量 | -.247 | 3.726 |
| 新 ring shaping 分量 | .381 | .895 |
| entropy | 2.114 | 1.251 |
| post-update KL mean / max | .00290 / .01639 | .01201 / .03747 |
| actor gradient norm mean | .840 | 2.319 |
| value_loss mean | .0733 | .0709 |
| KL early-stop 记录占比 | 0% | 18% |

![R1训练诊断](../../artifacts/2026-10-11_cocap_mappo_batch02/r1_training_diagnostics.png)

可确认：reward上升而最终 normal下降，末点以P-P碰撞为主，更新幅度/KL后期增大；训练/评估都走同一个 R1 reward hook，raw/shaped分量可重构。R1仅有早期资源回滚（85,752→75,000，丢弃10,752未保存decision），不是1M之前的临时换源。

诊断推测：仍存在较强 native global驻留奖励，叠加几何shaping可能鼓励近敌聚集却未保证安全角分布；后期PPO漂移可能同时影响保持。这些不是因果归因，不能据此改reward/减系数/改KL。缺少独立seed、正式selection/final及行为反事实验证。本轮停止于诊断，R2继续按成功IQN原奖励恢复；R1不作为其前置成功门槛。

## 能力边界

已正式有历史 T0 证据支持 TERL 场景下 MAPPO 可训练；Batch01 进一步提供 PROVISIONAL 支持：continuous physical AW 可学到捕获、CoCap-style backbone 可在不增加native信息时训练、配对较低target_kl在此续训样本末点保持较好。

仍未证明：MAPPO 在 CoCap V2 environment、完整 CR-MS/support/safety reward、完整 VorAdj/NormSense/Z observation，以及 capture→recovery→CE 全生命周期可稳定训练。N1/C0的高峰不是全任务证据；R1/T1都没有证明末点稳定迁移成功。BASE也仍未取得独立验收。

下一步仅为[环境合同](TERL_COCAP_ENVIRONMENT_CONTRACT_20261011_ZH.md)、[E1实施计划](BATCH02_E1_IMPLEMENTATION_PLAN_20261011_ZH.md)和[科学选择](BATCH02_SCIENTIFIC_CHOICES_20261011_ZH.md)。
