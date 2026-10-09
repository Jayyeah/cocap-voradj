# TERL-MAPPO Stage1最终结果：1m精确续训完成

本次结论为 **`TERL_MAPPO_STAGE1_LEARNABLE`**。random-init、无teacher/BC的TERL-backbone MAPPO，在原生Stage1学出了重复出现的正常cooperative capture。预声明screen→held-out selection选出的775k checkpoint，在隔离final中argmax/sample均45/50正常捕获、5/50碰撞。该结论适用于本次seed9训练及原生Stage1合同；训练末尾1m点出现退化，不能据此宣称优化始终稳定或所有训练seed均成功。

## 训练与实际精确续训

TERL采用并再次核验远程HEAD：`143359b2722d49c29b4fecc0ad1fd8d46326e45a`。CoCap parent `80debaef0f5983d8128b5059c90d65acc9216c10`，corrected MAPPO lineage `25dd0f8`。独立分支 `experiment/terl-backbone-mappo-20261008`。

原100k run完成后，按用户最新授权直接恢复到累计1m，不等待gate结论。100k早期final仅两模式各1/50正常捕获、49/50碰撞，分类PARTIAL；该gate没有阻止续训。CUDA32-step完整rollout/PPO/Adam/ValueNorm/env/RNG逐位一致与真实CLI续训预检均通过。

实际resume anchor SHA256：`781c391a175e51dfabead66bac645a6efb1bc230747276768d998ccab1f9153e`。续训manifest与原manifest的config仅budget100k→1m，所有科学source hashes及runtime合同相同。parent metrics的392行原样保留；第一个续训update为100,256 decisions、300,768 active transitions、update393、2,358 paired minibatches，未重置计数、优化器或环境。

1m训练于2026-10-09 10:10:55+08完成；筛选、selection、final及自动远程同步于10:21完成。总计1,000,000 joint decisions、3,000,000 active agent transitions、3,920 rollout PPO calls、22,994 paired actor/critic minibatches（两optimizer各22,994次）。GPU0 PyTorch peak allocated约616MiB。训练/evaluator/controller已结束，无训练或评估failure工件。

## Capture、几何与碰撞

全41点表见 `TERL_MAPPO_1M_LEARNING_20261008_ZH.md`。独立审计覆盖980局screen、120局selection和100局final；没有未完成partial。以下为代表性的screen节点，argmax/sample各自独立报告，n是每模式局数。

| checkpoint | n | normal capture argmax/sample | collision argmax/sample | ring3 argmax/sample |
|---:|---:|---:|---:|---:|
| 0 | 20 | 0% / 0% | 5% / 10% | 0% / 0% |
| 100k | 20 | 5% / 0% | 95% / 100% | 5% / 0% |
| 250k | 20 | 30% / 10% | 70% / 85% | 60% / 20% |
| 500k | 20 | 95% / 65% | 5% / 20% | 100% / 100% |
| 750k | 20 | 80% / 70% | 20% / 30% | 85% / 75% |
| 775k | 10 | 100% / 100% | 0% / 0% | 100% / 100% |
| 1m | 20 | 30% / 30% | 50% / 60% | 60% / 50% |

250k起有31个screen checkpoint至少一种模式出现≥2次normal capture；不是孤立随机成功。几何从初始ring2/ring3/strict全0逐步改善，不能仅用reward上涨解释学会围捕。100k的高碰撞是早期限制，不构成MAPPO在该原生任务不可学习的结论。

预声明候选排序：两模式平均normal rate、低collision、strict、ring3、较早step；screen前三候选为775k、450k、500k。selection使用独立2036100800起的20个物理seed，两模式各20局。775k selection正常率90%/80%，高于450k的95%/65% pooled结果和500k的100%/55% pooled结果，按既定排序选775k。final从2046101800起，未参与selection；与screen、selection以及旧100k final seed域均隔离。

| final模式 | normal/capture | collision | ring2/ring3/strict | 成功时间mean/median/p90（秒） | 成功n/删失n |
|---|---:|---:|---:|---:|---:|
| argmax | 45/50 / 45/50 | 5/50 | 46/46/45 | 90.69 / 70.50 / 160.50 | 45 / 5 |
| sample | 45/50 / 45/50 | 5/50 | 49/46/45 | 125.78 / 93.00 / 227.10 | 45 / 5 |

argmax失败为5局P-E；sample为3局P-E、2局P-P。两模式均0局纯time-limit censor，所有未capture都是collision failure。保留evaluator“所有未capture均censored”的原汇总口径，同时在审计JSON分别列出collision failure与time-limit censor。正常捕获逐局验证为capture且无同时碰撞，并出现原生strict encirclement。argmax/sample配对相同50个物理初态，不能将100局视为100个独立物理seed；两种动作模式的采样流独立。

## 最佳checkpoint与保留

best是775k，文件：`runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/best.pt`，hardlink指向 `step_000775000.pt`。

SHA256：`590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4`。

best alias、selected、final模型hash一致，当前实际文件重新hash通过。latest及所有保留checkpoint的actor/V参数均finite、source hashes一致。只保留latest、100/250/500/750/1000k milestones、screen前三候选及best alias；30个其它本自动线checkpoint在完整screen/hash验证后按ledger清理。删除记录、metadata hash与evaluation hash互相匹配；parent及其它实验路径未清理。模型只保留服务器，轻量metrics/episodes/evaluations/retention与图已归档云端。

## 数值健康与剩余问题

**Reward。** 保持公开time/distance/global/emergency/collision/goal分量；每step原reward重构assert以及每episode总和重算均通过。初始argmax global负值给出约98.4%越界agent时间占比下界；原boundary不是collision/terminal。到1m最后rollout reward mean≈8.93，但screen normal只有30%，说明高reward不是capture学习的充分证据。Stage1内每agent处于8m内且3人满足related18m时，普通reward可达-1+5+5=9；在可无限维持该状态的理想反事实下，gamma.99的折现驻留收益9/(1-.99)=900，超过3人等角capture一次120π≈377。实际维持能力和风险未受控验证，不能把这项潜在激励矛盾直接写成末尾退化的因果结论；未修改shaping。

**PPO。** 全3,920行完整指标finite，更新后ratio、KL、clip、梯度和ValueNorm均已记录。target KL是minibatch监控的提前停更新条件，PPO clip也不是ratio硬上限；末尾post-update KL≈.0273、ratio .412–2.160、clip31.35%必须保留，不能以“没有NaN”代替稳定性。KL提前停止比例从0–100k的0%升到775k–1m的13.15%，该guard减少了更新次数，没有停止1m训练。

**Critic。** rollout EV中位数随阶段约.881、.322、.050、-.0058、-.043（0–100k、100–250k、250–500k、500–775k、775k–1m）。晚期EV较差/波动，最小约-11.93，虽value loss与参数finite，仍有校准问题需要诊断。低target variance也可能放大负EV；没有保存逐batch return variance，无法将其单独归因于representation或认定它导致性能退化。

**Exploration。** rollout entropy中位数从早期2.164下降至晚期1.052（AW9最大log9≈2.197）；best final entropy argmax/sample≈1.126/1.080，sample仍90%正常捕获，成功不是只依赖argmax。末尾更低entropy与性能下降同现，但没有探索消融，不作因果排序。

**Representation。** 保留原TERL entity/type/Transformer/self+unmasked-max/Target-Selection/fusion，正式256/8/4为真实入口值；必要finite mask修复已验证，未修padding/APF/current的公开有效行为。现有结果证明本次表示可支持学习，不能证明它优于CoCap backbone；第三组尚未实现/训练。

**Safety。** best final仍有10% collision，P-E为主要类型。原boundary与global cooperation合并，positive global只能使越界占比下界变为0，不能证明没有越界。未记录直接boundary计数，不把低collision或positive global写成全域安全保证。公开APF过滤索引、duplicate-current逻辑、padding设计意图及后续多目标dense重复计算仍按原合同保留/标记UNRESOLVED。

分阶段完整min/median/mean/max和每checkpoint同一步数PPO指标见 `completion_audit_20261009.json`，图见 `training_health_1m.png` / `learning_curves.png`。2250个训练完成episode中capture846、collision1326，是整条探索训练历史；不并入独立评估成功率，训练ring字段仅为terminal-step snapshot。

## 迁移、测试与后续决定

原生合同是3P1E0obs4currents、120×120、AW9、.05×10=.5秒、capture8m、related18m、native末步collision及3001-step horizon。保留真实global分支、驻留dense reward、goal信用和完整enemy token；与CoCap的CR-MS/Z/NormSense/VorAdj/swept碰撞是不同任务。不能直接用历史M-CAP/M-COV成功率判断算法优劣。

迁移完成categorical9替代IQN quantile/cosine/Q后端、参数共享decentralized actor与action-free central V、corrected PPO/GAE/ValueNorm/active loss、独立优化器、严格runtime/RNG checkpoint。必要修改包括finite空mask/no-target、防dropout似然漂移、原trainer episode生命周期产生正确terminal/truncation、pre-reset V bootstrap；actor不读取central state，critic顺序等变。

43项基础correctness tests已通过，实际256/8/4 CPU/CUDA smoke及额外完整CUDA budget-resume/CLI/retention预检均通过。完成审计在源码完全相同的基础上重算全部1200局评估、3920条训练记录、选择规则与保留模型hash，所有检查通过。未因docs收尾重复GPU测试或再启动训练。

建议下一步先在相同MarineEnv/奖励/信息/evaluation协议下复核多个训练seed，并做TERL-IQN original / TERL-backbone MAPPO / CoCap-backbone MAPPO的同任务对照。另独立诊断驻留激励、critic校准与late-update稳定性；任何shaping/pooling改变均应是新版本。当前1m目标已完成；后续完整7M课程需要新的Stage1稳定性/预算判断，不由本轮自动启动。

中央worktree当前HEAD `381b61dba6c94478ed23b712d39018fc70e2341f`（只读核验，clean），handoff更新在本分支，不写中央DAG/state。自动结果已在 `5eeb3c855351b9c4867a3320280ef4935797e576` push并核验；独立closeout审计及本报告另行提交/核验remote HEAD。

主要交付：[Migration Contract](TERL_MAPPO_MIGRATION_CONTRACT_20261008_ZH.md)、[Implementation](TERL_MAPPO_IMPLEMENTATION_20261008_ZH.md)、[1m完整Learning table](TERL_MAPPO_1M_LEARNING_20261008_ZH.md)、本Final report/Follow-up、[Requirement Audit](TERL_MAPPO_REQUIREMENT_AUDIT_20261008_ZH.md)、[Central Handoff](TERL_MAPPO_CENTRAL_HANDOFF_20261008_ZH.md)及[完整独立审计](../artifacts/2026-10-08_terl_mappo_1m/completion_audit_20261009.json)。最佳checkpoint与最终分类已从PENDING转为上述实证结果。
