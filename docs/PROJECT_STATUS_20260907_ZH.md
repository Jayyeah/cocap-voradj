# 项目宏观状态快照（2026-09-07）

> **2026-09-08 canonical main / BC parity补充：** [完整逐项合同表与奖励/信息审计](FINAL_IQN_MAIN_MAPPO_BC_CONTRACT_PARITY_20260908_ZH.md)。main@a5814f4才是Final parent；BC环境是B0/K10 corrected Pure-Capture，未迁移Final完整任务。下一条改为A同合同三策略冻结评估，B Final-transfer随后单独定义；不是直接把旧BC放进混合修改后的“Final-side”环境。

> **2026-09-08 P0 更正，优先于下文历史结论：**见 [最新合同/信息审计](P0_CONTRACT_INFORMATION_AUDIT_20260908_ZH.md)。原 PPO dropout/log-prob 合同不成立；BC 与 native PPO 实际 env seeds 相差10000，撤回 BC↔PPO 配对显著性及“显著 erosion”，warm-up未排除cold-start。旧 AC legacy_voradj 不是 Final 的半径局部敌方感知，BC100%仅在旧AC argmax capture合同有效。旧 outer-ring/首事件 latency不是新 same-target closure。support11仍为无效NOOP，AW/VXY物理预算不等价；仅使用GPU1，暂停所有旧GPU0/长训/teacher-KL路线建议。下文数字作为历史记录保留，不作为新 Gate。


## 1. 本轮边界与现场

本轮只审计、归档和汇报，不恢复训练、不修改实验配置、不启动新任务。2026-09-07 16:01 CST 现场：

- 分支 `experiment/small-step-ac-migration-20260828` 的本地/远端起点均为 `9733670df84954e3045bf23a634fe3a28f6da5fd`；
- 没有 CoCap 训练、rollout、supervisor PID 或 tmux；GPU0/1 分别约 `466/3791 MiB`，没有本项目占用；
- 主机最近启动时间为 2026-09-06 19:53 CST；VXY Stage3 的 metrics/status 最后更新时间为19:49/19:50，故中断与主机重启吻合；
- 动态 raw rollout、dataset shards、训练 checkpoint 继续留在本机，不纳入 Git。

## 2. 宏观目标坐标

| 主线 | 已完成 | 当前结论 | 当前坐标 |
| --- | --- | --- | --- |
| GPU0 IQN-VXY 性能对标 | Final-AW↔VXY Stage2/3 matched formal100完成 | VXY能完成capture+coverage，但相对AW主要缺口仍是capture/support latency、Stage3成功率和boundary safety；servo/action dynamics仍是matched证据最强的机制候选 | baseline/差距定位完成；第一条“support 1/1”因配置层级错误未真正执行，correction因果验证仍为0次有效实验 |
| GPU1 IQN→MAPPO | teacher qualification、600回合dataset、Actor distillation、pre-PPO formal100 Gate、Direct/Warm PPO及终点formal100全部完成 | BC直接复现teacher；PPO后两支均有可测capture保持损失，critic warm-up没有改善，random-critic cold start不是主要解释 | exploration问题已大幅隔离；下一研究问题转为PPO policy retention，teacher-KL只作为待决候选，不自动启动 |
| 运行状态 | 所有BC任务自然结束；VXY Stage1/2完成、Stage3中断 | 当前无后台任务 | 静态审计点，等待用户重新确定下一条有效实验 |

## 3. IQN teacher→MAPPO 最终结果

四者均在 MAPPO-9-v2 exact target capture 环境；BC、Direct、Warm formal100 使用同一 seeds `2026090301..0400`，deterministic categorical argmax。

| Actor | capture | collision | 2+ / 3+ | mean length |
| --- | ---: | ---: | ---: | ---: |
| IQN-AW teacher | `1.00` | `.01` | `.66/.14` | `73.25` |
| distilled BC（PPO=0） | `1.00` | `.03` | `.64/.12` | `71.16` |
| BC→Direct PPO 100k | `.96` | `.04` | `.87/.28` | `80.32` |
| BC→12,032 critic warm-up→PPO，total 110k | `.92` | `.08` | `.92/.27` | `72.18` |

同 seed paired bootstrap 95%区间：

- Direct−BC：capture `-4pp [-8,-1]`，collision `+1pp [-4,+6]`，2+/3+ `+23/+16pp`，length `+9.16 [-1.42,+21.56]`；
- Warm−BC：capture `-8pp [-14,-3]`，collision `+5pp [-1,+12]`，2+/3+ `+28/+15pp`，length `+1.02 [-5.33,+7.35]`；
- Warm−Direct：capture `-4pp [-10,+2]`，collision `+4pp [-2,+10]`，length `-8.14 [-19.94,+1.93]`，三项区间均跨0。

结论：

1. BC Gate成功，说明MAPPO scratch弱的重要原因确实包含“从零找不到强策略”；
2. Direct PPO仍是强策略，但capture相对BC有显著的4pp保持损失；
3. warm-up没有优于Direct，point estimate反而更低成功、更高碰撞；因此random critic cold start不是主要瓶颈；
4. 最符合预设解释的是“BC很好，PPO产生中等policy erosion，critic warm-up不能消除”。若以后继续，优先保留BC作为冻结基线、Direct作为当前最佳PPO分支，再以小规模annealed teacher KL检验retention；本轮不启动。

关键资产：

- BC Actor SHA：`bb8f971201f6e0a55af3ccb62bcae117f96afa1b07fad2c95c3da1dc552fa35e`；
- Direct终点：`ppo_branches/direct_ppo/checkpoints/step_000100000.pt`，formal full-resume SHA `74e69c115711ae6b8f686c9b81b42c69e279e9a1b18b899f5d012a522740561b`；
- Warm终点：`ppo_branches/critic_warmup_ppo/checkpoints/step_000110000.pt`，formal full-resume SHA `fce7527aee710167b3ac0ff3744d6857e394aa2b12f07c8e09a3b25564d75a87`；
- 两个 supervisor 均自然停在 `WAITING_FOR_RESULT`，formal于2026-09-04 00:09 CST完成。

## 4. VXY “support11”线的合同审计

### 4.1 因果标签无效

候选配置把权重写成：

`reward.support_reward_capture_weight=1.0`
`reward.support_reward_coverage_weight=1.0`

但环境 `VorAdjEnv._vct_ls_support_reward_weights()` 实际读取：

`voradj.support_reward_capture_weight`
`voradj.support_reward_coverage_weight`

训练器自身 `load_config` 的最终解析证明，Stage1/2实际 `voradj` 权重仍为 `.5/.5`。进一步的bitwise证据：

| checkpoint | 旧 VXY SHA | “support11” SHA |
| --- | --- | --- |
| Stage1 850k | `3df7a1c3…eef8` | `3df7a1c3…eef8` |
| Stage1 1M | `bbf0ffe1…858c7` | `bbf0ffe1…858c7` |

两组同step screening指标也逐项完全一致。因此该线必须标记为：

`INVALID_NOOP: requested support .5/.5→1/1 was not applied`

后续结果仍可作为“原 .5/.5 VXY 延长训练”资产，但不能支持或反驳support reward correction。

### 4.2 Stage1（完成，Gate FAIL后授权override）

- 1M自然完成，40个25k checkpoints/screenings；
- selected `step_850000.pt`，SHA `3df7a1c3e41f020b92a7efa7cf505d987178e012e9c201326476bcdc0b55eef8`；
- selected screen：capture `.45`、coverage CE `0`、mixed capture/CE `.65/.05`、max collision `0`；
- 原Gate：capture FAIL、coverage CE FAIL、mixed CE FAIL；mixed capture/safety/checkpoint PASS；
- formal20：capture `.20`、coverage CE `.05`、mixed capture/CE `.65/.05`、max collision `.05`；
- supervisor按用户预先授权的全课程override晋级；这不是Gate PASS。

### 4.3 Stage2（完成，作为原合同延长训练有效）

- 从上述Stage1 850k开始，1M自然完成，40个25k checkpoints/screenings；
- selected `step_950000.pt`，SHA `4d2137a9d92af9e515c27b471a6ea4f2fbc01b2b48b2761a15b884864bc7b39b`；
- selected screen：capture/coverage CE/mixed capture/mixed CE均为 `1.00/1.00/1.00/.95`，collision `0`；
- formal20：capture/coverage CE/mixed capture/mixed CE均 `1.00`，collision/boundary `0/0`，capture length `269.6`；
- 说明在原 `.5/.5` support合同下，更长Stage2训练和该lineage能得到强8v2 joint skill；不能归因于1/1。

### 4.4 Stage3（主机重启中断）

- 训练日志最后到 episode 1509 / global step `472425`；supervisor最后快照为472k；
- 18个25k checkpoint/screening完整落盘，最后持久化点为 `step_450000.pt`，SHA `bc4baaa9a670601691ac1783d93d681d00f249eb1c3acda67c8630f34adcbbd0`；
- 450k screen：capture/collision/length `.95/.05/188.9`，coverage CE `0`，mixed capture/CE/collision `.95/0/.05`；
- 中途能力明显振荡：300k为capture/coverage/mixed `1.00/.85/.80`且collision 0；400k为 `1.00/.65/.85`且collision 0；425–450k的standalone/mixed CE均跌到0；
- 训练窗口最近100回合指标在472k附近仍约capture `.97`、coverage strict `.95`、collision `.04`，但它与fixed-policy screening冲突，不能替代formal；
- 主机于2026-09-06 19:53重启，`/dev/shm/iqn_vxy_support11_1m_20260903` full-resume已丢失。只能从450k轻量权重近似续训，不能恢复optimizer/replay/env/RNG的精确状态；
- 本轮按用户要求不恢复、不finalize、不从不完整轨迹选best。

## 5. 当前应保留的结论与下一决策

有效结论：

- AW↔VXY真实性能差距已量化；servo/action dynamics仍是G0最有证据的机制候选；
- IQN→BC成功，证明强策略可被MAPPO Actor表达；
- PPO产生可测保持损失，critic warm-up没有解决，故G1已从exploration问题推进到policy-retention问题；
- 原 `.5/.5` VXY 的延长Stage2 checkpoint 950k是新的强8v2资产。

尚未完成：

- 真正的support `1/1` 单变量实验一次都没有执行；
- Stage3仅到472k且无正式selection/formal；
- annealed teacher KL未实现/未启动；
- 没有证据允许把错误配置线写成support correction成功或失败。

恢复工作前需要用户在两件事上重新选择优先级：

1. G0：修正为 `voradj.support_reward_*=1.0` 后从头重跑，或回到matched证据更强的servo最小修复；
2. G1：接受Direct PPO为当前在线学习基线，或启动小规模annealed teacher-KL retention实验。

NEXT WAKE-UP: first verify this snapshot and choose one G0 correction plus whether G1 should stop at Direct PPO or test annealed teacher KL; do not resume the interrupted mislabeled support11 Stage3 as a valid 1/1 experiment.
