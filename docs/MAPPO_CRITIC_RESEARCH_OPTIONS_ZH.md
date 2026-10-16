# CoCap MAPPO Critic 研究选项与创新性风险（2026-10-15）

## 当前critic与证据边界

Batch02使用action-free centralized V：训练时读取central state，部署时每个pursuer actor仍以局部CoCap observation独立执行。实现沿用TERL CentralValueNetwork并适配输入尺寸；V3没有修改critic。它是合理的MAPPO/CTDE baseline，但不能仅凭“centralized”断言已有足够协调credit。

现有证据：

1. canonical BC actor能完成CoCap full task，scratch MAPPO未成功；起点与数据分布不同，不能归因于critic。
2. frozen BC P2发现early-recovery value正偏差；P3 advantage sign大体一致，pure coverage early value error也高。
3. 既有critic审计中，central CQ/V的held-out realized-policy MC fit弱于local LQ/NQ。
4. IQN teacher-Q与action ranking可作参照但不是真值；counterfactual rollout受碰撞、失败分支污染，尚无ranking-qualified critic。
5. 双Value Head未改善recovery profile calibration，已有停止加head的结果。

Critic identifiability held-out具体值：LQ RMSE34.835 / EV.593，NQ32.592 / .643，CQ42.393 / .398，V43.277 / .375；这是Final IQN/BC审计bank上的监督拟合，不是Batch02同一PPO训练因果比较。shared-trunk 16-minibatch gradient audit中，joint raw negative-gradient一阶方向提高phase loss的次数为pre-capture 0/16、post-capture 6/16、pure coverage 7/16。该结果支持post-capture conflict线索，不证明它导致策略退化。

## 七个问题

| 问题 | 判断 | 依据 |
|---|---|---|
| 1. action-free centralized V缺少关键协调信息吗？ | **UNKNOWN** | Action-free V不输出action-specific Q，但PPO仍可用state baseline估计advantage。没有同actor/同数据、仅替换critic输入的因果对照。 |
| 2. Actor局部obs与Critic全局state语义错配吗？ | **SUPPORTED as risk；causality UNKNOWN** | actor接收focal self/neighbor/target/obstacle tokens，critic接收集中状态；role、可见性、phase与时间上下文可能不对齐。中央拟合较差只是间接证据。 |
| 3. Capture/Search/Recovery要条件化value吗？ | **HYPOTHESIS** | early recovery bias明显但样本小；coverage early RMSE相似。可能与phase、clock、样本量或输入有关。双头失败不支持简单多head。 |
| 4. Credit/Advantage而非表示/探索是主因吗？ | **尚无证据** | P3 sign一致约91.3%；成功BC表明actor可以表达强策略，scratch N3没有收益；BC-PPO的CE下降不能定位critic。 |
| 5. Topology/role conditioned/value decomposition值得吗？ | **有动机，证据不足以开训练** | 必须证明比local和简洁central baseline更好并定位增量。图网络、attention或额外head本身不是新机制。 |
| 6. IQN teacher Q-ranking能检验credit吗？ | **可以作诊断，不是地面真值** | 可比较critic偏好与teacher高Q动作的排序；按Q-gap与phase分层。teacher Q不等于当前MAPPO team advantage。 |
| 7. 新critic能改善分布式actor的capture/collision/time吗？ | **UNKNOWN** | 校准或ranking改善不保证PPO行为改善；需要matched multi-seed完整任务结果。 |

## 方向对比与论文新颖性风险

| 方向 | 可检验增益 | 风险/约束 | 判断 |
|---|---|---|---|
| 加大V/attention trunk | joint state拟合更准 | central V历史上不如local fit；容量变大不是新机制 | 不建议 |
| 更多phase Value Heads | phase return分开拟合 | 双头实验已失败，不能无新证据扩head数 | 不重复 |
| Actor-Critic语义对齐 | V共享focal role、可见集/邻接和phase字段 | 要匹配参数量和信息量，才能归因于对齐 | 未来可作一次输入诊断 |
| Topology/role conditioned V | 利用邻接和agent角色生成baseline | graph attention已有近邻；标签不等于新机制 | 先证明具体credit failure |
| Counterfactual/value decomposition | 从team return导出individual advantage | COMA、DOP、QMIX、QPLEX是直接近邻；现有AW9 branch有collision污染 | 暂不提算法稿 |
| Teacher Q-ranking audit | 用成功IQN偏好评估critic排序 | teacher动作差异多为near-tie，Q不是当前策略的真实advantage | 适合作offline诊断 |

相关近邻包括 [MAPPO](https://arxiv.org/abs/2103.01955) 的centralized training/decentralized execution，[COMA](https://arxiv.org/abs/1705.08926) 的central action-value与counterfactual baseline，[QMIX](https://arxiv.org/abs/1803.11485) 的monotonic value factorization，[QPLEX](https://arxiv.org/abs/2008.01062) 的duplex dueling decomposition，[DOP](https://openreview.net/forum?id=6FqKiVAdI3Y) value-decomposition actor-critic，以及 [Actor-Attention-Critic](https://proceedings.mlr.press/v97/iqbal19a.html)。因此，仅更名为topology-aware/role-conditioned、增加attention或多head不足以作为论文贡献。

若将来需要新critic机制，待验证的候选是：**针对capture→recovery阶段，把central team credit与每个分布式actor实际可执行的局部动作集合对齐，并用成功IQN的相对Q偏好和matched counterfactual rollout校验。**还需明确它与COMA/value decomposition机制差异；当前不能宣称创新成立。

## 未来重开时只先做一项离线诊断

当前建议STOP/ARCHIVE，不执行训练。若IQN-Z主线未来需要policy-gradient baseline，信息量最大的是固定数据上的phase-conditioned value calibration + Q-ranking audit：

- **可证伪假设：**现有V误差与AW9排序错误集中在capture transition/early recovery；仅增加canonical phase与recovery clock输入，可同时改善held-out MC偏差/RMSE和高Q-gap teacher pairwise ranking。若错误不集中或排序不改善，则假说失败。
- **唯一变量：**central V是否可见已有phase和recovery-clock字段；模型宽度、参数预算、loss、target、数据与actor固定。
- **matched control/parent：**现有central V输入；同一个canonical BC actor SHA 7916a970…bf2cd、同一held-out bank与Final contract，actor冻结。已有MC-fitted V作为cold/fitted参照，不跑PPO。
- **预算/早期点：**0 online training；复用bank，最多20k train / 5k held-out focal-agent rows。5k检查phase覆盖，10k看train/heldout gap，20k封闭分析。用rows/episodes取代在线5k/10k/25k步骤。
- **指标：**phase-stratified held-out MC bias/RMSE/EV，高Q-gap pairwise ranking；次级看matched branch return、collision分支比例和mission time。critic离线改善不算capture改善。
- **继续/停止：**early recovery RMSE至少改善10%、bias绝对值降低，且高Q-gap排序在至少两个phase同向改善，才讨论下一层；否则停止MAPPO critic支线。
- **对IQN-Z主论文价值：**低。它首先解释成功IQN/BC actor为何可能被PPO更新损害；只有后续变成安全、更快且跨seed的actor增益，才可能进入主论文对照或后续方法稿。

这是一项未来重开门槛，不是当前实验授权。不得用它绕过双Value Head的失败结果。
