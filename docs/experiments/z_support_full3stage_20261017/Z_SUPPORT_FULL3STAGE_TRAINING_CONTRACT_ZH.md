# Z-Guided Support 完整三阶段训练合同

本轮用户已授权：最终验收为PASS或工程阻断已解决的CONDITIONAL_PASS后，自动执行全部课程；阶段衔接、独立评价、轨迹、事件与GIF无需重复确认。当前任务覆盖AGENTS.md中2026-09-15旧任务的预算和路线限制；保留其独立工作树、不得干扰其他任务、状态真实及Git可追溯要求。

## 冻结的研究变量

源分支`experiment/iqn-z-guided-support-20261016`启动fetch后确认HEAD为`4d7005d2bc80c21d1798b04ba711d258be2b8956`。执行分支`experiment/iqn-z-guided-support-full3stage-20261017`，独立工作树`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017`。原训练事实源`f0c0dfca8a508233f30ddbb5158699400d92a00f`。

相对每个原Z05阶段，YAML仅允许：新的run seed、run_name、series_label和`voradj.z_guided_support_enabled=true`。运行时另填独立output_root、device及本线跨阶段selected路径。逐阶段完整resolved与差异JSON随代码登记。没有改环境、网络、动作、reward门控、capture或CE的数学定义；增加的代码负责保存、排队、独立调用原评估器和登记。

| 阶段 | 原生Mixed规模 | 环境决策预算 | run seed | 初始化 |
|---|---|---:|---:|---|
| Stage1 | 4P1E1O | 2,000,000 | 2026101701 | scratch；不加载pilot或任何历史权重 |
| Stage2 | 8P2E2O | 700,000 | 2026101702 | 本线Stage1 selected模型权重；新建optimizer/replay/env/Z/RNG |
| Stage3 | 12P3E3O | 700,000 | 2026101703 | 本线Stage2 selected模型权重；新建optimizer/replay/env/Z/RNG |

Coverage任务仍为各阶段原配置的0E，原mixed/coverage episode交替及recovery reset pool不变。跨阶段保持`shape_compatible`模型传递和原fresh-stage Gate；同阶段重启使用完整状态exact resume，二者严格区分。

## 原Z05训练合同

Transformer-IQN单共享策略，hidden256、8 heads、4 layers，self9、friend7、最多8友机/8目标/5障碍，AW9。无独立Gate、MoE、集中式Critic或角色策略网。gamma=.99，batch128，train_freq=4，train/target quantiles=8/8、action32，target_update_freq=10,000，gradient_clip_norm=.5，Huber kappa1；保留原优化器和replay容量及类别。

Stage1 epsilon .6→.05/500k；LR 1e-4，200k后3e-5。Stage2/3 epsilon .28→.05/420k，LR3e-5。保留原CE控制权重schedule。语义batch计数为pursuing64、pre_capture_cover16、post_capture_real32、recovery_pure16；类别最低3000及原post不足时采样替代逻辑不变。早期填充可能产生0次更新，不能把环境决策数误作优化次数。

Z同步邻居传播、lambda=eta=.5、hard-zero .1；原捕获奖励、环形点、CE/PBRS、安全碰撞、stationary fallback、目标失活和post-capture规则全部继承。决策dt=.5s，物理dt=.05s，synchronized_swept_v1。没有缩短episode horizon、降低capture或CE门槛。

Support只在旧角色条件下，用动作前self和真正进入policy observation的有效friend tokens选最高Z；self在1e-6内相等优先，alpha0；否则最高Z友机中最近，距离同值按实际token可观测几何、最后内部ID破平。不平均友机。alpha=z_j；友机动作前固定位置作锚点，clip3、approach系数1、static_capture_scale1；乘完整CE的权重为1-alpha，PBRS边界/终止消项使用同一权重。

多跳Z不自动激活Support；非Support机器人继续原capture/coverage路由。alpha常为1及动态加权PBRS残差保留并披露，不擅自改公式。

## 连续训练、保存与独立队列

`CoCapTrainer.train`每次真实env.step后global_step+=1，与成功、normal capture、CE或episode结束无关。原trainer已经在每个决策末尾提交current_*与env/Z/replay状态，并可中途完整恢复；不需重写训练循环。

执行器`QueueTrainer`复用该循环，只在安全决策边界保存：100k间隔不可变模型快照；首次决策和中间25k边界完整备份，100k也完整备份。完整状态包含online/target、optimizer、replay、recovery pool、env、当前observation与episode累计项、Z/token映射、Python/NumPy/Torch/CUDA RNG。每阶段仅保留最新`resume_latest.pt`，原子替换且没有历史硬链接；所有模型快照保留、只读、核验SHA。

顺序为Training → Immutable Checkpoint → SQLite独立Evaluation Queue → Artifact Registration。训练进程不执行evaluation或render。SQLite WAL/FULL事务记录任务、真实job PID与进程启动时间、尝试次数、错误、checkpoint和资产SHA；工作进程死亡后最多3次尝试。快照已保存但入队未完成时，下次重启扫描并补登记。训练锁与监督器锁阻止重复任务。

选模用两个独立CPU worker；正式评价、历史参考复评和GIF用另一个CPU worker。CPU worker设置`CUDA_VISIBLE_DEVICES=''`，不占训练显存。正式评价优先于参考复评，二者优先于GIF；不生成非必要中期GIF。任务积压不会停训练。只有阶段训练结束且需要原selection结果才能科学交接时等待选模队列；不等待GIF。

训练初始使用物理GPU0、进程内cuda:0；GPU1不用于本线评价。其他用户两GPU进程保留。启动要求至少6GB空闲显存、30GiB磁盘；运行每1000决策检查磁盘，低于20GiB或系统可用RAM低于8GiB安全暂停；原子备份前还预留至少8GiB或上一文件1.5倍的临时写入空间。资源、数值非有限或科学合同错误立即暂停，普通进程失败有限重试，不以性能低提前停掉课程。

原GPU数值诊断保留正式网络，但使用16+1步、小batch/replay以触发真实优化，仅作诊断；它不参与正式Stage1初始化或训练结果。

## 模型选择、正式评估与GIF

每100k快照按原`select_balanced`函数评价20 episodes/scene；selection seed_base=2026091900，scene偏移100000，模式固定midpoint32。所有Stage1 20个、Stage2/3各7个候选均参与；原BalancedFloor、tie-break和全零CE fallback原样保留。不会预设1.3M/100k/600k最佳步数。

阶段预算全部完成、selection文件/快照SHA/config签收后自动启动下一阶段，前一阶段正式评价/GIF可继续后台运行。交接文件记录selected来源、SHA、selection理由；从不加载旧optimizer/replay/env/RNG。

Final independent seed_base=2026102700，每场景+100000，index0..19，和selection不重叠。各Stage各模式三场景20个完整rollout，全部纳入统计，保存episode JSON/JSONL、summary、全部60份trajectory与60份event log。主模式argmax/fixed-midpoint32；次模式sample/随机32分位均值Q argmax，非categorical actor。多目标normal主标签沿用历史evaluator口径（全部捕获且至少一次loose事件），并额外登记all_targets_normal_rate，避免把夹有stationary的完成误称为全部normal。原Z05 `run_episode`和独立评估器的环境factory、summary、Z诊断与renderer复用；复制源及纯计时字段兼容差异已审计，不另建简化物理。

每场景预定偶数episode index 0,2,...,18渲染10个GIF，即每阶段每模式30个、全线180个。选择与成功或失败无关；每份登记seed、checkpoint/trajectory SHA、renderer源码、300帧上限、trail_window70、100ms、CE targets开关和命令。300帧只影响GIF抽帧，完整trajectory与episode horizon不截断。

历史Z05正式selected另在同一最终seed族上做20×3 argmax复评，补全存活人数，核对与Treatment初态fingerprint。保留原历史正式报告；新旧训练seed不同，统一复评仍不是matched训练因果实验。Repeated-arrival仅登记下一步，不开展完整多波研究。

## 持久化、同步与恢复命令

服务器原始目录：`/home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017-runtime`。正式状态`status.json`、实时进程`live_processes.json`、队列`queue.sqlite3`、各阶段`training/progress.json`及完整恢复文件。大型checkpoint、trajectory与GIF留此目录；Git保存代码、YAML、验收/执行报告、轻量metrics/curve/registry。

启动与恢复都使用同一入口；它先认领已有训练或exact resume，不重复scratch。

```bash
cd /home/yjq/rl/CoCap1/iqn-z-guided-support-full3stage-20261017
/usr/local/anaconda3/bin/python3 tools/z_support_full3stage_20261017.py supervise --gpu 0
```

正式外部执行在独立tmux会话`cocap-zsupport-full3stage`中持续。聊天会话不承担训练生命周期。每100k快照及Stage/最终里程碑自动commit/push并核验远端HEAD；Git同步失败只登记pending，不停GPU训练。最终状态只有三阶段预算和全部队列任务闭合后才写COMPLETE，失败资产写明缺失。
