# TERL-backbone MAPPO 实现与正确性验证

原生任务与算法迁移已经实现；当前科学结果由独立Stage1评估决定，不能把 smoke 或训练reward上升写成capture成功。

## 实现

- `vendor/terl/`：TERL `143359b2722d49c29b4fecc0ad1fd8d46326e45a` 的原始 Git 字节，保留LICENSE。环境、APF、动力学、观测、reward和capture源码未修改。启动断言检查counts/map/spawn/AW9/physics/reward/horizon全部runtime getters并写入manifest。完整阅读模型、agent、trainer、replay、环境、P/E/perception、robot、APF、config及入口；重点测试其实际行为，而非README。
- `src/terl_mappo/native.py`：加载固定原生配置，截取Stage1 schedule。wrapper实现原trainer的collision deactivation、active<3/capture joint reset和3001-decision timeout。不修改MarineEnv.step返回值；单独产生PPO terminated/truncated/episode_end，终止优先，pre-reset next state计算V。每步重构time/distance/global/emergency/collision/goal并断言与原生reward一致。关闭wandb网络日志仅影响记录。
- `src/terl_mappo/model.py`：继承原encoder/type/Transformer/target-selection及feature fusion，删除quantile/cosine参数与buffer；hidden-layer/ReLU/LayerNorm接9-logit categorical。hidden256、**8 heads/4 layers为原正式入口有效值**，构造dropout.1；PPO rollout和更新均eval，autograd开启，避免两次似然因dropout不一致。输出head orthogonal gain.01初始化用于均匀探索；不加载teacher/BC/IQN权重。
- self token + **原始unmasked max pooling保留**。key padding mask没有消除padding query输出；masked_pool开关仅用于独立反事实测试，正式false。存在目标时feature与原TERL逐浮点parity；无目标权重零，全mask临时开放self，使forward/backward有限。
- critic复用`cocap_voradj.models.small_step_ac.CentralValueNetwork`：256/8/4、PreLN Transformer、dropout0、action-free per-agent V。Stage1全状态是focal25（x/y、速度、sin/cos、speed、pursuing、时间、4个排序洋流核）、P3×7、E1×7及空障碍slot。只对critic作固定物理尺度归一化，actor/reward保持原生数值。无agent位置embedding；P排列改变时输出同样排列，已有等变测试。4个核位置/方向/强度和horizon进入V，避免忽略current和时间；actor没有global输入。
- PPO复用`cocap_voradj.training.small_step_ac.MAPPOTrainer`及`compute_gae` / `ValueNorm`。未改历史learner文件。独立Adam LR3e-5/1e-4，eps1e-5；gamma.99、lambda.95、clip.2、3epochs/2minibatches、entropy.01、gradclip.5、KL.02、ValueNorm beta.99999。raw rewards/raw-return GAE；rollout-old ValueNorm统计先反归一化，更新一次return stats，value clipping比较原normalized旧预测；active-only归一化advantage/loss。
- `run.py`：rollout256 joint decisions；环境steps、active agent transitions、rollout update calls、paired optimizer minibatches分开计数。固定25k checkpoint切点会使尾rollout不足256，明确记为同一on-policy update。完整checkpoint含actor/V/Adam/ValueNorm、环境对象/PRNG、Python/NumPy/CPU/CUDA RNG、计数、source hashes。atomic落盘与严格resume来源校验；CPU exact resume与CUDA action/logprob replay验证。
- `evaluate.py`：独立CPU evaluator、2 episode workers，argmax/sample；相同物理seed初态、隔离sample动作RNG。输出每episode fingerprint、capture/normal、time均值中位数p90/成功n/删失n、ring2/3访问、strict geometry、collision类型、reward组件、entropy。模型hash前后核验，partial heartbeat可检查。
- `supervise.py`：固定100k bounded run，screening异步，不因早期0capture暂停/改reward/改超参；正常结束后进行selection-heldout与final隔离测试，生成曲线和分类、提交轻量artifact并直接push本实验分支，验证remote HEAD。既有DAG/state不写。

## 审计问题与保留理由

1. global合作reward的>=5 elif不可达，但后续全局-10仍生效；3/4/5/6人数反事实测试有效值分别+5/+3.5/-8/-9.5。计数不排除inactive，Stage1首collision即joint end；保持公开行为。
2. capture sparse reward中的pursuer_count是helper数量而非总人数。3人均匀几何每参与者得到120π≈376.99，不是配置表面120。该credit和done时间顺序均保留。
3. dense reward非Δdistance：active每步-1，近8m内+5，远端硬编码5×exp；多目标逐项循环且远端用min distance，可能重复放大，Stage1仅1目标。附近<4m另-5；emergency友军距离使用post-step位置、障碍距离使用缓存perception。不改shaping。
4. enemy观测没有20m过滤；友军/障碍才过滤，pursuing nearest-enemy≤18。因此本任务actor保留原生全敌方token，不能描述为严格radius-local。
5. 原APF的positive acceleration filtered-array argmin缺少全数组index映射，可能选成0（negative acceleration）；current的“skip duplicate”内层continue也没有跳过核。两者保留，归为公开合同/设计意图待解决；不影响本次对原源码的parity。
6. timeout实际3001 decisions；capture步done可全false；原trainer负责episode end。迁移若直接使用done会错误bootstrap，wrapper按实际生命周期明确生成terminal并保留原horizon。原边界只罚-5，不terminal；未混入CoCap swept碰撞、安全reward、stationary capture或CR-MS。

## 验证

第一批：35 passed；覆盖byte identity、随机seed动作转移/reward/done、3/4/5/6合作分支、capture/碰撞/timeout/越界、actor特征原始parity、padding反事实、no-target/all-mask、argmax/sample/logprob/entropy/共享参数/梯度、critic等变与隔离、GAE、ValueNorm、CPU exact checkpoint resume、CUDA有限update和RNG resume，及既有corrected learner/terminal-row回归。补充测试包括独立PPO clipping计算、inactive-reward不可影响loss、事件原生parity、checkpoint选择/分类与并行完整episode evaluator；补充后 **42 passed**（35基础+5扩展+2追加），4个warning（有限性检查通过）；compileall/diff-check通过。最终结果见轻量artifacts和学习报告。

实际网络（不是缩小网络测试）512-step CPU与CUDA smoke均完成2个PPO update / 12 paired minibatches，finite；CUDA峰值分配589.99 MiB。最终1024-step CUDA preflight完成4个PPO updates/24 paired minibatches，18.74 decisions/s、peak589.99MiB；reward重构与post-update ratio telemetry有限。对应progress/manifest和checkpoint hash已落盘。

## 资源、运行与复现

共享GPU0上其它用户job util≈97%，仍按本轮用户授权正常启动，显存约24GiB余量；不操作任何其它进程。GPU1也繁忙；本实验仅用physicalGPU0（process-local cuda:0）。磁盘初始143GiB free；本实验100k仅0/25/50/75/100k、latest、best hardlink，峰值保守<1GiB。不积累replay，不上传大模型，不清理其它实验。

Python3.12/PyTorch2.9.1+cu128；原缺gym，通过实验内`.runtime-deps/`安装gym0.26.2/gym_notices0.0.8，复用现有NumPy/SciPy/wandb（关闭网络日志）。复现：

```bash
python -m pip install --no-deps --target .runtime-deps gym==0.26.2 gym_notices==0.0.8
PYTHONPATH=.runtime-deps:src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q test/test_terl_native_mappo_20261008.py test/test_small_step_ac_migration_contract.py test/test_mappo_terminal_rows_20260923.py
PYTHONPATH=.runtime-deps:src python -m terl_mappo.supervise --gpu 0 --output runs/terl_mappo_stage1_seed9_100k
```

原生TERL的sensing/current/APF/reward与CoCap Final PureCapture明显不同；不能将两套成功率直接解释为算法/backbone优劣。首轮只实现TERL-backbone模型，后续同任务CoCap-backbone对照尚未实现。未运行7M课程或修改任务难度。
