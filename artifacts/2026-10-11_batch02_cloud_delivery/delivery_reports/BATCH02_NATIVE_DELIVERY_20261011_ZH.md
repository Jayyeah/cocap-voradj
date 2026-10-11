# Batch02 N3/R3/V3 交付

观测时间 2026-10-11T21:51:24.404016+08:00。PROVISIONAL；长训运行不等于能力PASS。

A. 版本树：V0 TERL baseline → V1 E1 bridge8827c11（独立PASS/冻结）→ V2 N2/R2能力验证8ce5593（运行）→ representation N3 28a3337 → reward账本R3 7cb8bc0 → ARM V3/minimal mix3575b74（对应路线V4）。

B. 逐步代码parent/child：N3 8ce5593→28a3337；R3 28a3337→7cb8bc0（算法父线R2/8ce5593）；V3 7cb8bc0→3575b74（集成N3+R3）。最终三线独立config/process/checkpoint共用固定3575b74，不混用训练状态。代码与中央9ead763均已授权发布并核对远端。

N3保留CoCap self9/22slots原输入，原成功IQN encoders/Transformer/target与summary attention替换TERL表示；actor3,568,137→3,963,913；原AW9 policy head和critic3,236,865保持。R3保留R2 TERL self4/19slots/网络，仅增加成功native reward六分类ledger；数值delta0。V3保留完整CoCapenv/obs/reward+N3backbone+TERL PPO，追加R3ledger。相对IQN仅是on-policy categorical actor/central V/optimizer等明确MAPPO adaptation；任务源唯一为成功1a9da05、resolved6955ecf2…，没有failed MAPPO provider继承。

C. 测试：三线import/config、64scripted transition parity、256步PPO、16步fresh-process exact resume；N3/V3原IQN feature等值、mask padding/batch行独立/inactive finite；R3/V3 8固定mixed fixtures reward重建最大残差2.78e-17。最终SHA三线scratch step0 evaluator各2原episode PASS。

各pilot1024、4次PPO/24minibatches、checkpoint恢复与2episode evaluator均PASS，梯度/参数finite，无阻断bug。另16步continuation与16步replay标为恢复诊断；不用于学习判断。

D. 当前训练：

| 线 | GPU | PID | phase | decisions | last checkpoint | throughput | ETA |
|---|---:|---:|---|---:|---:|---|---|
| N2 | 0 | 179017 | TRAINING | 117664 | 100000 | 22.79 decisions/s | 10.75h |
| R2 | 1 | 179417 | checkpoint_evaluation | 100000 | 100000 | 19.65 decisions/s | 12.73h |
| N3 | 0 | 214952 | initial_evaluation | 0 | 0 | 尚未测得 | 初始评估中 |
| R3 | 1 | 216982 | TRAINING | 5376 | 0 | 13.69 decisions/s | 20.18h |
| V3 | 1 | 219016 | initial_evaluation | 0 | 0 | 尚未测得 | 初始评估中 |

每线固定1M joint decisions、25k保存与screen评估，initial/selection/final各域固定；独立selection选best，final比较initial/selected/last。pilot权重不加载进long。已有N2/R2原PID、源码、配置、预算、合同未改。

E. 已执行顺序：N3实现/commit/DAG → R3实现/commit/DAG → V3集成/commit/DAG → 三线必要smoke → N3/R3/V3串行CPU pilot → 每线PASS后按GPU0/GPU1/GPU1错峰启动long。资源足够，未停止旧作业。之后继续固定预算与逐checkpoint评价，不自动启动新任务。

F. 风险：R3是reward decomposition而非新reward tuning；V3与N3只差telemetry。两者不能制造额外因果结论。1024 pilot不证明学习；单训练seed、混合任务泛化/退化/稳定性等待full curves及final评估。未覆盖新lifecycle、repeated arrival、多target、coverage扩展或规模课程。

N3/V3同CPU/seed pilot checkpoint的完整trainer state相同：True（记录等价性，不作为性能门禁）。

NEXT WAKE-UP：2026-10-11T22:21:24.404016+08:00，或任一新线首25k checkpoint。

本地runtime与交付JSON保存完整config hash、checkpoint清单/sha、PID、吞吐/ETA与pilot数据；每线progress.json为实时来源。
