# Z-Guided Support warm-start 已完成回执（2026-10-17）

**Scratch 未启动；Stage2/3 未启动。** 两臂100k预算和三个匹配评价窗口均已完成，02:00左右正常退出。服务器13:02核查无本线训练进程，两GPU空闲。

## 训练与工程 Gate

train seed=2026101601。共同Z05 Stage1 selected，仅传policy权重，fresh optimizer/replay。

| 窗口 | Control/Treatment优化器更新 | 结果 |
|---|---:|---|
| 25k | 0 / 0 | checkpoint字节hash相同，能力保持检查，不能视为学习结果 |
| 50k | 4225 / 4225 | 完成匹配评价，无>10pp告警 |
| 100k | 16725 / 16725 | 完成匹配评价，无>10pp告警 |

每窗口×每臂×三场景×两种推理模式×50 episodes，共1800次执行；同一seed复用，不能称为1800独立场景。900对初态fingerprint已重新核验相等。argmax固定midpoint32；sample为IQN随机32分位均值Q取argmax，不是categorical actor。

## 预定100k窗口（非best screen）

| 模式 | 场景 | Normal capture C/T | Collision C/T | CE success C/T | 捕获均时秒 C/T |
|---|---|---:|---:|---:|---:|
| argmax | capture | 50/50 → 50/50 | 0/50 → 0/50 | — | 31.96 / 34.28 |
| argmax | mixed | 48/50 → 49/50 | 2/50 → 2/50 | 0/50 → 46/50 | 32.18 / 34.94 |
| argmax | coverage | — | 0/50 → 0/50 | 0/50 → 47/50 | — |
| sample | capture | 49/50 → 50/50 | 1/50 → 0/50 | — | 32.90 / 33.05 |
| sample | mixed | 48/50 → 49/50 | 2/50 → 2/50 | 9/50 → 47/50 | 32.36 / 35.59 |
| sample | coverage | — | 0/50 → 0/50 | 38/50 → 49/50 | — |

C=Control；T=Treatment。CE列在mixed为post-capture恢复，在coverage为纯覆盖成功。捕获时间只在完成episode上统计，分母可能不同，不能做无条件速度比较。mixed的unfinished字段仅衡量未完成capture，完整任务失败还需查看CE，不能用capture完成代替全任务完成。

50/50成功的Wilson95%区间约[92.9%,100%]；49/50约[89.5%,99.6%]；48/50约[86.5%,98.9%]。捕获差1个episode不能证明统计优势或非劣性。

## 机制与下一步判断

100k Treatment捕获场景共10214个Support转移，其中10211次有效吸引、self tie仅3次（0.029%），friend tie4705次，ID fallback为0。alpha直方图见JSON；原门控下仍几乎都是alpha=1。当前未出现self-tie大量退化为纯CE，不混入friend-first消融。

主要积极信号是Treatment保持Coverage与post-capture CE，Control继续训练后的argmax CE大幅退化。捕获成功率与碰撞未触发工程告警，但100k Treatment捕获均时略慢，不能宣称围捕更快。单训练seed无法排除Control继续训练遗忘或优化波动，不能将本结果升级为已验证Z05正式方法改进。

建议下一阶段做4P1E1O fresh-init匹配Control/Treatment，预算参照原Stage1 2M，优先normal capture和安全；本次状态查询没有启动新长训。先完成独立学习确认，再决定8P/12P课程扩展。

## 产物

100k checkpoint已重新核验hash，留服务器，各臂路径为runtime/<arm>/checkpoints/final_step_100000.pt。checkpoint_registry.csv、training_curves.csv、paired_evaluation_curves.csv、rollout_index.csv和pilot_status.json已刷新。完整六个窗口报告位于runtime/<arm>/evaluations/step_*/report.json。
