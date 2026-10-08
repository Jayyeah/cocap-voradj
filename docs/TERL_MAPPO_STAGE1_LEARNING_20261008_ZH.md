# TERL-MAPPO Stage1 学习验证

更新时间：2026-10-08T22:20:22.640674+08:00（Asia/Shanghai）。
状态：`READY_TO_LAUNCH`；科学分类：`PENDING：训练/独立评估未完成`。

预算：100,000 joint environment decisions（Stage1上限2M；不自动扩步）。实际进度：0。
运行目录：`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_100k`。最新已落盘数值健康：`pending`；训练PID `pending`。

## Checkpoint screening / held-out

| steps | domain | mode | normal/capture n | ring2/ring3 | strict | collision | capture mean/median/p90 s | censored |
|---:|---|---|---|---|---|---|---|---|
| — | screen | pending | — | — | — | — | — | — |

0/25k/50k/100k 每模式20局，75k每模式10局。screen同组seed对齐初态，sample有独立动作RNG。训练seed9；screen2026100800起；selection-heldout2036100800起；final2046100800起。

预声明selection：screen pooled normal rate最高，其次collision最低、strict最高、ring3最高、较早checkpoint；screen capture候选和最终screen-best扩展selection-heldout各20局，再按同规则选取；final各50局不参与selection。

最佳checkpoint：`PENDING`；SHA256：`PENDING`。

## 数值健康与解释

最新PPO telemetry：`{}`。

仅凭reward上升不认定学会围捕。Strong要求多个checkpoint重复normal capture、相对初始改善且碰撞不主导；partial包括孤立capture或ring改善。100k无信号不代表MAPPO不可学习；2M内是否扩步需审查环形几何、exploration、critic和原生驻留reward。

## 后续决定

当前只等待已启动的bounded训练/独立评估；不启动CoCap-backbone对照或7M课程。

中央DAG/state只读；handoff在本分支。模型文件留在服务器，不提交大checkpoint。
