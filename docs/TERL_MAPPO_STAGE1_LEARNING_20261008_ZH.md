# TERL-MAPPO Stage1 学习验证

更新时间：2026-10-08T22:32:43.465922+08:00（Asia/Shanghai）。
状态：`RUNNING_HEALTHY_PENDING_EVALUATION`；科学分类：`PENDING：训练/独立评估未完成`。

预算：100,000 joint environment decisions（Stage1上限2M；不自动扩步）。实际进度：5,120。
运行目录：`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_100k`。最新已落盘数值健康：`True`；训练PID `992932`。

## Checkpoint screening / held-out

| steps | domain | mode | normal/capture n | ring2/ring3 | strict | collision | capture mean/median/p90 s | censored |
|---:|---|---|---|---|---|---|---|---|
| — | screen | pending | — | — | — | — | — | — |

0/25k/50k/100k 每模式20局，75k每模式10局。screen同组seed对齐初态，sample有独立动作RNG。训练seed9；screen2026100800起；selection-heldout2036100800起；final2046100800起。

预声明selection：screen pooled normal rate最高，其次collision最低、strict最高、ring3最高、较早checkpoint；screen capture候选和最终screen-best扩展selection-heldout各20局，再按同规则选取；final各50局不参与selection。

最佳checkpoint：`PENDING`；SHA256：`PENDING`。

## 数值健康与解释

最新PPO telemetry：`{"steps": 5120, "agent_transitions": 15360, "optimizer_steps": 120, "reward_mean": -1.6895107745701783, "reward_min": -85.97153033337285, "reward_max": 2.1126165022636263, "actor_loss": -0.022321425067881744, "value_loss": 0.04630030815800031, "entropy": 2.1925788720448813, "clip_fraction": 0.0, "approx_kl": 2.175228094832467e-05, "actor_grad_norm": 0.9731611907482147, "value_grad_norm": 1.7293937305609386, "explained_variance": 0.1465732455253601, "value_norm_mean": -191.6474151611328, "value_norm_std": 101.24028015136719, "kl_early_stop": 0.0, "ppo_epochs_completed": 3.0, "minibatch_updates": 6.0, "actor_update_l2": 0.032214491433277224, "actor_update_relative_l2": 0.00033963068858463235, "approx_kl_max": 5.926238372921944e-05, "update_count": 20.0, "post_update_ratio_mean": 1.0000741481781006, "post_update_ratio_min": 0.9752854108810425, "post_update_ratio_max": 1.0209031105041504, "post_update_clip_fraction": 0.0, "post_update_kl": 0.00010100395593326539, "episodes": 1, "training_capture": 0, "training_collision": 1}`。

仅凭reward上升不认定学会围捕。Strong要求多个checkpoint重复normal capture、相对初始改善且碰撞不主导；partial包括孤立capture或ring改善。100k无信号不代表MAPPO不可学习；2M内是否扩步需审查环形几何、exploration、critic和原生驻留reward。

## 后续决定

当前只等待已启动的bounded训练/独立评估；不启动CoCap-backbone对照或7M课程。

中央DAG/state只读；handoff在本分支。模型文件留在服务器，不提交大checkpoint。
