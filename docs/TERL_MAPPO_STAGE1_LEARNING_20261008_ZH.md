# TERL-MAPPO Stage1 学习验证

更新时间：2026-10-08T23:58:32.928946+08:00（Asia/Shanghai）。
状态：`COMPLETE_BOUNDED_GATE`；科学分类：`TERL_MAPPO_STAGE1_PARTIAL`。

预算：100,000 joint environment decisions（Stage1上限2M；不自动扩步）。实际进度：100,000。
运行目录：`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_100k`。最新已落盘数值健康：`True`；训练PID `992932`。

## Checkpoint screening / held-out

| steps | domain | mode | normal/capture n | ring2/ring3 | strict | collision | capture mean/median/p90 s | censored |
|---:|---|---|---|---|---|---|---|---|
| 100000 | final | argmax | 1/1/50 | 27/2 | 1 | 49/50 | 40.50/40.50/40.50 | 49 |
| 100000 | final | sample | 1/1/50 | 18/4 | 1 | 49/50 | 338.50/338.50/338.50 | 49 |
| 0 | screen | argmax | 0/0/20 | 0/0 | 0 | 1/20 | —/—/— | 20 |
| 0 | screen | sample | 0/0/20 | 0/0 | 0 | 2/20 | —/—/— | 20 |
| 25000 | screen | argmax | 0/0/20 | 0/0 | 0 | 1/20 | —/—/— | 20 |
| 25000 | screen | sample | 0/0/20 | 0/0 | 0 | 3/20 | —/—/— | 20 |
| 50000 | screen | argmax | 0/0/20 | 0/0 | 0 | 0/20 | —/—/— | 20 |
| 50000 | screen | sample | 0/0/20 | 0/0 | 0 | 9/20 | —/—/— | 20 |
| 75000 | screen | argmax | 0/0/10 | 0/0 | 0 | 0/10 | —/—/— | 10 |
| 75000 | screen | sample | 0/0/10 | 0/0 | 0 | 1/10 | —/—/— | 10 |
| 100000 | screen | argmax | 1/1/20 | 9/1 | 1 | 19/20 | 29.50/29.50/29.50 | 19 |
| 100000 | screen | sample | 0/0/20 | 5/0 | 0 | 20/20 | —/—/— | 20 |
| 100000 | selection | argmax | 0/0/20 | 11/0 | 0 | 20/20 | —/—/— | 20 |
| 100000 | selection | sample | 0/0/20 | 9/1 | 0 | 20/20 | —/—/— | 20 |

0/25k/50k/100k 每模式20局，75k每模式10局。screen同组seed对齐初态，sample有独立动作RNG。训练seed9；screen2026100800起；selection-heldout2036100800起；final2046100800起。

预声明selection：screen pooled normal rate最高，其次collision最低、strict最高、ring3最高、较早checkpoint；screen capture候选和最终screen-best扩展selection-heldout各20局，再按同规则选取；final各50局不参与selection。

最佳checkpoint：`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_100k/checkpoints/step_000100000.pt`；SHA256：`781c391a175e51dfabead66bac645a6efb1bc230747276768d998ccab1f9153e`。

## 数值健康与解释

最新PPO telemetry：`{"steps": 100000, "agent_transitions": 300000, "optimizer_steps": 2352, "reward_mean": 1.6797435261123224, "reward_min": -3.513938830050974, "reward_max": 9.0, "actor_loss": -0.02082460808257262, "value_loss": 0.023641031545897324, "entropy": 2.0894074042638144, "clip_fraction": 0.007936508394777775, "approx_kl": 0.0007648032733413856, "actor_grad_norm": 0.7820917765299479, "value_grad_norm": 1.6211454619963963, "explained_variance": 0.7598110437393188, "value_norm_mean": -205.0832061767578, "value_norm_std": 151.17315673828125, "kl_early_stop": 0.0, "ppo_epochs_completed": 3.0, "minibatch_updates": 6.0, "actor_update_l2": 0.08767334039533786, "actor_update_relative_l2": 0.0009239974443068081, "approx_kl_max": 0.0024926781188696623, "update_count": 392.0, "post_update_ratio_mean": 1.0034762620925903, "post_update_ratio_min": 0.7088078856468201, "post_update_ratio_max": 1.3822875022888184, "post_update_clip_fraction": 0.0376984141767025, "post_update_kl": 0.002340493258088827, "episodes": 0, "training_capture": 0, "training_collision": 0}`。

仅凭reward上升不认定学会围捕。Strong要求多个checkpoint重复normal capture、相对初始改善且碰撞不主导；partial包括孤立capture或ring改善。100k无信号不代表MAPPO不可学习；2M内是否扩步需审查环形几何、exploration、critic和原生驻留reward。

## 后续决定

保留本次单seed、有限预算证据；先复核最佳checkpoint与final独立测试，再决定Stage1扩展或同任务backbone对照。

中央DAG/state只读；handoff在本分支。模型文件留在服务器，不提交大checkpoint。

补充诊断与后续决定：见 `TERL_MAPPO_FOLLOWUP_DECISION_20261008_ZH.md`；完整训练窗口统计和checkpoint证据审计已归档。
