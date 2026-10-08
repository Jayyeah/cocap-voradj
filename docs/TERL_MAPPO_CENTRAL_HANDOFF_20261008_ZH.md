# CENTRAL HANDOFF：EXP-TERL-MAPPO-01

该独立agent不写中央 `docs/ops/AC_MASTER_DAG_20260921_ZH.md` / `artifacts/2026-09-21_ac_master_dag/state.json`。本轮用户明确授权TERL-native scratch训练及直接push，并覆盖GPU高占用等待条件；历史CoCap路线限制不适用于此独立原生任务。

- branch：`experiment/terl-backbone-mappo-20261008`。
- worktree：`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008`。
- TERL reference：`143359b2722d49c29b4fecc0ad1fd8d46326e45a`；remote HEAD本轮核验相同。
- CoCap MAPPO/report parent：`80debaef0f5983d8128b5059c90d65acc9216c10`；复用corrected PPO/GAE/ValueNorm/central V，不改历史learner。
- 合同/实现/学习报告：`docs/TERL_MAPPO_{MIGRATION_CONTRACT,IMPLEMENTATION,STAGE1_LEARNING}_20261008_ZH.md`。
- run：`runs/terl_mappo_stage1_seed9_100k`；只TERL backbone模型，Stage1 3P1E0obs4cores、100k bounded，未进入7M课程/CoCap-backbone对照。
- launch code HEAD：`c6ce0398febfa7b886fb7c836948a5b53078179c`（启动前push且remote HEAD核验一致）；后续报告提交不改变scientific source hashes。
- launch：独立tmux `terl_mappo_stage1_20261008`；physicalGPU0，local cuda:0；1 training CPU thread，2 independent CPU evaluation workers。
- 初始35 tests passed；最终43项tests通过（42项suite+1项bootstrap独立测试）；actual256/8/4 CPU+CUDA512 smoke有限、CUDApeak≈590MiB。
- baseline与0/25/50/75/100k checkpoints会screen；0/25/50/100k双模式各20局，75k各10；selection-heldout双模式各20、selected final各50，seed域分离。
- checkpoints：初始/4 milestones/latest/best hardlink；严格runtime+RNG resume；hash记录。所有轻量结果将由bounded supervisor归档并commit/push本实验分支，remote HEAD核验；不触碰其它工作树或进程。
- 学习分类当前PENDING，不能登记为learnable/partial/no-signal；完成后4-way分类按预声明规则产生。进度、PID、throughput、ETA见本分支artifact summary与live progress；不用长期交互监控。

恢复检查：先读run的progress、supervisor/evaluation_status、日志及failure字段，核对PID/cmdline；不要重复启动或因0capture重置。完整resume命令：

```bash
PYTHONPATH=.runtime-deps:src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m terl_mappo.run --output runs/terl_mappo_stage1_seed9_100k --device cuda:0 --resume runs/terl_mappo_stage1_seed9_100k/checkpoints/latest.pt
```

仅用于已确认原训练PID退出、source/config hash一致的恢复；不得并发写同一run。独立evaluator失败保留train继续，故需要检查supervisor_failure与checkpoint backlog后再恢复评估；不要重启训练。100k完成不自动扩2M；允许2M内后续审查，但不根据本轮中间单次成功随意改reward/超参。

正式launch：2026-10-08 22:27:50+08:00，supervisor PID992900，training PID992932，initial evaluator PID993509；首次核验512/100000、2 PPO updates、12 paired minibatches、finite，后续快照见summary。NEXT WAKE-UP：25k checkpoint（按约20 decisions/s约22:49），或任何failure.json/supervisor_failure.json出现时；训练完成预计约23:50，evaluation可能更晚。所有时间是启动早期吞吐估计，不是完成承诺。

退出交互前运行快照：2026-10-08T22:32:37.005104+08:00，5120/100000、20 PPO updates、finite。latest resumable step/hash：`4096` / `38f6d7da0f6ea5aeee8839bbe6ef4688d22430ce6d2cb4781bfa6b7332588a13`；完整runtime/RNG/source hash校验通过。

2026-10-08 22:53更新：25k checkpoint已生成并由独立CPU evaluator PID1007667筛选；0-step完整40局已归档，capture/ring2/ring3均0，argmax/sample collision分别1/20、2/20，仅作为随机初始化基线。训练与评估进程仍存活，无failure文件，最新进度请读summary/live progress。每个完成checkpoint的evaluation与同一步PPO指标由 `tools/audit_terl_mappo_results_20261008.py` 只读关联，输出 `artifacts/2026-10-08_terl_mappo/checkpoint_evidence_audit.json`；尚未完成的评估保持pending。轻量metrics.jsonl和episodes.jsonl有本实验专用Git忽略例外，后台最终sync也会归档推送。

自动证据收尾已启动：tmux `terl_mappo_finalize_20261008`，PID1020175，等待现有supervisor PID992900正常结束后生成Follow-up诊断、验证完整评估并push/核验remote HEAD。实际状态见run内 `evidence_finalizer_status.json`，日志 `evidence_finalizer.log`。该报告程序不重启训练、不加载或更新模型，科学source hash仍与launch一致。25k双模式各20局完整：capture/ring均0，collision1/20与3/20；50/75/100k及selection/final继续后台执行。
