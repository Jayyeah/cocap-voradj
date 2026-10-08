# CENTRAL HANDOFF：EXP-TERL-MAPPO-01

该独立agent不写中央 `docs/ops/AC_MASTER_DAG_20260921_ZH.md` / `artifacts/2026-09-21_ac_master_dag/state.json`。本轮用户明确授权TERL-native scratch训练及直接push，并覆盖GPU高占用等待条件；历史CoCap路线限制不适用于此独立原生任务。

- branch：`experiment/terl-backbone-mappo-20261008`。
- worktree：`/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008`。
- TERL reference：`143359b2722d49c29b4fecc0ad1fd8d46326e45a`；remote HEAD本轮核验相同。
- CoCap MAPPO/report parent：`80debaef0f5983d8128b5059c90d65acc9216c10`；复用corrected PPO/GAE/ValueNorm/central V，不改历史learner。
- 合同/实现/学习报告：`docs/TERL_MAPPO_{MIGRATION_CONTRACT,IMPLEMENTATION,STAGE1_LEARNING}_20261008_ZH.md`。
- run：`runs/terl_mappo_stage1_seed9_100k`；只TERL backbone模型，Stage1 3P1E0obs4cores、100k bounded，未进入7M课程/CoCap-backbone对照。
- launch：独立tmux `terl_mappo_stage1_20261008`；physicalGPU0，local cuda:0；1 training CPU thread，2 independent CPU evaluation workers。
- 初始35 tests passed；补充40 tests passed；actual256/8/4 CPU+CUDA512 smoke有限、CUDApeak≈590MiB。
- baseline与0/25/50/75/100k checkpoints会screen；0/25/50/100k双模式各20局，75k各10；selection-heldout双模式各20、selected final各50，seed域分离。
- checkpoints：初始/4 milestones/latest/best hardlink；严格runtime+RNG resume；hash记录。所有轻量结果将由bounded supervisor归档并commit/push本实验分支，remote HEAD核验；不触碰其它工作树或进程。
- 学习分类当前PENDING，不能登记为learnable/partial/no-signal；完成后4-way分类按预声明规则产生。进度、PID、throughput、ETA见本分支artifact summary与live progress；不用长期交互监控。

恢复检查：先读run的progress、supervisor/evaluation_status、日志及failure字段，核对PID/cmdline；不要重复启动或因0capture重置。完整resume命令：

```bash
PYTHONPATH=.runtime-deps:src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m terl_mappo.run --output runs/terl_mappo_stage1_seed9_100k --device cuda:0 --resume runs/terl_mappo_stage1_seed9_100k/checkpoints/latest.pt
```

仅用于已确认原训练PID退出、source/config hash一致的恢复；不得并发写同一run。独立evaluator失败保留train继续，故需要检查supervisor_failure与checkpoint backlog后再恢复评估；不要重启训练。100k完成不自动扩2M；允许2M内后续审查，但不根据本轮中间单次成功随意改reward/超参。
