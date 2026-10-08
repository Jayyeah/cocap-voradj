# TERL-MAPPO：无视100k gate的累计1m自动续训

2026-10-08最新用户指令：若当前总步数只有100k，马上增加自动线，在100k结束时续训到1m，忽略gate；只有无法精确续训时才开完全一致的并行1m线。本次确认原预算100,000，采用**精确续训**。总目标为1,000,000 joint environment decisions，即parent 100k＋续训900k。该授权覆盖原报告的“不自动扩步”决定及旧CoCap路线中的long-extension限制。

## 已启用的自动线

- parent：`runs/terl_mappo_stage1_seed9_100k`，training PID992932。
- 自动控制器：tmux `terl_mappo_1m_continue_20261008`，当前PID1040959。
- 状态：`ARMED_WAITING_PARENT_100K`；核验时parent 86,520步，数值有限，实际进程仍存活。
- 新output：`runs/terl_mappo_stage1_seed9_1m_continuation`。
- 新配置：`configs/experiments/terl_mappo_20261008/stage1_1m.json`。
- 实际状态/日志：新output内 `continuation_status.json`、`controller.log`；launch与CUDA证明归档到 `artifacts/2026-10-08_terl_mappo_1m`。

控制器核验parent trainer PID/cmdline/start ticks，等待**该训练进程**退出和100k不可变checkpoint完整落盘。它不等待100k evaluator、selection、final test或gate分类。无capture、partial/no-convincing-signal不阻止训练；CPU evaluator失败与云端暂时同步失败不触发performance early-stop。真实训练异常保留诊断。

## 精确恢复与配置合同

未修改任何冻结的 `src/terl_mappo/*.py`、corrected learner或vendored TERL源文件。原 `load_checkpoint()` 的strict合同检查明确允许仅改变 `budget`，其余config和source hashes必须逐项相同。

恢复actor、central V、两个Adam、ValueNorm、update counters、完整MarineEnv/adapter、Python/NumPy/CPU/CUDA RNG。从100k不可变checkpoint恢复，不加载teacher或BC权重。原run和新run独立目录；新run保留parent全量metrics/episodes曲线，随后从100k追加，累计计数连续、不重复初始化。manifest记录parent/source/anchor hash与resume counters。

配置唯一变化：`budget: 100000 → 1000000`。seed9/actor seed109、Stage1 3P1E0obs4cores、地图/current/APF/reward/dynamics、256/8/4 actor/critic、AW9、PPO/GAE/ValueNorm、rollout256、25k checkpoint切点、物理GPU0均保持。保存latest沿用原runner的触发条件；25k切点后的latest并非严格每2k刷新，至少随每个25k milestone更新。本次保持该原始行为，保证同配置训练序列。

恢复执行命令由控制器在100k boundary自动启动：

```bash
PYTHONPATH=.runtime-deps:src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m terl_mappo.run \
  --config configs/experiments/terl_mappo_20261008/stage1_1m.json \
  --output runs/terl_mappo_stage1_seed9_1m_continuation \
  --device cuda:0 \
  --resume runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000100000.pt
```

已有控制器会启动该命令；手动执行前必须确认其状态，避免并发写同一run。新trainer独立process session，不依赖report控制器存活。

## 实际验证

`tools/verify_terl_mappo_cuda_budget_resume_20261008.py` 使用正式50k不可变checkpoint、实际256/8/4 CUDA网络，分别用100k/1m预算严格load。后续32 joint decisions、一次完整PPO update（6 paired minibatches）**逐位一致**：load前后actor/critic/Adam/ValueNorm/counters、全部rollout字段、PPO telemetry、序列化完整环境及所有RNG。checkpoint hash未变，source严格匹配；约9.68s完成。

checkpoint SHA256：`d618a7c4fe5016138deded570a89b24a0c96d0a259b07feb48bc147763665e48`；权威路径/hash见 `exact_resume_validation.json`。本验证范围是实测32步和一次完整update，不将尚未发生的1m结果写成完成。

额外真实CUDA CLI从50k续至50,032，用于验证entrypoint、累计agent transitions、rollout updates与optimizer计数没有重置。非budget超参变化会被strict guard拒绝；retention验证覆盖其他路径、pending、top3与固定milestones保护。证据见 `orchestration_preflight.json`。已有基础43项correctness tests保持原结论。

## 评估、retention与同步

复用parent已完成screen，记录provenance。后续每25k点screen双模式各10局；100/250/500/750/1000k各20局。独立CPU evaluator两workers，训练正常继续。最终screen排序前三点各20局selection-heldout，再按预声明normal/低collision/strict/ring3/较早step选best；最终各50局，seed2046101800起，与旧100k final及screen/selection隔离。

保存latest、100/250/500/750/1000k milestones、screen前三候选、待评估点和最终best。其它本自动线checkpoint只在完整screen落盘且hash验证后清理；不删除parent或其它实验路径。完整轻量metric/episode/evaluation/retention记录归档。Git同步等待旧100k报告writer结束，训练启动不受此等待影响；后续报告直接push本实验分支并核验remote HEAD。

1m学习结果见 `TERL_MAPPO_1M_LEARNING_20261008_ZH.md`。旧100k报告仍是早期gate证据，其后续预算建议由当前用户明确1m授权覆盖。中央DAG/state只读，提供handoff。
