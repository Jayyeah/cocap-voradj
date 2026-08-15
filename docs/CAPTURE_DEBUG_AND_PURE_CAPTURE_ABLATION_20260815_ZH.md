# Capture Debug 与 Pure-Capture Ablation（2026-08-15）

## 状态

- CF3@400k、PC0@100k均已结束；不再扩步。
- 两张GPU均被其他项目使用。按用户指令，本项目不使用GPU、不挂GPU waiter、不启动B正式训练。
- A1--A4采用有限、串行、自结束的CPU队列；B只完成实现、测试和CPU smoke。

## A1：collision默认合同修复

旧AW实现虽配置`collision_check_each_substep=true`，实体碰撞却只在decision末检测；同时`_refresh_collisions()`边遍历边deactivate，导致同点双机通常只失活低索引一台，且boundary/entity接触阈值与episode统计不一致。

新默认` synchronized_swept_v1`记录同步子步trace，在immutable轨迹上做swept-circle检测，再atomic apply：pursuer-pursuer双方失活；pursuer-evader保留历史“只失活pursuer”；统一接触判定，并输出`agent_agent/obstacle/evader_contact/boundary` provenance。`legacy_end_step`只保留作paired audit。旧CF3/PC0必须标记为legacy collision semantics。

回归覆盖顺序不变、双机/三机atomic、精确相切、AW swept擦碰、近平行不误报及evader/boundary统计；综合聚焦测试`61 passed`。一份修复前旁证CF3@400k legacy deterministic20为`1 normal/20、0 stationary、20 collision/20`；严格CPU pre/post paired由自动队列另行输出。

## A2--A4 CPU诊断

- `tools/probe_cf3_policy_temperature.py`：同seed、同epsilon流比较`T=0/0.25/1`，分开normal/stationary、2+/3+/hold与collision type。
- `tools/evaluate_iqn_corrected_capture.py`：公平重测IQN@125k（SHA256 `5a0ad1c1400d0004334669908c85db6f7b8496fb2987f36f992040c26bd0344d`），修复旧脚本seed+43、整数动作直传AW、APF判断错误。
- `tools/audit_cf3_success_consolidation.py`：保存normal capture、3+→collision、长2+失败最后50步，统计angular gap/separation、clearance、径向/切向速度和`(a,w)`饱和/振荡。
- `tools/run_capture_debug_cpu_queue.py`：A1 legacy/fixed→A2→A3（normal≥4/20才扩100）→A4 legacy/fixed；强制隐藏CUDA、单进程、flock、原子status/error、失败即停并自结束。

## B：Pure-Capture / All-Capture

配置：`configs/experiments/parallel_ce_legacy_voradj_20260809/legacy_voradj_b_pure_capture_local_allcapture_4p1e1obs_100k_aw.yaml`。

冻结CF3 corrected local合同：4P1E1obs、moving APF、K3+stationary fallback、K10/min-active2、`(a,w)`、all-agent MASAC self+mean、no clip、MSE、UTD=.25、LR=1e-4、tau=.005、batch128、horizon1000、capture terminal、scratch/new replay、seed`2026081304`；新实验显式使用fixed swept semantics。

唯一任务/reward变化：

- raw self-visible direct：`approach + mean-shift + front`；
- 看不到enemy但一跳友邻raw/effective pursuing且可解析真实target：`approach only`；
- uninformed：capture dense=0、coverage/PBRS=0，只保留公共safety/collision/boundary；
- normal/stationary terminal reward同值共享给所有仍active pursuers。

K10核心`task_label/reward_role`保持不变；B另设当前可观测的`capture_objective_role`，禁止K10-held agent获得oracle geometry。开关关闭时CF3回归一致。新增normal/stationary、2+/3+/hold、angular gap、agent-agent collision、三角色比例和reward decomposition。

验证：综合`61 passed`、compile/diff-check通过；32-step CPU scratch smoke为`step/replay=32/32`、updates=0（5k warmup内）、all_finite、0 termination/collision、约`13.07 step/s`；checkpoint-screen 1 episode×2 step CPU smoke也通过。未加载旧checkpoint、未启动正式B训练。未来GPU可用时必须显式`--seed 2026081304`从scratch启动。

