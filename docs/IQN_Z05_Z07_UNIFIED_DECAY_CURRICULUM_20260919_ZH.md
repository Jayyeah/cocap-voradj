# IQN Z05/Z07 Unified-Decay 全课程实验交接

日期：2026-09-19

## 科学问题

比较唯一 target-evidence policy state 下的两条完整课程线：

- Z05：`lambda = eta = 0.5`
- Z07：`lambda = eta = 0.7`

两线共享同一 NormSense-V2、reward、CE、CR-MS、support blend、replay、recovery、网络结构、seed、评估 seed 和课程预算。唯一科学差异是 alpha；`hard_zero_threshold=0.10` 两线相同。

## Z-v2 语义

每个完整 environment decision 只更新一次，physics substep 不更新：

```text
z_raw_i = max(direct_i, alpha * z_i(previous), alpha * max_neighbor(z_j(previous)))
z_i = 0 if z_raw_i < 0.10 else z_raw_i
```

传播读取 immutable previous-z，所有 agent 同步写入下一 decision state。清零时强制：`source=zero`、`lineage_hops=-1`、`source_age_steps=-1`。

`cocap-z-state-v2` 保存并校验：`enabled`、`lambda`、`eta`、`hard_zero_threshold`、values、direct、neighbor max、source、lineage hops、source age、neighbors、update count。旧 v1、alpha 不匹配或 threshold 不匹配均拒绝 resume。

决策周期：`physics_dt=0.05s`、10 个 physics substeps、`decision_dt=0.5s`。理论释放：Z05 约 2.0s，Z07 约 3.5s；空间 hop 使用相同 alpha 衰减。

## 代码与配置

- 分支：`experiment/iqn-z-unified-decay-dual-curriculum-20260919`
- Worktree：`/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919`
- 配置目录：`configs/experiments/iqn_z_unified_decay_curriculum_20260919/`
- 单线执行器：`tools/iqn_z_unified_decay_curriculum_20260919.py`
- 双线协调器：`tools/supervise_iqn_z_unified_decay_dual_20260919.py`

阶段：

- Stage1：4 pursuers / 1 evader / 1 obstacle，scratch，2,000,000 steps
- Stage2：8 / 2 / 2，700,000 steps，继承本线 Stage1 selected model weights
- Stage3：12 / 3 / 3，700,000 steps，继承本线 Stage2 selected model weights

跨 stage 只传 policy/model weights；不传 replay、optimizer、RNG、env runtime、当前 z 或 recovery pool。每阶段每 100k 保存 checkpoint、exact full-resume、formal evaluation、runtime evidence；阶段完成后使用 BalancedFloor 选择，strict coverage 全零时使用记录完整计算过程的 capture-retention + normalized CE-RMS/area-CV maximin fallback。无 performance promotion gate。

## Preflight 结果

2026-09-19 18:47 +08:00 已通过：

- 初始网络 bit-exact SHA：`67c6126a0b282eb3d691f4ef455fb11f4eb087f8f7a53e7de070adb7dbe24924`
- Z05/Z07 非 alpha 合同差异：`0`
- hard floor：`0.10`
- 每 decision 恰好一次 Z update：通过
- 16-step 真实 optimizer smoke：两线各 15 updates
- loss finite：通过
- target sync：两线各 15 次
- checkpoint load：通过
- full resume：通过，含 Z-v2 snapshot
- formal evaluator smoke：通过，update-count diagnostics 无违规

Preflight 产物：`artifacts/2026-09-19_iqn_z_unified_decay_curriculum/preflight/`

## 资源与排队

当前 GPU 状态：

- GPU0：被外部 PID `3248997` 占用
- GPU1：被同一外部 PID `3248997` 占用

当前磁盘：

- 根盘 `/`：约 57 GB 可用，不足以安全承载双线全课程 exact-resume
- `/data/disk2`：约 4.5 TB 可用，但当前账户无法在目标路径创建目录
- `/data/disk1/shared_data/overleaf`：可写权限属于 Overleaf 服务，不得用于训练

目标训练存储路径登记为：

`/data/disk2/home/yjq/cocap-runs/iqn-z-unified-decay-dual-curriculum-20260919`

协调器在该路径可写且至少保留 200 GiB 后才启动训练；在此之前持续写轻量状态并排队，避免根盘写满。

当前 detached supervisor：

- tmux：`iqn_z_unified_decay_dual_20260919`
- PID：`3319724`
- 状态：`queued / waiting_for_storage`
- 当前没有训练 PID、optimizer updates 或 throughput；两条线尚未启动

## Supervisor 与产物

协调器状态文件：

`artifacts/2026-09-19_iqn_z_unified_decay_curriculum/status.json`

训练产物（资源恢复后）：

- `<storage>/z05/`
- `<storage>/z07/`
- `<storage>/Z05_VS_Z07_FINAL_COMPARISON.json`

每条线最终报告：

- `Z05_CURRICULUM_FINAL_REPORT.json`
- `Z07_CURRICULUM_FINAL_REPORT.json`

## Resume 规则

同一 stage 只允许 exact full-resume，要求 code/contract hash、alpha、threshold、Z-v2 schema、replay/runtime、RNG 和断点可读且一致。跨 stage 只允许本线 selected model 权重 warm-start。任何 NaN/Inf、checkpoint/replay corruption、contract drift、wrong GPU、错误 alpha/threshold、physics substep 更新或 ancestry 错误均 fail-closed。

## Git 与已知风险

当前 HEAD：`81be35ca56ad067d34fc8e417ff75dffef2d6fde`。远端最新 Z 事实源内容树已核对一致，但当前 fetch 因本机代理不可用及无 CA 证书失败；待修改 commit 后按最多两次安全 push 规则处理。

已知运行前置风险：大容量盘目录需要管理员或存储服务提供可写路径；两张 GPU 需外部 Isaac Sim 进程释放。协调器不会抢占或停止外部进程。
