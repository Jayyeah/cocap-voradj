# B3 — Direct Local Evidence State Replacement（2026-09-18）

状态：**`B3_FAIL`**。scalar `z_i/z_j` 能显著改善 direct/support 离线模仿，但没有恢复 pure coverage 或 mixed recovery；本轮按预设 STOP，不启动 warm RL。

## 1. Branch / baseline

- branch：`experiment/iqn-zstate-b3-20260918`。
- baseline：远端 `experiment/iqn-cleanup-emergent-b1-20260917`，HEAD `9c8f603ea81cb8bb1104549e360058e8eaa41f7c`；fetch 后无更新。
- 独立 worktree：`/home/yjq/rl/CoCap1/cocap-voradj-iqn-zstate-b3-20260918`。
- 未修改 B1.5、A2 或 canonical 双线规划文档。

## 2. z 更新语义与参数

固定实现：

`z_i^t = max(d_i^t, 0.95 z_i^(t-1), 0.85 max_{j in N_i^t} z_j^(t-1))`

每个 decision boundary：

1. 从当前 physical/sensing state 同时计算所有 `d_i^t`；
2. 冻结上一 decision 的完整 `z^(t-1)`；
3. 所有 agent 同步计算 `z^t`；
4. 再 pack policy observation；
5. actor 选动作并执行 env step。

当前直接看见 active enemy 的 agent 在同一个 decision 立即得到 `z=1`；邻居只读取上一 decision 的 z，因此不存在同 step algebraic loop。reset 先清零；初始 physical state 随后进行一次正式同步更新。capture 后无 active source，z 只按 decay/propagation 自然下降。

实测 decision timestep=0.5 s。`lambda=0.95` 的解析 half-life=13.51 decisions / 6.76 s；`eta=0.85`。该时间尺度合理，因此没有做工程校准，也没有根据 test 结果扫参。

## 3. Token / architecture

- self：role-free physical 8 dims → `[physical_8, z_i]`，共 9 dims。
- friend：relative physical 6 dims → `[relative_physical_6, z_j]`，共 7 dims。
- friend ordering 保持 physical-only，不使用 role 或 z 排序。
- 保持删除 self/friend `is_pursuing`、role-dependent ordering、`pursuing_embed` 和 late-fusion role branch。
- z 仅经过原 entity MLP/Transformer/summary/IQN head；没有 special z embedding、独立 branch、classifier 或 global information。
- reward、sensing、communication/Voronoi topology 均未改。旧 Final IQN checkpoint 与 `include_is_pursuing=true` 路径保持兼容。

环境保存当前 z、direct/source/hop/age 诊断；observation/replay 同时记录 current z 与 next z。`z_state_dict/load_z_state_dict`、deepcopy、pickle 和 reset 均有回归测试。

## 4. Teacher / student transfer

- frozen Final IQN SHA-256：`2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`。
- 旧 C1 数据缺少可严格恢复 friend identity 的时序映射，因此未猜测重建、未复用旧 replay；重新采集 frozen-teacher matched trajectories并原生记录 z。
- dataset：29,864 rows；direct 1,810、support 1,936、post-capture 12,384、pure coverage 10,812、ring2 64、ring3 16。固定 20 mixed + 20 coverage seeds；自然样本没有 ring3，因此增加一条明示的 geometry-only ring3 类别覆盖轨迹，未改 reward/topology/policy。
- transfer：所有 shape-compatible backbone weights bit-exact 复用；self/friend 第一层末列显式从 teacher role 列迁移为普通 z 列；删除 `pursuing_embed`，`single_action_feature` 只保留 hidden columns。
- 4 epochs BC，0 RL updates。student SHA-256：`e8f38734c462488afea992ada065bd96f8a98c536077ca079ff043c47ed6de9b`。

## 5. Offline action agreement vs B1

| subset | B3 | B1 | delta |
|---|---:|---:|---:|
| overall | 0.4144 | 0.3738 | +0.0406 |
| pre-capture | 0.4976 | 0.3982 | +0.0994 |
| direct | 0.6686 | 0.4944 | +0.1742 |
| support | 0.4954 | 0.2773 | +0.2180 |
| post-capture | 0.3874 | 0.3652 | +0.0222 |
| pure coverage | 0.3941 | 0.3689 | +0.0252 |
| coverage role | 0.3908 | 0.3725 | +0.0184 |
| student-trajectory teacher agreement | 0.1839 | 0.1722 | +0.0116 |

B3 overall categorical KL=1.5740，Q-ranking agreement=0.7695。ring2/ring3 agreement=0.5938/0.6875；normal-capture-transition=0.5595。

恢复主要集中在 direct/support，而 coverage/post-capture 与 B1 的差距很小，不满足“整体显著恢复”。

## 6. Matched rollout（20 seeds / scene / policy）

| scene | policy | capture | normal | collision | CE | safe complete | mean capture s | mean recovery s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pure coverage | Final | — | — | 0.00 | 1.00 | 1.00 | — | — |
| pure coverage | B3 | — | — | 0.10 | 0.00 | 0.00 | — | — |
| pure capture | Final | 1.00 | 1.00 | 0.00 | — | 1.00 | 36.2 | — |
| pure capture | B3 | 0.70 | 0.70 | 0.30 | — | 0.70 | 68.0 | — |
| mixed lifecycle | Final | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 43.0 | 70.3 |
| mixed lifecycle | B3 | 0.75 | 0.75 | 0.35 | 0.00 | 0.00 | 67.9 | — |

B3 pure coverage 的 episode duration mean=1,352.6 s、median=1,500 s，说明多数到完整 horizon 仍未恢复 CE。Mixed 即使捕获也没有一例完成 recovery。

## 7. z 传播 / 衰减诊断

- direct-visible invariant：4,789 checks，`z=1` rate=1.000。
- support-role z：mean/p50/p90=0.834/0.850/0.850。
- coverage-role z：mean/p50/p90=0.0067/0/约 0；pure-coverage z 全程最大值=0。
- neighbor-dominant：6,905 samples，mean z=0.766；max lineage hop=2。
- all-active `z>=0.9` fraction=0.00205；`z>=0.5` fraction=0.0450。无 direct source 时 all-high 最长仅 2 steps，未观察到长期全 swarm `z≈1` 或隐藏 global role-bit 式饱和。
- 解析 `max z<0.1` 需要 45 decisions。13 个有充分 post-capture horizon 的 mixed captures 全部恰在 45 decisions（22.5 s）释放；2 个在 25/32 steps 时因 collision terminal 右删失；有充分 horizon 仍不释放=0。
- 因而失败不是 z 永不 release 或全局泛洪，而是 scalar z 所携带的信息不足以恢复 coverage/recovery policy。

典型 trace：

- capture success：`z_traces/student_capture_2026192801.npz`；
- capture collision failure：`z_traces/student_capture_2026192803.npz`；
- mixed capture-but-no-recovery：`z_traces/student_mixed_2026292801.npz`；
- mixed post-capture collision/right-censored：`z_traces/student_mixed_2026292802.npz`；
- pure-coverage horizon failure：`z_traces/student_coverage_2026092801.npz`。

## 8. Gate / next

分类：**`B3_FAIL`**。

原因：

- overall/off-policy 与 on-policy agreement 相比 B1 仅 +0.0406 / +0.0116；
- pure coverage 0/20；
- mixed recovery 0/20；
- capture 只保留 0.70–0.75，并出现 0.30–0.35 collision；
- z 自身传播/衰减语义正确且无新 role shortcut，但 observation-side scalar replacement 不足。

BC 不够强，formal rollout 又明确失败，因此不满足“中等 gap 但明显优于 B1”的 warm fine-tune 条件；本轮 STOP，不做 short IQN RL，更不做 scratch 2M。

**当前不值得进入 local Voronoi。** 先重审缺失的历史状态表示；不能从本次失败推断局部 Voronoi 能补足该缺口。

机器可读：

- `artifacts/2026-09-18_iqn_zstate_b3/B3_FINAL_AUDIT.json`
- `artifacts/2026-09-18_iqn_zstate_b3/student_report.json`
- `artifacts/2026-09-18_iqn_zstate_b3/rollout_report.json`
- `artifacts/2026-09-18_iqn_zstate_b3/z_diagnostics.json`
