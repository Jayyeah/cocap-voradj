# AC-0 — BC / Critic Lineage Recovery（2026-09-20）

范围：只读恢复、代码静态审计、已有结果整理；本轮没有训练、rollout、critic fitting、GPU 实验或代码/config 修改。

仓库：`/home/yjq/rl/CoCap1/cocap-voradj`。`git fetch --all --prune` 已按要求执行，但远端连接被当前环境代理拒绝；以下 refs 是本地已存在的事实，不冒充 2026-09-20 的远端刷新结果。

## A. Canonical strong BC

### 身份与 lineage

当前恢复分支为 `audit/ac-bc-critic-lineage-20260920`，基于
`experiment/critic-identifiability-audit-20260917@53dfa8344104640bbaf4993a6edd5dbf4e0b205d`。
Full-Task BC 的正式 ancestry 为：

- `f890b33a3ae67722e470f0528fb5ed2b985cc6b7`：canonical Final IQN 与旧 BC task lineage/parity 审计；
- `f9532d2dcc49aad36d729bfc8e0eb393e4ed59c8`：Forward Final teacher + categorical distillation bridge；
- `64e11b140288bd7a71f5b24eb3f98ce376d7da1b`：C3 formal100 完成并通过 frozen-task gate；
- `515d54d66d91d42393b69b5961fd49ffda511b71`：full-task efficiency / critic-calibration gate 记录。

### frozen actor checkpoint

当前 AC critic audit 若要冻结 BC actor，应使用这个 categorical actor：

- 逻辑 canonical path：`artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt`；
- SHA-256：`7916a970226a0452a8f71a22235524f6ba4511aa4c9bda907a9c8368a60bf2cd`；
- 当前 audit worktree 的可用、同 SHA 副本：`artifacts/2026-09-17_critic_identifiability_audit/frozen_policy/actor_epoch_030.pt`；
- 原始 C2 path 在当前 worktree 中未保留二进制，故“原始路径文件存在”标记为 `NOT PROVEN`；同 SHA 副本已实际校验。

它就是历史 strong Full-Task BC，不是 later IQN cleanup/Z student，也不是 current critic bank 的 generator。

### architecture / environment contract

- Teacher：Final IQN Stage1 4v1@2M，`artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt`，SHA-256 `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`。
- Student：9-action categorical `CategoricalGridActor`；严格复制 IQN decision backbone 的 86 个 keys 并冻结；只训练 `policy` hidden head 与 9-way `logits_head`。
- Task：120×120，mixed 4v1/1 obstacle；pure coverage 4v0/1 obstacle；完整 capture→post-capture→coverage lifecycle。
- Sensing：enemy/obstacle 20m surface-clearance；friend-only VorAdj/free-mask projected topology；无 global enemy token；map-known / friend-topology 语义沿用 Final。
- Reward：Final ring capture、support `0.5×approach + 0.5×CE`、coverage CE/PBRS；`gamma=.99`；capture 后不立即 terminal，post window 500，CE RMS≤.05/max≤.10 持续 30 步；总 horizon 3000。BC loss 不使用 reward，只使用 teacher Q/action。
- Action：AW9 categorical grid，`a∈{-0.4,0,0.4}`、`ω∈{-π/6,0,π/6}`，物理 `vmax=3`、dt `.05×10=.5s`。

### historical rollout evidence

`artifacts/2026-09-08_forward_final/c3_formal100/comparison.json` 的 frozen-task gate 为 `PASS`，同一 evaluator、seed/fingerprint 配对：

| 指标 | Final IQN greedy | BC argmax | BC sample |
|---|---:|---:|---:|
| mixed normal capture | 100/100 | 99/100 | 99/100 |
| mixed CE / safe | 99/100 | 98/100 | 98/100 |
| mixed collision | 1/100 | 1/100 | 2/100 |
| mixed mission mean / P90 (s) | 114.48 / 160.9 | 117.02 / 163.55 | 112.98 / 157.9 |
| pure coverage CE / safe | 99/100 | 100/100 | 100/100 |
| pure coverage P50 / P90 (s) | 63 / 106.3 | 62.5 / 102.2 | 64 / 102 |

因此“历史 strong Full-Task BC”已确认；该证据不是 PPO 增益或大规模泛化证明。

主要文档与 artifacts：

- `docs/FORWARD_FINAL_MAPPO_BRIDGE_LEDGER_20260908_ZH.md`
- `docs/FINAL_IQN_MAIN_MAPPO_BC_CONTRACT_PARITY_20260908_ZH.md`
- `docs/FORWARD_FINAL_CORRECTED_PPO_LEDGER_20260909_ZH.md`
- `docs/FORWARD_FINAL_PPO_ROOT_CAUSE_20260909_ZH.md`
- `artifacts/2026-09-08_forward_final/{c0_formal20,c1_dataset,c2_distillation,c3_formal100}`
- trainer：`tools/collect_forward_final_dataset_20260908.py`、`tools/distill_forward_final_actor_20260908.py`

## B. BC pipeline：当前仓库有几套

按独立 trainer/contract 计为 3 套；按用户要求的“strong BC vs later cleanup student BC”计为 2 类：

| pipeline | 用途 / contract | 是否 historical strong BC | 是否 later loss-risk line |
|---|---|---:|---:|
| 2026-09-03 `mappo-iqn-distillation` | 旧 MAPPO-9-v2、capture-only、Stage2 8v2@300k teacher；600 episodes / 171,052 rows；student categorical，旧合同 formal gate PASS | 否，旧 capture-only BC | 否 |
| 2026-09-08 `forward-final-categorical-bc-v1` | Final IQN→categorical Actor，mixed + pure coverage；200 episodes / 146,612 rows；C3 full-task gate PASS | 是 | 否 |
| 2026-09-17 B1 / 2026-09-18 B3 `iqn-cleanup` / `z-state` | IQN student 做 no-is-pursuing 或 z-state ablation；复用/重采 C1 teacher data；不是 MAPPO actor | 否 | 是 |

### BC trainer contract 静态结论

Strong Full-Task C2：

- Teacher Q/action：`fixed_midpoint_q` 使用 `tau=(j+0.5)/32`、32 个 IQN quantiles，`Q_T=mean_j Z_tau_j`，`greedy=argmax_a Q_T[a]`；epsilon 为 0。
- Dataset：完整 teacher trajectories，mixed/pure 交替、无 success filtering；C1 100 pairs = 200 episodes，146,612 active-agent rows；train 116,316，validation 30,296；episode-pair split，不按 agent row 随机切分。
- Student output：9-way categorical logits，不是 IQN quantile output。
- Loss：
  `p_T=softmax((Q_T-max(Q_T))/T)`，`p_S=softmax(logits_S)`，
  `L=KL(p_T || p_S)`；hard action CE 只用于报告，不是训练附加项。
- Teacher temperature：`T=0.0031884169327952754`，只进入 teacher target distribution；这是 categorical-logit distillation contract，不是遗漏的 student IQN temperature。
- Quantile count：teacher 32τ；student 无 quantile count（categorical head）。因此 `teacher 32τ vs student 8τ` 不适用于 strong Full-Task BC。
- Frozen/trainable：backbone 86 keys frozen；policy head + logits head trainable；forward 保持 eval mode。
- Optimizer/budget：Adam `lr=3e-4, eps=1e-5`，grad clip 1.0，batch 2048，30 epochs；按 validation macro-phase agreement 选 best，epoch 30。

Later B1/B3 student line：teacher Q 用 32τ；student 训练 forward 用 8τ、评估再用 32τ；loss 是
`KL(softmax(Q_T/T) || softmax(Q_S)) + 0.25×centered-Q smooth-L1`，student 分支没有对应的 `/T`。这正是后者的静态 contract 风险，不应回写到 strong categorical BC。

## C. strong BC 是否有明显 loss contract 问题

静态判断：**没有发现足以否定 strong Full-Task BC 的明显 loss contract 问题。**

但“teacher 与 student 输出完全相同”必须按同一输出空间解释：

- 若指 `p_S == p_T`（即 student logits 与 `Q_T/T` 只差 action-independent 常数），则 `KL=0`，理论 gradient=0；这是正确 stationary condition。
- 若指 student logits 原样等于 raw `Q_T`，则因 `T=.003188...`，一般并不满足 `p_S=p_T`，loss/gradient 不必为 0。这是显式 temperature transform 的结果，不是 strong BC 已被证明错误。
- 后来 IQN student 的 `32τ→8τ` 与 teacher-only temperature 则使“raw teacher-copy stationary”不成立；该问题只归属于 later cleanup/Z student line。

## D. current critic bank：由谁生成

当前 bank：`artifacts/2026-09-17_critic_identifiability_audit/trajectory_bank.npz`。

- manifest：`artifacts/2026-09-17_critic_identifiability_audit/trajectory_bank_manifest.json`；bank SHA-256 `ad5607c82ba077c1cbde99e852c8f8201bbf13b7dbfef816691136b206e3a5ee`。
- generator actor：Final IQN Stage1 4v1@2M，policy `iqn_epsilon05`，checkpoint SHA `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`；actor state hash before/after 相同 `242cc13f4c40b69afa670ca2991aca079bc66df7449e15a5224c072fdbde73cc`。
- **不是 canonical strong categorical BC**。它由 IQN 以 epsilon=.05 生成；`frozen_policy/actor_epoch_030.pt` 只作为 bank worktree 中的 BC checkpoint copy 存在，没有被该 bank 的 collector 使用。
- contract：NormSense-V2 4v1；60 episodes（49 success / 11 failure），16,728 transitions，66,912 active-agent rows。
- MC target：完整真实 trajectory empirical discounted return，`G_t=Σ_k γ^k r_(t+k)`，`γ=.99`；不使用 learned target；episode 重算最大误差 `1.5232e-5`。

事件总数（active-agent rows）：

| event | rows | event | rows |
|---|---:|---|---:|
| pure coverage | 29,764 | coverage steady | 21,300 |
| ordinary pursuit | 8,288 | ring2 | 1,620 |
| ring3 | 188 | normal capture | 164 |
| capture terminal | 72 | early recovery | 3,440 |
| late recovery | 23,612 | pure coverage restart | 1,440 |
| collision | 16 | stationary capture | 0 |

split：train 11,572 transitions / 46,288 rows；validation 1,284 / 5,136；test 3,872 / 15,488。episode/seed 无交集；collision rows 为 train/validation/test=`4/4/8`；phase/event labels 已在 manifest 中保存。

## E. critic audit：最新做到哪

最新 critic branch：`experiment/critic-identifiability-audit-20260917`，HEAD
`53dfa8344104640bbaf4993a6edd5dbf4e0b205d`；本地未发现后续 A2/bootstrap branch 或已执行的 A2 artifact。

A0 gate PASS；A1 只做 MC-supervised fitting，formal classification 为 **`LOCAL_SUFFICIENT`**。A2 native bootstrap、A3 frozen-actor critic pretrain、A4 online policy improvement 均未执行；actor updates=0，bootstrap=未使用。

### Heldout test metrics（15,488 active-agent rows）

| critic | RMSE | MAE | EV | Pearson | Spearman | calib slope / intercept | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| LQ | 34.835 | 12.414 | .593 | .775 | .398 | .905 / 2.851 | 3.141 |
| NQ | 32.592 | 11.350 | .643 | .802 | .445 | .989 / -0.778 | 1.636 |
| CQ | 42.393 | 12.844 | .398 | .670 | .426 | .747 / .924 | 5.717 |
| V | 43.277 | 13.109 | .375 | .651 | .406 | .747 / -.183 | 5.429 |

phase/event RMSE（LQ / NQ / CQ / V）：

- pure coverage：`4.04 / 3.63 / 2.29 / 2.46`；ordinary pursuit：`86.30 / 84.00 / 102.75 / 105.95`；
- ring2：`79.08 / 72.34 / 121.14 / 119.34`；ring3/normal capture：`102.29 / 100.21 / 147.12 / 149.92`；
- collision：`66.30 / 96.03 / 74.55 / 76.93`；early recovery：`65.79 / 49.46 / 66.84 / 68.47`；
- late recovery：`3.82 / 2.64 / 2.67 / 2.08`。

action-ranking（twin AW9 counterfactual stability，不是真实 counterfactual return）：

| Q | twin rank Spearman mean / p10 | twin top-1 | predicted success>collision |
|---|---:|---:|---:|
| LQ | .045 / -.867 | .185 | .600 |
| NQ | .107 / -.817 | .208 | .594 |
| CQ | .360 / -.222 | .283 | .125 |
| V | N/A | N/A | .225 |

真实 success>collision pairwise 为 `.713`。因此 `LOCAL_SUFFICIENT` 只说明 local input 可拟合 realized-policy MC return，不说明反事实动作排序可靠。

机器证据：`artifacts/2026-09-17_critic_identifiability_audit/{a0_gate.json,trajectory_bank_manifest.json,critic_comparison.json,predicted_vs_mc_scatter.png}`；comparison SHA-256 `170d79de4014971036afb59757d1365acb2f890fe9ad0b24e293e7456552436b`。

## F. Missing evidence

1. 远端 fetch 未成功，不能证明本地 refs 已包含远端 2026-09-20 之后的任何更新。
2. 当前 bank 不是 canonical strong BC bank；若 AC critic audit 的 frozen actor 定义必须是 strong BC，最小缺口是重新建立由 `actor_epoch_030.pt` 同 SHA 生成的 matched bank，并保存真实 env/reward/sensing/action contract、generator SHA、source snapshot、独立 split 与 MC return manifest。
3. A2/bootstrap 尚无执行证据；本轮不启动 A2，不把 A1 `LOCAL_SUFFICIENT` 扩写成 bootstrap 或 online-RL 结论。
