# IQN-ROLE-TOKEN scratch control（2026-09-18）

## 目的与边界

本线回答：在相同当前 IQN 任务合同、算法、训练预算和网络主体下，binary `is_pursuing` token 与连续 propagated `z` token 哪一种更容易从 scratch 学出 coverage → capture → recovery。

本 worktree / branch：

- worktree：`/home/yjq/rl/CoCap1/iqn-role-token-scratch-20260918`
- branch：`experiment/iqn-role-token-scratch-20260918`
- 起点：`origin/experiment/iqn-role-architecture-ablation-20260918`
- 对照线：`origin/experiment/iqn-zstate-b3-20260918`
- 启动 HEAD、PID、GPU、tmux/session 和最新状态：`artifacts/2026-09-18_iqn_role_token_scratch/launch.json`、`status.json`

聊天摘要与仓库 runtime 不一致时，以本仓库当前代码和 resolved artifact 为准。

## 当前合同裁决

最新 IQN/Z runner 实际消费的是 Final CR-MS + VCT-LS + CE 的 4v1 full-lifecycle IQN 合同：120×120 map、4 pursuers/1 evader、AW9、hard boundary/death、synchronized swept collision、CR-MS direct capture、VCT-LS support blend、CE coverage、post-capture recovery、现有 replay/epsilon/target-network 语义。

NormSense-V2 当前已接入独立 Forward-Final/MAPPO 流程，但本次 fetch 的 IQN/Z 配置没有消费 `forward_final_v2` 或 density-normalized sensing resolver。按“最新仓库 contract 优先”规则，本线与 Z 严格匹配当前实际 IQN/Z runtime，并保留此边界，未把 NormSense-V2 配置文字强行混入 IQN runner。

ROLE 与 Z 的 resolved `reward`、`env`、`voradj`、任务 schedule、动力学、AW9、termination、replay 和 IQN hyperparameters 完全一致；允许差异只有 policy observation 的最后一维语义，以及 ROLE 为显式 `friend_ordering_mode: physical_only` 的等效实现开关。

## ROLE architecture

- self token：physical self features + binary `is_pursuing_i`；shape `(9,)`。
- friend token：relative physical features + binary `is_pursuing_j`；shape `(8, 7)`。
- enemy / obstacle token：沿用原 entity encoder 输入。
- 保留原 entity MLP、Transformer、summary、single-head IQN。
- 删除 `pursuing_embed`、专门 late-fusion branch 和其他 role shortcut；`pursuing_late_fusion=false`、`pursuing_embed_dim=0`。
- friend ordering 强制 `physical_only`，不得按 role、capture/support 身份或 `is_pursuing` 排序/截断。
- Z 线的 role-free branch 已是相同 physical-only ordering；preflight 对同 seed 的 physical friend token 顺序做 exact compare。

## Scratch 训练合同

配置：`configs/experiments/iqn_role_token_scratch_20260918/role_token.yaml`

- total target：`200000` environment steps；自动停止，不超过该值。
- checkpoints/evaluation：25k、50k、75k、100k、125k、150k、175k、200k。
- batch `128`；gamma `0.99`；Adam；base LR `1e-4`；沿用 Z 的 optimizer、epsilon、n-step/quantile、replay、target update 和 curriculum 设置。
- epsilon：`0.6 → 0.05`，沿用 Z 的 `1,000,000`-step schedule；run 在200k前不进入后续 LR schedule。
- checkpoint exact resume：`checkpointing.full_resume=true`，`resume_latest.pt`。
- 禁止 Final actor/checkpoint warm-start、BC、distillation、C0/C1/B3 权重初始化。
- 训练主体从统一随机初始化开始；teacher 只可用于历史审计，不能进入本次 actor。

## 启动前 sanity / preflight

机器可读结果：`artifacts/2026-09-18_iqn_role_token_scratch/preflight.json`。

状态：`PASS`。

已确认：

1. ROLE self/friend token 存在，shape 正确，token 为 binary；
2. late fusion/pursuing embed 不存在；
3. ROLE/Z physical friend ordering exact match；
4. reward、env/termination、voradj 与 Z exact match；
5. 无 pretrained path，随机初始化参数非零；
6. 真实 VorAdj env rollout 64 steps；
7. 真实 IQN optimizer updates `63` 次；
8. loss finite，最后 loss `0.4174326062`，loss EMA `0.7097943232`；
9. 参数发生变化；
10. replay insert/sample 正常，sample 8 rows；
11. target network update 正常；
12. checkpoint load exact、full-resume exact，resume step `64`；
13. formal evaluator smoke complete；
14. GPU1 为 NVIDIA RTX A6000，preflight allocated/reserved 显存约 `229/270 MiB`。

## Formal evaluator 与 trend ledger

入口：`tools/evaluate_iqn_role_token_formal_20260918.py`。

每个 milestone 对固定 seed base `2026092801`，每 scene 至少20局，deterministic fixed-midpoint-32 quantiles，统一记录：

- pure coverage：strict CE、collision、CE RMS、area CV、time-to-CE；
- pure capture：normal/stationary capture、ring2/ring3、collision、capture time；
- mixed lifecycle：capture、collision、post-capture CE、safe complete、capture/recovery/mission time；
- action histogram 及逐局 episode record / mission events。

supervisor：`tools/supervise_iqn_role_token_scratch_20260918.py`。

- trend ledger：`artifacts/2026-09-18_iqn_role_token_scratch/trend_ledger.jsonl`；
- status/heartbeat：`artifacts/2026-09-18_iqn_role_token_scratch/status.json`；
- train/eval logs 与 checkpoint 只留本地，不进 Git；
- supervisor 不会自动越过 200k，也不会因25k暂时无最终成功而早停。

## 长时运行与恢复

启动命令：

```bash
tmux new-session -d -s iqn_role_token_scratch_20260918 \
  "cd /home/yjq/rl/CoCap1/iqn-role-token-scratch-20260918 && \
   python3 tools/supervise_iqn_role_token_scratch_20260918.py \
   --device cuda:1 --episodes 20 \
   --preflight artifacts/2026-09-18_iqn_role_token_scratch/preflight.json"
```

最短查看：

```bash
tmux attach -t iqn_role_token_scratch_20260918
tail -f /home/yjq/rl/CoCap1/iqn-role-token-scratch-20260918/artifacts/2026-09-18_iqn_role_token_scratch/status.json
```

若会话退出，使用同一 supervisor 命令恢复；它读取 `status.json` / `resume_latest.pt`，按下一个未完成 milestone 执行 exact resume。不要直接用普通 model checkpoint 恢复 optimizer/replay 状态。

ETA 使用 supervisor 已完成 milestone 的实际稳定 steps/s 动态计算；启动初期若稳定窗口尚未形成，status 中明确记为 unavailable，不用配置猜测替代。

## 已知 caveat

- NormSense-V2 与当前 IQN/Z runner 的接入边界已记录；若后续将 NormSense-V2 纳入 IQN，必须另建 matched contract，不得把本结果与不同 sensing runtime 合并。
- ROLE token 是 policy observation 输入，但 evaluator 仍可事后记录 effective role 作为分析 metadata；metadata 不进入 action tensor 以外的额外 branch。
- preflight 的64步 smoke 使用缩短 replay threshold/单 `voradj` task 以验证 optimizer plumbing；正式 run 使用完整 `voradj_mixed_coverage` 和生产 replay/config。
