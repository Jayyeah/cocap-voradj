# IQN ROLE-token vs Z-token matched scratch 总账

## 核心裁决

- ROLE HEAD：`9828ca7475522f401c14f24331dd22a46f95988d`
- Z HEAD：`57bb0aa2e5e80fb0b34da5880e059bbcf60d7b6c`
- NormSense-V2 runtime observation hash：`6c2af0df8ebb4fbca9c52a29fb4008a84e0ff90a5804ba633054745bef6f2bcd`
- matched non-token contract hash：`12029ad757c1efe975ea3d04971d7d12f0fb8e00ca5209782dbdd95534777843`
- non-token diff count：`0`
- 初始权重 SHA：`22a7deaa4b7265909a8b6c340f84d8741b69194fd3cc5c643f487ddeb3908ec8`，双边 bit-exact
- physical friend rows matched：`10`；ROLE binary checks：`14`；Z `z_j` checks：`10`

允许训练差异只有五个 evidence-token 开关：`iqn/perception.include_is_pursuing`、`iqn/perception.include_z_state`、`z_state.enabled`。实验 metadata 中 arm 名称与 evidence 描述只作审计，不进入训练合同。

## 正式运行

固定 GPU：ROLE→物理 GPU0，Z→物理 GPU1；两边进程内均使用 `cuda:0`。正式运行于 `2026-09-19T01:09:49+0800` 从 clean scratch step 0 启动。

| Arm | PID | step | replay | updates | loss | loss EMA | steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROLE | 2935643 | 16000 | 64000 | 3304 | 1.4405 | 2.5562 | 28.82 |
| Z | 2935727 | 16000 | 64000 | 3278 | 1.3790 | 2.6811 | 28.82 |

两边 target 均已在 step 10k 同步一次；loss/metrics finite；replay 正常增长；动作分布非固定。首个 production checkpoint/full-resume 与 formal evaluation 在 25k 生成。

## Detached coordinator

- tmux：`ROLE_Z_MATCHED_SUPERVISOR`
- 首次健康启动观测 PID：`2938476`；重启后的当前 PID 以 `status.json` 为准
- status：`artifacts/2026-09-18_iqn_role_z_matched_supervisor/status.json`
- matched ledger：`artifacts/2026-09-18_iqn_role_z_matched_supervisor/matched_ledger.jsonl`
- startup audit：`artifacts/2026-09-18_iqn_role_z_matched_supervisor/STARTUP_AUDIT.json`
- 磁盘 fail-closed 下限：20 GiB；启动窗口剩余约 78 GiB

自动恢复仅允许：训练进程死亡、HEAD/config/runtime hash 未变化、checkpoint 可读且 full-resume 合同有效。合同漂移、错误 GPU、NaN/Inf、replay/checkpoint 损坏或 runtime assertion failure 均停止对应 run 并记录 `failed_closed`。

当前 ETA 为短窗口估计：ROLE/Z 25k 约 `2026-09-19 01:24 +0800`、50k `01:39`、100k `02:08`、200k `03:05`。

```bash
jq '{status,role_runtime,z_runtime,fail_reasons}' artifacts/2026-09-18_iqn_role_z_matched_supervisor/status.json
tmux attach -t iqn_role_token_matched_20260919
tmux attach -t iqn_z_token_matched_20260919
tmux attach -t ROLE_Z_MATCHED_SUPERVISOR
```
