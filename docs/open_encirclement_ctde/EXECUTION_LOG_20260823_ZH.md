# Open Encirclement CTDE 整合复现执行记录（2026-08-23）

## 隔离与事实快照

- worktree：`/home/yjq/rl/CoCap1/cocap-voradj-open-ctde`
- branch：`repro/open-ctde-20260823`
- base：`3ee4d94909a163b5f54689b6a4b1e62ac4baf286`
- 未改动参考线 `ablation/all-agent-oldmix-20260810`、主线和历史 `repro/maadpg-20260823`；未读取或 warm-start 既有 CoCap replay/checkpoint。
- 初始资源：2×RTX A6000 48 GiB 空闲；RAM 约 116 GiB available；根盘约 83 GiB available；无其他 Python 训练进程。
- 依赖：见 `requirements-lock.txt`。项目 `.venv` 使用 `--system-site-packages` 复用 PyTorch 2.9.1+cu128，只增装 Gym/Gymnasium、TensorBoardX 与 imageio-ffmpeg。

## R-1：上游冻结

| repo | commit | snapshot | license | LFS |
|---|---|---|---|---|
| light_mappo | `c503d89b6f28c9687ce9e45304fe66b57322ce1e` | `vendor/upstream/light_mappo/<sha>/` | MIT | none |
| MADDPG_Multi_UAV_Roundup | `15309de231f639c62d2049b0ad5b07b8975309c9` | `vendor/upstream/MADDPG_Multi_UAV_Roundup/<sha>/` | MIT | none |
| KF_AA_MARL | `c8d68cab016ce9e6b18f8435b2faa0048d59da41` | `vendor/upstream/KF_AA_MARL/<sha>/` | MIT | none |

完整 URL、entrypoint、依赖、模型清单、size 与逐文件 SHA256：`upstream_manifest.json`。快照总计 140 个文件、约 7.7 MiB；clone 的历史对象不进入本工程。

## R0：Upstream-Raw

命令：

```bash
.venv/bin/python tools/open_encirclement_ctde/evaluate_upstream_raw.py \
  --snapshot vendor/upstream/MADDPG_Multi_UAV_Roundup/15309de231f639c62d2049b0ad5b07b8975309c9 \
  --output-dir artifacts/open_encirclement_ctde/r0_upstream_raw \
  --episodes-per-mode 20 --gif-count 2
```

结果：

| mode | episodes | raw capture rate | mean length | containment fraction | collision fraction |
|---|---:|---:|---:|---:|---:|
| deterministic | 20 | 60% | 88.05 | 0.1115 | 0.0000 |
| upstream original noise | 20 | 15% | 95.50 | 0.0225 | 0.1445 |

四个 actor/critic/target 网络集合均能在当前 CUDA/PyTorch 加载；观测维度 `[26,26,26,23]`，动作均为二维，最大动作 norm 约 0.04。报告为 `artifacts/open_encirclement_ctde/r0_upstream_raw/report.json`，共保存 4 个 GIF。这里保留 reward slice、精确面积相等、全矩阵速度 norm 等 raw bug，结果不得与 corrected baseline 混算。

另以相同 deterministic seeds 7100–7101、noise seeds 8100–8101 重跑；逐 episode JSON 与正式 R0 对应四条完全相等，固定 seed Gate 通过。

## R1：corrected_roundup_v1

实现路径：`src/open_encirclement_ctde/`。固定合同为 3 learned hunters + 1 scripted target、2×2、3 静态圆障碍、world `[ax,ay]`、dt=.5、100 steps、26 维 full-target-information、三 hunter 非退化三角形包含 target 且全部距离 ≤.3 的 instantaneous success。

明确修正：

1. stage/terminal reward 覆盖全部三名 hunter；
2. 速度逐 agent 独立限幅；
3. 策略输出、环境声明、执行与 replay 使用同一逐元素 ±.04 物理动作；
4. 使用带尺度容差的 oriented-area/barycentric 判定，退化三角形不算包含；
5. target 固定为 light_mappo 实际代码的 nearest-hunter flee + wall repulsion；
6. capture=`terminated`，100-step timeout=`truncated`；
7. metrics 暴露 hull/radius/hold/angular gap/distance/progress/collision/action/speed；
8. 环境 RNG、obstacle、状态和 runtime 可完整保存/恢复。

未加入 Legacy-VorAdj、K10、APF evader、body `(a,w)`、support/coverage、swept collision 或 post-capture。

Oracle 使用 3 秒有限前视的三角 slot 拦截器，仅作为环境可解性 sanity check；moving scripted target、无障碍的 seeds 0–9 为 10/10 成功。20-seed random policy success 不超过预注册上限 2/20。

## R2：算法与恢复

MAPPO 直接复用冻结 light_mappo 的 MLP actor/critic、PPO trainer、shared rollout buffer、GAE、Huber loss 与 ValueNorm。由于 upstream Box policy 是无界 Normal，本地 bridge 只替换为 tanh-squashed、Jacobian-corrected log-prob 的物理动作头；shared actor 输入 26，centralized V 输入 concat obs 78。实际 max grad norm 明确为 10.0。

MADDPG 只训练三 hunter：三个 independent local actors 输入 26，三个 centralized critics 各读取 joint obs 78 + joint physical action 6；128×128、actor LR 1e-4、critic LR 3e-3、gamma .99、tau .01、replay 1M、batch 256。Scheduler 明确沿用 upstream：actor StepLR(1000,.8)，critic StepLR(5000,.33)。teammate replay action 停止梯度，critic target 只用 `terminated` 阻断 bootstrap，`truncated` 的 terminal next observation 保留 bootstrap。

完整 checkpoint 使用原子 replace；MAPPO 包含 actor/critic、optimizers、ValueNorm、RNG、runtime、环境状态；MADDPG 另包含 actors/critics/targets、schedulers 与 replay ring。25k milestone 为 model-only；训练中只有一个 `rolling-full.pt`，最终 `final-full.pt` 验证后删除被取代 rolling 并写 cleanup manifest。

## 验证记录

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/open_encirclement_ctde
18 passed in 4.14s
```

| smoke | device | steps | updates | throughput | 关键健康指标 |
|---|---|---:|---:|---:|---|
| MAPPO CPU | cpu | 32 | 1 | 22.25 steps/s | value/policy/entropy/grad finite；ValueNorm active |
| MADDPG CPU | cpu | 32 | 1 | 24.40 steps/s | Q/TD/actor/critic grad finite；replay=32 |
| MAPPO CUDA | cuda:0 | 2000 | 10 | 233.49 steps/s | 从 step 600 rolling full 恢复；all finite |
| MADDPG CUDA | cuda:1 | 2000 | 98 | 233.66 steps/s | replay=2000；full replay bundle 实际加载；all finite |

CUDA smoke 暴露并修复两项恢复问题：ValueNorm `denormalize` 返回 NumPy；CUDA `map_location` 会把 CPU RNG ByteTensor 移到 GPU。两项均通过真实 checkpoint resume 验证。

## L0 调度与预算

六条正式配置位于 `configs/open_encirclement_ctde/`。双 lane：

- GPU0：MAPPO seed1 → MADDPG seed2 → MAPPO seed3；
- GPU1：MADDPG seed1 → MAPPO seed2 → MADDPG seed3。

supervisor 使用非阻塞文件锁、PID/run state、原子状态写入和 fail-closed 队列；任一 run 失败不会停止另一张卡上已运行进程，但不会启动新的 queued run。

按 smoke 吞吐，单条 150k 纯训练约 10.7 分钟；25k/50k/75k/100k/125k/150k 的 20-rollout deterministic eval 会增加墙时，保守按每条 12–18 分钟估算。每张卡串行三条约 36–54 分钟。MADDPG 在 150k 有效 replay 约 193 MiB；单 seed 同时保留 rolling/final 的峰值约 400 MiB，final 验证后降至约 200 MiB。六条正式线加模型、日志和 GIF 预计低于 1.5 GiB，显著低于启动前 83 GiB 可用空间。

## L0 正式启动验收

- 启动代码提交：`0547dc35ebbabf8fc3bdb43685da10539f85946f`。
- 启动时间：2026-08-23 11:50:28 +08:00；tmux：`open_ctde_l0_20260823`；supervisor PID：233110。
- GPU0 worker PID 233196：`l0_mappo_seed1_150k`；队列为 MADDPG seed2 → MAPPO seed3。
- GPU1 worker PID 233197：`l0_maddpg_seed1_150k`；队列为 MAPPO seed2 → MADDPG seed3。
- supervisor 状态为 `RUNNING`、failed 为空；队列使用 fail-closed 语义，当前 run 成功后自动接续，失败时不会误启同 lane 的后续 seed。

固定观测窗为 11:50:28–11:53:32 +08:00：

| lane | 固定快照 step/update | 实测吞吐 | steps/hour | 首个 25k | 150k 绝对 ETA |
|---|---:|---:|---:|---|---|
| MAPPO seed1 / cuda:0 | 55,000 / 275 | 303.06 steps/s | 1,091,002 | 11:51:54 已完成评估 | 11:58:45 +08:00 |
| MADDPG seed1 / cuda:1 | 38,312 / 3,729 | 210.95 steps/s | 759,406 | 11:52:30 已完成评估 | 12:02:21 +08:00 |

两次间隔观测均显示 step/update 单调增长，且两条线跨过 25k milestone eval 后继续训练。MAPPO 的 value/policy/entropy/actor-grad/critic-grad 均为有限值且 `value_normalizer=true`；MADDPG 的 Q/TD/actor/critic 指标均为有限值，固定快照 replay=38,312。两份 `rolling-full.pt` 均由训练进程原子写入并标记 `rolling_full_verified=true`，随后又在独立 CPU 进程中直接 `torch.load`：

- MAPPO bundle schema 为 `open-encirclement-full-v1`，包含 actor/critic、两 optimizer、ValueNorm、RNG、runtime、8 个环境状态和 observations；
- MADDPG bundle 使用相同 schema，包含 actors/critics/targets、optimizers、schedulers、完整 replay、RNG、runtime、8 个环境状态和 observations；25k 载入快照的 replay count 精确为 25,000。

验收时 GPU0/GPU1 显存分别为 371/373 MiB（总计各 49,140 MiB），温度 64/72°C；系统 RAM 125 GiB、available 114 GiB、swap 0；根盘 available 82 GiB。没有 OOM、traceback、NaN/Inf、checkpoint 或 supervisor 错误。至此未启动的四个 seed 已可靠挂入自动调度，后续唯一事项是等待正式 step 增长。

## L0 最终完成与性能结论（2026-08-27 回填）

六条 150k 训练已于 2026-08-23 12:26:19 +08:00 全部完成；supervisor=`COMPLETED`、6/6 completed、failed 为空、队列为空。MAPPO 三 seed deterministic/stochastic capture 均为 2/60（3.33%）；MADDPG 分别为 1/60（1.67%）和 0/60。MAPPO 平均碰撞约 9%，MADDPG 约 63%。

工程与恢复 Gate 全部通过，但“稳定围捕”性能 Gate 未通过。MAPPO 更常让三机同时接近 target，却几乎不形成 containment；MADDPG 的 seed2/seed3 出现高碰撞退化。完整逐线、逐模式、里程碑表现与结论见 `L0_FINAL_RESULTS_20260823_ZH.md`。
