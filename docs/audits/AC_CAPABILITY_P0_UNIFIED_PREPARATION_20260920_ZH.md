# CoCap Actor-Critic Capability Baseline：P0 统一前期准备

日期：2026-09-20（Asia/Shanghai）
状态：准备完成；本轮没有启动 25k/50k/75k/100k 正式训练。

## 1. Base / HEAD

- `BASE_SELECTION`: 新分支从 `audit/ac-bc-critic-lineage-20260920` 的 `4ad4cb72060142ed2bce6b33ccd7fe9c9f123b` 创建，而不是从 `origin/main` 直接创建。
- 理由：服务器本地 `git fetch --all --prune` 后，`origin/main` 为 `a5814f49fa29d869cdc3fb8d8e0df4722aa11f00`（2026-08-04），而当前 AC audit branch 含最新 AC-5B2 审计上下文；Final IQN contract/config 也在该 audit lineage 可直接复核。没有丢弃或 cherry-pick 不必要的历史修复。
- 实验分支：`experiment/shared-local-offpolicy-ac-capability-20260920`。
- 相关历史 refs 已读取：Final IQN / IQN branches、`small-step-ac-migration-20260828`、`td3-local-aw-stage1-20260916` 等。
- Git fetch 使用了 mihomo `127.0.0.1:17892`；没有修改 shell、global git config 或 mihomo 配置。

## 2. Final IQN recovered contract

事实源为：

- `configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/common.yaml`
- `configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/stage1_4p1e1obs_scratch2m.yaml`
- 其父配置 `configs/experiments/voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml`
- `src/cocap_voradj/envs/voronoi_adjacency.py`
- `src/cocap_voradj/training/trainer.py`

恢复结果：Stage1 是 4 pursuers / 1 evader，map 120×120，one obstacle，episode horizon 3000；pure coverage scene 是 4 pursuers / 0 evader。Robot runtime defaults 是 `dt=0.05`、`N=10`、decision interval 0.5 s，pursuer `v_max=3.0`，APF evader 保留 Final IQN 的 APF-v2 fixed 合同。碰撞语义显式固定为 `legacy_end_step`，不是 swept-collision 迁移。

Observation 保留 Final IQN 的 VorAdj contract：self width 9（包含 `is_pursuing`）、friend/enemy width 7、obstacle width 5、max pursuers 8 / evaders 8 / obstacles 5；VCT-LS friendly-only topology、free-mask projected obstacle mode、enemy/obstacle sensing radius 20、surface distance 和 release-delay 设置保留。没有引入 Z05/Z07、local Voronoi cleanup、通信 cleanup 或删除 role/status feature。

Reward 保留 CR-MS ring capture、CE `centroid_energy_v0`、CE PBRS `phase_and_all_terminal`、Stage1 strict CE thresholds（RMS .05、max .10、hold 30）、capture ring preferred radius 8 / outer 10.5 / radial sigma 2 / progress clip 3。Full Mix capture 后继续 post-capture coverage，post window 500。

## 3. New learner equation

新增 learner 为 `PS-Local-Discrete-AC`（Parameter-Shared Local Discrete Off-Policy Actor-Critic）：

$$
y_i=r_i+\gamma(1-d_i)\sum_{a'=0}^{8}
\pi_{\bar\theta}(a'|o'_i)Q_{\bar\phi}(o'_i,a')
$$

其中 `d_i` 使用 `terminated`，time-limit truncation 不会错误地清掉 bootstrap。

$$
J_\pi=\mathbb E_{o\sim D}\left[\sum_{a=0}^{8}
\pi_\theta(a|o)Q_\phi(o,a)\right],
\qquad L_\pi=-J_\pi
$$

第一版是 single Q、target actor、target critic；没有 twin/min-Q、SAC entropy target 或 distributional critic。Actor 更新时冻结 critic 参数，target 使用 Polyak `tau=0.005`，Adam actor/critic lr 均为 `1e-4`、grad clip `0.5`；这些 optimizer/target 参数来自 repo 现有 local off-policy baseline infrastructure。

## 4. Shared Actor / Shared Local-Q architecture

- Shared Actor：一个网络，对所有 active pursuer rows 应用同一组权重，输出 9 logits。
- Shared Local-Q：一个独立网络，对所有 active pursuer rows 应用同一组权重，一次输出 `Q(o,a0)...Q(o,a8)`。
- Actor 与 critic 不共享参数；两者可使用相同 Final-IQN-compatible local encoder architecture，但权重独立。
- agent ID 不进入 actor/critic，只存为 replay provenance metadata。
- formal network：hidden 256、8 heads、4 transformer layers、AW9、self 9、max P/E/O=8/8/5。
- formal parameter counts：Actor `3,965,977`，Critic `3,965,977`。
- 三条线均 scratch actor + scratch critic；没有 BC actor、IQN encoder 或 AC audit LQ warm-start。

AW9 live runtime assert 在 `src/cocap_voradj/dynamics/robot.py` 上通过，顺序为：

```text
0 (-0.4, -pi/6)   1 (-0.4, 0)       2 (-0.4, +pi/6)
3 ( 0.0, -pi/6)   4 ( 0.0, 0)       5 ( 0.0, +pi/6)
6 (+0.4, -pi/6)   7 (+0.4, 0)       8 (+0.4, +pi/6)
```

## 5. Exploration

Environment behavior policy 显式为：

$$
\mu(a|o)=(1-\epsilon)\pi_\theta(a|o)+\epsilon/9
$$

因此每个 AW9 action 的概率下界是 `epsilon/9`。直接 transfer Final IQN schedule：`epsilon=0.6 → 0.05 over 500,000 env steps`。AC actor objective 不加隐式 entropy tuning；探索只发生在 collection behavior policy。

replay row 保存 `behavior_probability`、`epsilon`、`actor_probability`；第一版不做 importance sampling。

## 6. Replay

所有 active pursuer rows 进入一个 shared replay，单 row 保存：`obs/action/reward/next_obs/done/terminated/truncated/phase/replay_class/behavior_probability/epsilon/actor_probability/episode_id/agent_id/metadata`。没有 per-agent buffer。

Formal Full Mix semantic sampler 固定为：

```text
pursuing          64
pre_capture_cover 16
post_capture_real 32
recovery_pure     16
total             128
```

正式配置 `replay_fallback: defer`：class 不足时等待，不静默挪用其它 class。短 smoke 仅为验证 learner 更新而显式允许 `uniform_fallback`，并在 sampler report 中记录 fallback；没有伪造 transition。

## 7. Pure Coverage contract

配置：`configs/experiments/ac_capability_20260920/ac_capability_cov_stage1.yaml`。

只使用 `voradj_coverage`，0 evader/0 capture，`inner_random_cluster` 初始化，coverage reward 直接继承 Final CE contract；不使用 Full Mix capture snapshot pool。transition 记为 `recovery_pure`，但采样使用 shared uniform（没有人为套 Full Mix 64/16/32/16）。目标指标为 strict CE success、CE RMS、area CV、collision、episode return、completion time。

## 8. Pure Capture contract

配置：`configs/experiments/ac_capability_20260920/ac_capability_cap_stage1.yaml`。

只使用 `voradj`，1 evader，coverage CE reward、coverage PBRS、post-capture coverage reward 全部为 0；capture terminal 为 true。新增的 `pure_capture_reward_contract: final_ring` 只服务本 AC capability CAP，不改变旧 `pure_capture_all_capture_enabled` strong-BC contract：直接 VCT-LS detector 使用 Final CR-MS ring reward；one-hop informed support 使用 approach-only attraction；uninformed support 不得收到 oracle dense reward。CAP 只写 `pursuing` rows，不保留无意义 coverage rows。目标指标为 normal/stationary capture、2+ ring、3+ ring、collision、capture time、return。

## 9. Full Mix contract

配置：`configs/experiments/ac_capability_20260920/ac_capability_mix_stage1.yaml`。

除 learner 外尽量与 Final IQN Stage1 一致：mixed 1 evader capture scene 与 pure coverage scene 交替；capture 不在 capture event 立即结束，而是进入 post-capture coverage；Final IQN recovery pool、reward、horizon、observation、AW9、碰撞语义全部复用。目标指标为 capture rate、normal capture、post-capture CE、safe/full completion、early-recovery duration、collision、mission time、replay class fill counts。

## 10. Episode alternation

`FINAL_IQN_EPISODE_SCHEDULER_RESOLVED`

来源：`src/cocap_voradj/training/trainer.py` 的 `CoCapTrainer._reset_task`；`train_mode==voradj_mixed_coverage` 时 `task_order=["voradj", "voradj_coverage"]`，每个 episode 完成后 cursor 加一，按 episode 而不是 env-step 切换。因此实际序列为：

```text
voradj, voradj_coverage, voradj, voradj_coverage, ...
```

AC runner 对 MIX 复用这个 episode-level 1:1 scheduler。smoke 前 24 次 reset 的实际序列与该 reference 完全一致。

## 11. Cross-init

来源：`trainer.py` 的 `recovery_init_pool` 与 Final IQN common config：

- capacity = 1000，`deque(maxlen=1000)`；
- capture snapshot 只在 episode 内全部 evader deactivated 时写入，snapshot 只含 step、pursuer positions、active mask；
- pool 非空时 capture snapshot = 75%；非 capture 部分中 map-random = 50%、synthetic cluster = 50%，所以约为 75% / 12.5% / 12.5%；
- pool 为空时直接是 map-random / synthetic cluster 50% / 50%；
- reset 时恢复 episode counters、capture snapshot、post-capture state、role/status；base reset 做合法性和 collision-free placement 检查。

AC CAP/COV 不使用 Full Mix pool。AC MIX 只在 coverage reset 使用它。recovery pool 始终 resident/bounded in-memory，不写单独 snapshot 文件。

独立 fixture-only smoke `tools/smoke_ac_cross_init.py` 已验证 snapshot、ordinary map-random、synthetic cluster 三路：episode_step=0、active mask 合法、state counters 清零、collision-free init、snapshot 清理均通过。

## 12. Smoke results

每条线运行 1200 env steps，仅为 pipeline smoke，不作学习性能结论。runner 在 smoke 使用 hidden=64/heads=4/layers=1、batch=32（MIX 保持 exact 128-row batch），formal config 仍为 hidden=256/heads=8/layers=4。

| line | updates | action min / entropy | actor / critic / target changed | finite | save/load | eval |
|---|---:|---:|---|---|---|---|
| COV | 20 | 359 / 2.1428 | yes / yes / yes | pass | pass | pass |
| CAP | 20 | 405 / 2.1270 | yes / yes / yes | pass | pass | pass |
| MIX | 20 | 391 / 2.1106 | yes / yes / yes | pass | pass | pass |

checkpoint smoke 使用 atomic temp + `os.replace`；model-only payload 没有 replay；latest runtime payload 含 optimizer/RNG。三条线 deterministic eval 均可运行。MIX 短 smoke 没有自然生成 `post_capture_real`，因此 semantic sampler 明确报告 uniform fallback；正式配置会 defer 等待，不会 silently fill。

## 13. Tests

通过：

- `PYTHONPATH=src pytest -q tests/test_shared_local_ac_capability.py`：5 passed；
- `PYTHONPATH=src pytest -q tests/test_pure_capture_all_capture_contract.py tests/test_iqn_deterministic_eval.py`：10 passed；
- AW9 mapping、shared actor permutation/consistency、shared local Q output `[B,9]`、actor/critic parameter independence；
- behavior mixture sum=1、epsilon floor；
- exact `64/16/32/16` semantic sampler；
- target construction、single-Q update、finite losses、target update、save/resume；
- recovery pool capacity/source contract；
- all three effective configs load and instantiate live AW9 environment；
- `tools/smoke_ac_cross_init.py`：pass；
- episode scheduler smoke：pass。

## 14. Disk/GPU audit

磁盘：root `/dev/nvme0n1p2` 915G total / 782G used / 87G free / 90%；inode 60,981,248 total / 57,450,149 free。`/data/disk2` 有 4.5T free，`/data/disk1` 有 2.6T free。正式 run 不应把 replay 或大 checkpoint 放到 root 而不检查 output_root。

serialized static estimate（formal hidden=256）：model-only checkpoint 63,617,179 bytes；runtime latest without replay 63,631,893 bytes；one replay row 2,453 bytes；4 historical model checkpoints约 254,468,716 bytes；4 model + one latest runtime 约 318,100,609 bytes。100k/200k active rows 分别约 400k/800k。若显式把 replay 塞入 latest resume，约 1.30GB / 2.28GB；所以默认 `save_replay=false`，latest-only atomic overwrite，禁止 rotating full-resume copies。

GPU：两张 RTX A6000，各 49,140MiB；审计时各只有约 815MiB 已用，但 GPU0 有 active Z05，GPU1 有 active Z07；因此当前安全可用 GPU 数为 0（不能杀现有任务）。`tmux ls` 中 Z05/Z07 及历史 TD3 session 均保留。

磁盘安全合同：model checkpoints 只保存 state dict/contract metadata；optimizer/RNG 只进入一个 latest resume；replay 不进入 periodic checkpoints；eval 只写 fixed small JSON summary，不写 full rollout tensors/GIF；recovery pool 仅内存 bounded；每 run 大型历史 resume copy ≤1。

## 15. Exact launch commands

以下是主会话获得 GPU 资源后使用的 100k formal commands；本轮未执行：

```bash
PYTHONPATH=src python3 tools/run_shared_local_ac_capability.py --config configs/experiments/ac_capability_20260920/ac_capability_cov_stage1.yaml --device cuda:0 --total-timesteps 100000 --run-name ac_capability_cov_stage1

PYTHONPATH=src python3 tools/run_shared_local_ac_capability.py --config configs/experiments/ac_capability_20260920/ac_capability_cap_stage1.yaml --device cuda:0 --total-timesteps 100000 --run-name ac_capability_cap_stage1

PYTHONPATH=src python3 tools/run_shared_local_ac_capability.py --config configs/experiments/ac_capability_20260920/ac_capability_mix_stage1.yaml --device cuda:0 --total-timesteps 100000 --run-name ac_capability_mix_stage1
```

resume 命令在需要时追加：`--resume-path runs/<run_name>/resume_latest/full_resume.pt`。25k/50k/75k/100k 会保存 model-only checkpoint 并写 deterministic fixed summary；不会自动扩展至 200k。

## 16. Recommended parallel schedule

当前两张 GPU 都有 Z05/Z07 active process，因此先不 launch。资源释放后：

1. 若同时有 ≥3 张安全 free GPU：COV、CAP、MIX 并行；
2. 若有 2 张：先 COV + CAP，MIX 接下一张；
3. 若有 1 张：COV → CAP → MIX，且每条线先检查 root/data disk 与 latest resume policy。

本机当前是 0 张“可安全占用”的 free GPU；不能把“显存尚有空闲”误当成“没有现有长训”。

## 17. Known risks

- Full Mix 的 `post_capture_real` 在 1200-step smoke 中没有自然填充，这是稀有事件，不是 sampler bug；unit fixture 已证明 exact sampler，formal 采用 defer/warmup。
- root 已使用 90%；默认 output_root 应迁移到有空间的 data disk，并在正式 launch 前重新运行 `df -h` / `df -i`。
- replay row 是 agent-row，100k env steps 约 400k rows；若未来启用 replay-in-resume，必须重新审计并只保留 latest 一份。
- 当前 GPU 资源被 Z05/Z07 占用；本轮没有 kill、暂停或抢占任何任务。
- smoke 的小网络只验证实现通路；不能用于比较 COV/CAP/MIX capability 或推断正式训练趋势。
