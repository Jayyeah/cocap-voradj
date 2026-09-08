# Final IQN-AW 完整合同审计（2026-08-30）

> **2026-09-08 P0 更正，优先于下文历史结论：**见 [最新合同/信息审计](P0_CONTRACT_INFORMATION_AUDIT_20260908_ZH.md)。原 PPO dropout/log-prob 合同不成立；BC 与 native PPO 实际 env seeds 相差10000，撤回 BC↔PPO 配对显著性及“显著 erosion”，warm-up未排除cold-start。旧 AC legacy_voradj 不是 Final 的半径局部敌方感知，BC100%仅在旧AC argmax capture合同有效。旧 outer-ring/首事件 latency不是新 same-target closure。support11仍为无效NOOP，AW/VXY物理预算不等价；仅使用GPU1，暂停所有旧GPU0/长训/teacher-KL路线建议。下文数字作为历史记录保留，不作为新 Gate。


## 1. 最终锚点

仓库中真正的最终 integrated AW 主线不是早期 A3，也不是 Pure-Capture B0，而是：

```text
configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/
  common.yaml
  stage1_4p1e1obs_scratch2m.yaml
  stage2_8p2e2obs_700k.yaml
  stage3_12p3e3obs_700k.yaml

artifacts/2026-08-04_crms_vctls_ce_final/
  manifest.json
  checkpoints/stage1_4v1_step_2000000.pt
  checkpoints/stage2_8v2_step_300000.pt
  checkpoints/stage3_12v3_step_700000.pt
```

它从 `voradj_a3_apfnew_sqrtn_20260723/a3_apfnew_sqrtn_4v1_scratch_mix_2m.yaml` 继承环境、IQN 与 mixed-episode 基线，在 `common.yaml` 扁平加入最终 CR-MS、VCT-LS、CE、support approach 与最新 replay/recovery 分布。`docs/CRMS_VCTLS_CE_FINAL_RELEASE_20260804_ZH.md` 和 artifact manifest 对该身份给出一致声明。

`STATUS: FINAL_AW_CONTRACT_RESOLVED / UNEXPLAINED=0 / VXY_DIFF_TESTED`

`IMPLEMENTATION_COMMIT: dda2a23f20c4b857bc958c856d76bf869b50df02`

## 2. Final IQN-AW Contract Table

| 项目 | AW final behavior | VXY required change | should remain identical? | actual implementation path | test status |
| --- | --- | --- | --- | --- | --- |
| action count/head | 9 个 `(a,w)` Cartesian actions，IQN `action_size=9` | 仍为9 logits，index映射到 VXY9 | head shape是 | `iqn.action_size`; `vxy9_body_grid()` | PASS |
| AW physics | pursuer unicycle，`a={-.4,0,.4}`，`w={-π/6,0,π/6}`，max speed 3 | 替换为 body-frame desired `(vx,vy)` + rate-limited servo | 否，唯一主要变量 | `pursuer.action_mode`; `envs/base.py` | PASS |
| policy architecture | `voradj_single_head`，hidden256，8 heads，4 layers，dropout .1 | 无 | 是 | `CoCapIQN` / stage effective config | PASS |
| distributional IQN | 32 quantiles，64 cosine；train/target sample各8，action sample32 | 无 | 是 | `iqn.*quantiles`; `update_rule=distributional_iqn` | PASS |
| optimization | batch128，replay1M，γ=.99，min replay3k，train每4步，target每10k，Huber1，grad clip.5 | 无 | 是 | historical stage YAML + inherited trainer | PASS |
| local observation | local VorAdj Transformer；敌人/障碍 surface distance 20m | 无 | 是 | `voradj.perception_topology_version=friendly_voronoi_comm_v0` | PASS |
| decentralized execution | policy不看global enemy；只允许friendly一跳通信 | 无 | 是 | `perception.global_evader_visibility=false`; VCT-LS | PASS |
| direct pursuing reward | `ring_importance_ms_v0`；preferred 8、outer10.5、σ2、weight2；旧global approach/MS/front均0 | 无 | 是 | `reward.capture_reward_mode` 与 `ring_ms_*` | PASS |
| support reward | 非直接探测的一跳support：`.5×neighbor-visible approach-only + .5×CE` | 无 | 是 | `voradj.support_reward_*` | PASS |
| role/K10 | effective pursuing从capture转coverage延迟10 action transitions；observation/reward/hold/replay共享该role | 无 | 是 | `is_pursuing_release_delay_steps=10`; `vct_ls_apply_release_delay=true` | PASS |
| normal capture | `k_required=3`，`r_e=8`，角度/环形capture判定 | 无 | 是 | inherited reward/env capture logic | PASS |
| stationary capture | enabled；speed≤.2保持10步、至少2 pursuers | 无 | 是 | `capture_stationary_*` | PASS |
| CE reward | centroid-energy PBRS，scale10，κ1；200k后轻速度cost `.0005` | 无 | 是 | `coverage_objective_version=centroid_energy_v0` | PASS |
| CE success | RMS≤.05、max≤.10、连续30步；min active按4/8/12 | 无 | 是 | `coverage_ce_success_*` | PASS |
| loose coverage | area CV<.15只作诊断，不进reward、不替代strict CE | 无 | 是 | `coverage_cv_loose_area_cv_threshold=.15` | PASS |
| post-capture | capture后继续coverage，window按stage为500/600/700；旧额外latch/repeat奖励被清零 | 无 | 是 | `post_capture_coverage_window_steps`; Final `common.yaml` overrides | PASS |
| episode mix | `voradj` 与 pure `voradj_coverage` 1:1 alternating | 无 | 是 | `train_mode=voradj_mixed_coverage`; `episode_schedule=alternating_1to1_scenes` | PASS |
| replay mix | pursuing64 / pre-capture-cover16 / real-post32 / recovery-pure16 | 无 | 是 | `three_semantic_four_physical_buffers` | PASS |
| recovery | inner cluster + capture snapshot + map random；pool1000，captured ratio.75，非capture中map ratio.5 | 无 | 是 | `voradj.recovery` | PASS |
| map/spawn/safety | 120×120，map-random pursuit，clustered pure coverage；hard boundary/death；collision/boundary -160，proximity -10 | 无 | 是 | inherited `env`, `tasks`, APF-v2-fixed | PASS |
| collision semantics | historical `legacy_end_step`，Final早于 synchronized-swept correction | 首轮仍显式保留；corrected transfer另做 | 是 | `env.collision_semantics=legacy_end_step` in VXY overlay | PASS |
| stage curriculum | 4v1 scratch2M → 8v2 warm700k → 12v3 warm700k | 无 | 是 | three historical stage YAML | PASS |
| checkpoint selection | 全stage跑满后按capture/collision、mix CE、pure CE、mix CV、pure CV排序 | cadence从100k加密至25k仅为观察 | 排序是 | `finalize_screened_run.py`; VXY supervisor | PASS |
| evaluation | capture、coverage、mix；formal deterministic；历史20 rollouts/10 GIF | fixed midpoint τ并加密validation；不改环境 | 场景与指标是 | watcher/finalizer/fixed-τ evaluator | PASS |
| Zone | 仅外部分支的泛化/适应实验，不在Final主训练task | 无 | 是，不纳入首轮 | `configs/demos/zonedemo_v0`, separate tools | PASS |
| resume | 历史 milestone偏模型态，不含完整在线runtime | 新增单一rolling full resume | 否，`IMPLEMENTATION_ONLY` | `cocap_iqn_full_resume_v1` | PASS |

## 3. 训练、课程与真实 lineage

### 3.1 Stage 1：4v1 scratch

- seed `2026080201`，2,000,000 env steps；
- LR `1e-4`，200k后 `3e-5`；epsilon `.6→.05` / 500k；
- 4 pursuers、1 evader、1 obstacle；pure coverage则0 evader；
- 历史每100k screening，但不 early stop；必须跑满2M再选优；
- release checkpoint：`stage1_4v1_step_2000000.pt`，SHA-256 `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89`。

### 3.2 Stage 2：8v2 warm course

- seed `2026080202`，700,000 env steps；
- 从stage1 selected checkpoint shape-compatible注入；
- LR `3e-5`；epsilon `.28→.05` / 420k；
- 8 pursuers、2 evaders、2 obstacles；CE min active 8，post-capture window600；
- 当前release展示checkpoint为300k，SHA-256 `ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e`。

300k与500k不能混称：历史课程按旧排序选中500k并用它启动stage3；2026-08-04 的同seed、固定 midpoint τ、capture/coverage/mix各20回合CE-first重评才选中300k。300k与500k均为capture/mix capture 100%、全场景0 collision，但300k的pure/mix CE strict为100%，500k为45%。release选择300k用于综合展示，没有伪造stage3重训。

### 3.3 Stage 3：12v3 warm course

- seed `2026080203`，700,000 env steps；
- 真实 warm-start 是当时选中的stage2@500k，不是后来release选择的300k；
- 12 pursuers、3 evaders、3 obstacles；CE min active12，post-capture window700；
- release checkpoint：`stage3_12v3_step_700000.pt`，SHA-256 `5e4173eac94c921685091d60033bffb8798180e974b72ec02437e3de265848ee`。

### 3.4 “mixed/generalization”的含义

历史主课程没有第四个 mixed training stage。每个stage本身已经在 `voradj` 与 `voradj_coverage` 间1:1混训；finalizer再分别评估：

- `capture`：追捕/围捕能力；
- `coverage`：无敌人的pure CE；
- `mix`：capture后进入post-capture coverage。

ZoneDemo是单独的泛化评估/适应分支。把它加进VXY Full首轮训练会同时改变task distribution，破坏“只换动作”的因果问题。

## 4. 历史 stage gate 的准确表述

历史 supervisor 的推进条件是“完整stage训练结束、所有预期screening存在、finalizer生成selected checkpoint”，而不是显式的数值 PASS/FAIL rejection threshold。finalizer使用词典序排名：

```text
min(capture, mix capture)
→ mean capture
→ lower max collision
→ mix CE strict
→ pure CE strict
→ mix/pure loose CV
→ later checkpoint
```

因此不能把今天为VXY安全迁移新增的 `.50/.10/.50` promotion threshold 倒写成历史事实。VXY supervisor 的显式 gate 是新的 `IMPLEMENTATION_ONLY/SAFETY` 决策层：stage失败即停止，不跳下一阶段；它不参与训练loss或checkpoint排序。

## 5. 差异分类与审计闭环

| 分类 | 本轮项目 | 处理 |
| --- | --- | --- |
| `ALGORITHM_REQUIRED` | 无；Line A仍是同一IQN | freeze |
| `ACTION_REQUIRED` | AW9→VXY9 dynamics/action semantics | 唯一主要实验变量 |
| `TASK_REQUIRED` | 4→8→12 historical stage规模变化 | 完全照历史 |
| `IMPLEMENTATION_ONLY` | 25k milestone、fixed-τ validation、atomic full resume、resource supervisor | 有测试，不改训练update |
| `LIKELY_MIGRATION_ERROR` | continuous mode默认world-axis yaw、误用corrected swept collision、vector/integer action混用 | 已通过explicit legacy yaw/collision与typed VXY9入口消除 |
| `UNEXPLAINED` | 0 | strict flattened diff test守护 |

## 6. 可审计证据与测试

- source configs：`configs/experiments/cr_ms_support_approach_ce_curriculum_20260802/`；
- release identity/hash：`artifacts/2026-08-04_crms_vctls_ce_final/manifest.json`；
- 300k/500k事实：`docs/CRMS_8V2_300K_VS_500K_FORMAL_20260804_ZH.md`；
- VXY overlay：`configs/experiments/iqn_vxy_full_migration_20260830/`；
- strict diff：`test/test_iqn_vxy_full_config_contract.py`；
- action contract：`test/test_iqn_vxy_full_action_contract.py`；
- resume：`test/test_iqn_full_resume_contract.py`；
- orchestration/gate：`test/test_20260830_supervisor_contract.py`。

最终合并定向回归为 `78 passed`，2-step Full-VXY CUDA smoke与resume roundtrip均PASS。该结论只证明“迁移合同准确且可执行”，不提前声称Full-VXY性能成功。
