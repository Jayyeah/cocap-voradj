# A1 + A2 shared CUDA root-cause audit

结论：`SHARED_ROOT_CAUSE_CONFIRMED`；独立审计 worktree 内最小修复完成，分类 `ROOT_CAUSE_CONFIRMED_FIXED`。这是工程结论，A1/A2 均没有新增 formal 科学结果。

## 真实 first bad row

所有索引为 **0-based**；collection step 是调用 env.step 前、runner 自增前的 global_step。

| 项目 | A1 | A2 |
|---|---|---|
| 原 branch | `experiment/ac-entropy-cov-20260921` | `experiment/ac-discrete-sac-preflight-20260921` |
| 原 HEAD | `dc14560e1f2761ee92e36ee8ce769fe0caa8eb11`，实际 runtime 有未提交修改 | `e383c03`；原 formal 源码快照 `135eca3` |
| 本轮首次测得失败 | step **760**，第 **3** 次 update | step **808**，第 **15** 次 update |
| entrypoint | `target_actor.probabilities(next_obs)` | **online** `actor.logits(next_obs)`；没有独立 target actor |
| batch index | **31** | **69** |
| replay slot / episode / agent | **1254 / 1 / 2** | **3099 / 2 / 3** |
| collection step | **313** | **774** |
| task / phase / replay class | `voradj_coverage / pure_coverage / recovery_pure` | 相同 |
| reset source | `environment_default` | 相同 |
| active before / after | true / false | true / false |
| terminated / truncated / done | true / false / true | true / false / true |
| environment event | boundary collision，reward_safety=-160 | 相同 |
| logits finite | **1143/1152** | **1143/1152** |

两行均来自真实 replay：inactive 的 `next_obs=None` 被 runner 按既有逻辑转成全零 observation，六个字段全部归零，**22 个 valid-token mask 全 false，包括 self**。原始字段完整保存在 [failing_row_manifest.json](failing_row_manifest.json) 和两个 `failure.pt`。原始 feature 全零；embedding 后 token 为有限非零值，问题是没有任何可见 key。

A1 的原日志没有记录 global_step；只存在 step-0 evaluation artifact，不能据此认定故障发生在 step0。此前 `_gate_evaluation(0) -> 第一次 _collect_step()` 尚未跨过正式 replay warmup（3000 rows，约750 env steps）。本轮保留完整初始评估并运行到 update，捕获到了 step760 的 first source。原始退出进程没有保存失败权重，本报告不宣称恢复了它的 bit-exact state。

A1 未提交的 entropy/runtime 实现已按实际文件保存，见 [source_manifest.json](source_manifest.json)。两条原 encoder 字节相同，SHA256：`567855d626403bcea549c4806ccae0951f57cc24540925d92ca6197d530b1de5`。

## First bad tensor / layer

- raw/normalized packed observation、所有 embedding projections、type embedding、Transformer 第0层输入及模型权重都 finite。
- exact production forward 首次坏输出：`encoder.transformer.layers.0` 的 `torch._transformer_encoder_layer_fwd`，shape `[128,22,256]`，恰好坏行的 `22*256=5632` 个数非有限。
- 独立 native-attention 分解：norm1、QKV、Q/K/V、未屏蔽 QK scores 都 finite；**`aten._masked_softmax`** 在无可见 key 的行首先产生 NaN，然后传到 attention output、residual、norm2、后续层、self token、fusion、decision feature、actor head 和全部9个 logits。
- mask/屏蔽后的 scores 内 `-inf` 是有意屏蔽值；这里区分它与首次非预期 NaN。最后有限 attention 张量是未屏蔽 QK scores。
- native 分解与真实 fused layer 的有限部分最大误差为 **0**，非有限位置完全一致。方法 wrapper trace 与无 trace 的 logits 也完全一致；全部 **4** 个 fused Transformer 层仍执行。

没有安装会关闭该 PyTorch fastpath 的 module hooks，没有新增 backend flags。生产捕获在真实 logits 返回后检查并 dump，在污染梯度前停止。详见 [first_bad_tensor.json](first_bad_tensor.json)；[ROOT_CAUSE_PROOF.md](ROOT_CAUSE_PROOF.md) / [ROOT_CAUSE_PROOF.json](ROOT_CAUSE_PROOF.json) 在修复前写入。

两条 learner 都先执行 next-state forward，再乘 `~terminated`。`0 * NaN` 仍为 NaN，不能保护 terminal target；后续 loss/gradient 可受污染，使问题最终表现为 actor probability assert。此根因不涉及 entropy/SAC 科学效果。

## CPU/CUDA matched replay

使用同一权重、input、mask、eval/no_grad、float32、无 autocast。修复前 CPU 与 CUDA 均在相同层/行失败，没有 CPU finite / CUDA non-finite 分岔。

| 单行 CUDA 重放 | A1 原始 / 修复后失败数 | A2 原始 / 修复后失败数 |
|---|---:|---:|
| 1 run | 1 / 0 | 1 / 0 |
| 10 runs | 10 / 0 | 10 / 0 |
| 100 runs | 100 / 0 | 100 / 0 |
| 1000 runs | 1000 / 0 | 1000 / 0 |

原始首次失败 iteration 均为1，失败模式确定。原 batch、倒序 batch、重复坏行各10次：修复前均10/10失败，修复后均0/10。证据确认 row-local mask bug；CUDA-only、batch position、偶发 kernel、未初始化输入不是本次故障所需的解释。

修复后全部模型输出 finite；逐层 CPU/CUDA 最大绝对误差 A1 `2.86102294921875e-6`、A2 `3.814697265625e-6`，最终 logits 均 `9.5367431640625e-7`。每条 batch 的其余 **127** 个有效行，修复前后在各设备上均 **bit-exact，最大差0**。

逐层记录：[cpu_cuda_layer_compare.json](cpu_cuda_layer_compare.json)。重复次数、failure count、首次失败 iteration：[minimal_reproducer_result.json](minimal_reproducer_result.json)。完整共享/差异路径：[shared_forward_diff.json](shared_forward_diff.json)。

## 最小修复及科学合同

生产代码仅改 [LocalObservationEncoder.features](../../src/cocap_voradj/models/shared_local_ac.py)：复制 mask，仅对全部无效的行在 encoder 内保留 self slot，attention 与 pooling 使用该内部 mask。原始 observation/replay mask 不变，有效行的输入及 mask 不变。终止占位行没有物理下一状态，其有限编码仍被既有 terminated multiplier 排除于 bootstrap。

没有修改 entropy alpha、Q 数量、SAC 方程、reward、replay、epsilon、LR、gamma、tau、observation packing、初始化或 BC；没有新增 `nan_to_num`、logits clamp 或 skip batch。完整 diff：[production_fix.patch](production_fix.patch)。

历史 `530ecb9/62f1f8f/766283d` 有同类修复，但都不是 A1/A2 的 ancestor；当前真实坏行独立证明了这一遗漏。详细 [indexing_mask_audit.json](indexing_mask_audit.json) 区分了合法 `(B,)` action / `(B,1)` gather、agent-slot mapping、mask broadcast、padding 与 test-only shape bug。summary attention 至少有 mean/max 两个有效槽，空 enemy 分支有 guard，二者不是 first bad op。

## 验证与资源

- 新增5个 terminal-row 回归：修复前全部失败，修复后全部通过。
- 新回归 + 既有 SAC/AC 合同测试：**14 passed**，包含 checkpoint/target update 检查。
- A1 原有 entropy 与 action-shape 测试：**7 passed**。
- A1、A2 各自 CPU/CUDA 的原正式源码路径：**四次均1000 env steps / 63 updates**，包含完整初始 evaluation、H256/L4、batch128、min replay3000、episode max3000。最终网络参数和 telemetry 全 finite，实际收集了 inactive terminal 全零行。
- A2 当前 e383c03 runner 附加 smoke：**900 steps / 38 updates 通过**，最终参数和 telemetry 全 finite；保留其已有 backend/finite settings，没有新增切换。详见 [tests.txt](tests.txt)。

审计工具曾有两个已记录问题：首次 trace 对 bool 调用 abs；当前 A2 settings 下 aggregate precision getter 报混合 API 状态。这两次都属于诊断工具问题，保留原日志，修正 metadata 读取后才补验；不据此宣称生产修复失败或成功。

GPU 使用物理GPU1/逻辑cuda:0；每次新任务前后10×1秒采样，记录于 `resource_*.json` / `*.launch.json`。只管理本轮新开的有时限 diagnostic。IQN PID 17097/19555 保持健康，未改科学合同或中央 DAG/state。未调用正式 runner.run()，没有启动100k/300k长训。

建议下一轮 MASTER 集成 encoder-only fix 后重新尝试 A1/A2。已知 first source 已消除；长程稳定性与科学效果仍需后续正式预算回答，本轮不启动。使用有限的初始状态，不能恢复已受 NaN 污染的权重。

## 复现与交付

在审计 worktree 运行；label 必须全新，包装器先检查 GPU/heartbeat 门槛：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/a1_a2_bounded_launch.py --label review_a2_original --timeout 120 -- \
python tools/a1_a2_cuda_replay.py \
  --artifact artifacts/2026-09-22_a1_a2_cuda_root_cause/a2_original/failure.pt \
  --label review_a2_original --implementation original
```

改为 `--implementation current` 验证修复；修改 label。将 `a2_original` 改为 `a1_original` 验证A1。命令读取 captured weights/input，不重训、不生成合成失败行。

- branch：`audit/a1-a2-cuda-root-cause-20260922`。
- worktree：`/home/yjq/rl/CoCap1/ac-a1-a2-cuda-root-cause-20260922`。
- 审计启动时中央 HEAD：`4538aca38571b1f6aec8a9bfa7f69bca2c871562`；本轮未编辑中央文件。
- 提交及结构化 handoff：[handoff.json](handoff.json)。
- 目录约34MiB，两个 `failure.pt` 是 gitignored 的本机 runtime artifact；manifest 保存 SHA256。迁移机器时需同时复制这两个完整权重/输入文件。

最小生产修复及回归测试 commit：`4e210c090ec696bda73ff36877f8e33270ef8a86`。其余代码仅为诊断工具与证据。最终交付 HEAD 记录在本机 `delivery.json`，也在最终回复中列出。
