# IQN / MASAC Rollout GIF 审计与正式默认合同（2026-08-11）

## 结论

旧 IQN 可视化链完整：单回合入口为 `tools/rollout_voradj_visual.py`，批量入口为 `tools/batch_rollouts.py` / `tools/batch_rollouts_parallel.py`，历史 stage-best finalizer 使用确定性 IQN midpoint、每场景 20 rollout / 10 GIF。

仓库也存在 `tools/run_continuous_formal_rollout.py`，且已经生成过早期 P6 连续策略 GIF；但它不能直接作为当前正式 MASAC 可视化入口：它按 checkpoint contract 重建固定的 128-step、零障碍 P6 场景，且 loader 只接受早期 `formal_p6|smoke` profile，不接受当前 `formal_ctde` checkpoint。若直接用于当前 Pure-CE、Legacy Old-Mix 或 Stage4 VCT-LS，会改变场景合同。

新增 `tools/run_masac_rollout_gifs.py` 作为当前 CTDE MASAC 正式入口。它必须同时接收 checkpoint 和该 run 保存的 `effective_config.yaml`，并在 rollout 前严格检查 root config hash、各场景 effective hash、action mode、`a_max/w_max`；不匹配即拒绝运行。策略固定 deterministic actor，不读取 replay。

## 旧 IQN 默认与历史正式口径

`rollout_voradj_visual.py` 的裸 CLI 默认：100 ms/帧，即 10 FPS；尾迹、邻接线、感知圈均关闭；trail window 70；最多渲染 1000 帧；mix 1200、capture 600、coverage 900 步。批量脚本会显式覆盖这些值。

最终 IQN stage-best pipeline 的常用正式覆盖值为：每场景 20 rollout / 10 GIF，capture 1000 步；4v1 coverage 1200、mix 2200，8v2 coverage 1500、mix 2500，12v3 coverage 1800、mix 2800；100 ms/帧、最多 1000 GIF 帧、邻接线和感知圈开启、尾迹默认关闭。部分较早历史 artifact 曾显式打开尾迹，不能把它误认为当前默认。

## 当前 MASAC 统一输出合同

- 每个 checkpoint、每个 preset 场景固定 20 个 deterministic rollout；前 5 个固定 seed 生成 GIF，不按成功与否事后挑图，避免展示偏差。
- last 与 best checkpoint 必须使用同一 preset、同一 20 个 seed，形成 paired comparison。
- 默认 100 ms/帧（10 FPS），GIF 循环播放；每个 GIF 最多 300 帧，长 episode 在完整轨迹上等间距抽帧，首尾和整体演化均保留。
- 默认关闭 faded trail；若后续确需观察局部振荡，可另开显式对照，不改变正式默认。
- 输出保存 `run_args.json`、resolved preset、effective config、20回合逐条记录、汇总，以及5个 GIF/逐帧 episode JSON。
- 默认 `device=cpu`、4 workers，避免在训练期抢 GPU；正式线结束后可显式改用空闲 GPU，但同一 paired comparison 必须保持执行合同一致。

## 三类 preset

| preset | 场景与上限 | GIF诊断层 | seed |
|---|---|---|---:|
| `pure_coverage.yaml` | pure_ce 1500步 | Voronoi cell + CE centroid；邻接线关、感知圈关、尾迹关 | 2026081101--2026081120 |
| `old_mix.yaml` | capture 1000、pure_ce 1500、mixed_crms 1500步 | Legacy VorAdj邻接线开；感知圈关；coverage阶段CE centroid开；尾迹关 | 2026081201--2026081220 |
| `vct_ls_capture.yaml` | capture 1000步 | pursuer-only邻接线开、20m敌/障局部感知圈开、尾迹关 | 2026081301--2026081320 |

因此每个 checkpoint 的实际产物量为：Pure-CE 20 rollout / 5 GIF；Old-Mix 三场景合计 60 rollout / 15 GIF（每场景各20/5）；VCT-LS capture 20 rollout / 5 GIF。

Pure-CE 关闭邻接线是为了避免 Delaunay 边与 CE cell/centroid 重叠污染主观察；底层 policy 仍使用 effective config 中的 friendly Voronoi communication。Old-Mix 的敌我可见性由 legacy VorAdj 邻接决定，因此显示邻接线而不画会误导的半径圈。VCT-LS 同时依赖友方一阶通信邻接与局部 surface-distance 感知，两个诊断层都显示。若具体 checkpoint 的 `global_evader_visibility=true`（例如 Stage4A/C诊断线），该事实会写入备份的 effective config；感知圈只表示配置半径，不伪称它是该线唯一敌人来源。

## 调用模板

```bash
python3 tools/run_masac_rollout_gifs.py \
  --preset-config configs/evaluation/masac_rollout_gif_20260811/pure_coverage.yaml \
  --config <run-or-checkpoint>/effective_config.yaml \
  --checkpoint <checkpoint>/trainer.pt \
  --output-root artifacts/2026-08-11_masac_20rollout5gif/<line>/<best-or-last>
```

Old-Mix 和 VCT-LS 分别把 preset 换成 `old_mix.yaml` 与 `vct_ls_capture.yaml`。可先加 `--dry-run` 验证 config/checkpoint/preset 合同而不执行环境 rollout；开发 smoke 可加 `--episodes 1 --gif-count 1 --workers 1 --horizon-cap 2 --max-gif-frames 2`，这些参数不得用于正式结果。

## 本批模型计划

- Stage4A：last=200k；best=50k（既有20-rollout为19/20 capture，是当前充分验证最佳）。
- Stage4C：last=200k；best待按全部里程碑的接敌/碰撞排序确定，若无有效候选则只把“统计最不差点”明确标作诊断best，不伪称成功模型。
- Pure-CE：last=200k；当前临时候选 best=75k，待200k及完整20-rollout后最终确定。
- Legacy Old-Mix Focal A、All-Agent B：各自 last=200k；best必须在相同20-rollout三场景合同下选，不用4-episode diagnostic直接定夺。

正式批处理放到对应训练完成后执行，当前不启动，避免与在训线竞争CPU/GPU。

## 2026-08-11 14:15 正式批处理状态

已启动完结线的 paired 评估，统一输出到 artifacts/2026-08-11_completed_lines_20rollout5gif/。Stage4A 的 50k 候选最佳与 200k last 已各自完成 20 rollout / 5 GIF：两点均为 capture 20/20、collision 0/20；50k 平均完成 65.85 步，200k 平均完成 42.85 步。因此在本组完全相同 seed 上，200k 不仅保持成功率，而且完成更快；最终 best 选择应结合这批 paired rollout 修正，不能沿用另一组 seed 下的旧标签。

Stage4C 的 150k 诊断点与 200k last、Baseline A 的 100k coverage-diagnostic 点与 200k last 正在独立 tmux 中自动完成。VCT-LS 每点为 20 rollout / 5 GIF；Old-Mix 每点为 capture、pure-CE、mixed 各 20 rollout / 5 GIF，即总计 60 rollout / 15 GIF。任务完成时会写出各目录的 all_summaries.json 与 timing.json，无需保持交互窗口。

首次并发启动暴露出 PyTorch 默认 CPU thread pool 过度创建：6 个父进程各约 66 threads，使一分负载瞬时达到 370。只停止了这6个 rollout，不触碰训练；随后在 run_masac_rollout_gifs.py 的父进程和 spawned worker 内显式固定 torch intra-op/inter-op 为1，并用 OMP_NUM_THREADS=1、MKL_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、nice=15 重启。合同测试 5/5 通过。安全版每个父进程3 threads，每个worker约2 threads且最多占1个CPU核；当前剩余4个任务共8个计算worker，GPU rollout占用为0，系统一分负载约20/128核。

## 与旧 IQN 并行脚本的实现差异

旧 IQN 的 batch_voradj_rollouts_parallel_20260721.py / tools/batch_rollouts_parallel.py 已经是并行实现，但准确说是多进程而非 Python 多线程：默认4个worker，以子进程分别运行 batch脚本，每个worker加载一份 IQN，历史默认放在 cuda:1，结果先写 _workers/ 再合并。旧实现没有显式设置 PyTorch/BLAS CPU线程上限，因此4 workers并不严格等于只创建4个OS线程；空闲服务器通常表现为约4个持续占用的CPU核，但库线程池仍有在高并发时放大 runnable thread 数的风险。

当前 MASAC 脚本同样采用多进程，使用 spawn ProcessPoolExecutor；每个worker加载一份与 checkpoint contract/effective config 严格匹配的 central MASAC trainer，并直接按固定episode index合并结果。正式preset默认4 workers，但本批为保护在训线显式使用2 workers/任务，并新增单线程池限制。它还与旧 IQN 有三项语义差异：MASAC走确定性连续actor而非IQN midpoint动作；按 Pure-CE/Old-Mix/VCT-LS 三套精确场景合同分别执行；运行前校验root/scene/action hash，拒绝错配checkpoint。
