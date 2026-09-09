# Forward Final BC / PPO rollout GIF：标准调用与显示合同

本页适用于Final Full-Task categorical Actor，**不适用于旧Pure-Capture BC**。执行环境与C3一致；复用旧`tools/rollout_voradj_visual.py`的快照格式和renderer，不复制另一套动力学/评估循环。

## 本批数量、策略与输出

每个checkpoint分别做 **mixed 20 rollout / 10 GIF + coverage 20 rollout / 10 GIF**，即40回合、20张GIF。固定取各场景最前10个seed，不按成功/失败、好看与否挑图。BC与已有PPO低LR512对照合计80回合、40张GIF。

| 名称 | checkpoint | 执行 |
|---|---|---|
| Final BC | `artifacts/2026-09-08_forward_final/c2_distillation/actor_epoch_030.pt` | categorical sample、零PPO更新 |
| PPO low-LR512 | `artifacts/2026-09-09_forward_final_d/lr_probe/low_lr/actor_step_000512.pt` | categorical sample、仅既有512步diagnostic；不是25k或已通过健康Gate的模型 |

输出根目录 `artifacts/2026-09-09_forward_final_visual/`，两个run为`bc_sample/`、`ppo_low_lr512_sample/`。每个run含：

- `index.html`：已完成GIF的可点击预览，刷新页面可看到新增图。
- `mix/`、`coverage/`：`rollout_*_seed_*.gif`、对应完整逐帧`episode_*_seed_*.json`和`summary_*_seed_*.json`。
- `rollout_report.json`：全部已完成rollout及同C3定义的capture/CE/safety/timing/event统计。不能拿只展示的10局代替20局统计。
- `launch.json`、`runtime_preflight.json`、`run_args.json`：resolved显示参数、实际env合同、checkpoint与source SHA。
- `gif_manifest.json`：已完成GIF路径、SHA、源帧数/渲染帧数、耗时。
- `progress.json`、`active_episode.json`、`process.json`、`worker.log`：进度、当前回合、后台进程与错误。

原始GIF/逐帧JSON及模型保留本地；Git跟踪配置、代码、台账、汇总和SHA。尚在运行时，Git中的进度是提交时快照，不等于实时完成状态。

## 标准配置

唯一显示preset：`configs/evaluation/forward_final_rollout_20260909/standard.yaml`。

| 参数 | 值/来源 |
|---|---|
| episodes / gif_count | 每场景20 / 10 |
| mixed seeds | 2026098101..2026098120 |
| coverage seeds | mixed seed +100000，即2026198101..2026198120 |
| inference | 默认`bc_sample`，Actor eval forward；`bc_argmax`必须显式指定，不能混称 |
| GIF duration / loop | 100 ms每帧（10FPS），无限循环 |
| max GIF frames | 1000；全轨迹均匀抽帧，保留首尾；不是截取前1000步 |
| trail_window / trails | 70 / 关闭 |
| 邻接线 / 感知圈 | 开启 / 开启 |
| CE centroid | 开启，coverage阶段显示C编号黑叉 |
| 画布 / 字体 | 旧renderer的7×6.6英寸、110dpi，即770×726；标题8.4 |
| CPU rendering | 每run 2个spawn进程，Torch/OMP/MKL/OpenBLAS线程数各1 |
| 模型/物理/task | runtime preflight的Forward Final；AW9，decision dt=.5s；synchronized-swept collision |
| 真正终止合同 | horizon3000，capture后coverage窗口500，原生CE hold/terminal；没有重置并拼接两个phase |

显示参数沿用历史Final IQN stage-best 20rollout10gif；**不复制历史可视化脚本的mix2200/coverage1200截断**，因为那会改变当前Forward Final任务。也不采用旧MASAC的5 GIF / 300帧preset。若要复现历史短horizon，应另立诊断合同，不覆盖本批。

## 标准调用脚本

入口：`tools/run_forward_final_rollout_gifs_20260909.py`。在仓库根目录执行；输出目录必须是新的，不能覆盖已有有效结果。两条可独立运行；GPU1本批在critic诊断完成后才使用。

BC：

```bash
visual_out=artifacts/2026-09-09_forward_final_visual/bc_sample
mkdir -p "$visual_out"
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  flock -n /tmp/cocap_gpu0.lock \
  python3 tools/run_forward_final_rollout_gifs_20260909.py \
  --preset configs/evaluation/forward_final_rollout_20260909/standard.yaml \
  --output-root "$visual_out" --device cuda:0 \
  --label 'Final BC sample' \
  --reference-report artifacts/2026-09-09_forward_final_d/smoke_task_eval20/bc_parent/report.json \
  > "$visual_out/worker.log" 2>&1 < /dev/null &
echo $! > "$visual_out/launch.pid"
```

已有PPO-512：

```bash
visual_out=artifacts/2026-09-09_forward_final_visual/ppo_low_lr512_sample
mkdir -p "$visual_out"
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  flock -n /tmp/cocap_gpu1.lock \
  python3 tools/run_forward_final_rollout_gifs_20260909.py \
  --preset configs/evaluation/forward_final_rollout_20260909/standard.yaml \
  --output-root "$visual_out" --device cuda:0 \
  --actor-checkpoint artifacts/2026-09-09_forward_final_d/lr_probe/low_lr/actor_step_000512.pt \
  --label 'PPO low-LR512 sample (diagnostic)' \
  --reference-report artifacts/2026-09-09_forward_final_d/lr_probe/low_lr_eval20/report.json \
  > "$visual_out/worker.log" 2>&1 < /dev/null &
echo $! > "$visual_out/launch.pid"
```

后续新PPO checkpoint只替换`--actor-checkpoint`、输出目录和label，保持同preset/mode/seed。`--reference-report`只能指向**同checkpoint、同mode、同seeds**的已有结果；新模型尚无reference时省略，输出为`NOT_CHECKED`，不能伪称与旧模型轨迹一致。BC parent仍验证固定SHA；PPO loader检查schema、Final contract、BC parent与teacher SHA，拒绝旧AC模型。

只补画已完成rollout中缺失的GIF，无需GPU、无需重跑策略：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
python3 tools/run_forward_final_rollout_gifs_20260909.py \
  --render-only --output-root artifacts/2026-09-09_forward_final_visual/bc_sample
```

这会跳过已存在GIF。对现有GIF修改显示参数应另复制输出做重渲染，保留原证据。`--smoke`仅用于新目录的3步/2帧管线检查，不能当正式20回合结果。

## 必须记住的易错点

1. **只复用render，不复用旧环境构建。** 旧continuous/MASAC/AC wrapper可能改变reward、APF、信息、collision或capture terminal。本入口走同一个C3 `run_episode`，Final capture后自然继续coverage。
2. **绘图不能改环境。** 旧`snapshot_env`会写label/cache；现在对env深拷贝取快照。回归测试验证Actor采样、事件与RNG不变；本批逐回合与已有eval20比fingerprint、capture/CE/safety、长度/时延、完整事件及action histogram。只忽略同时事件的列表排列，不忽略任何事件内容或次数。
3. **GIF不是实时。** 物理一步=.5s，GIF一帧=.1s，未抽帧时约5倍速；长局均匀抽帧后加速更多。比较秒数以JSON为准，不能用播放长度比较控制速度。
4. **success文字沿用历史“整回合最终摘要”。** 它贯穿所有帧，不表示目标在第0帧已捕获；phase/step/位置/CE数值是当前帧。episode success使用safe full-mission；pure coverage的capture显示N/A。标题已分为3行，防止CE RMS/max被画布裁掉，尺寸/字体/帧率未改。
5. **感知圈是名义20m参考，不是中心距离hard gate。** runtime按surface distance判断，精确边界还取决于双方半径；全局观察者图上画出的敌机不代表Actor收到全局敌情。实际actor输入仍是Final local contract。
6. **友军Voronoi图与策略计算勿混称。** sites断言friend-only；图示是边界裁剪的连续Voronoi，runtime CE centroid来自实际free-mask/projected网格。C黑叉读真实runtime centroid，不能用屏幕多边形肉眼反算并断言reward错误。
7. **颜色不是完整role分类。** 旧renderer按`is_pursuing`两色显示，绿色不等于“没有support”；sidecar保留task_label，正式role/phase统计看report中的policy diagnostics。为了保持历史外观，此轮没另造三色role图例。
8. **不要切断尾部或筛成功图。** 首10seed固定，失败也保留；保留初始/终止帧及collision/CE状态。没有capture即停、失败重置后接图或删去碰撞片段。
9. **不要把sample、argmax、teacher greedy混用。** 本批BC/PPO均sample；teacher仅用于同C3动作agreement诊断，IQN使用固定midpoint quantiles。PPO-512明确标作diagnostic，不是假装已完成25k。
10. **后台不轮询等待。** 启动后检查PID/launch/runtime和首回合无报错，继续CPU工作；其余任务完成后只做一次交接检查，记录每卡PID/进度/ETA/NEXT WAKE-UP、commit/push即结束。不抢占无关进程。

首次启动的事件排序断言记录在`startup_event_order_fail/`；标题裁切修复前的输出在`startup_title_clip/`。正式结果仅使用顶层两个run，不将这些启动记录混入统计。

## 本批完成状态

2026-09-09 13:58 CST：两个模型各40回合/20GIF全部完成。BC worker480288、PPO worker480289及其CPU render workers均结束；两卡空闲。相同seed/fingerprint、capture/CE/collision/时间/event/action histogram对照全部PASS。该批是既有eval20的可视化重放，不是新增80个独立样本。总索引`artifacts/2026-09-09_forward_final_visual/index.html`；每个run的index包含所有20张图。ETA=0，无需等待或定时唤醒。
