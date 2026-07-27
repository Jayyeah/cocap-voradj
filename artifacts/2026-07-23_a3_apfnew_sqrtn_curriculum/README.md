# 2026-07-23 A3 APF-v2 + CenterSqrtN Curriculum

This directory is the unified milestone home for A3 curriculum artifacts started on 2026-07-23. Future A3 formal rollout GIFs and summaries are written under `best_20rollout10gif/`; screening checks are written under `screening/`. All jobs use GPU1 unless explicitly noted.

Policy: stage checks start at 300k and repeat every 100k; early promotion is allowed only at or after 800k; otherwise each stage trains to cap and the historical best tested checkpoint is selected for handoff. The selected checkpoint's formal `20 rollout / 10 GIF` job starts asynchronously before the next stage begins.

### 2026-07-23 A3 curriculum output directory update

用户指定：后续 A3 curriculum 的 rollout GIF 统一放入今天日期的新 milestone 目录，便于管理查看。已将新的 A3 curriculum supervisor 输出根目录改为 `artifacts/2026-07-23_a3_apfnew_sqrtn_curriculum/`。其中 `best_20rollout10gif/` 保存各阶段 best checkpoint 的正式 `20 rollout / 10 GIF`，`screening/` 保存训练中 quick check 结果。旧的单阶段 A3 supervisor 已停止，新 supervisor `voradj_a3_apfnew_sqrtn_curr_gpu1_0723` 已接管当前 stage1 训练 PID `668659`，不会重开或丢失 4v1 scratch 进度。

### A3 stage1 check-start adjustment

A3 stage1 is a 4v1 scratch run. To avoid noisy early offline checks, stage1 quick screening starts at `1M`; later stages start at `300k` and repeat every `100k`. This policy is encoded in `tools/supervise_a3_curriculum.py`.

