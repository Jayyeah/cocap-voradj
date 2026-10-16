# CoCap实验与可视化工具索引（2026-10-15）

本轮只归档已存在工具，不改evaluator、不增加第二套可视化系统、不运行训练或rollout。命令示例默认写入新的临时目录，避免覆盖历史结果。

## 1. 工具清单

| 脚本 / entrypoint | 配置 | 输入 | 输出 | 参考命令 / 限制 |
|---|---|---|---|---|
| /home/yjq/rl/CoCap1/iqn-z05-independent-20260923/tools/evaluate_iqn_z05_independent_20260923.py；entrypoint子命令run/audit/recover | Stage配置YAML；checkpoint与selection_report；规模、seed、post-window等CLI覆盖 | checkpoint、YAML、seed manifest、环境源码 | report.json、compact_summary.json、episodes.jsonl、progress.json；指定gif-indices时另写gif_sources与GIF | 见下方Stage3命令。run会执行environment rollout；禁止将其当作纯renderer调用。run source SHA由报告head提供。 |
| 同上 evaluator内置GIF writer | 与evaluator相同，GIF indices默认为固定episode索引 | 新生成episode中的帧记录 | 现有Z05八个主批次每批每场景10份GIF；10月配对批次每场景3份 | 这些GIF已存在；本轮没有重做。没有单独render-config文件，版本应与evaluator source SHA一起归档。 |
| /home/yjq/rl/CoCap1/iqn-z05-independent-20260923/tools/render_coverage_voronoi_gifs.py；entrypoint main | CLI trail-window/frame-duration/max-gif-frames | 旧格式episode_seed_<seed>.json，内含env、rollout与pursuer trajectories | output-root/rollout_seed_<seed>.gif、render_manifest.json | 只适用于该旧episode JSON格式；当前主批次gif_sources/episode_<scene>_seed_<seed>.json不是同名输入契约。先核对schema，不能直接指向现有目录。 |
| /home/yjq/rl/CoCap1/iqn-z05-independent-20260923/tools/rollout_voradj_visual.py；entrypoint main / render_gif API | YAML或run目录、checkpoint、scenario、seed、最大步数与显示开关 | 新环境rollout；render_gif API可渲染既有帧对象 | output-root下GIF/可视化时间线 | CLI会重跑episode，不是从现有Z05 JSONL恢复轨迹的纯plotter。缺GIF时才可在新输出目录作复现渲染，并记录seed。 |
| /home/yjq/rl/CoCap1/iqn-z05-independent-20260923/tools/visualize_ce_coverage.py；entrypoint main | YAML/run目录、checkpoint、seed、max-steps、post-success-steps | 从固定seed重新运行的coverage episode | CE coverage GIF/分析输出 | 会加载策略并重新step环境；不能用于“仅展示已有轨迹”。用前先确认输出root新建，且这次执行确为必要。 |
| /home/yjq/rl/CoCap1/iqn-z05-independent-20260923/tools/rerender_rollout_gifs.py；entrypoint main | root、workers、draw-trails | 特定旧stage/scenario目录下保存的episode_*_seed_*.json及run_args | 同路径替换GIF，并改写episode/summary JSON及worker copies | 脚本用os.replace覆盖源GIF，还会改写JSON。不得对论文原始资产root运行；若必须重渲染，先复制到隔离副本并核对hash。 |
| /home/yjq/rl/CoCap1/z05-repeated-arrival-demo-20261006/tools/run_iqn_repeated_arrival_demo_20261006.py；entrypoint main | Stage3 YAML、selection report、episode数量、seed base、regime、device | Stage3 checkpoint与环境；每episode逐波状态 | summary.json、逐局events、trajectory.jsonl、最多每regime前5索引GIF | formal执行一次为60个rollout；本轮未执行。source commit b308050f…为strict formal evaluator；relaxed代码提交906e2bc…仅diagnostic。 |
| /home/yjq/rl/CoCap1/evidence-local-global-20261006/tools/evaluate_iqn_evidence_checkpoint_20261006.py；entrypoint main | variant/stage/checkpoint/output/episodes/seed-base/workers | 每checkpoint、paired seed与环境 | run report、progress、逐局结果 | 子评估入口；直接运行会新评估。 |
| /home/yjq/rl/CoCap1/evidence-local-global-20261006/tools/supervise_iqn_evidence_comparison_20261006.py；entrypoint main | 当前Evidence configs及selection reports | Z05、Local Binary、Global Oracle Stage3 selected checkpoint | --final-only应写final_heldout_50_corrected下各臂report与paired结果 | 当前代码先check-final-readiness，再fresh process --final-only；本轮未启动。旧final_heldout_50不是corrected目录。 |
| evaluator内summary writer | 同上Z05 evaluator | 各局episode结果 | report.json、compact_summary.json、episodes.jsonl | 当前没有独立的Z05 capture/coverage静态summary plotter；summary JSON由evaluator写出。 |
| 通用轨迹图 / 静态指标图 | 未发现独立、匹配当前JSONL的trajectory plotter或capture/coverage PDF/PNG入口 | —— | —— | 当前可复用的论文图资产是seed绑定的GIF与机器可读报告。没有静态CE/capture plot时，应由论文分析阶段从报告生成并另存source/config/hash，而不要误称仓库已有。 |

## 2. 论文参考运行命令

### 2.1 重现Z05 Stage3 native评估（示例，不在本轮执行）

    cd /home/yjq/rl/CoCap1/iqn-z05-independent-20260923
    python tools/evaluate_iqn_z05_independent_20260923.py run \
      --mode native \
      --config configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml \
      --selection-report /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/selection_report.json \
      --expected-step 600000 \
      --episodes 20 \
      --seed-base 2026091900 \
      --device cpu \
      --output /tmp/cocap_paper_reproduction/stage3_native_20261015

实际历史批次使用evaluator report中绑定的source SHA 5c146f74752073414af416b0db6c686e1c588a07。上述是参考重跑示例；输出目录应保持新建。跨规模示例将mode设为large，并显式传--pursuers 24 --evaders 6 --obstacles 5、使用2026092303 seed base；原有G3报告仍以其resolved_config_hash为准。

### 2.2 Repeated-arrival正式合同示例（不在本轮执行）

    cd /home/yjq/rl/CoCap1/z05-repeated-arrival-demo-20261006
    python tools/run_iqn_repeated_arrival_demo_20261006.py \
      --config configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage3_12p3e3obs_700k.yaml \
      --selection-report /home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages/stage3/selection_report.json \
      --expected-step 600000 \
      --episodes-per-regime 20 \
      --output /tmp/cocap_paper_reproduction/persistent_strict_20261015 \
      --device cpu

这会执行60个episode。已有正式strict 20×3资产，不要为复现而重复运行；该命令只供未来需要独立重跑时参考。

### 2.3 Evidence corrected close-out（当前仍缺，不在本轮执行）

    cd /home/yjq/rl/CoCap1/evidence-local-global-20261006
    PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python tools/supervise_iqn_evidence_comparison_20261006.py --check-final-readiness
    PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python tools/supervise_iqn_evidence_comparison_20261006.py --final-only

Expected destination is runtime/final_heldout_50_corrected. 这要求新的450个paired seed episode；旧报告能继续用于已记录端点，但不能补出缺失的first-strict-CE transition时间。正式运行前先保留旧report不动，不覆盖它。

## 3. GIF与trajectory归档约定

- Z05主批次GIF均留在原始artifacts路径，rollout CSV每行绑定scenario、seed、initial-state fingerprint、checkpoint SHA、report SHA、episodes JSONL、trajectory source JSON及GIF路径。
- 10月重复到达strict/relaxed共30份GIF已有逐局summary映射；旧3×3 demo的3份GIF没有summary seed关联，标记为seed mapping missing，不作论文代表案例。
- repeated-arrival runner对formal/relaxed每组前5个episode index渲染GIF，不按成功率或最短时间挑选。论文展示的strict/relaxed成功样例在这前5个中按最小episode index规则选取，并明确这是case illustration，不是平均表现。Z07报告明示取episode index 0，每场景一份。
- 渲染参数若未随artifact写入，不补造参数；登记为no standalone render config，并引用当时evaluator commit。
- 任意再生成GIF须输出到新目录，记录原轨迹SHA、seed、renderer source SHA和渲染参数；不得对已有source GIF执行覆盖式rerender。

## 4. 当前工具缺口

1. 当前独立评估JSON可直接生成统计表，但没有与其schema相匹配的通用逐局轨迹静态绘图CLI。
2. Z05主要指标保存在report与compact_summary中，没有论文版捕获时间/CE误差置信区间图输出脚本。
3. Existing old-schema Evidence report缺少corrected CE transition time；这属于结果闭合问题，不能用画图或离线推测代替。
4. 本轮没有新增plotter：重要图形素材已有GIF，汇总数字已有机器可读JSON；缺少的严谨统计图应在决定最终实验合同之后一次性从源JSON生成，并绑定分析代码commit。
