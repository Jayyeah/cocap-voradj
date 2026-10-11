"""Local bounded closeout: all checkpoints, best AND last, never promote."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from terl_mappo.run import atomic_json
from terl_mappo.batch01.evaluation import isolated_rng
from .evaluation import mean_metric, selection_key


def read_json(path):
    return json.loads(Path(path).read_text())


def refresh(run):
    run = Path(run)
    screens = [read_json(p) for p in sorted((run / 'evaluations').glob('screen_*.json'))
               if not p.name.endswith('.progress.json')]
    rows = []
    for report in screens:
        rows.append(dict(decisions=report['decisions'], checkpoint_sha256=report['checkpoint_sha256'],
                         capture_completion_rate=mean_metric(report, 'capture_completion_rate'),
                         fully_capture_rate=mean_metric(report, 'fully_capture_rate'),
                         episode_reward=mean_metric(report, 'episode_reward_mean'),
                         collision_rate=mean_metric(report, 'collision_rate'),
                         boundary_rate=mean_metric(report, 'boundary_rate'),
                         episode_length=mean_metric(report, 'episode_length_mean'),
                         capture_time_seconds=mean_metric(report, 'capture_time_seconds_mean')))
    atomic_json(run / 'checkpoint_performance.json', rows)
    if rows:
        with (run / 'checkpoint_performance.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    metrics = []
    if (run / 'metrics.jsonl').exists():
        metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    if metrics:
        fields = ['decisions', 'reward_mean_active', 'actor_loss', 'value_loss', 'entropy', 'approx_kl',
                  'actor_grad_norm', 'value_grad_norm', 'gradient_finite', 'training_capture_completions',
                  'training_collision_episodes', 'training_boundary_episodes']
        with (run / 'training_curve.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in metrics)
    if screens:
        best, last = max(screens, key=selection_key), max(screens, key=lambda r: r['decisions'])
        atomic_json(run / 'screen_best_last.json', dict(classification='PROVISIONAL',
                    best_screen_decisions=best['decisions'], best_screen_sha256=best['checkpoint_sha256'],
                    last_screen_decisions=last['decisions'], last_screen_sha256=last['checkpoint_sha256'],
                    last_minus_best_capture=mean_metric(last, 'capture_completion_rate') - mean_metric(best, 'capture_completion_rate'),
                    note='Screen best is descriptive; independent selection/final domains are used at budget closeout.'))
    # Export scientific plots; rendering trouble is a warning, not a new gate.
    try:
        with isolated_rng():
            plot_curves(run, rows, metrics)
    except Exception as error:
        atomic_json(run / 'plot_warning.json', dict(warning=repr(error), blocks_training=False))
    return screens


def plot_curves(run, checkpoints, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if checkpoints:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        x = [r['decisions'] for r in checkpoints]
        for field, label in [('capture_completion_rate', 'Capture completion'),
                             ('collision_rate', 'Collision'), ('boundary_rate', 'Boundary')]:
            axes[0].plot(x, [r[field] for r in checkpoints], marker='.', label=label)
        axes[0].set(ylabel='Evaluation rate', ylim=(-.03, 1.03))
        axes[0].legend()
        axes[1].plot(x, [r['episode_reward'] for r in checkpoints], color='tab:blue')
        axes[1].set(xlabel='Training joint decisions', ylabel='Fixed-evaluation episode reward')
        for axis in axes:
            axis.grid(alpha=.2)
        fig.tight_layout()
        for suffix in ['png', 'svg']:
            fig.savefig(run / f'checkpoint_curve.{suffix}', dpi=150)
        plt.close(fig)
    if metrics:
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
        x = [r['decisions'] for r in metrics]
        for axis, field in zip(axes.flat, ['reward_mean_active', 'entropy', 'actor_loss',
                                         'value_loss', 'approx_kl', 'actor_grad_norm']):
            axis.plot(x, [r[field] for r in metrics], linewidth=.8)
            axis.set(ylabel=field)
            axis.grid(alpha=.2)
        for axis in axes[-1]:
            axis.set_xlabel('Training joint decisions')
        fig.tight_layout()
        for suffix in ['png', 'svg']:
            fig.savefig(run / f'training_curve.{suffix}', dpi=150)
        plt.close(fig)


def closeout(run, config, selected, final_reports):
    run = Path(run)
    screens = refresh(run)
    baseline = min(screens, key=lambda r: r['decisions'])
    last = max(screens, key=lambda r: r['decisions'])
    best_screen = max(screens, key=selection_key)
    late = sorted(screens, key=lambda r: r['decisions'])[-config['evaluation']['late_window']:]
    rates = [mean_metric(r, 'capture_completion_rate') for r in late]
    base_rate = mean_metric(baseline, 'capture_completion_rate')
    best_rate = mean_metric(best_screen, 'capture_completion_rate')
    final_base = mean_metric(final_reports['initial'], 'capture_completion_rate')
    final_best = mean_metric(final_reports['selected'], 'capture_completion_rate')
    final_last = mean_metric(final_reports['last'], 'capture_completion_rate')
    if best_rate <= base_rate and final_best <= final_base:
        learning = '从未观察到相对初始化的 capture 能力提升'
        recommendation = 'C. 回查 observation/reward 消费、角色与失败轨迹；不立即修改 reward'
    elif final_best <= final_base:
        learning = 'screen 曾有提升，未在未参与选择的 final seeds 上复现；泛化不足或采样波动'
        recommendation = 'C. 回查输入信息和训练轨迹，并评估跨 seed 复现'
    elif final_last < final_best:
        learning = '出现 held-out capture 能力提升，但 last 低于 selected；存在退化或不稳定信号'
        recommendation = 'D. 先分析退化时段和失败模式，再决定迁移路线'
    else:
        learning = '观察到 held-out capture 能力提升；单训练 seed 的 PROVISIONAL 证据'
        recommendation = '完成 native backbone capability 复核后讨论下一步；本 runner 不自动启动其他任务'
    findings = dict(classification='PROVISIONAL', status='COMPLETE_BUDGET_AND_EVALUATION', arm=config['arm'],
                    budget=config['budget'], selected_checkpoint_decisions=selected['decisions'],
                    selected_checkpoint_sha256=selected['checkpoint_sha256'],
                    best_selection_basis=config['evaluation']['selection_order'],
                    best_screen_decisions=best_screen['decisions'], last_decisions=last['decisions'],
                    screen_initial_capture_rate=base_rate, screen_best_capture_rate=best_rate,
                    final_initial_capture_rate=final_base, final_selected_capture_rate=final_best,
                    final_last_capture_rate=final_last, final_last_minus_selected=final_last - final_best,
                    late_capture_mean=float(np.mean(rates)), late_capture_min=min(rates),
                    late_capture_max=max(rates), late_capture_std=float(np.std(rates)),
                    learning_assessment=learning, next_stage_recommendation= recommendation,
                    next_stage_started=False, final_domain_used_for_selection=False,
                    warnings=['One training seed; no scale/lifecycle experiment or formal promotion.',
                              'Capture completion is separate from full CoCap episode_success.',
                              'Simple rate differences are descriptive signals, not statistical proof of regression.'],
                    final_reports={key: {k: v for k, v in value.items() if k != 'episodes'}
                                   for key, value in final_reports.items()})
    atomic_json(run / 'capability_report.json', findings)
    text = [f'# Batch02-A {config["arm"]} capability validation', '',
            '**PROVISIONAL；预算与评估已完成，不启动下一批。**', '',
            f'学习判断：{learning}。', '',
            '最佳 checkpoint 由独立 selection seed 域按 capture completion、collision、boundary、capture time、较早 step 选择；final seed 域不参与选择。', '',
            f'Selected={selected["decisions"]}，last={last["decisions"]}；final capture completion：initial={final_base:.3f}，selected={final_best:.3f}，last={final_last:.3f}。', '',
            '| decisions | capture | collision | boundary | reward | length | capture time(s) |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for row in read_json(run / 'checkpoint_performance.json'):
        duration = '-' if row['capture_time_seconds'] is None else f'{row["capture_time_seconds"]:.1f}'
        text.append(f'| {row["decisions"]} | {row["capture_completion_rate"]:.3f} | {row["collision_rate"]:.3f} | {row["boundary_rate"]:.3f} | {row["episode_reward"]:.3f} | {row["episode_length"]:.1f} | {duration} |')
    text += ['', f'Late checkpoint capture range={min(rates):.3f}–{max(rates):.3f}，std={np.std(rates):.3f}。', '',
             f'下一阶段建议：{recommendation}。仅建议，未授权或启动。', '',
             '完整曲线：training_curve.csv / metrics.jsonl；每个 checkpoint 的 argmax/sample、失败类型与条件 capture time 见 evaluations/。', '',
             '本报告是新 held-out capability evaluator 的结果，不是工程 self-test；执行者仍为 Commander，保持 PROVISIONAL。']
    (run / 'CAPABILITY_REPORT_ZH.md').write_text('\n'.join(text) + '\n')
    return findings

