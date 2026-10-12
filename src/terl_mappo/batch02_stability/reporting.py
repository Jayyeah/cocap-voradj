"""Describe paired continuation screens, including late retention and KL activation."""
import csv
import json
from pathlib import Path

import numpy as np
from terl_mappo.run import atomic_json


def refresh(run):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    baseline = Path(manifest['baseline_run'])
    rows = []
    for condition, root in [('baseline', baseline), ('stability', run)]:
        for path in sorted((root / 'evaluations').glob('screen_*.json')):
            if path.name.endswith('.progress.json'):
                continue
            report = json.loads(path.read_text())
            if report['decisions'] < 500000:
                continue
            for mode, metrics in report['modes'].items():
                rows.append(dict(condition=condition, decisions=report['decisions'], mode=mode,
                                 capture_rate=metrics['capture_completion_rate'],
                                 collision_rate=metrics['collision_rate'],
                                 mean_reward=metrics['episode_reward_mean'],
                                 episode_length=metrics['episode_length_mean'],
                                 capture_time=metrics['capture_time_seconds_mean'],
                                 episodes=metrics['episodes'], failure_types=metrics['failure_types'],
                                 checkpoint=report['checkpoint'], checkpoint_sha256=report['checkpoint_sha256']))
    fields = ['condition','decisions','mode','capture_rate','collision_rate','mean_reward',
              'episode_length','capture_time','episodes','failure_types','checkpoint','checkpoint_sha256']
    with (run / 'paired_screen_curve.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    diagnostics = []
    for condition, root in [('baseline', baseline), ('stability', run)]:
        path = root / 'metrics.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['decisions'] > 500000:
                    diagnostics.append(dict(condition=condition, **row))
    diagnostic_fields = ['condition','decisions','reward_mean_active','entropy','approx_kl','approx_kl_max',
                         'actor_loss','value_loss','explained_variance','kl_early_stop',
                         'minibatch_updates','gradient_finite','parameter_finite',
                         'actor_update_l2','actor_update_relative_l2','value_norm_mean','value_norm_std']
    with (run / 'optimization_curve.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic_fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(diagnostics)
    paired = {}
    summaries = {}
    for mode in manifest['config']['evaluation']['modes']:
        curves = {condition:{r['decisions']:r for r in rows if r['mode']==mode and r['condition']==condition}
                  for condition in ['baseline','stability']}
        common = sorted(curves['baseline'].keys() & curves['stability'].keys())
        paired[mode] = [dict(decisions=step,
                            capture_delta=curves['stability'][step]['capture_rate']-curves['baseline'][step]['capture_rate'],
                            collision_delta=curves['stability'][step]['collision_rate']-curves['baseline'][step]['collision_rate'],
                            mean_reward_delta=curves['stability'][step]['mean_reward']-curves['baseline'][step]['mean_reward'])
                        for step in common]
        summaries[mode] = {}
        for condition, curve in curves.items():
            if not curve:
                continue
            ordered = sorted(curve.values(),key=lambda r:r['decisions'])
            # A descriptive screen peak is never a parent checkpoint or formal selected model.
            best = max(ordered, key=lambda r:(r['capture_rate'],-r['collision_rate'],-r['decisions']))
            late = [r for r in ordered if 800000 <= r['decisions'] <= 1000000]
            summaries[mode][condition] = dict(best_screen=best, last_screen=ordered[-1],
                checkpoint500k=curve.get(500000), checkpoint550k=curve.get(550000),
                checkpoint800k=curve.get(800000), checkpoint1m=curve.get(1000000),
                late_800k_1m_capture_mean=float(np.mean([r['capture_rate'] for r in late])) if late else None,
                late_800k_1m_collision_mean=float(np.mean([r['collision_rate'] for r in late])) if late else None,
                late_checkpoints=len(late))
    kl = {}
    for condition in ['baseline','stability']:
        subset = [r for r in diagnostics if r['condition']==condition]
        kl[condition] = dict(updates=len(subset),
            target_kl=.02 if condition=='baseline' else .01,
            kl_stop_update_fraction=float(np.mean([r['kl_early_stop'] for r in subset])) if subset else None,
            approx_kl_mean=float(np.mean([r['approx_kl'] for r in subset])) if subset else None,
            approx_kl_max=float(max(r.get('approx_kl_max',r['approx_kl']) for r in subset)) if subset else None,
            PPO_minibatches=sum(r.get('minibatch_updates',0) for r in subset))
    complete = all(any(r['decisions']==1000000 and r['condition']==condition and r['mode']==mode for r in rows)
                   for condition in ['baseline','stability'] for mode in manifest['config']['evaluation']['modes'])
    report = dict(classification='PROVISIONAL', line_id=manifest['line_id'],
                  status='PAIRED_SCREENS_COMPLETE' if complete else 'PAIRED_SCREENS_RUNNING',
                  parent_fixed500k=manifest['parent_checkpoint'], scientific_delta=manifest['scientific_delta'],
                  curves=rows, aligned_differences=paired, checkpoint_summary=summaries,
                  KL_control_diagnostics=kl, capability_success_declared=False,
                  causal_conclusion='PENDING_REVIEW' if complete else 'PENDING_COMPLETE_CONTINUATIONS',
                  interpretation=['A requires consistent late retention improvement and effective KL control; provisional support only.',
                                  'B: no improvement does not by itself prove reward/network failure; KL control may not activate.',
                                  'C: differing modes/arms or capture-safety tradeoffs require a mixed/inconclusive assessment.',
                                  'Single training seed; screen metrics, no formal promotion; not selecting a final best model.',
                                  'Baseline and stability log frequency may differ if KL early-stop activates; screens compare aligned absolute steps.'])
    atomic_json(run / 'stability_report.json', report)
    lines = [f"# {manifest['line_id']} paired continuation",'',
             f"PROVISIONAL — {report['status']}; conclusion: {report['causal_conclusion']}.",'',
             'Fixed parent500k, not best; target_kl .02 -> .01; all other contracts/states retained.','',
             '| mode | condition | best screen step | best capture | last step | last capture | last collision |',
             '|---|---|---:|---:|---:|---:|---:|']
    for mode, variants in summaries.items():
        for condition, summary in variants.items():
            best,last = summary['best_screen'],summary['last_screen']
            lines.append(f"| {mode} | {condition} | {best['decisions']} | {best['capture_rate']:.1%} | {last['decisions']} | {last['capture_rate']:.1%} | {last['collision_rate']:.1%} |")
    lines += ['', 'Full 500k→1M checkpoint/optimization metrics: paired_screen_curve.csv and optimization_curve.csv.',
              'Absence of improvement cannot identify reward/network as the cause. Check actual KL early-stop activation.', '']
    (run / 'STABILITY_REPORT.md').write_text('\n'.join(lines))
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(2,3,figsize=(14,7),sharex=True)
        for i,mode in enumerate(manifest['config']['evaluation']['modes']):
            for condition in ['baseline','stability']:
                curve = sorted([r for r in rows if r['mode']==mode and r['condition']==condition],key=lambda r:r['decisions'])
                for j,metric in enumerate(['capture_rate','collision_rate','mean_reward']):
                    axes[i,j].plot([r['decisions']/1000 for r in curve],[r[metric] for r in curve],label=condition)
                    axes[i,j].set_title(mode+' '+metric); axes[i,j].legend(); axes[i,j].set_xlabel('decisions (k)')
        fig.tight_layout(); fig.savefig(run/'paired_screen_curves.png',dpi=140); plt.close(fig)
    return report
