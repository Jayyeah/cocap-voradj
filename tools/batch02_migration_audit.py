#!/usr/bin/env python3
"""Read existing Batch01 evidence; never evaluate policies or launch training."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
B01 = ROOT / 'artifacts/2026-10-09_terl_mappo_batch01'
OUT = ROOT / 'artifacts/2026-10-11_cocap_mappo_batch02'


def read(path):
    return json.loads(Path(path).read_text())


def hf(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def csvwrite(path, rows):
    if not rows:
        return
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def numeric(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'max': max(values) if values else None, 'min': min(values) if values else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = read(B01 / 'commander_v3r2/status_20261011/snapshot.json')
    registry = read(B01 / 'run_registry.json')
    registered = {r['run_id']: r for r in registry['pilot_runs']}
    result = {'schema': 'cocap.batch02.migration-audit.v1',
              'time_Asia_Shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'classification': 'PROVISIONAL_LONG_DESCRIPTIVE_ONLY',
              'formal_evidence': False, 'selection_or_final_consumed': False,
              'central_head': git(ROOT, 'rev-parse', 'HEAD'),
              'central_branch': git(ROOT, 'branch', '--show-current'),
              'base': {k: snapshot[k] for k in ['source_candidate_sha', 'canonical_lock_sha256',
                       'core_status', 'QA_status', 'base_freeze_status', 'BASE_FROZEN']}, 'arms': []}
    train_rows, screen_rows = [], []
    for row in snapshot['runs']:
        wt, output = Path(row['worktree']), Path(row['output_root'])
        manifest, delta = read(output / 'manifest.json'), read(output / 'delta.json')
        assert git(wt, 'rev-parse', 'HEAD') == row['head']
        assert not git(wt, 'status', '--porcelain', '--', 'src', 'configs')
        source_checks = []
        for change in delta['source_changes']:
            actual = hf(wt / change['path'])
            assert actual == change['after'], change['path']
            source_checks.append({'path': change['path'], 'sha256': actual})
        cp = row['checkpoint']
        assert hf(cp['path']) == cp['sha256']
        side = read(Path(cp['path']).with_suffix('.batch01.json'))
        assert side['checkpoint_sha256'] == cp['sha256'] and side['steps'] == row['step']
        patch = git(wt, 'diff', result['base']['source_candidate_sha'], row['head'], '--', 'src', 'configs', 'tools')
        (OUT / 'diffs').mkdir(exist_ok=True)
        (OUT / 'diffs' / (row['variant'] + '.patch')).write_text(patch + '\n')
        # A strict recovery can repeat an unsaved step range. Keep last record
        # for each step; raw files and recovery receipts remain authoritative.
        records = {}
        previous = registered.get(registered[row['run_id']].get('previous_run_id'))
        origins = ([Path(previous['output_root'])] if previous else []) + [output]
        for origin in origins:
            for line in (origin / 'metrics.jsonl').read_text().splitlines():
                record = json.loads(line)
                records[record['step']] = record
        records = [records[k] for k in sorted(records)]
        for rec in records:
            metrics = rec['metrics']
            train_rows.append({'arm': row['variant'], 'step': rec['step'], 'update': rec['update'],
                'target_kl': rec['live_target_kl'], **{k: metrics.get(k) for k in
                ['actor_loss', 'value_loss', 'entropy', 'approx_kl', 'approx_kl_max', 'clip_fraction',
                 'explained_variance', 'actor_grad_norm', 'actor_update_l2', 'kl_early_stop']},
                **{k: rec.get('post_update_policy', {}).get(k) for k in ['post_update_kl', 'post_update_ratio_max']},
                **{'reward_' + k: rec.get('reward_components_mean', {}).get(k) for k in
                   ['time', 'distance', 'global', 'emergency', 'collision', 'goal', 'raw', 'shaping', 'shaped']}})
        screens = {}
        for origin in origins:
            for path in (origin / 'evaluations').glob('screen_*.json'):
                if '.partial.' in path.name:
                    continue
                ev = read(path)
                assert ev['seed_domain'] == 'screen' and ev['status'] == 'PROVISIONAL_SCREEN'
                assert ev['evaluation_protocol'] == manifest['evaluation_protocol']
                screens[ev['steps']] = (path, ev)
        assert max(screens) == row['step']
        for step, (path, ev) in sorted(screens.items()):
            for mode, values in ev['modes'].items():
                screen_rows.append({'arm': row['variant'], 'step': step, 'mode': mode,
                    'n': values['episodes'], **{k: values[k] for k in
                    ['normal_capture_rate', 'collision_rate', 'ring2_rate', 'ring3_rate', 'strict_geometry_rate',
                     'capture_time_mean', 'censored_rate', 'episode_reward_mean', 'action_entropy']},
                    'checkpoint_sha256': ev['checkpoint_sha256'], 'source': str(path), 'sha256': hf(path)})
        modes = screens[max(screens)][1]['modes']
        best = {}
        for mode in modes:
            step = max(screens, key=lambda k: (screens[k][1]['modes'][mode]['normal_capture_rate'],
                      -screens[k][1]['modes'][mode]['collision_rate'], -k))
            path, ev = screens[step]
            checkpoint = Path(ev['checkpoint'])
            exists = checkpoint.exists()
            if exists:
                assert hf(checkpoint) == ev['checkpoint_sha256']
            best[mode] = {'step': step, 'checkpoint': str(checkpoint),
                          'checkpoint_sha256': ev['checkpoint_sha256'], 'checkpoint_present': exists,
                          'evaluation': str(path), 'evaluation_sha256': hf(path), 'metrics': ev['modes'][mode]}
        windows = {}
        for name, values in [('first100', records[:100]), ('last100', records[-100:])]:
            windows[name] = {'start_step': values[0]['step'], 'end_step': values[-1]['step'],
                'metrics': {k: numeric([v['metrics'].get(k) for v in values]) for k in
                 ['value_loss', 'entropy', 'approx_kl', 'approx_kl_max', 'clip_fraction', 'explained_variance',
                  'actor_grad_norm', 'kl_early_stop']},
                'reward': {k: numeric([v.get('reward_components_mean', {}).get(k) for v in values]) for k in
                          ['raw', 'shaping', 'shaped', 'distance', 'global']},
                'post_update_kl': numeric([v.get('post_update_policy', {}).get('post_update_kl') for v in values])}
        result['arms'].append({'variant': row['variant'], 'run_id': row['run_id'], 'status': row['status'],
            'head': row['head'], 'branch': row['branch'], 'worktree': str(wt), 'output_root': str(output),
            'delta_hash': row['delta_hash'], 'source_changes_verified': source_checks,
            'config_changes': delta['config_changes'], 'extensions': delta.get('extensions'),
            'resolved_config': manifest['resolved_config'], 'last_checkpoint': cp,
            'best_observed_screen': best, 'last_screen': modes, 'last_metric': records[-1],
            'training_windows': windows, 'screen_count': len(screens),
            'training_records_dedup': len(records), 'raw_training_records': sum(len((o/'metrics.jsonl').read_text().splitlines()) for o in origins),
            'recovery_history': row.get('resource_resume_history'), 'formal_evidence': False})
    write(OUT / 'batch01_audit.json', result)
    csvwrite(OUT / 'training_curves.csv', train_rows)
    csvwrite(OUT / 'screen_curves.csv', screen_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, arm in zip(axes.flat, result['arms']):
        label = arm['variant']
        for mode in arm['last_screen']:
            pts = [p for p in screen_rows if p['arm'] == label and p['mode'] == mode]
            ax.plot([p['step']/1000 for p in pts], [p['normal_capture_rate'] for p in pts], label=mode+' normal')
            ax.plot([p['step']/1000 for p in pts], [p['collision_rate'] for p in pts], linestyle='--', label=mode+' collision')
        ax.set(title=label, ylim=(-.02, 1.02), xlabel='Joint decisions (k)')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Batch01 PROVISIONAL_LONG: all observed screens, n=10/20; no selection/final')
    fig.tight_layout(); fig.savefig(OUT/'all_screen_curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    rows = [r for r in train_rows if r['arm']=='r1']
    for ax, keys in zip(axes.flat, [('entropy',), ('approx_kl_max','post_update_kl'), ('value_loss','explained_variance'), ('reward_raw','reward_shaped','reward_shaping','reward_global')]):
        for key in keys:
            # Fixed 50-record blocks are a plotting summary, not a new metric.
            blocks = [rows[i:i+50] for i in range(0,len(rows),50)]
            ax.plot([b[-1]['step']/1000 for b in blocks], [numeric([r[key] for r in b])['mean'] for b in blocks], label=key)
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_xlabel('Joint decisions (k)')
    fig.suptitle('R1 training diagnostics: mean of consecutive 50-update blocks; raw CSV preserved')
    fig.tight_layout(); fig.savefig(OUT/'r1_training_diagnostics.png', dpi=150); plt.close(fig)
    print(json.dumps({'arms':len(result['arms']),'screens':len(screen_rows),'training_records':len(train_rows),
                      'output':str(OUT),'formal_evidence':False}))


if __name__ == '__main__':
    main()
