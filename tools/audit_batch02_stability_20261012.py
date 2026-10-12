#!/usr/bin/env python3
"""Recover actual P1 history and pin existing 500k continuation parents."""
import copy
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import torch
from terl_mappo.batch02_native.runtime import sources as native_sources
from terl_mappo.batch02_a_long.runtime import sources as long_sources
from terl_mappo.run import atomic_json, file_hash

BASE = ROOT.parent
OUT = ROOT / 'artifacts/2026-10-12_batch02_stability'
CONFIGS = ROOT / 'configs/experiments/batch02_stability_20261012'


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [row for key in sorted(a.keys() | b.keys())
                for row in differences(a.get(key), b.get(key), f'{path}.{key}'.strip('.'))]
    return [] if a == b else [dict(path=path, before=a, after=b)]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def checkpoint_metadata(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['trainer']
    return payload, dict(path=str(path), sha256=file_hash(path), bytes=path.stat().st_size,
                         schema=payload.get('schema', payload.get('version')),
                         decisions=payload.get('decisions', payload.get('steps')),
                         trainer_keys=list(state), update_count=state['update_count'],
                         actor_optimizer_groups=state['actor_optimizer']['param_groups'],
                         critic_optimizer_groups=state['value_optimizer']['param_groups'],
                         actor_optimizer_state_count=len(state['actor_optimizer']['state']),
                         critic_optimizer_state_count=len(state['value_optimizer']['state']),
                         value_norm_present=state['value_normalizer'] is not None,
                         rng_keys=list(payload['rng']),
                         cuda_rng_count=len(payload['rng']['cuda']),
                         runtime_keys=list(payload['runtime']))


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    history = {}
    for line in ['control', 'treatment']:
        wt = BASE / f'batch01-p1-{line}-long-v3r2-20261010'
        run = next((wt / 'runs').glob('B01-P1-*-PROVLONG-V3R2-20261010'))
        manifest = json.loads((run / 'manifest.json').read_text())
        sha = manifest['source_git_head']
        actual_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=wt, text=True).strip()
        assert actual_sha == sha
        checkpoint = run / 'checkpoints/step_001000000.pt'
        payload, metadata = checkpoint_metadata(checkpoint)
        assert payload['config']['ppo'] == manifest['resolved_config']['ppo']
        assert manifest['active_runtime_values']['ppo'] == manifest['resolved_config']['ppo']
        sidecar = json.loads(checkpoint.with_suffix('.batch01.json').read_text())
        assert metadata['sha256'] in json.dumps(sidecar)
        destination = OUT / 'history' / line
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['manifest.json', 'fork_receipt.json']:
            shutil.copyfile(run / name, destination / name)
        shutil.copyfile(checkpoint.with_suffix('.batch01.json'), destination / 'checkpoint_1m_sidecar.json')
        atomic_json(destination / 'checkpoint_metadata.json', metadata)
        atomic_json(destination / 'resolved_config.json', manifest['resolved_config'])
        config_path = 'configs/experiments/terl_mappo_batch01_20261009/p1-' + line + '_delta.json'
        (destination / 'delta_config.json').write_text(git('show', f'{sha}:{config_path}'))
        history[line] = dict(commit_SHA=sha, config_path=config_path,
                             resolved_config=manifest['resolved_config'], checkpoint=metadata,
                             manifest=str(run / 'manifest.json'))
    scientific = differences(history['control']['resolved_config']['ppo'], history['treatment']['resolved_config']['ppo'])
    assert scientific == [dict(path='target_kl', before=.02, after=.01)]
    # Archive the full before/after configs; experiment identity is separately visible.
    before = json.dumps(history['control']['resolved_config'], indent=2, sort_keys=True).splitlines(True)
    after = json.dumps(history['treatment']['resolved_config'], indent=2, sort_keys=True).splitlines(True)
    (OUT / 'history/P1_full_config_diff.patch').write_text(''.join(difflib.unified_diff(before, after, 'P1-control', 'P1-treatment')))
    (OUT / 'history/P1_implementation_diff.patch').write_text(git('diff', '9cc2d47c532c76239a61a0f6a19c8603358c92bf', history['treatment']['commit_SHA'], '--', 'src/terl_mappo/batch01/arms/p1.py', history['treatment']['config_path']))
    central = BASE / 'ac-master-dag-20260921'
    for name in ['TERL_MAPPO_PARALLEL_MATRIX_20261009_ZH.md', 'BATCH01_SCIENTIFIC_CHANGELOG.md', 'BATCH01_MIGRATION_AUDIT_20261011_ZH.md']:
        shutil.copyfile(central / 'docs/experiments' / name, OUT / 'history' / name)
    dag = json.loads((central / 'artifacts/2026-09-21_ac_master_dag/state.json').read_text())
    atomic_json(OUT / 'history/P1_DAG_nodes.json', [row for row in dag['tasks'] if row['task_id'] in ['BATCH01-P1', 'BATCH01-P1-control']])
    runs = {}
    for arm in ['R2', 'R3', 'V3']:
        old = arm == 'R2'
        wt = BASE / ('terl-mappo-batch02-a-long-20261011' if old else 'terl-mappo-batch02-native-20261011')
        run = wt / 'runs' / ('batch02_a_capability_20261011' if old else 'batch02_native_capability_20261011') / arm
        manifest = json.loads((run / 'manifest.json').read_text())
        config = manifest['config']
        checkpoint = run / 'checkpoints/step_000500000.pt'
        payload, metadata = checkpoint_metadata(checkpoint)
        assert metadata['decisions'] == 500000
        assert payload['config'] == config
        assert payload['sources'] == (long_sources() if old else native_sources())
        registry = json.loads((run / 'checkpoint_list.json').read_text())
        entry = next(row for row in registry if row['decisions'] == 500000)
        assert entry['sha256'] == metadata['sha256']
        stability = copy.deepcopy(config)
        stability['learner']['ppo']['target_kl'] = history['treatment']['resolved_config']['ppo']['target_kl']
        delta = differences(config, stability)
        assert delta == [dict(path='learner.ppo.target_kl', before=.02, after=.01)]
        path = CONFIGS / f'{arm.lower()}.json'
        atomic_json(path, stability)
        atomic_json(OUT / f'{arm}_baseline_config.json', config)
        atomic_json(OUT / f'{arm}_stability_config.json', stability)
        a = json.dumps(config, indent=2, sort_keys=True).splitlines(True)
        b = json.dumps(stability, indent=2, sort_keys=True).splitlines(True)
        (OUT / f'{arm}_config_diff.patch').write_text(''.join(difflib.unified_diff(a, b, f'{arm}-baseline', f'{arm}-stability')))
        atomic_json(OUT / f'{arm}_parent_checkpoint_metadata.json', metadata)
        runs[arm] = dict(line_id=f'{arm}-stability', baseline_arm=arm,
                         baseline_SHA=manifest['experiment_SHA'], baseline_run=str(run),
                         parent_checkpoint=metadata, config_path=str(path.relative_to(ROOT)),
                         config_sha256=file_hash(path), scientific_delta=delta,
                         evaluator_version=manifest['evaluator_version'], evaluator_sha256=manifest['evaluator_sha256'],
                         physical_gpu=int(manifest['physical_gpu']), seed_manifest=manifest['seed_manifest'],
                         contract=payload['contract'], starting_step=500000, end_step=1000000,
                         additional_budget=500000, checkpoint_interval=25000, screen_interval=25000,
                         initialization='strict full-state fork; no optimizer/ValueNorm/RNG/environment reset',
                         status='AUDIT_PASS_VALIDATION_PENDING', classification='PROVISIONAL')
    audit = dict(status='HISTORY_AND_PARENT_AUDIT_PASS', classification='PROVISIONAL', history=history,
                 historical_scientific_delta=scientific, lr_scheduler='none; no scheduler introduced',
                 unchanged_parameters={k:v for k,v in history['control']['resolved_config']['ppo'].items() if k != 'target_kl'},
                 checkpoint_choice='fixed user-prespecified500k, before observed550k screen peak; no best-screen selection',
                 label_correction={'R3':'original TERL observation/backbone + native reward telemetry',
                                   'V3':'CoCap observation/backbone + reward telemetry; original central critic'},
                 no_new_tuning=True, runs=runs)
    atomic_json(OUT / 'historical_audit.json', audit)
    atomic_json(OUT / 'run_plan.json', dict(classification='PROVISIONAL', runs=runs, no_auto_extension=True,
                 no_next_stage=True, original_baselines_untouched=True, historical_setting_only=True))
    print(json.dumps(dict(status=audit['status'], historical_delta=scientific,
                         checkpoint_hashes={k:v['parent_checkpoint']['sha256'] for k,v in runs.items()})))


if __name__ == '__main__':
    main()
