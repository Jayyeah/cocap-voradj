#!/usr/bin/env python3
"""Offline BASE engineering only: lock, verify, prepare, and recover evidence."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from terl_mappo.batch01.contracts import (ROOT, LOCK_PATH, PARENT_SHA, create_lock, fingerprint,
    verify_base, verify_committed_pin, validate_resources)
from terl_mappo.batch01.interfaces import Hooks, assemble
from terl_mappo.batch01.provenance import runtime_manifest
from terl_mappo.batch01.evaluation import performance_report, isolated_rng
from terl_mappo.run import atomic_json, file_hash, source_hash

ORIGINAL = Path('/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008')
SELECTED_HASH = '590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4'


def read(path):
    return json.loads(Path(path).read_text())


def recover(output):
    out = ORIGINAL / 'runs/terl_mappo_stage1_seed9_1m_continuation'
    checkpoint = out / 'checkpoints/step_000775000.pt'
    digest = file_hash(checkpoint)
    if digest != SELECTED_HASH: raise ValueError('selected anchor hash mismatch')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    manifest = read(out / 'manifest.json')
    if ck['sources'] != source_hash() or ck['steps'] != 775000 or ck['config'] != manifest['config']:
        raise ValueError('selected checkpoint source/config/counter mismatch')
    if any(not torch.isfinite(v).all() for part in ('actor', 'value') for v in ck['trainer'][part].values()):
        raise ValueError('selected actor/critic nonfinite')
    evaluations = [read(p) for p in sorted((out / 'evaluations').glob('*.json'))
                   if not p.name.endswith('.partial.json')]
    by_domain = {d: [r for r in evaluations if r['seed_domain'] == d] for d in ('screen', 'selection', 'final')}
    if len(by_domain['screen']) != 41 or len(by_domain['selection']) != 3 or len(by_domain['final']) != 1:
        raise ValueError('incomplete original 1m evaluation evidence')
    report = performance_report(by_domain['screen'], by_domain['selection'], by_domain['final'])
    if report['selected_heldout']['checkpoint_sha256'] != digest:
        raise ValueError('selected checkpoint/evaluation mismatch')
    report = {k: {key: v for key, v in value.items() if key != 'episodes'}
              if isinstance(value, dict) else value for k, value in report.items()}
    # The archived closeout is preserved; no rewriting of the original report.
    audit = ROOT / 'artifacts/2026-10-08_terl_mappo_1m/completion_audit_20261009.json'
    result = {'schema': 'terl.batch01.recovery.v1', 'parent_sha': PARENT_SHA,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
              'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_steps': ck['steps'],
              'terl_sha': ck['terl_sha'], 'source_hashes_verified': True,
              'original_manifest_sha256': file_hash(out / 'manifest.json'),
              'original_completion_audit': str(audit), 'original_completion_audit_sha256': file_hash(audit),
              'original_completion_audit_passed': read(audit)['all_checks_passed'],
              'original_full_state_resume_evidence': read(ROOT / 'artifacts/2026-10-08_terl_mappo_1m/exact_resume_validation.json')
                  if (ROOT / 'artifacts/2026-10-08_terl_mappo_1m/exact_resume_validation.json').exists() else
                  {'note': 'see original 1m final report and test evidence'},
              'training_progress': read(out / 'progress.json'), 'performance': report,
              'evaluation_counts': {d: {'checkpoints': len(rs), 'episodes': sum(len(r['episodes']) for r in rs)}
                                    for d, rs in by_domain.items()},
              'evaluation_file_hashes': {str(p): file_hash(p) for p in sorted((out / 'evaluations').glob('*.json'))
                                         if not p.name.endswith('.partial.json')},
              'storage': 'original verified file reused read-only; no model copies',
              'science_status': 'original single-seed Stage1 result preserved; candidate requires independent QA'}
    atomic_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('performance', 'training_progress', 'original_full_state_resume_evidence', 'evaluation_file_hashes')}, ensure_ascii=False))


def prepare(args):
    lock, delta = read(args.lock), read(args.delta)
    resolved = verify_base(lock, delta, args.pin)
    verify_committed_pin(lock, delta)
    resources = validate_resources(args.output, args.device, args.threads)
    torch.set_num_threads(args.threads)
    extension = delta.get('extensions', {})
    hooks = Hooks()
    if extension:
        entry = extension['entrypoint']
        module, name = entry.split(':')
        if not module.startswith('terl_mappo.batch01.extensions.'):
            raise ValueError('extension entrypoint must be declared inside batch01/extensions')
        hooks = getattr(importlib.import_module(module), name)(extension.get('parameters', {}))
        if not isinstance(hooks, Hooks): raise ValueError('extension must return Hooks')
    if delta['line'] != 'T0' and hooks == Hooks() and not any(
        e['path'].split('.')[0] not in {'experiment', 'seed', 'actor_seed', 'budget', 'stage1_ceiling'}
        for e in delta.get('config_changes', [])):
        raise ValueError('experiment line has no implemented scientific delta')
    with isolated_rng():
        runtime = assemble(resolved, args.device, hooks, delta['line'])
        manifest = runtime_manifest(lock, delta, resolved, runtime, resources)
    output = Path(args.output)
    if output.exists(): raise ValueError('refuse to overwrite a prepared manifest')
    atomic_json(output, manifest)
    print(json.dumps({'status': manifest['status'], 'output': str(output),
                      'base': manifest['base'], 'manifest_fingerprint': fingerprint(manifest)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('lock')
    r = sub.add_parser('recover'); r.add_argument('--output', required=True)
    v = sub.add_parser('prepare')
    v.add_argument('--lock', default=str(ROOT / LOCK_PATH)); v.add_argument('--delta', required=True)
    v.add_argument('--pin', required=True, help='canonical lock fingerprint explicitly frozen by the experiment')
    v.add_argument('--output', required=True); v.add_argument('--device', default='cpu'); v.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if args.command == 'lock':
        output = ROOT / LOCK_PATH
        if output.exists(): raise ValueError('refuse to refresh existing BASE lock')
        lock = create_lock(); atomic_json(output, lock)
        print(json.dumps({'lock': str(output), 'canonical_sha256': fingerprint(lock), 'status': lock['status']}))
    elif args.command == 'recover':
        validate_resources(args.output); torch.set_num_threads(1); recover(args.output)
    else: prepare(args)


if __name__ == '__main__': main()
