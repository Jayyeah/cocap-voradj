#!/usr/bin/env python3
"""Read-only completion audit of existing Evidence artifacts and checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
import yaml
from tools.iqn_evidence_comparison_20261006 import contract_diff_report, resolved
from tools.iqn_z_unified_decay_curriculum_20260919 import MILESTONES

RUNTIME = Path('/home/yjq/rl/CoCap1/iqn-evidence-comparison-20261006-runtime')
Z05 = Path('/home/yjq/rl/CoCap1/iqn-z-unified-decay-dual-curriculum-20260919-runtime/z05/stages')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def diff(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in left.keys() | right.keys():
            path = f'{prefix}.{key}'.strip('.')
            if key not in left or key not in right: result.append(path)
            else: result.extend(diff(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def audit():
    static = contract_diff_report()
    assert static['status'] == 'pass'
    result = {'schema': 'evidence-completion-audit-v1',
              'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'runtime': str(RUNTIME), 'static_contract': static, 'training': [], 'z05_reference': [],
              'training_launched': False, 'evaluation_launched': False}
    for variant in ('local_binary', 'global_oracle'):
        parent = None
        for index, stage in enumerate(('stage1', 'stage2', 'stage3')):
            directory = RUNTIME / variant / 'stages' / stage
            launch = read(directory / 'training/launch.json')
            cfg_path = directory / 'training/effective_config.yaml'
            cfg = yaml.safe_load(cfg_path.read_text())
            expected = resolved(variant, stage)
            expected.setdefault('pretrained', {})
            operational = ('output_root', 'run_name', 'device', 'pretrained.path', 'checkpointing.full_resume')
            changes = diff(expected, cfg)
            unexpected = [p for p in changes if not any(p == a or p.startswith(a + '.') for a in operational)]
            assert not unexpected, (variant, stage, unexpected)
            assert cfg['seed'] == launch['seed'] == 2026091901 + index
            assert launch['warm_start_checkpoint'] == parent
            if parent is not None: assert cfg['pretrained']['path'] == parent
            assert not cfg.get('checkpointing', {}).get('resume_path')
            metrics = [json.loads(line) for line in open(directory / 'training/metrics.jsonl')]
            final_step = max(MILESTONES[stage])
            assert metrics[-1]['global_step'] == final_step
            assert [m['global_step'] for m in metrics] == list(range(1000, final_step + 1, 1000))
            log = (directory / 'training/stdout.log').read_text(errors='replace')
            assert 'final_checkpoint=' in log
            suspicious = [line for line in log.splitlines() if any(w in line.lower() for w in ('traceback', 'nonfinite', 'nan', 'error', 'resume'))]
            assert not suspicious
            checkpoint = directory / 'training/checkpoints' / f'final_step_{final_step}.pt'
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert payload['extra']['global_step'] == final_step
            assert all(torch.isfinite(t).all().item() for t in payload['state_dict'].values())
            selection = read(directory / 'selection_report.json')
            complete, missing = [], []
            for step in MILESTONES[stage]:
                report = read(directory / 'evaluations' / f'step_{step:09d}' / 'report.json')
                if report and report.get('status') == 'complete' and report.get('episodes_per_scene') == 20:
                    assert len(report['records']) == 60
                    assert report['variant'] == variant and report['stage'] == stage
                    assert report['checkpoint_sha256'] == sha(directory / 'training/checkpoints' / f'step_{step}.pt')
                    complete.append(step)
                else: missing.append(step)
            selected = selection['selected'] if selection else None
            if selected:
                assert sha(selected['checkpoint']) == selected['checkpoint_sha256']
                assert selection['fallback_used'] is False
                assert 'BalancedFloor' in selection['selection_reason']
                assert len(selection['candidates']) == len(MILESTONES[stage])
                parent = selected['checkpoint']
            status = read(directory / 'status.json')
            result['training'].append({'variant': variant, 'stage': stage, 'training_status': 'TRAINING_COMPLETE',
                'actual_final_step': final_step, 'seed': cfg['seed'], 'launch': launch,
                'effective_config': str(cfg_path), 'effective_config_sha256': sha(cfg_path),
                'effective_operational_differences': changes, 'unexpected_scientific_differences': unexpected,
                'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'finite_parameters': True,
                'resume_requested': False, 'anomaly_log_lines': suspicious, 'metrics_monotonic_and_complete': True,
                'selection_status': 'SELECTION_COMPLETE' if selected and not missing else 'SELECTION_PENDING',
                'candidate_steps': list(MILESTONES[stage]), 'screening_complete_steps': complete,
                'screening_missing_steps': missing, 'selection_rule': 'balanced_floor',
                'selection': selected, 'selection_report': str(directory / 'selection_report.json'),
                'heartbeat_snapshot': status})
    for stage in ('stage1', 'stage2', 'stage3'):
        selection = read(Z05 / stage / 'selection_report.json')['selected']
        assert sha(selection['checkpoint']) == selection['checkpoint_sha256']
        result['z05_reference'].append({'stage': stage, **selection})
    assert result['z05_reference'][-1]['checkpoint_sha256'] == '8ee5c162c32883984f72aa4be4b86e82338ae1e8d8c912011d181987a476d095'
    result['final_heldout'] = {'status': 'FINAL_EVAL_PENDING', 'episodes_per_scene': 50,
        'seed_base': 2026100601, 'seed_rule': 'seed_base + scene_index*100000 + episode_index',
        'expected_seed_manifest': [2026100601 + j * 100000 + i for j in range(3) for i in range(50)],
        'variants': {}}
    final_reports = {}
    for variant in ('local_binary', 'z05', 'global_oracle'):
        path = RUNTIME / 'final_heldout_50_corrected' / variant / 'report.json'
        report = read(path)
        if report:
            assert report['episodes_per_scene'] == 50
            assert report['seed_manifest'] == result['final_heldout']['expected_seed_manifest']
            assert report['optimizer_updates'] == report['replay_updates'] == 0
            final_reports[variant] = report
        result['final_heldout']['variants'][variant] = {'report': str(path), 'exists': path.exists(),
            'status': report.get('status') if report else 'MISSING',
            'diagnostic_implementation': report.get('diagnostic_implementation') if report else None,
            'coverage_timing_implementation': report.get('coverage_timing_implementation') if report else None}
    final_complete = len(final_reports) == 3 and all(
        r.get('status') == 'complete' and r.get('diagnostic_implementation') == 'corrected_global_target_token_occupancy_and_shortfall_v2'
        and r.get('coverage_timing_implementation') == 'first_strict_ce_success_transition_v1' for r in final_reports.values())
    if final_complete:
        assert len({tuple(r['initial_state_fingerprint'] for r in v['records']) for v in final_reports.values()}) == 1
        result['final_heldout']['status'] = 'FINAL_EVAL_COMPLETE'
    selection_complete = all(t['selection_status'] == 'SELECTION_COMPLETE' for t in result['training'])
    result['status'] = ' / '.join(['TRAINING_COMPLETE', 'SELECTION_COMPLETE' if selection_complete else 'SELECTION_PENDING',
                                  'FINAL_EVAL_COMPLETE' if final_complete else 'FINAL_EVAL_PENDING'])
    result['classification'] = 'Unified final comparison verified.' if final_complete else 'No unified final comparison or scientific ordering is available at this snapshot.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stages': len(result['training']),
                      'selected': sum(t['selection'] is not None for t in result['training'])}))
