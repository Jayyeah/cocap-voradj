#!/usr/bin/env python3
"""CPU derivations from fixed-checkpoint gradient audit; no neural fitting."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
from tools.audit_forward_final_critic_gradients_20260909 import BASE, CONTEXT, PHASES, sha, time_bins, save


def summarize(root):
    report_path = root / 'report.json'
    report = json.loads(report_path.read_text())
    result = {'classification': 'EXPERIMENT RESULT', 'report_sha256': sha(report_path),
              'source_sha256': sha(__file__), 'minibatch_summary': {}, 'parameter_groups': {}, 'bias_time_diagnostic': {}}
    batches = report['minibatches_at_fixed_final_checkpoint']
    for group in ('trunk', 'head', 'all'):
        rows = [b['gradients'][group] for b in batches]
        result['minibatch_summary'][group] = {
            'count': len(rows),
            'joint_raw_negative_gradient_would_raise_phase_loss_first_order': {p: sum(r['signed_projection_on_global'][p] < 0 for r in rows) for p in PHASES},
            'cosines': {pair: {'negative_count': sum(r['cosine'][pair] < 0 for r in rows),
                'p10_p50_p90': np.percentile([r['cosine'][pair] for r in rows], [10, 50, 90]).tolist()}
                for pair in rows[0]['cosine']},
            'norm_mass_share_nonadditive': {p: np.percentile([r['norm_mass_share_nonadditive'][p] for r in rows], [10, 50, 90]).tolist() for p in PHASES},
            'global_norm_p10_p50_p90': np.percentile([r['global_norm'] for r in rows], [10, 50, 90]).tolist(),
        }
    result['minibatch_summary']['would_clip_global_norm_at_0_5'] = sum(b['gradients']['all']['global_norm'] > .5 for b in batches)
    for split in ('train', 'heldout'):
        row = report['snapshots']['context_fitted'][split]
        path = ROOT / row['gradient_vectors']['path']
        assert sha(path) == row['gradient_vectors']['sha256']
        vectors = dict(np.load(path))
        groups, offset = {}, 0
        for param in row['gradient_vectors']['parameter_layout']:
            end = offset + param['numel']
            name = param['name']
            group = ('output_linear' if name.startswith('value_head.3.') else
                     'head_hidden' if name.startswith('value_head.') else name.split('.')[0])
            groups.setdefault(group, []).append((offset, end))
            offset = end
        result['parameter_groups'][split] = {}
        for group, spans in groups.items():
            values = [np.concatenate([vectors[p][start:end] for start, end in spans]) for p in PHASES]
            norms = [float(np.linalg.norm(v)) for v in values]
            result['parameter_groups'][split][group] = {
                'phase_norm': dict(zip(PHASES, norms)),
                'cosine': {PHASES[i] + '/' + PHASES[j]: float(values[i] @ values[j] / (norms[i] * norms[j]))
                           if norms[i] * norms[j] > 1e-30 else None
                           for i in range(3) for j in range(i + 1, 3)},
            }
        data = dict(np.load(BASE / split / 'fixed_bank.npz'))
        prediction = np.load(CONTEXT / split / 'predictions.npz')['context_fitted'].astype(float)
        bins = time_bins(data)
        result['bias_time_diagnostic'][split] = {}
        for phase, name in enumerate(PHASES):
            mask = data['active'] & (data['phase'] == phase)[:, None]
            y = data['target'].astype(float)
            residual = prediction - y
            train_bias = report['snapshots']['context_fitted']['train']['statistics'][name]['predictors']['context']['bias']
            corrected = residual[mask] - train_bias
            early = mask & (bins == 0)[:, None]
            mse = float(np.mean(residual[mask] ** 2))
            between_bin_sse = sum(float(np.sum(m)) * float(np.mean(residual[m])) ** 2
                for b in np.unique(bins[data['phase'] == phase])
                for m in [mask & (bins == b)[:, None]])
            result['bias_time_diagnostic'][split][name] = {
                'train_only_bias_removed_rmse': float(np.sqrt(np.mean(corrected ** 2))),
                'train_bias_subtracted': train_bias,
                'early_50_steps_row_fraction': float(early.sum() / mask.sum()),
                'early_50_steps_sse_fraction': float(np.sum(residual[early] ** 2) / np.sum(residual[mask] ** 2)),
                'time_bin_mean_residual_sse_fraction': between_bin_sse / (int(mask.sum()) * mse),
                'note': 'Bias subtraction is a closed-form diagnostic, not a changed checkpoint. Time-bin group residual uses this split for descriptive decomposition only, not heldout fitting.',
            }
    save(root / 'derived_summary.json', result)


if __name__ == '__main__':
    summarize(ROOT / 'artifacts/2026-09-09_root_cause/critic_gradient_audit_v2')
