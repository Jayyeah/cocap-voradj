#!/usr/bin/env python3
"""Describe the completed two-head control; never train or select new weights."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
from tools import audit_forward_final_critic_gradients_20260909 as audit


def summarize(out):
    report_path = out / 'report.json'
    report = json.loads(report_path.read_text())
    result = {'classification': 'EXPERIMENT RESULT', 'report_sha256': audit.sha(report_path),
              'source_sha256': audit.sha(__file__), 'temporal_decomposition': {}, 'minibatch_gradient_comparison': {}}
    old = json.loads((ROOT / 'artifacts/2026-09-09_root_cause/critic_gradient_audit_v2/report.json').read_text())
    for split in ('train', 'heldout'):
        data = dict(np.load(audit.BASE / split / 'fixed_bank.npz'))
        predictions = dict(np.load(out / (split + '_predictions.npz')))
        bins = audit.time_bins(data)
        result['temporal_decomposition'][split] = {}
        for i, phase in enumerate(audit.PHASES):
            mask = data['active'] & (data['phase'] == i)[:, None]
            y = data['target'].astype(float)
            old_p, new_p = predictions['single_head'].astype(float), predictions['two_head'].astype(float)
            row = {'two_minus_single_prediction': audit.distribution((new_p - old_p)[mask]), 'strata': {}}
            for label, time_mask in (('first_50', bins == 0), ('after_50', bins > 0)):
                m = mask & time_mask[:, None]
                row['strata'][label] = {'target': audit.distribution(y[m]),
                    'predictors': {k: {**audit.calibration(y[m], p[m].astype(float)),
                        'prediction_mean': float(p[m].mean()), 'residual_median': float(np.median(p[m] - y[m]))}
                        for k, p in predictions.items()}}
            a = report['scores'][split][phase]['predictors']['single_head']
            b = report['scores'][split][phase]['predictors']['two_head']
            delta_mse = b['rmse'] ** 2 - a['rmse'] ** 2
            row['mse_increase'] = delta_mse
            row['squared_bias_increase'] = b['squared_bias'] - a['squared_bias']
            row['centered_error_variance_increase'] = b['centered_error_variance'] - a['centered_error_variance']
            row['bias_fraction_of_mse_increase'] = row['squared_bias_increase'] / delta_mse if abs(delta_mse) > 1e-12 else None
            result['temporal_decomposition'][split][phase] = row
    for snapshot, batches in [('single_head', old['minibatches_at_fixed_final_checkpoint']),
                               ('two_head', report['minibatches_at_fixed_final_checkpoint'])]:
        result['minibatch_gradient_comparison'][snapshot] = {}
        for group in ('trunk', 'head', 'all'):
            rows = [b['gradients'][group] for b in batches]
            result['minibatch_gradient_comparison'][snapshot][group] = {
                'negative_cosine_count_of_16': {pair: sum(r['cosine'][pair] < 0 for r in rows) for pair in rows[0]['cosine']},
                'median_cosine': {pair: float(np.median([r['cosine'][pair] for r in rows])) for pair in rows[0]['cosine']},
                'median_phase_norm_mass_share_nonadditive': {p: float(np.median([r['norm_mass_share_nonadditive'][p] for r in rows])) for p in audit.PHASES},
                'joint_raw_negative_gradient_would_raise_phase_loss_first_order_count': {p: sum(r['signed_projection_on_global'][p] < 0 for r in rows) for p in audit.PHASES}}
    assert all(report['checks'][s + '/' + p]['rmse_ratio'] > 1 for s in ('train', 'heldout') for p in audit.PHASES[1:])
    assert all(report['gradients'][s]['two_head']['trunk']['cosine'][pair] > 0
               for s in ('train', 'heldout') for pair in ('pre_capture/post_capture', 'pre_capture/pure_coverage'))
    result['decision'] = 'CASE_3_STOP_HEAD_SPLITTING_NO_RECOVERY_CALIBRATION_BENEFIT'
    result['interpretation_label'] = 'HYPOTHESIS'
    result['interpretation'] = 'At 100 matched updates head isolation shifts recovery predictions downward without learning the recovery time profile. Removing direct head conflict is insufficient; no causal support that head sharing is the main calibration bottleneck.'
    result['next_single_recommendation'] = 'Audit early-recovery transition state and full-return construction, including phase-boundary/reset event order and MC recurrence; no further critic architecture or optimizer intervention in this run.'
    result['gates'] = {'p2': 'HOLD', 'p3': 'HOLD', 'ppo_25k': 'HOLD', 'continuous': 'HOLD'}
    result['limits'] = 'One training initialization/budget and 5 heldout episodes per phase. Routing makes disjoint-head cosine zero by construction. Added head parameters and resulting global clipping/Adam trajectory are inseparable from this intervention. Gradient cosines are not a value-health gate.'
    audit.save(out / 'analysis.json', result)


if __name__ == '__main__':
    summarize(ROOT / 'artifacts/2026-09-09_root_cause/two_head_value_20260910')
