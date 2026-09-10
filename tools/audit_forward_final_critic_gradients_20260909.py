#!/usr/bin/env python3
"""Read-only fixed-BC-bank value loss/gradient audit. No rollout or optimizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from cocap_voradj.models.small_step_ac import CentralValueNetwork
from cocap_voradj.training.small_step_ac import ValueNorm, tensor_tree, MAPPOConfig
from cocap_voradj.training.forward_final_value_context import augment, expand_value_input, SCHEMA, NAMES
from tools.summarize_forward_final_root_cause_20260909 import calibration, PHASES

BASE = ROOT / 'artifacts/2026-09-09_forward_final_d/fixed_mc_critic'
CONTEXT = ROOT / 'artifacts/2026-09-09_root_cause/context_critic_lineage_fixed'
CHECKPOINT_SHA = '53af9775acc2e04d1527c55310468822453c74603c4c9504640b695e73d7a1fe'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def distribution(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()),
                **dict(zip(('p10', 'p50', 'p90'), map(float, np.percentile(x, [10, 50, 90])))))


def time_bins(data):
    age = np.zeros(len(data['phase']), dtype=int)
    for episode in np.unique(data['episode']):
        for phase in range(3):
            ids = np.flatnonzero((data['episode'] == episode) & (data['phase'] == phase))
            age[ids] = np.arange(len(ids))
    return age // 50


def baseline_tables(data):
    bins = time_bins(data)
    means, phase_means = {}, {}
    for phase in range(3):
        mask = data['active'] & (data['phase'] == phase)[:, None]
        phase_means[phase] = float(data['target'][mask].mean())
        for b in np.unique(bins[data['phase'] == phase]):
            mask = data['active'] & ((data['phase'] == phase) & (bins == b))[:, None]
            means[phase, int(b)] = float(data['target'][mask].mean())
    return means, phase_means


def baseline_predictions(data, tables):
    means, phase_means = tables
    predictions = {}
    for name, values in (
        ('phase_constant', [phase_means[int(p)] for p in data['phase']]),
        ('phase_time', [means.get((int(p), int(b)), phase_means[int(p)])
                        for p, b in zip(data['phase'], time_bins(data))]),
    ):
        predictions[name] = np.broadcast_to(np.asarray(values)[:, None], data['target'].shape)
    return predictions


def gradient_summary(gradients, names):
    """Inputs are gradients of phase loss / GLOBAL active count, not phase means."""
    output = {}
    for group in ('trunk', 'head', 'all'):
        ids = [i for i, name in enumerate(names)
               if group == 'all' or name.startswith('value_head.') == (group == 'head')]
        vectors = [torch.cat([g[i].reshape(-1) for i in ids]) for g in gradients]
        total = sum(vectors)
        norm = [float(v.norm()) for v in vectors]
        output[group] = {
            'global_norm': float(total.norm()),
            'phase_norm': dict(zip(PHASES, norm)),
            'norm_mass_share_nonadditive': dict(zip(PHASES, [n / max(sum(norm), 1e-30) for n in norm])),
            'signed_projection_on_global': dict(zip(PHASES, [float(v @ total / total.square().sum().clamp_min(1e-30)) for v in vectors])),
            'cosine': {PHASES[i] + '/' + PHASES[j]: float(vectors[i] @ vectors[j] / max(norm[i] * norm[j], 1e-30))
                       for i in range(3) for j in range(i + 1, 3)},
        }
    return output


def phase_gradients(model, central, data, norm, device, chunk=64, progress=None):
    params = list(model.parameters())
    count = int(data['active'].sum())
    grads, losses = [], []
    prediction = np.zeros_like(data['target'])
    for phase in range(3):
        accum = [torch.zeros_like(p, dtype=torch.float64) for p in params]
        loss_sum = 0.
        ids = np.flatnonzero(data['phase'] == phase)
        for offset in range(0, len(ids), chunk):
            take = ids[offset:offset + chunk]
            batch = tensor_tree({k: v[take] for k, v in central.items()}, device)
            active = torch.as_tensor(data['active'][take], device=device)
            target = norm.normalize(torch.as_tensor(data['target'][take], device=device))
            value = model(batch)
            prediction[take] = norm.denormalize(value.detach()).cpu().numpy()
            loss = .5 * (value[active] - target[active]).square().sum() / count
            part = torch.autograd.grad(loss, params)
            for acc, g in zip(accum, part):
                acc.add_(g.double())
            loss_sum += float(loss.detach())
        grads.append(accum)
        losses.append(loss_sum)
        if progress:
            progress(phase)
    return grads, losses, prediction


def direct_gradient(model, central, data, norm, device):
    active = torch.as_tensor(data['active'], device=device)
    target = norm.normalize(torch.as_tensor(data['target'], device=device))
    value = model(tensor_tree(central, device))
    return torch.autograd.grad(.5 * (value[active] - target[active]).square().mean(), list(model.parameters()))


def prediction_stats(data, predictions, losses=None):
    result = {}
    bins = time_bins(data)
    for phase, name in enumerate(PHASES):
        mask = data['active'] & (data['phase'] == phase)[:, None]
        y = data['target'][mask].astype(np.float64)
        result[name] = {'target': distribution(y), 'joint_states': int((data['phase'] == phase).sum()),
                        'episodes': int(len(np.unique(data['episode'][data['phase'] == phase]))), 'predictors': {}}
        if losses is not None:
            result[name]['global_loss_contribution'] = losses[phase]
            result[name]['global_loss_share'] = losses[phase] / sum(losses)
        for label, prediction in predictions.items():
            p = prediction[mask].astype(np.float64)
            result[name]['predictors'][label] = {
                **calibration(y, p), 'residual': distribution(p - y), 'prediction': distribution(p),
                'overestimate_fraction': float(np.mean(p > y)),
                'centered_error_variance': float(np.var(p - y)), 'squared_bias': float(np.mean(p - y) ** 2),
            }
        result[name]['time_profile'] = []
        for b in np.unique(bins[data['phase'] == phase]):
            m = mask & (bins == b)[:, None]
            result[name]['time_profile'].append({'elapsed_phase_step_bin_50': int(b), 'rows': int(m.sum()),
                'target_mean': float(data['target'][m].mean()),
                **{label + '_mean': float(p[m].mean()) for label, p in predictions.items()}})
        result[name]['episode_residuals'] = []
        for episode in np.unique(data['episode'][data['phase'] == phase]):
            m = mask & (data['episode'] == episode)[:, None]
            result[name]['episode_residuals'].append({'episode': int(episode), 'rows': int(m.sum()),
                **{label: calibration(data['target'][m].astype(float), p[m].astype(float)) for label, p in predictions.items()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(1)
    checkpoint = CONTEXT / 'context_critic_100.pt'
    assert sha(checkpoint) == CHECKPOINT_SHA
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['schema'] == SCHEMA and list(payload['names']) == list(NAMES)
    assert MAPPOConfig().value_coef == 1. and MAPPOConfig().max_grad_norm == .5
    norm = ValueNorm(device=args.device)
    norm.load_state_dict(payload['normalizer'])
    assert tuple(payload['normalization']) == (float(norm.mean), float(norm.std))
    sources = {str(checkpoint.relative_to(ROOT)): sha(checkpoint), str(Path(__file__).relative_to(ROOT)): sha(__file__)}
    banks, centers, refs = {}, {}, {}
    for split in ('train', 'heldout'):
        data = dict(np.load(BASE / split / 'fixed_bank.npz'))
        context_path = CONTEXT / split / 'context.npz'
        ctx = np.load(context_path)['context']
        manifest = json.loads((CONTEXT / split / 'context_manifest.json').read_text())
        assert sha(context_path) == manifest['context_sha256']
        assert sha(BASE / split / 'fixed_bank.npz') == manifest['old_bank_sha256']
        assert manifest['all_geometry_target_phase_episode_bit_exact']
        assert np.isin(data['phase'], [0, 1, 2]).all()
        np.testing.assert_array_equal(ctx[:, :3].argmax(1), data['phase'])
        np.testing.assert_array_equal(data['active'], data['global_active_mask'])
        banks[split] = data
        centers[split] = augment({k[7:]: v for k, v in data.items() if k.startswith('global_')}, ctx)
        refs[split] = {'context': np.load(CONTEXT / split / 'predictions.npz')['context_fitted'],
                       'geometry': np.load(BASE / split / 'predictions.npz')['fitted']}
        for path in (BASE / split / 'fixed_bank.npz', context_path, CONTEXT / split / 'predictions.npz', BASE / split / 'predictions.npz'):
            sources[str(path.relative_to(ROOT))] = sha(path)
    tables = baseline_tables(banks['train'])
    baseline_reference = json.loads((ROOT / 'artifacts/2026-09-09_root_cause/cpu/existing_mc_enrichment.json').read_text())
    report = {'classification': 'EXPERIMENT RESULT', 'head_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol': {'actor_updates': 0, 'critic_updates': 0, 'rollouts': 0, 'optimizer_constructed': False,
            'normalization': list(payload['normalization']), 'loss': '0.5 * sum(active normalized residual squared) / global active rows; value_coef=1',
            'gradients': 'raw autograd BEFORE global clipping or Adam; global phase contributions sum to global gradient',
            'trunk': 'all parameters except value_head.*', 'head': 'value_head.*',
            'norm_share': 'norms are not additive; norm mass is NOT a causal update percentage; signed global projection is additive',
            'baseline': 'train-only phase x 50 elapsed-phase steps; absent bin falls back to train phase mean',
            'limitations': 'Fixed checkpoints, not historical optimizer steps. Correlated agent/time rows in 30 train + 10 heldout all-safe episodes. No repeated conditional outcomes; irreducible variance not identified.',
            'gpu_visible': os.environ.get('CUDA_VISIBLE_DEVICES')}, 'sources': sources, 'snapshots': {}}
    initial_state = None
    for snapshot, splits in (('context_fitted', ('train', 'heldout')), ('context_initial', ('train',))):
        torch.manual_seed(2026097101)
        model = expand_value_input(CentralValueNetwork(hidden_dim=256, num_heads=8, num_layers=4,
                         self_feature_dim=9, max_agents=4, max_evaders=8, max_obstacles=5)).to(args.device)
        if initial_state is None:
            initial_state = state_sha(model)
        if snapshot == 'context_fitted':
            model.load_state_dict(payload['value'], strict=True)
        model.eval()
        before = state_sha(model)
        names = [name for name, _ in model.named_parameters()]
        report['snapshots'][snapshot] = {}
        for split in splits:
            data, central = banks[split], centers[split]
            def progress(phase):
                save(args.out / 'progress.json', {'status': 'gradient_audit_no_updates', 'snapshot': snapshot,
                     'split': split, 'phase_complete': PHASES[phase], 'elapsed_seconds': time.monotonic() - start, 'pid': os.getpid()})
                print(snapshot, split, PHASES[phase], 'complete', flush=True)
            gradients, losses, prediction = phase_gradients(model, central, data, norm, args.device, progress=progress)
            predictions = dict(refs[split], **baseline_predictions(data, tables))
            if snapshot == 'context_fitted':
                max_error = float(np.max(np.abs(prediction - refs[split]['context'])))
                assert max_error < .0002, max_error
            else:
                max_error = None
                predictions['initial_same_frozen_normalizer'] = prediction
            stats = prediction_stats(data, predictions, losses)
            for phase in PHASES:
                expected = baseline_reference['scores'][split][phase]['safe_success']['phase_time_baseline']['rmse']
                assert abs(stats[phase]['predictors']['phase_time']['rmse'] - expected) < 1e-4
            gs = gradient_summary(gradients, names)
            for group in gs.values():
                group['within_phase_mean_norm'] = {p: group['phase_norm'][p] * int(data['active'].sum()) / stats[p]['target']['count'] for p in PHASES}
            row = {'statistics': stats, 'gradients': gs, 'global_loss': sum(losses), 'prediction_parity_max_abs': max_error}
            vectors_path = args.out / (snapshot + '_' + split + '_gradients.npz')
            np.savez_compressed(vectors_path, **{p: torch.cat([g.flatten() for g in gradients[i]]).cpu().numpy() for i, p in enumerate(PHASES)})
            row['gradient_vectors'] = {'path': str(vectors_path.relative_to(ROOT)), 'sha256': sha(vectors_path),
                'parameter_layout': [{'name': n, 'shape': list(p.shape), 'numel': p.numel()} for n, p in model.named_parameters()]}
            report['snapshots'][snapshot][split] = row
            save(args.out / 'partial_report.json', report)
            del gradients
        if snapshot == 'context_fitted':
            rng = np.random.default_rng(2026099301)
            batches = [rng.choice(len(banks['train']['target']), 64, replace=False) for _ in range(100)]
            batch_hash = hashlib.sha256(b''.join(b.tobytes() for b in batches)).hexdigest()
            original_report = json.loads((CONTEXT / 'report.json').read_text())
            assert batch_hash == original_report['batch_indices_sha256']
            report['minibatches_at_fixed_final_checkpoint'] = []
            for index, take in enumerate(batches[:16]):
                data = {k: v[take] for k, v in banks['train'].items()}
                central = {k: v[take] for k, v in centers['train'].items()}
                gradients, losses, _ = phase_gradients(model, central, data, norm, args.device)
                row = {'original_batch_index': index + 1, 'global_loss': sum(losses), 'gradients': gradient_summary(gradients, names)}
                if index == 0:
                    direct = direct_gradient(model, central, data, norm, args.device)
                    diff = sum(float((sum(g[i] for g in gradients) - d.double()).square().sum()) for i, d in enumerate(direct)) ** .5
                    magnitude = sum(float(d.double().square().sum()) for d in direct) ** .5
                    row['phase_sum_vs_direct_relative_l2'] = diff / max(magnitude, 1e-30)
                    assert row['phase_sum_vs_direct_relative_l2'] < 1e-4
                report['minibatches_at_fixed_final_checkpoint'].append(row)
                del gradients
            report['original_100_batch_indices_sha256'] = batch_hash
        assert state_sha(model) == before
        report['snapshots'][snapshot]['state_unchanged_sha256'] = before
        del model
    assert sha(checkpoint) == CHECKPOINT_SHA
    for path, expected in sources.items():
        assert sha(ROOT / path) == expected
    report['all_source_files_unchanged'] = True
    report['elapsed_seconds'] = time.monotonic() - start
    save(args.out / 'report.json', report)
    save(args.out / 'progress.json', {'status': 'complete', 'actor_updates': 0, 'critic_updates': 0,
                                     'elapsed_seconds': report['elapsed_seconds'], 'pid': os.getpid()})
    print('COMPLETE', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
