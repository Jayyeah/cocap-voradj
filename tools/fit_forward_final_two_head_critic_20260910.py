#!/usr/bin/env python3
"""One same-bank/init/batches/Adam/100-update two-head value causal control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from tools import audit_forward_final_critic_gradients_20260909 as audit
from tools.train_forward_final_ppo_20260909 import make_trainer, tensor_hash, BC_SHA
from cocap_voradj.training.forward_final_value_context import augment, expand_value_input, SCHEMA, NAMES
from cocap_voradj.training.forward_final import preflight
from cocap_voradj.training.small_step_ac import tensor_tree
from cocap_voradj.models.forward_final_two_head_value import TwoHeadCentralValue


def score_groups(data, predictions):
    result = audit.prediction_stats(data, predictions)
    bins = audit.time_bins(data)
    for phase, name in enumerate(audit.PHASES):
        mask = data['active'] & ((data['phase'] == phase) & (bins == 0))[:, None]
        result[name]['first_50_steps'] = {
            'rows': int(mask.sum()), 'target': audit.distribution(data['target'][mask]),
            'predictors': {label: {**audit.calibration(data['target'][mask].astype(float), p[mask].astype(float)),
                                  'residual_median': float(np.median(p[mask] - data['target'][mask])),
                                  'prediction_mean': float(p[mask].mean())} for label, p in predictions.items()}}
    return result


def gradient_head_routing(gradients, names):
    result = {}
    for group in ('pre', 'recovery'):
        ids = [i for i, n in enumerate(names) if n.startswith('value_head.' + group + '.')]
        result[group] = {phase: float(torch.cat([gradients[j][i].flatten() for i in ids]).norm())
                         for j, phase in enumerate(audit.PHASES)}
    assert result['pre']['post_capture'] == result['pre']['pure_coverage'] == 0.
    assert result['recovery']['pre_capture'] == 0.
    return result


def bootstrap_rmse_delta(data, single, two, phase, repetitions=2000):
    # Episode-resampled paired row-weighted RMSE, not iid agent/transition rows.
    ids = np.unique(data['episode'][data['phase'] == phase])
    counts, old_sse, new_sse = [], [], []
    for e in ids:
        mask = data['active'] & ((data['episode'] == e) & (data['phase'] == phase))[:, None]
        counts.append(mask.sum())
        y = data['target'][mask].astype(float)
        old_sse.append(np.sum((single[mask] - y) ** 2))
        new_sse.append(np.sum((two[mask] - y) ** 2))
    rng = np.random.default_rng(2026091001 + phase)
    take = rng.integers(0, len(ids), size=(repetitions, len(ids)))
    n = np.asarray(counts)[take].sum(1)
    delta = np.sqrt(np.asarray(new_sse)[take].sum(1) / n) - np.sqrt(np.asarray(old_sse)[take].sum(1) / n)
    return {'episodes': len(ids), 'rmse_two_minus_single_ci95': np.percentile(delta, [2.5, 97.5]).tolist(),
            'limits': 'Paired episode bootstrap on a small fixed bank; does not measure variation across training seeds.'}


def run(out, device):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(1)
    runtime = preflight()
    audit.save(out / 'runtime_preflight.json', runtime)
    checkpoint = audit.CONTEXT / 'context_critic_100.pt'
    assert audit.sha(checkpoint) == audit.CHECKPOINT_SHA
    reference = torch.load(checkpoint, map_location='cpu', weights_only=False)
    old_report = json.loads((audit.CONTEXT / 'report.json').read_text())
    old_audit_path = ROOT / 'artifacts/2026-09-09_root_cause/critic_gradient_audit_v2/report.json'
    old_audit = json.loads(old_audit_path.read_text())
    assert reference['schema'] == SCHEMA and list(reference['names']) == list(NAMES)
    trainer = make_trainer(device, 2026097101)
    trainer.actor.eval()
    for p in trainer.actor.parameters():
        p.requires_grad_(False)
    actor_sha = tensor_hash(trainer.actor.state_dict())
    assert actor_sha == reference['actor_sha256']
    single_initial = expand_value_input(trainer.value).eval()
    initial_sha = audit.state_sha(single_initial)
    assert initial_sha == old_audit['snapshots']['context_initial']['state_unchanged_sha256']
    model = TwoHeadCentralValue(single_initial).to(device)
    assert audit.state_sha(model.shared) == initial_sha
    assert audit.state_sha(model.shared.value_head) == audit.state_sha(model.recovery_head)
    assert not set(map(id, model.shared.value_head.parameters())) & set(map(id, model.recovery_head.parameters()))
    banks, centers, fingerprints, pools, sources = {}, {}, {}, {}, {}
    for path in (checkpoint, old_audit_path, Path(__file__), ROOT / 'src/cocap_voradj/models/forward_final_two_head_value.py',
                 ROOT / 'src/cocap_voradj/models/small_step_ac.py', ROOT / 'src/cocap_voradj/training/forward_final_value_context.py',
                 ROOT / 'tools/audit_forward_final_critic_gradients_20260909.py'):
        sources[str(path.relative_to(ROOT))] = audit.sha(path)
    for split in ('train', 'heldout'):
        bank_path = audit.BASE / split / 'fixed_bank.npz'
        context_path = audit.CONTEXT / split / 'context.npz'
        manifest_path = audit.CONTEXT / split / 'context_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        data = dict(np.load(bank_path))
        context = np.load(context_path)['context']
        assert audit.sha(bank_path) == manifest['old_bank_sha256']
        assert audit.sha(context_path) == manifest['context_sha256']
        assert manifest['all_geometry_target_phase_episode_bit_exact']
        np.testing.assert_array_equal(data['active'], data['global_active_mask'])
        np.testing.assert_array_equal(context[:, :3], np.eye(3, dtype=np.float32)[data['phase']])
        central = augment({k[7:]: v for k, v in data.items() if k.startswith('global_')}, context)
        central['phase'] = data['phase']
        banks[split], centers[split] = data, central
        fingerprints[split] = {r['initial_fingerprint'] for r in manifest['lineage']}
        pools[split] = {h for r in manifest['lineage'] for h in r['pool_snapshot_hashes']}
        for path in (bank_path, context_path, manifest_path, audit.CONTEXT / split / 'predictions.npz'):
            sources[str(path.relative_to(ROOT))] = audit.sha(path)
    assert not fingerprints['train'] & fingerprints['heldout']
    assert not pools['train'] & pools['heldout']
    trainer.value_normalizer.update(torch.as_tensor(banks['train']['target'], device=device),
                                    torch.as_tensor(banks['train']['active'], device=device))
    norm = trainer.value_normalizer
    normalization = (float(norm.mean), float(norm.std))
    assert normalization == tuple(reference['normalization'])
    optimizer_config = {k: v for k, v in reference['optimizer']['param_groups'][0].items() if k != 'params'}
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    assert not optimizer.state
    assert optimizer_config['lr'] == 1e-4 and optimizer_config['eps'] == 1e-5
    assert trainer.config.max_grad_norm == .5 and trainer.config.value_coef == 1.
    rng = np.random.default_rng(2026099301)
    batches = [rng.choice(len(banks['train']['target']), 64, replace=False) for _ in range(100)]
    batch_hash = hashlib.sha256(b''.join(b.tobytes() for b in batches)).hexdigest()
    assert batch_hash == old_report['batch_indices_sha256']
    launch = {'pid': os.getpid(), 'gpu_visible': os.environ.get('CUDA_VISIBLE_DEVICES'), 'schema': 'forward-final-two-head-value-v1',
        'base_commit': 'dad17b4', 'actor_parent_file_sha256': BC_SHA, 'sources': sources,
        'initial_single_state_sha256': initial_sha, 'batch_indices_sha256': batch_hash,
        'optimizer': optimizer_config, 'optimizer_initial_state': 'empty, as original 100-update run',
        'normalization': normalization, 'loss': 'same global active mean normalized 0.5 MSE; value_coef=1; max_grad_norm=.5',
        'updates': 100, 'batch': 64, 'actor_updates': 0, 'new_rollouts': 0,
        'trunk': 'same class/forward/initial parameters; trainable exactly as single-head baseline',
        'routing': 'true phase 0 -> original copied pre head; phases 1/2 -> identical cloned recovery head',
        'only_intervention': 'independent pre/recovery value-head parameters and hard phase routing; trunk executes twice through unchanged original forward',
        'screen_not_population_significance': {'meaningful_recovery_rmse_reduction': .05, 'pre_rmse_tolerance': .02,
            'scope': 'full phase and early50 train/heldout; report raw values and paired episode uncertainty; no automatic PPO'},
        'gates': {'p3': 'HOLD', 'ppo_25k': 'HOLD', 'continuous': 'HOLD'}}
    audit.save(out / 'launch.json', launch)
    # Verify every bank state starts with the same function, not just a few samples.
    initial_error = 0.
    with torch.no_grad():
        for split in ('train', 'heldout'):
            for offset in range(0, len(banks[split]['phase']), 64):
                b = tensor_tree({k: v[offset:offset + 64] for k, v in centers[split].items()}, device)
                initial_error = max(initial_error, float((model(b) - single_initial(b)).abs().max()))
    assert initial_error < 1e-6
    random.seed(2026099301)
    np.random.seed(2026099301)
    torch.manual_seed(2026099301)
    losses, grad_norms = [], []
    for update, take in enumerate(batches, 1):
        batch = tensor_tree({k: v[take] for k, v in centers['train'].items()}, device)
        active = torch.as_tensor(banks['train']['active'][take], device=device)
        target = norm.normalize(torch.as_tensor(banks['train']['target'][take], device=device))
        loss = .5 * (model(batch)[active] - target[active]).square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
        grad_norms.append(float(grad_norm))
        if update % 10 == 0:
            audit.save(out / 'progress.json', {'status': 'fitting_two_head', 'update': update, 'total_updates': 100,
                'elapsed_seconds': time.monotonic() - start, 'pid': os.getpid()})
            print('update', update, 'loss', losses[-1], flush=True)
    fitting_seconds = time.monotonic() - start
    assert actor_sha == tensor_hash(trainer.actor.state_dict())
    assert all(p.grad is None for p in trainer.actor.parameters())
    assert normalization == (float(norm.mean), float(norm.std))
    assert all(int(state['step']) == 100 for state in optimizer.state.values())
    state_before_audit = audit.state_sha(model)
    ckpt = out / 'two_head_critic_100.pt'
    torch.save({'value': model.state_dict(), 'optimizer': optimizer.state_dict(), 'normalizer': norm.state_dict(),
        'schema': launch['schema'], 'context_schema': SCHEMA, 'context_names': NAMES, 'normalization': normalization,
        'actor_sha256': actor_sha, 'batch_indices': batches, 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, 'updates': 100, 'launch': launch}, ckpt)
    tables = audit.baseline_tables(banks['train'])
    result = {'classification': 'EXPERIMENT RESULT', 'launch': launch, 'initial_all_bank_max_abs_error': initial_error,
        'actor_bit_exact': True, 'all_source_pools_and_fingerprints_disjoint': True, 'critic_updates': 100,
        'checkpoint_sha256': audit.sha(ckpt), 'fitting_including_preflight_seconds': fitting_seconds,
        'losses': losses, 'preclip_gradient_norms': grad_norms, 'scores': {}, 'gradients': {}, 'paired_bootstrap': {}}
    names = model.audit_parameter_names()
    for split in ('train', 'heldout'):
        def progress(phase):
            audit.save(out / 'progress.json', {'status': 'post_fit_gradient_audit', 'update': 100, 'split': split,
                'phase_complete': audit.PHASES[phase], 'elapsed_seconds': time.monotonic() - start, 'pid': os.getpid()})
            print('gradients', split, audit.PHASES[phase], flush=True)
        gradients, phase_losses, fitted = audit.phase_gradients(model, centers[split], banks[split], norm, device, progress=progress)
        single = np.load(audit.CONTEXT / split / 'predictions.npz')['context_fitted']
        predictions = {'two_head': fitted, 'single_head': single, **audit.baseline_predictions(banks[split], tables)}
        np.savez_compressed(out / (split + '_predictions.npz'), **predictions)
        result['scores'][split] = score_groups(banks[split], predictions)
        for phase, name in enumerate(audit.PHASES):
            result['scores'][split][name]['global_loss_share'] = phase_losses[phase] / sum(phase_losses)
            expected = old_audit['snapshots']['context_fitted'][split]['statistics'][name]['predictors']['context']['rmse']
            assert abs(result['scores'][split][name]['predictors']['single_head']['rmse'] - expected) < 1e-5
        gs = audit.gradient_summary(gradients, names)
        routing = gradient_head_routing(gradients, names)
        path = out / (split + '_gradients.npz')
        np.savez_compressed(path, **{name: torch.cat([g.flatten() for g in gradients[i]]).cpu().numpy() for i, name in enumerate(audit.PHASES)})
        result['gradients'][split] = {'two_head': gs, 'single_head': old_audit['snapshots']['context_fitted'][split]['gradients'],
            'head_routing_norms': routing, 'phase_loss_global_contribution': phase_losses,
            'vectors_path': str(path.relative_to(ROOT)), 'vectors_sha256': audit.sha(path),
            'layout': [{'name': n, 'shape': list(p.shape), 'numel': p.numel()} for n, p in zip(names, model.parameters())]}
        result['paired_bootstrap'][split] = {name: bootstrap_rmse_delta(banks[split], single, fitted, phase)
            for phase, name in enumerate(audit.PHASES)}
        del gradients
        audit.save(out / 'partial_report.json', result)
    result['minibatches_at_fixed_final_checkpoint'] = []
    for index, take in enumerate(batches[:16]):
        data = {k: v[take] for k, v in banks['train'].items()}
        central = {k: v[take] for k, v in centers['train'].items()}
        gradients, phase_losses, _ = audit.phase_gradients(model, central, data, norm, device)
        row = {'original_batch_index': index + 1, 'gradients': audit.gradient_summary(gradients, names),
               'head_routing_norms': gradient_head_routing(gradients, names)}
        if index == 0:
            direct = audit.direct_gradient(model, central, data, norm, device)
            diff = sum(float((sum(g[i] for g in gradients) - d.double()).square().sum()) for i, d in enumerate(direct)) ** .5
            mag = sum(float(d.double().square().sum()) for d in direct) ** .5
            row['phase_sum_vs_direct_relative_l2'] = diff / max(mag, 1e-30)
            assert row['phase_sum_vs_direct_relative_l2'] < 1e-4
        result['minibatches_at_fixed_final_checkpoint'].append(row)
        del gradients
    assert audit.state_sha(model) == state_before_audit
    for path, expected in sources.items():
        assert audit.sha(ROOT / path) == expected
    result['model_unchanged_during_gradient_audit'] = True
    result['source_files_unchanged'] = True
    result['checks'] = {}
    for split in ('train', 'heldout'):
        for phase in audit.PHASES:
            r = result['scores'][split][phase]
            ratio = r['predictors']['two_head']['rmse'] / r['predictors']['single_head']['rmse']
            early_ratio = r['first_50_steps']['predictors']['two_head']['rmse'] / r['first_50_steps']['predictors']['single_head']['rmse']
            result['checks'][split + '/' + phase] = {'rmse_ratio': ratio, 'early50_rmse_ratio': early_ratio,
                'pre_preserved_or_recovery_improved': ratio <= (1.02 if phase == 'pre_capture' else .95),
                'early_recovery_improved': None if phase == 'pre_capture' else early_ratio <= .95,
                'beats_phase_time_baseline': r['predictors']['two_head']['rmse'] <= r['predictors']['phase_time']['rmse']}
    result['elapsed_seconds'] = time.monotonic() - start
    result['decision'] = 'COMPLETE_REQUIRES_LEDGER_CLASSIFICATION_NO_PPO'
    audit.save(out / 'report.json', result)
    audit.save(out / 'progress.json', {'status': 'complete', 'update': 100, 'actor_updates': 0,
        'elapsed_seconds': result['elapsed_seconds'], 'eta_seconds': 0, 'pid': os.getpid()})
    (out / 'partial_report.json').unlink(missing_ok=True)
    print('COMPLETE', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        run(args.out, args.device)
    except Exception:
        launch_path = args.out / 'launch.json'
        if launch_path.exists() and json.loads(launch_path.read_text()).get('pid') == os.getpid():
            audit.save(args.out / 'failure_audit.json', {'traceback': traceback.format_exc(), 'pid': os.getpid()})
            audit.save(args.out / 'progress.json', {'status': 'failed', 'pid': os.getpid()})
        raise
