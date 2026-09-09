#!/usr/bin/env python3
"""Fixed-BC MC critic fit; independent reset pools for train and heldout, no PPO."""
import argparse, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from tools.train_forward_final_ppo_20260909 import make_trainer, FinalMissionStream, collect_transition, tensor_hash, BC_SHA
from tools.evaluate_forward_final_critic_20260909 import full_returns, calibration
from tools.run_forward_final_bridge_20260908 import atomic_json
from tools.collect_iqn_aw_teacher_dataset_20260903 import sha256_file
from cocap_voradj.training.forward_final import preflight
from cocap_voradj.training.small_step_ac import tensor_tree
PHASES = ('pre_capture', 'post_capture', 'pure_coverage')


def predict(trainer, central):
    values = []
    with torch.no_grad():
        for offset in range(0, len(central['active_mask']), 64):
            batch = tensor_tree({k: v[offset:offset+64] for k, v in central.items()}, trainer.device)
            values.append(trainer.value_for_gae(batch).cpu().numpy())
    return np.concatenate(values)


def score(data, prediction):
    result = {}
    for index, phase in enumerate(PHASES):
        mask = data['active'] & (data['phase'][:, None] == index)
        if not mask.any():
            result[phase] = {'rows': 0, 'episodes': 0}
            continue
        result[phase] = dict(calibration(data['target'][mask], prediction[mask]), rows=int(mask.sum()),
                             episodes=len(np.unique(data['episode'][np.any(mask, axis=1)])))
    return result


def collect(trainer, seed, count, out, progress):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    stream = FinalMissionStream(seed, out)
    central_rows, targets, masks, phases, ids, reports = [], [], [], [], [], []
    for episode in range(count):
        rows, phase = [], []
        while True:
            phase.append(2 if stream.task == 'voradj_coverage' else
                         0 if any(not e.deactivated for e in stream.env.evaders) else 1)
            row, done = collect_transition(trainer, stream); rows.append(row)
            if done is not None: break
        excluded = bool(np.stack([r['truncated'] for r in rows]).any())
        reports.append(dict(done, length=len(rows), truncated_excluded=excluded))
        if not excluded:
            central_rows.extend(r['global_obs'] for r in rows)
            targets.append(full_returns(np.stack([r['rewards'] for r in rows]),
                                        np.stack([r['terminated'] for r in rows])))
            masks.extend(r['active_mask'] for r in rows)
            phases.extend(phase); ids.extend([episode] * len(rows))
        progress(episode + 1)
    if not central_rows: raise ValueError('No complete trajectory targets')
    central = {k: np.stack([r[k] for r in central_rows]) for k in central_rows[0]}
    data = dict(target=np.concatenate(targets).astype(np.float32), active=np.asarray(masks, bool),
                phase=np.asarray(phases), episode=np.asarray(ids))
    np.savez_compressed(out / 'fixed_bank.npz', **{'global_' + k: v for k, v in central.items()}, **data)
    atomic_json(out / 'bank_manifest.json', dict(seed=seed, episodes=reports,
        sha256=sha256_file(out / 'fixed_bank.npz'), steps=len(phases), active_rows=int(data['active'].sum()),
        reset_pool_scope='only this split; no train/heldout pool sharing', reward_ce_speed=.0005))
    return central, data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--train-episodes', type=int, default=30)
    p.add_argument('--heldout-episodes', type=int, default=10)
    args = p.parse_args(); out = args.output_root
    if (out / 'launch.json').exists(): raise ValueError('Fresh output required')
    assert 0 < args.updates <= 100
    assert args.train_episodes > 0 and args.heldout_episodes > 0
    seeds = dict(train=2026099201, heldout=2027099201)
    atomic_json(out / 'runtime_preflight.json', preflight())
    trainer = make_trainer(args.device, 2026097101)
    trainer.actor.eval(); trainer.value.eval()
    for p in trainer.actor.parameters(): p.requires_grad_(False)
    actor_sha = tensor_hash(trainer.actor.state_dict())
    atomic_json(out / 'launch.json', dict(bc_parent_sha256=BC_SHA, seeds=seeds,
        episodes=dict(train=args.train_episodes, heldout=args.heldout_episodes), updates=args.updates,
        source_sha256=sha256_file(Path(__file__)), gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        critic_initial_sha256=tensor_hash(trainer.value.state_dict()), target='full MC gamma .99, true terminal; exclude truncation',
        normalization='one train-only ValueNorm update, then frozen', actor_updates=0,
        gate='post/pure in both splits: >=3 episodes, RMSE <= .75 cold, within-phase EV > 0; diagnostic only, no automatic PPO',
        comparison_limit='not target-only A/B against prior bootstrap warm-up: fixed data and optimizer schedule also differ'))
    start = time.monotonic(); banks = {}; cold = {}
    for split, count in [('train', args.train_episodes), ('heldout', args.heldout_episodes)]:
        def progress(n):
            atomic_json(out / 'progress.json', dict(status='collecting', split=split, completed=n,
                total=count, elapsed_seconds=time.monotonic()-start))
        central, data = collect(trainer, seeds[split], count, out / split, progress)
        banks[split] = central, data
        cold[split] = predict(trainer, central)
    central, data = banks['train']
    trainer.value_normalizer.update(torch.as_tensor(data['target'], device=trainer.device),
                                    torch.as_tensor(data['active'], device=trainer.device))
    norm = (float(trainer.value_normalizer.mean), float(trainer.value_normalizer.std))
    random.seed(2026099301); np.random.seed(2026099301); torch.manual_seed(2026099301)
    rng = np.random.default_rng(2026099301); losses = []
    for update in range(1, args.updates + 1):
        take = rng.choice(len(data['target']), size=min(64, len(data['target'])), replace=False)
        batch = tensor_tree({k: v[take] for k, v in central.items()}, trainer.device)
        active = torch.as_tensor(data['active'][take], device=trainer.device)
        target = trainer.value_normalizer.normalize(torch.as_tensor(data['target'][take], device=trainer.device))
        loss = .5 * (trainer.value(batch)[active] - target[active]).square().mean()
        trainer.value_optimizer.zero_grad(set_to_none=True)
        (trainer.config.value_coef * loss).backward()
        torch.nn.utils.clip_grad_norm_(trainer.value.parameters(), trainer.config.max_grad_norm)
        trainer.value_optimizer.step(); losses.append(float(loss.detach()))
        assert np.isfinite(losses[-1])
        if update % 10 == 0:
            atomic_json(out / 'progress.json', dict(status='fitting', update=update, total=args.updates,
                value_loss=losses[-1], elapsed_seconds=time.monotonic()-start))
    assert norm == (float(trainer.value_normalizer.mean), float(trainer.value_normalizer.std))
    assert actor_sha == tensor_hash(trainer.actor.state_dict())
    report = {}; checks = {}
    for split, (central, data) in banks.items():
        fitted = predict(trainer, central)
        report[split] = dict(cold=score(data, cold[split]), fitted=score(data, fitted))
        episodes = json.loads((out / split / 'bank_manifest.json').read_text())['episodes']
        safe_ids = [e['episode'] for e in episodes if e['ce_success'] and not e['collision']
                    and (e['task'] == 'voradj_coverage' or e['captured'])]
        for label, keep_safe in [('safe_completion', True), ('other_outcome', False)]:
            subset = dict(data, active=data['active'] & (np.isin(data['episode'], safe_ids)[:, None] == keep_safe))
            report[split][label] = dict(cold=score(subset, cold[split]), fitted=score(subset, fitted))
        np.savez_compressed(out / split / 'predictions.npz', cold=cold[split], fitted=fitted)
        for phase in PHASES[1:]:
            a, b = report[split]['cold'][phase], report[split]['fitted'][phase]
            checks[split+'/'+phase] = bool(b['episodes'] >= 3 and b['rmse'] <= .75*a['rmse']
                                           and b['within_phase_ev'] is not None and b['within_phase_ev'] > 0)
    path = out / 'critic_mc_100updates.pt'
    torch.save(dict(trainer=trainer.state_dict(), actor_sha256=actor_sha, normalization=norm,
                    schema='forward-final-fixed-mc-critic-diagnostic-v1'), path)
    decision = 'MC_FIT_SCREEN_PASS_NOT_PPO_PASS' if all(checks.values()) else 'HOLD_MC_FIT_CALIBRATION'
    atomic_json(out / 'report.json', dict(decision=decision, checks=checks, scores=report,
        actor_bit_exact=True, normalization=norm, losses=losses, checkpoint_sha256=sha256_file(path),
        elapsed_seconds=time.monotonic()-start,
        limits='MC is one noisy realized return, not exact state V; correlated rows; safe outcome strata are post-hoc only. No Actor update or PPO advancement.'))
    atomic_json(out / 'progress.json', dict(status='complete', decision=decision, eta_seconds=0))

if __name__ == '__main__': main()
