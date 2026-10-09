"""Shared definitions and seed isolation; metrics retain native T0 semantics."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import random

import numpy as np
import torch

from ..evaluate import summarize, evaluate
from ..run import rng_state
from ..supervise import selection_score

METRIC_DEFINITIONS = {
    'normal_capture': 'all native evaders captured AND no collision at joint end',
    'ring2/ring3': 'full-episode visitation: >=2/>=3 active pursuers within <8m of an evader',
    'strict_geometry': '>=3 near pursuers; max angle gap<=pi and <=3*min gap',
    'collision_type': 'per-episode incidence of native end-of-decision collision categories',
    'capture_time': 'conditional on ANY native capture; decisions * decision_seconds',
    'censored': 'native T0 convention: all noncapture; collision failure reported separately',
    'critic_ev': 'post-update denormalized V vs pre-update GAE raw return targets on active rows; 1-var(return-V)/var(return)',
    'ppo_kl': 'sample estimator mean((ratio-1)-log_ratio); minibatch and post-update separately',
    'ppo_ratio_clip': 'ratio=exp(new-old); clip fraction=mean(abs(ratio-1)>clip_param)',
    'entropy': 'backend-labelled measure; categorical and continuous raw values never pooled',
    'multiscale': 'group by P/E/O/current/map/horizon label; rates and normalized time per scale',
}


def seed_manifest(config, protocol):
    result = {'train_environment': [config['seed']], 'train_actor': [config['actor_seed']],
              'sample_torch_offset': protocol['sample_torch_offset']}
    for name in ('screen', 'selection', 'final'):
        domain = protocol[name]
        result[name] = list(range(domain['seed_base'], domain['seed_base'] + domain['max_episodes_per_mode']))
    seen = set(result['train_environment'] + result['train_actor'])
    for name in ('screen', 'selection', 'final'):
        seeds = set(result[name])
        action_seeds = {s + protocol['sample_torch_offset'] for s in seeds}
        if seeds & seen or action_seeds & seen or seeds & action_seeds:
            raise ValueError('seed domains overlap (including sample action streams)')
        seen |= seeds | action_seeds
    return result


@contextmanager
def isolated_rng():
    """Restore every train RNG even on evaluator exceptions (including CUDA)."""
    state = rng_state()
    try:
        yield
    finally:
        random.setstate(state['python']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if state['cuda']:
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def evaluate_t0(checkpoint, output, config, protocol, domain='screen', step=0, workers=2):
    seed_manifest(config, protocol)
    if domain not in {'screen', 'selection', 'final'} or not 1 <= workers <= 2:
        raise ValueError('invalid evaluator domain/resources')
    setting = protocol[domain]
    n = setting['max_episodes_per_mode']
    if domain == 'screen' and step not in protocol['screen_milestones']:
        n = setting['regular_episodes_per_mode']
    from .contracts import validate_resources
    validate_resources(output, workers=workers)
    with isolated_rng():
        return evaluate(checkpoint, output, n, setting['seed_base'], workers, domain)


def scene_label(env):
    return {'pursuers': len(env.pursuers), 'evaders': len(env.evaders),
            'obstacles': len(env.obstacles), 'cores': len(env.cores),
            'map': [env.width, env.height], 'horizon_decisions': env.episode_max_length + 1,
            'decision_seconds': env.pursuers[0].dt * env.pursuers[0].N}


def summarize_scene(rows, scene, backend):
    if not rows or any(r.get('scene', scene) != scene for r in rows):
        raise ValueError('one nonempty scale per summary; never silently pool scenes')
    result = summarize(rows)
    result.update(scene=copy.deepcopy(scene), action_family=backend.family,
                  entropy_measure=backend.entropy_measure, logprob_measure=backend.logprob_measure,
                  collision_failure_n=sum(not r['capture'] and r['collision'] for r in rows),
                  time_limit_censored_n=sum(not r['capture'] and not r['collision'] for r in rows))
    horizon_seconds = scene['horizon_decisions'] * scene['decision_seconds']
    times = [r['capture_time'] / horizon_seconds for r in rows if r['capture']]
    result['capture_time_fraction_of_horizon_mean'] = float(np.mean(times)) if times else None
    return result


def performance_report(screens, selections, finals, late_points=5):
    """Report best AND last. Final never enters selection or stability sorting."""
    if not screens or late_points <= 0:
        raise ValueError('screen evidence and positive late window required')
    if any(r['seed_domain'] != 'screen' for r in screens) or any(r['seed_domain'] != 'selection' for r in selections):
        raise ValueError('wrong selection domain')
    selected = max(selections, key=selection_score) if selections else None
    if finals and (selected is None or len(finals) != 1 or finals[0]['seed_domain'] != 'final' or
                   (finals[0]['steps'], finals[0]['checkpoint_sha256']) !=
                   (selected['steps'], selected['checkpoint_sha256'])):
        raise ValueError('final must evaluate the held-out selected immutable checkpoint')
    ordered = sorted(screens, key=lambda r: r['steps'])
    late = ordered[-late_points:]
    rates = [sum(m['normal_capture_rate'] for m in r['modes'].values()) / 2 for r in late]
    return {'best_screen': max(screens, key=selection_score), 'last_screen': ordered[-1],
            'selected_heldout': selected, 'selected_final': finals[0] if finals else None,
            'late_stage': {'checkpoint_steps': [r['steps'] for r in late],
                           'normal_rate_mean': float(np.mean(rates)), 'normal_rate_min': min(rates),
                           'normal_rate_max': max(rates), 'normal_rate_std': float(np.std(rates)),
                           'last_minus_best': rates[-1] - max(sum(m['normal_capture_rate'] for m in r['modes'].values()) / 2 for r in screens)},
            'status': 'EVALUATION_COMPLETE' if finals else 'PENDING_FINAL'}
