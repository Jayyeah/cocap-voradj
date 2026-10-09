"""Startup assertions and reviewable runtime/config/delta manifests."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path

import torch

from .contracts import ROOT, environment_versions, fingerprint, declared_sources, git
from .source_guard import SourceGuard
from .evaluation import METRIC_DEFINITIONS, seed_manifest, scene_label, protocol_for_line
from ..native import NativeStage1
from ..model import TERLActor


def active_values(runtime):
    t, a, c = runtime.trainer, runtime.adapter, runtime.config
    if t.actor.training or t.value.training:
        raise ValueError('PPO actor/critic must stay in eval mode with autograd available')
    if asdict(t.config) != asdict(type(t.config)(**c['ppo'])):
        raise ValueError('PPO requested/runtime mismatch')
    for optimizer, lr in ((t.actor_optimizer, c['ppo']['actor_lr']), (t.value_optimizer, c['ppo']['critic_lr'])):
        if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]['lr'] != lr or optimizer.param_groups[0]['eps'] != 1e-5:
            raise ValueError('actor/critic optimizer contract mismatch')
    actor_ids = {id(p) for group in t.actor_optimizer.param_groups for p in group['params']}
    critic_ids = {id(p) for group in t.value_optimizer.param_groups for p in group['params']}
    if actor_ids != {id(p) for p in t.actor.parameters()} or critic_ids != {id(p) for p in t.value.parameters()} or actor_ids & critic_ids:
        raise ValueError('optimizer parameter isolation mismatch')
    if (t.value_norm is not None) != c['ppo']['value_norm']:
        raise ValueError('ValueNorm requested/runtime mismatch')
    if isinstance(a, NativeStage1):
        a.assert_contract()
    architecture = {}
    if hasattr(t.actor, 'transformer_encoder'):
        layers = t.actor.transformer_encoder.layers
        architecture['actor'] = {'class': type(t.actor).__qualname__, 'layers': len(layers),
                                 'heads': layers[0].self_attn.num_heads, 'hidden_dim': layers[0].self_attn.embed_dim}
        if runtime.hooks.actor_factory is TERLActor and (
            len(layers), layers[0].self_attn.num_heads, layers[0].self_attn.embed_dim) != (
                c['num_layers'], c['num_heads'], c['hidden_dim']):
            raise ValueError('actor requested/runtime architecture mismatch')
    if hasattr(t.value, 'attention'):
        layers = t.value.attention.layers
        architecture['critic'] = {'class': type(t.value).__qualname__, 'layers': len(layers),
                                  'heads': layers[0].self_attn.num_heads, 'hidden_dim': layers[0].self_attn.embed_dim,
                                  'self_feature_dim': t.value.self_feature_dim, 'max_agents': t.value.max_agents,
                                  'max_evaders': t.value.max_evaders, 'max_obstacles': t.value.max_obstacles}
    return {'environment': a.contract(), 'scene': scene_label(a.env),
            'action_backend': asdict(runtime.hooks.backend), 'ppo': asdict(t.config), 'architecture': architecture,
            'actor_mode': 'eval', 'critic_mode': 'eval',
            'actor_parameters': sum(p.numel() for p in t.actor.parameters()),
            'critic_parameters': sum(p.numel() for p in t.value.parameters()),
            'actor_optimizer': {'class': type(t.actor_optimizer).__name__, 'lr': c['ppo']['actor_lr'], 'eps': 1e-5},
            'critic_optimizer': {'class': type(t.value_optimizer).__name__, 'lr': c['ppo']['critic_lr'], 'eps': 1e-5},
            'value_norm_state': None if t.value_norm is None else
                {'beta': t.value_norm.beta, 'epsilon': t.value_norm.epsilon},
            'device': str(t.device)}


def assert_loaded_sources_covered(declared_sources):
    SourceGuard(declared_sources).check()


def runtime_manifest(lock, delta, resolved, runtime, resources):
    sources = declared_sources(lock, delta)
    if sources != runtime.source_guard.expected:
        runtime.source_guard.fail('runtime source guard differs from pinned BASE/delta')
    runtime.source_guard.check()
    active = active_values(runtime)
    if resources['device'] == 'cuda:0' and torch.cuda.max_memory_allocated() / 1024**2 > resources['max_gpu_allocated_mib']:
        raise ValueError('GPU allocated memory exceeds declared limit')
    if active['value_norm_state'] is not None and active['value_norm_state'] != {
        'beta': resolved['ppo']['value_norm_beta'], 'epsilon': resolved['ppo']['value_norm_epsilon']}:
        raise ValueError('ValueNorm active settings mismatch')
    raw_env = active['environment']
    reward = {k: v for k, v in raw_env.items() if any(s in k for s in ('reward', 'penalty', 'decay'))}
    action = {'backend': active['action_backend'],
              'physics': {k: v for k, v in raw_env.items() if k in
                          {'dt', 'integration_substeps', 'aw9', 'pursuer_max_speed', 'evader_max_speed'}}}
    return {'schema': 'terl.batch01.runtime.v1', 'status': 'OFFLINE_PREPARED_NO_TRAINING',
            'base': delta['base'], 'line': delta['line'], 'resolved_config': resolved,
            'delta_manifest': delta, 'common_source_hashes': sources,
            'generated_runtime_source_hashes': dict(runtime.source_guard.generated_sources),
            'seed_manifest': seed_manifest(resolved, protocol_for_line(lock, delta['line'])),
            'evaluation_protocol': protocol_for_line(lock, delta['line']), 'metric_definitions': METRIC_DEFINITIONS,
            'active_runtime_values': active, 'environment_fingerprint': fingerprint(raw_env),
            'action_fingerprint': fingerprint(action),
            'reward_fingerprint': fingerprint({'native': reward, 'extension': delta.get('extensions', {}),
                                               'source_delta': delta.get('source_changes', [])}),
            'initial_state_fingerprint': runtime.adapter.fingerprint(),
            'resources': resources, 'versions': environment_versions(),
            'pid': os.getpid(), 'source_git_head': git('rev-parse', 'HEAD'),
            'counters': {'steps': 'joint environment decisions', 'agent_transitions': 'active actor rows',
                         'updates': 'on-policy rollout calls', 'optimizer_steps': 'paired actor/critic minibatches'},
            'checkpoint': None, 'qa_status': 'PENDING_INDEPENDENT_QA'}
