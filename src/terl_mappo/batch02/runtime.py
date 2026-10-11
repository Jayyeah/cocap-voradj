"""Minimal CoCap rollout wiring; PPO and existing network forwards unchanged.

Input dimension/capacity binding is an explicit MAPPO adaptation, authorized
by the user with CoCap taking precedence over TERL when contracts conflict.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from terl_mappo.model import TERLActor
from terl_mappo.batch01.interfaces import multiscale_state
from terl_mappo.run import rng_state, restore_rng, file_hash
from cocap_voradj.models.small_step_ac import CentralValueNetwork
from cocap_voradj.training.small_step_ac import MAPPOConfig, MAPPOTrainer, tensor_tree
from .cocap_env import CoCapAdapter, ROOT, stable_hash


def central_state(env):
    """Existing action-free TERL geometry schema, bound to actual CoCap counts.

    CoCap has no vortex field: remove those absent current columns rather than
    inventing currents. This training-only V input never enters actor tokens.
    Physical scales and the remaining nine focal features stay as registered.
    """
    view = SimpleNamespace(pursuers=env.pursuers, evaders=env.evaders, obstacles=env.obstacles,
                           cores=(), episode_time_steps=env.episode_step)
    return multiscale_state(view, max_cores=0, map_scale=env.width,
                           horizon_scale=env.episode_max_length)


def build(config, adapter, device='cpu'):
    seed = config['actor_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    h, heads, layers = (config[key] for key in ('hidden_dim', 'num_heads', 'num_layers'))
    actor = TERLActor(h, heads, layers, seed, config['masked_pool'])
    self_dim = adapter._template['self'].shape[-1]
    actor.entity_encoders['self'][0] = nn.Linear(self_dim, h)
    actor.feature_dims = {**actor.feature_dims, 'self': self_dim}
    value = CentralValueNetwork(h, heads, layers, self_feature_dim=9,
                               max_agents=len(adapter.env.pursuers),
                               max_evaders=max(1, len(adapter.env.evaders)),
                               max_obstacles=max(1, len(adapter.env.obstacles))).eval()
    return MAPPOTrainer(actor, value, MAPPOConfig(**config['ppo']), device)


def collect(trainer, adapter, length):
    """Store physical pre-reset next V and unmodified task reward for GAE."""
    if length <= 0:
        raise ValueError('positive rollout length required')
    rows, episodes, trace = [], [], []
    for _ in range(length):
        local = adapter.packed_observations()
        central = central_state(adapter.env)
        actions, logp, latent, value = trainer.act(local, {key: v[None] for key, v in central.items()})
        _, reward, terminated, truncated, end, info = adapter.step(actions)
        next_central = central_state(adapter.env)
        with torch.no_grad():
            next_value = trainer.value(tensor_tree({key: v[None] for key, v in next_central.items()},
                                                 trainer.device)).cpu().numpy()[0]
        rows.append({'local_obs': local, 'global_obs': central, 'actions': actions,
                     'latent': latent, 'log_prob': logp, 'values': value[0],
                     'next_values': next_value, 'rewards': reward,
                     'terminated': terminated, 'truncated': truncated,
                     'episode_end': np.full(len(reward), end), 'active_mask': info['active']})
        trace.append({'actions': actions.tolist(), 'reward': reward.tolist(),
                      'terminated': terminated.tolist(), 'truncated': truncated.tolist(),
                      'end': end, 'evader_actions': info['evader_actions'],
                      'fingerprint_before_reset': adapter.fingerprint(),
                      'task_infos': info['task_infos']})
        if end:
            episodes.append(copy.deepcopy(info))
            adapter.reset()
    result = {key: {field: np.stack([row[key][field] for row in rows]) for field in rows[0][key]}
              if isinstance(rows[0][key], dict) else np.stack([row[key] for row in rows]) for key in rows[0]}
    return result, episodes, trace


def sources():
    paths = list((ROOT / 'src/terl_mappo/batch02').glob('*.py'))
    paths += [ROOT / rel for rel in ('src/terl_mappo/model.py', 'src/terl_mappo/run.py',
              'src/terl_mappo/native.py', 'src/terl_mappo/batch01/interfaces.py',
              'vendor/terl/policy/TERL_model.py', 'src/cocap_voradj/models/small_step_ac.py',
              'src/cocap_voradj/training/small_step_ac.py')]
    # Pin actual provider/control/reward/observation consumers, not just a YAML.
    paths += list((ROOT / 'src/cocap_voradj/envs').glob('*.py'))
    paths += [ROOT / rel for rel in ('src/cocap_voradj/control/apf.py',
              'src/cocap_voradj/dynamics/robot.py', 'src/cocap_voradj/training/trainer.py',
              'src/cocap_voradj/training/forward_final_v2.py')]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(set(paths))}


def save_checkpoint(path, trainer, adapter, config, decisions, agent_transitions, optimizer_steps):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema': 'terl_mappo.batch02.e1.v1', 'trainer': trainer.state_dict(),
               'runtime': adapter.state_dict(), 'rng': rng_state(), 'config': copy.deepcopy(config),
               'sources': sources(), 'contract': adapter.contract(), 'decisions': decisions,
               'agent_transitions': agent_transitions, 'optimizer_steps': optimizer_steps}
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary); temporary.replace(path)
    return file_hash(path)


def load_checkpoint(path, trainer, adapter, config):
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload['schema'] != 'terl_mappo.batch02.e1.v1' or payload['config'] != config:
        raise ValueError('E1 checkpoint config mismatch')
    # Resolved sensing radius depends on the saved obstacle map. A fresh
    # constructor's map is not the resumed map; compare immutable bindings
    # before loading, then verify derived runtime metadata after restoration.
    binding = lambda contract: {key: value for key, value in contract.items() if key != 'sensing'}
    if payload['sources'] != sources() or binding(payload['contract']) != binding(adapter.contract()):
        raise ValueError('E1 checkpoint provider/source mismatch')
    trainer.load_state_dict(payload['trainer'])
    adapter.load_state_dict(payload['runtime'])
    if payload['contract'] != adapter.contract():
        raise ValueError('E1 restored runtime contract mismatch')
    restore_rng(payload['rng'], trainer.device)
    return payload
