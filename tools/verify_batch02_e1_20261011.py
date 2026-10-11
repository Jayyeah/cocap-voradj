#!/usr/bin/env python3
"""Finite CPU E1 verification against the original successful IQN provider.

One 256-decision PPO rollout/update, then a 16-decision continuation/update
compared to restoration in a fresh Python process. No long-run entry point.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from terl_mappo.batch02.cocap_env import CoCapAdapter, REFERENCE_RESOLVED_HASH, pack_observations
from terl_mappo.batch02.runtime import build, collect, load_checkpoint, save_checkpoint, sources
from terl_mappo.run import atomic_json, file_hash, rng_state
from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.training.trainer import set_global_config
from cocap_voradj.training.small_step_ac import tensor_tree

IQN = Path('/home/yjq/rl/CoCap1/iqn-z05-independent-20260923')
CENTRAL = Path('/home/yjq/rl/CoCap1/ac-master-dag-20260921')
BASE = 'd6e201052f0eac319342c35cd4efd9171df3fd54'
SEED = 2056101100


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for first, second in zip(a, b): same(first, second)
    else: assert a == b, (a, b)


def digest_graph(value):
    """Canonical numerical state hash, ignoring process-specific addresses."""
    digest = hashlib.sha256(); seen = set()
    def add(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().numpy()
            digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
            digest.update(array.tobytes()); return
        if isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode()); digest.update(str(item.shape).encode())
            digest.update(item.tobytes()); return
        if isinstance(item, np.generic): add(item.item()); return
        if isinstance(item, (str, int, float, bool, type(None))):
            digest.update((type(item).__name__ + ':' + repr(item) + ';').encode()); return
        if id(item) in seen:
            digest.update(b'already-visited;'); return
        seen.add(id(item))
        digest.update((type(item).__module__ + '.' + type(item).__name__).encode())
        if isinstance(item, dict):
            for key in sorted(item, key=repr): add(key); add(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item: add(entry)
        elif isinstance(item, (set, frozenset)):
            for entry in sorted(item, key=repr): add(entry)
        elif isinstance(item, np.random.RandomState): add(item.get_state())
        elif hasattr(item, 'query') and hasattr(item, 'data'):
            add(item.data); add(item.indices); add(item.leafsize); add(item.boxsize)
        elif hasattr(item, '__dict__'): add(vars(item))
        else: raise TypeError('unsupported checkpoint state: ' + str(type(item)))
    add(value)
    return digest.hexdigest()


def robot_state(env):
    return [[r.x, r.y, r.theta, r.speed, *r.velocity, bool(r.deactivated), bool(r.collision)]
            for r in env.pursuers + env.evaders]


def original_oracle():
    path = IQN / 'tools/iqn_token_matched_20260919.py'
    spec = importlib.util.spec_from_file_location('successful_iqn_e1_oracle', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config = module.resolved(IQN / 'configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage1_4p1e1obs_2m.yaml')
    assert module.stable_hash(config) == REFERENCE_RESOLVED_HASH
    return module, config


def make_oracle(module, root, scene, seed):
    env, observations = module.make_env(root, scene, seed)
    return env, observations, [ApfAgent(evader.a, evader.w) for evader in env.evaders]


def oracle_step(env, agents, actions):
    set_global_config(env.config)
    evader_obs = env.get_evader_observations_for_apf()
    evader_actions = [None if evader.deactivated else int(agents[i].act(evader_obs[i]))
                     for i, evader in enumerate(env.evaders)]
    pursuer_actions = [None if p.deactivated else int(actions[i]) for i, p in enumerate(env.pursuers)]
    return env.step(pursuer_actions, evader_actions), evader_actions


def compare_transition(adapter, env, agents, actions):
    before = adapter.env._z_update_count
    native, evader_actions = oracle_step(env, agents, actions)
    local, reward, term, trunc, end, info = adapter.step(np.asarray(actions, dtype=np.int64))
    same(adapter.observations, native.observations)
    same(local, pack_observations(native.observations, adapter._template))
    same(reward, native.rewards); same(info['task_infos'], native.infos)
    same(info['evader_actions'], evader_actions)
    same(term, [row['terminated'] for row in native.infos])
    same(trunc, [row['truncated'] for row in native.infos])
    same(end, all(native.dones)); same(robot_state(adapter.env), robot_state(env))
    same(adapter.env.rng.get_state(), env.rng.get_state())
    same(adapter.env.z_state, env.z_state)
    assert adapter.env._z_update_count == before + 1
    count = adapter.env._z_update_count
    adapter.packed_observations(); adapter.env.get_observations()
    assert adapter.env._z_update_count == count
    return end, {'actions': list(map(int, actions)), 'reward': reward.tolist(),
                 'terminated': term.tolist(), 'truncated': trunc.tolist(), 'end': end,
                 'evader_actions': evader_actions, 'next_state': plain(robot_state(env)),
                 'phase': [row['replay_metadata']['phase'] for row in native.infos],
                 'post_capture_started': bool(env.post_capture_started)}


def inject(env, item):
    for robot, row in zip(env.pursuers + env.evaders, item['robots']):
        robot.x, robot.y, robot.theta, robot.speed = map(float, row[:4])
        robot.velocity = np.asarray(row[4:6], dtype=float)
        robot.deactivated, robot.collision = map(bool, row[6:8])
        robot.start = (robot.x, robot.y)
    env.obstacles[0].x, env.obstacles[0].y, env.obstacles[0].r = item['obstacle']
    env.episode_step = item['episode_step']
    env._invalidate_voronoi_cache(); env._pursuing_flags_initialized = False
    env._pursuing_release_counters = [0] * len(env.pursuers); env._reset_z_state()
    data = env._capture_voronoi_map()
    env.last_task_labels = env._task_labels_from_map(data, update_effective=True)
    env._advance_z_state(data)


def provider_parity(output):
    module, root = original_oracle(); results = []
    for scene, count in [('mixed', 32), ('capture', 16), ('coverage', 16)]:
        adapter = CoCapAdapter(SEED, scene)
        env, observations, agents = make_oracle(module, root, scene, SEED)
        same(adapter.root_config, root); same(adapter.config, env.config)
        same(adapter.observations, observations)
        records = []
        for step in range(count):
            actions = [(step + 2 * i) % 9 for i in range(len(env.pursuers))]
            end, row = compare_transition(adapter, env, agents, actions); records.append(row)
            if end:
                adapter.reset(); set_global_config(env.config)
                observations = env.reset(); agents = [ApfAgent(e.a, e.w) for e in env.evaders]
                same(adapter.observations, observations)
        results.append({'scene': scene, 'transitions': count, 'exact_task_parity': True, 'records': records})
    # Reuse the first-stage physical inputs, now compare the implemented bridge
    # to an independently constructed original successful provider.
    reference = json.loads((CENTRAL / 'artifacts/2026-10-11_cocap_mappo_batch02/reference_transitions.json').read_text())
    fixtures = []
    for case in reference['fixtures']:
        adapter = CoCapAdapter(SEED, 'mixed'); env, _, agents = make_oracle(module, root, 'mixed', SEED)
        item = case['input']; inject(adapter.env, item); inject(env, item)
        adapter.observations = adapter.env.get_observations()
        same(adapter.observations, env.get_observations())
        end, row = compare_transition(adapter, env, agents, item['pursuer_actions'])
        fixtures.append({'name': case['name'], 'input': item, 'exact_bridge_parity': True, 'result': row})
        if case['name'] == 'capture_to_recovery':
            assert adapter.env.evaders[0].deactivated and not end
            assert row['phase'] == ['pre_capture'] * 4
            assert adapter.env.last_capture_events
            end1, row1 = compare_transition(adapter, env, agents, [4] * 4)
            assert adapter.env.post_capture_started and adapter.env.post_capture_step == 1 and not end1
            assert row1['phase'] == ['post_capture'] * 4
            assert not adapter.env.last_capture_events
            fixtures.append({'name': 'recovery_starts_next_decision', 'exact_bridge_parity': True, 'result': row1})
            # Keep the observed captured state; force only the diagnostic clock
            # to the existing window boundary, never change a task parameter.
            adapter.env.post_capture_step = env.post_capture_step = 499
            end2, row2 = compare_transition(adapter, env, agents, [4] * 4)
            assert adapter.env.post_capture_step == 500 and end2
            fixtures.append({'name': 'recovery_window_499_to_500', 'exact_bridge_parity': True, 'result': row2})
        if case['name'] == 'boundary':
            # Repeat the physical input at simultaneous horizon: death wins.
            adapter.reset(); set_global_config(env.config); env.reset()
            agents = [ApfAgent(e.a, e.w) for e in env.evaders]
            both = {**item, 'episode_step': 2999}; inject(adapter.env, both); inject(env, both)
            adapter.observations = adapter.env.get_observations()
            end2, row2 = compare_transition(adapter, env, agents, item['pursuer_actions'])
            assert end2 and all(row2['terminated']) and not any(row2['truncated'])
            fixtures.append({'name': 'death_terminal_over_horizon', 'exact_bridge_parity': True, 'result': row2})
    atomic_json(output / 'provider_parity.json', plain({'scenes': results, 'fixtures': fixtures,
                'oracle_module': str(IQN / 'tools/iqn_token_matched_20260919.py'),
                'oracle_sha256': file_hash(IQN / 'tools/iqn_token_matched_20260919.py'),
                'reference_resolved_hash': REFERENCE_RESOLVED_HASH}))
    return module, root, {'scenes': 3, 'scripted_rollout_transitions': 64, 'edge_fixtures': len(fixtures),
                          'observation_reward_done_next_state_exact': True, 'Z_once_per_decision': True}


def check_sources():
    provider = {}
    for rel in sorted(sources()):
        algorithm_files = ('src/cocap_voradj/models/small_step_ac.py',
                           'src/cocap_voradj/training/small_step_ac.py')
        if rel.startswith('src/cocap_voradj/') and rel not in algorithm_files and (IQN / rel).exists():
            current = file_hash(ROOT / rel); original = file_hash(IQN / rel)
            assert current == original, rel
            provider[rel] = {'current': current, 'successful_IQN': original, 'equal': True}
    unchanged = {}
    for rel in ('src/terl_mappo/model.py', 'src/terl_mappo/native.py', 'src/terl_mappo/run.py',
                'src/terl_mappo/batch01/interfaces.py', 'vendor/terl/policy/TERL_model.py',
                'src/cocap_voradj/training/small_step_ac.py', 'src/cocap_voradj/models/small_step_ac.py'):
        original = subprocess.check_output(['git', 'show', BASE + ':' + rel], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == file_hash(ROOT / rel)
        unchanged[rel] = file_hash(ROOT / rel)
    return {'successful_provider_hashes': provider, 'unchanged_learner_model_sources': unchanged}


def continuation(trainer, adapter, config):
    batch, episodes, trace = collect(trainer, adapter, 16)
    metrics = trainer.update(batch, categorical=True)
    assert metrics['minibatch_updates'] > 0 and all(np.isfinite(v) for v in metrics.values())
    return {'batch_digest': digest_graph(batch), 'episodes_digest': digest_graph(episodes),
            'trace_digest': digest_graph(trace), 'metrics': metrics,
            'trainer_digest': digest_graph(trainer.state_dict()),
            'runtime_digest': digest_graph(adapter.state_dict()), 'rng_digest': digest_graph(rng_state()),
            'fingerprint': adapter.fingerprint()}


def restored_child(args):
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']; adapter = CoCapAdapter(SEED + 1, 'mixed'); trainer = build(config, adapter)
    loaded = load_checkpoint(args.checkpoint, trainer, adapter, config)
    actual = continuation(trainer, adapter, config)
    expected = json.loads(Path(args.expected).read_text())
    same(actual, expected)
    atomic_json(args.output / 'checkpoint_restore.json', {
        'status': 'FRESH_PROCESS_EXACT_CONTINUATION_PASS', 'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': file_hash(args.checkpoint), 'saved_decisions': loaded['decisions'],
        'saved_agent_transitions': loaded['agent_transitions'], 'saved_optimizer_steps': loaded['optimizer_steps'],
        'continuation_decisions': 16, 'batch_episode_trace_metrics_parameters_Adam_ValueNorm_runtime_rng_exact': True,
        'fresh_process_pid': os.getpid(), 'matches': actual})


def main(args):
    start = time.monotonic(); args.output.mkdir(parents=True, exist_ok=True)
    provenance = check_sources(); atomic_json(args.output / 'source_provenance.json', provenance)
    module, root, parity = provider_parity(args.output)
    print(json.dumps({'phase': 'provider_parity_complete', **parity}), flush=True)
    base = json.loads((ROOT / 'configs/experiments/terl_mappo_20261008/stage1.json').read_text())
    config = {key: base[key] for key in ('actor_seed', 'hidden_dim', 'num_heads', 'num_layers', 'masked_pool', 'ppo')}
    config['task_reference_resolved_hash'] = REFERENCE_RESOLVED_HASH
    adapter = CoCapAdapter(SEED, 'mixed'); trainer = build(config, adapter)
    batch, episodes, trace = collect(trainer, adapter, 256)
    logp_error = trainer.assert_behavior_log_probs(batch)
    # Replay actual sampled P actions through the original provider. APF, raw
    # reward and all task outputs remain independently generated by that path.
    env, observations, agents = make_oracle(module, root, 'mixed', SEED)
    template = copy.deepcopy(next(row for row in observations if row is not None))
    for step, row in enumerate(trace):
        same({key: batch['local_obs'][key][step] for key in batch['local_obs']},
             pack_observations(observations, template))
        outcome, evader_actions = oracle_step(env, agents, row['actions'])
        same(row['reward'], outcome.rewards.tolist()); same(row['task_infos'], outcome.infos)
        same(row['evader_actions'], evader_actions); same(row['end'], all(outcome.dones))
        observations = outcome.observations
        if row['end']:
            set_global_config(env.config); observations = env.reset(); agents = [ApfAgent(e.a, e.w) for e in env.evaders]
    actor_shapes = {name: list(value.shape) for name, value in trainer.actor.state_dict().items()}
    value_shapes = {name: list(value.shape) for name, value in trainer.value.state_dict().items()}
    before = digest_graph(trainer.state_dict())
    first = time.monotonic(); metrics = trainer.update(batch, categorical=True); update_seconds = time.monotonic() - first
    assert all(np.isfinite(v) for v in metrics.values()) and metrics['minibatch_updates'] > 0
    assert before != digest_graph(trainer.state_dict()) and metrics['actor_update_l2'] > 0
    assert all(torch.isfinite(p).all() for p in list(trainer.actor.parameters()) + list(trainer.value.parameters()))
    atomic_json(args.output / 'ppo_update.json', {'decisions': 256, 'agent_transitions': int(batch['active_mask'].sum()),
                'zero_update_log_prob_max_error': logp_error, 'metrics': metrics, 'update_seconds': update_seconds,
                'parameters_finite': True, 'real_optimizer_mutation': True,
                'actor_parameter_shapes': actor_shapes, 'critic_parameter_shapes': value_shapes,
                'input_binding': {'actor_self': 9, 'actor_slots': 22, 'critic_self': 9, 'critic_agents': 4},
                'contract': adapter.contract(), 'ppo': config['ppo'],
                'reward_mean_active': float(batch['rewards'][batch['active_mask']].mean()),
                'actual_sampled_rollout_replayed_through_successful_provider_exact': True})
    print(json.dumps({'phase': 'PPO_update_complete', 'metrics': metrics}), flush=True)
    checkpoint = ROOT / 'runs/e1_cpu_closed_loop_20261011/resume.pt'
    checkpoint_sha = save_checkpoint(checkpoint, trainer, adapter, config, 256,
                                    int(batch['active_mask'].sum()), int(metrics['minibatch_updates']))
    expected = continuation(trainer, adapter, config)
    expected_path = args.output / 'expected_continuation.json'; atomic_json(expected_path, expected)
    command = [sys.executable, str(Path(__file__).resolve()), '--restore-child', '--checkpoint', str(checkpoint),
               '--expected', str(expected_path), '--output', str(args.output)]
    subprocess.run(command, cwd=ROOT, check=True)
    recovery = json.loads((args.output / 'checkpoint_restore.json').read_text())
    result = {'status': 'E1_ENGINEERING_CLOSED_LOOP_PASS', 'E1_PASS': False,
              'independent_validation': False, 'formal_training_evidence': False,
              'verification_scope': 'Commander CPU task-provider parity, finite PPO and fresh-process recovery',
              'provider_parity': parity, 'sampled_rollout_parity_decisions': 256,
              'PPO_first_update': metrics, 'checkpoint_restore': recovery['status'],
              'checkpoint_sha256': checkpoint_sha, 'unique_smoke_decisions': 272,
              'continuation_replay_decisions': 16, 'total_PPO_calls_including_replay': 3,
              'long_training_started': False, 'reward_design_changed': False,
              'observation_design_changed': False, 'network_layer_topology_changed': False,
              'input_dimension_and_capacity_binding': 'explicit user-authorized MAPPO adaptation; CoCap takes precedence',
              'duration_seconds': time.monotonic() - start,
              'warnings': ['Independent reviewer sign-off absent; engineering closed-loop pass is not independent validation.',
                           'No capture/coverage performance conclusion from smoke.',
                           'Clock remains successful provider initialization at0; this smoke does not exercise200k reward schedule.']}
    atomic_json(args.output / 'result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restore-child', action='store_true'); parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--expected', type=Path)
    args = parser.parse_args(); args.output = args.output.resolve(); torch.set_num_threads(1)
    try:
        restored_child(args) if args.restore_child else main(args)
    except BaseException as error:
        atomic_json(args.output / 'failure.json', {'error': repr(error), 'pid': os.getpid(),
                    'status': 'STOPPED_FOR_BLOCKING_BUG_REVIEW', 'E1_PASS': False})
        raise
