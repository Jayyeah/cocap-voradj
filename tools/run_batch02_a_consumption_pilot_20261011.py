#!/usr/bin/env python3
"""One bounded shared N2/R2 qualification pilot; frozen E1 code is reused.

The user approved consumption qualification, since E1 already consumes both
complete CoCap contracts. Observers replay data and never change training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'tools')]
from terl_mappo.batch02.cocap_env import CoCapAdapter, pack_observations, stable_hash
from terl_mappo.batch02.runtime import build, collect, load_checkpoint, save_checkpoint, sources
from terl_mappo.run import atomic_json, file_hash, rng_state, restore_rng
from verify_batch02_e1_20261011 import (digest_graph, original_oracle, make_oracle,
                                       oracle_step, plain, robot_state, same)


def audit_batch(batch, trace, start_state, module, root):
    """Replay sampled actions using the successful IQN task provider.

    Restore the provider's full state at the saved rollout boundary. Only
    the replay instance is queried/stepped; the training runtime is untouched.
    """
    env, _, _ = make_oracle(module, root, 'mixed', 2056101100)
    env.__dict__.clear()
    env.__dict__.update(copy.deepcopy(start_state['env'].__dict__))
    agents = copy.deepcopy(start_state['apf'])
    template = start_state['_template']
    sums = defaultdict(float)
    nonzero = Counter()
    roles, phases = Counter(), Counter()
    events = Counter()
    finished = []
    residual_max = 0.0
    reset_count = 0
    observation_digest = []
    for step, row in enumerate(trace):
        z_before = env._z_update_count
        observations = env.get_policy_observations(include_is_pursuing=False, include_z_state=True)
        same(env._z_update_count, z_before)
        expected_local = pack_observations(observations, template)
        actual_local = {key: batch['local_obs'][key][step] for key in batch['local_obs']}
        same(actual_local, expected_local)
        observation_digest.append(digest_graph(actual_local))
        outcome, evader_actions = oracle_step(env, agents, row['actions'])
        same(batch['rewards'][step], outcome.rewards)
        same(row['task_infos'], outcome.infos)
        same(row['evader_actions'], evader_actions)
        same(row['terminated'], [info['terminated'] for info in outcome.infos])
        same(row['truncated'], [info['truncated'] for info in outcome.infos])
        same(row['end'], all(outcome.dones))
        for event in env.last_capture_events:
            events['stationary_capture' if event.get('capture_type') == 'stationary' else 'normal_capture'] += 1
        for index, info in enumerate(outcome.infos):
            meta = info['replay_metadata']
            same(float(batch['rewards'][step, index]), meta['reward_total'])
            roles[meta['reward_role']] += 1
            phases[meta['phase']] += 1
            for key, value in meta.items():
                if key.startswith('reward_') and isinstance(value, (int, float)):
                    sums[key] += float(value)
                    nonzero[key] += int(value != 0)
            total = sum(meta[key] for key in ('reward_capture', 'reward_coverage', 'reward_safety', 'reward_terminal'))
            residual_max = max(residual_max, abs(total - meta['reward_total']))
        if row['end']:
            collision_types = sorted({str(kind) for robot in env.pursuers + env.evaders
                                      for kind in getattr(robot, 'collision_types', ())})
            finished.append({'episode_step': int(env.episode_step),
                             'all_targets_captured': bool(env.evaders and all(e.deactivated and not e.collision for e in env.evaders)),
                             'post_capture_CE_success': bool(env.post_capture_coverage_success),
                             'collision_types': collision_types,
                             'terminated': row['terminated'], 'truncated': row['truncated']})
            from cocap_voradj.control.apf import ApfAgent
            from cocap_voradj.training.trainer import set_global_config
            set_global_config(env.config)
            env.reset()
            agents = [ApfAgent(e.a, e.w) for e in env.evaders]
            reset_count += 1
    return {'observation_exact': True, 'raw_reward_exact': True,
            'task_infos_APF_done_exact': True, 'Z_queries_advance_count': 0,
            'actor_observation_digest': stable_hash(observation_digest),
            'raw_reward_digest': digest_graph(batch['rewards']),
            'raw_reward_sums': dict(sums), 'reward_nonzero_agent_transitions': dict(nonzero),
            'top_level_component_residual_max': residual_max,
            'roles': dict(roles), 'phases': dict(phases), 'capture_events': dict(events),
            'finished_episodes': finished, 'reset_count': reset_count,
            'final_provider_robots': robot_state(env), 'final_provider_episode_step': int(env.episode_step)}


def main(args):
    started = time.monotonic()
    profile_path = ROOT / 'configs/experiments/terl_mappo_batch02_a_20261011/pilot.json'
    profile = json.loads(profile_path.read_text())
    qualification = json.loads(args.qualification.read_text())
    assert qualification['E1_PASS'] and qualification['independent_validation']
    assert qualification['tested_SHA'] == profile['frozen_bridge_sha']
    assert profile['additional_decisions'] == 1024 and profile['rollout_length'] == 256
    assert profile['device'] == 'cpu' and not profile['auto_extension']
    parent_path = Path(profile['parent_checkpoint'])
    assert file_hash(parent_path) == profile['parent_checkpoint_sha256']
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    assert sources() == parent['sources']
    config = parent['config']
    assert config['task_reference_resolved_hash'] == profile['reference_resolved_hash']
    if args.output.exists():
        raise ValueError('pilot output already exists; preserve the first run')
    args.output.mkdir(parents=True)
    adapter = CoCapAdapter(2056101100, 'mixed')
    trainer = build(config, adapter, 'cpu')
    loaded = load_checkpoint(parent_path, trainer, adapter, config)
    source_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = {'profile': profile, 'profile_sha256': file_hash(profile_path),
                'pilot_runner_sha256': file_hash(Path(__file__)), 'pilot_source_HEAD': source_sha,
                'frozen_E1_sources': sources(), 'learner_config': config,
                'independent_qualification_sha256': file_hash(args.qualification),
                'independent_qualification': str(args.qualification), 'initial_contract': adapter.contract(),
                'initial_fingerprint': adapter.fingerprint(), 'parent_decisions': loaded['decisions'],
                'budget_unique_additional_decisions': 1024, 'shared_evidence_arms': ['N2', 'R2']}
    atomic_json(args.output / 'manifest.json', manifest)
    module, root = original_oracle()
    rows, summaries, all_episodes = [], [], []
    extra_agent_steps = 0
    optimizer_steps = int(loaded['optimizer_steps'])
    for update in range(4):
        start_state = adapter.state_dict()
        batch, episodes, trace = collect(trainer, adapter, 256)
        logp_error = trainer.assert_behavior_log_probs(batch)
        # Replay is observational and must not change the learning RNG stream.
        learner_rng = rng_state()
        try:
            audit = audit_batch(batch, trace, start_state, module, root)
        finally:
            restore_rng(learner_rng, 'cpu')
        same(robot_state(adapter.env), audit['final_provider_robots'])
        same(adapter.env.episode_step, audit['final_provider_episode_step'])
        metrics = trainer.update(batch, categorical=True)
        assert all(np.isfinite(value) for value in metrics.values())
        assert metrics['minibatch_updates'] > 0 and metrics['actor_update_l2'] > 0
        assert all(torch.isfinite(parameter).all() for parameter in list(trainer.actor.parameters()) + list(trainer.value.parameters()))
        extra_agent_steps += int(batch['active_mask'].sum())
        optimizer_steps += int(metrics['minibatch_updates'])
        all_episodes.extend(audit['finished_episodes'])
        gradient = trainer.actor.entity_encoders['self'][0].weight.grad
        row = {'additional_decisions': (update + 1) * 256,
               'total_decisions': loaded['decisions'] + (update + 1) * 256,
               'active_agent_transitions': int(batch['active_mask'].sum()),
               'behavior_log_prob_max_error': logp_error,
               'reward_mean_active': float(batch['rewards'][batch['active_mask']].mean()),
               'self_input_weight_gradient_norm_by_column': gradient.norm(dim=0).tolist() if gradient is not None else None,
               'metrics': metrics, 'audit': audit}
        torch.save({'batch': batch, 'trace': trace, 'episodes': episodes}, args.output / f'rollout_{(update+1)*256:04d}.pt')
        rows.append(row); summaries.append(audit)
        atomic_json(args.output / 'training_curve.json', plain(rows))
        print(json.dumps({'phase': 'pilot_update', 'decisions': row['additional_decisions'],
                          'metrics': metrics, 'reward_mean_active': row['reward_mean_active']}), flush=True)
    last = args.output / 'last.pt'
    last_sha = save_checkpoint(last, trainer, adapter, config, loaded['decisions'] + 1024,
                              loaded['agent_transitions'] + extra_agent_steps, optimizer_steps)
    assert sources() == parent['sources']
    assert file_hash(parent_path) == profile['parent_checkpoint_sha256']
    common = {'shared_pilot_id': 'BATCH02-A-N2-R2-1024-CONSUMPTION-20261011',
              'classification': 'CONSUMPTION_QUALIFICATION_PILOT_PASS',
              'additional_decisions': 1024, 'new_PPO_calls': 4,
              'active_agent_transitions': extra_agent_steps,
              'frozen_bridge_sha': profile['frozen_bridge_sha'],
              'contract_delta_from_E1': None, 'contract_delta_from_successful_IQN': None,
              'causal_ablation': False, 'long_run_started': False,
              'long_run_authorized': False, 'last_checkpoint_sha256': last_sha}
    n2 = {**common, 'arm': 'N2', 'observation_exact_all_decisions': True,
          'full_local_schema': adapter.contract()['local_shapes'],
          'Z_query_updates': 0, 'extra_actor_information': None,
          'raw_observation_digests': [audit['actor_observation_digest'] for audit in summaries],
          'relative_TERL_observation_delta': 'self4/19 to successful self9/22 with full original CoCap sensing/VorAdj/centroid/boundary/Z05',
          'MAPPO_adaptation': 'existing authorized input binding and original on-policy TERLActor consumption; unchanged topology/PPO'}
    r2 = {**common, 'arm': 'R2', 'raw_reward_exact_all_decisions': True,
          'reward_transform': None, 'reward_tensor_dtype_in_existing_PPO': 'float32',
          'clock_range': [int(parent['runtime']['env'].total_steps), int(adapter.env.total_steps)],
          'coverage_ce_speed_weight': float(adapter.env.reward_cfg['coverage_ce_speed_weight']),
          'raw_reward_sums': dict(sum_counters([audit['raw_reward_sums'] for audit in summaries])),
          'reward_nonzero_agent_transitions': dict(sum_counters([audit['reward_nonzero_agent_transitions'] for audit in summaries])),
          'top_level_component_residual_max': max(audit['top_level_component_residual_max'] for audit in summaries),
          'relative_TERL_reward_delta': 'native time/distance/global reward to unchanged successful CR-MS/support/CE/PBRS/safety/events',
          'MAPPO_adaptation': 'unmodified task rewards enter existing GAE/PPO and ValueNorm; no R1 shaping or failed MAPPO reward'}
    warnings = ['Shared pilot qualifies both consumers; it is one training trajectory, not two independent causal experiments.',
                'No held-out or full-lifecycle learning-performance conclusion from1024 decisions.',
                'Original successful reward schedule200k boundary and recovery snapshot training distribution remain outside scope.']
    if r2['top_level_component_residual_max'] > 1e-6:
        warnings.append('Task metadata top-level component sum differs; raw reward parity passes. Preserve ledger residual for review.')
    result = {**common, 'E1_PASS': True, 'independent_validation_of_E1': True,
              'pilot_independent_validation': False, 'total_decisions': loaded['decisions'] + 1024,
              'total_new_minibatch_updates': optimizer_steps - loaded['optimizer_steps'],
              'first_metrics': rows[0]['metrics'], 'last_metrics': rows[-1]['metrics'],
              'finished_training_episodes': all_episodes,
              'unfinished_tail': {'censored_at_pilot_budget': True, 'episode_step': int(adapter.env.episode_step)},
              'last_checkpoint': str(last), 'frozen_sources_unchanged': True,
              'reward_design_changed': False, 'observation_design_changed': False,
              'actor_critic_topology_changed': False, 'entered_long_run': False,
              'long_run_decision': 'NO: only consumption qualification established, no causal/performance result or authorized long budget',
              'duration_seconds': time.monotonic() - started, 'warnings': warnings}
    for filename, data in [('n2_consumption.json', n2), ('r2_consumption.json', r2), ('result.json', result)]:
        atomic_json(args.output / filename, plain(data))
    print(json.dumps(result), flush=True)


def sum_counters(mappings):
    total = defaultdict(float)
    for mapping in mappings:
        for key, value in mapping.items(): total[key] += value
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); args.output = args.output.resolve()
    torch.set_num_threads(1)
    try:
        main(args)
    except BaseException as error:
        if args.output.exists():
            atomic_json(args.output / 'failure.json', {'status': 'STOPPED_FOR_BLOCKING_BUG_REVIEW',
                        'error': repr(error), 'entered_long_run': False, 'automatic_retry': False})
        raise
