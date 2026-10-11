"""Experiment wiring, never a replacement for the successful CoCap task.

N2 consumes the frozen CoCap observation. R2 calls the original TERL
perception implementation on isolated physical-state proxies. Native sensor
collision writes affect proxies only; the CoCap provider owns every terminal.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from terl_mappo.batch02.cocap_env import CoCapAdapter, ROOT
from terl_mappo.batch02.runtime import build as e1_build, central_state, sources as e1_sources
from terl_mappo.native import initialize_native, pack_local, empty_observation
from terl_mappo.run import file_hash, rng_state, restore_rng
from cocap_voradj.training.small_step_ac import tensor_tree
from cocap_voradj.training.trainer import CoCapTrainer, parse_scalar_step_schedule
from robots.pursuer import Pursuer as NativePursuer


def terl_observations(env):
    """Original self4/5P/8E/5O robot-frame contract, on the CoCap state.

    The global enemy visibility, 20m friendly/obstacle sensor, distance sort,
    raw metre units, pursuing bit, padding and types are native TERL semantics.
    Only physical state/radii/role values come from the real CoCap provider.
    """
    proxies = [NativePursuer(p.id) for p in env.pursuers]
    for proxy, physical in zip(proxies, env.pursuers):
        for key in ('x', 'y', 'theta', 'speed', 'velocity', 'r', 'detect_r',
                    'deactivated', 'collision', 'is_pursuing'):
            setattr(proxy, key, copy.deepcopy(getattr(physical, key)))
    return [p.perception_output(env.obstacles, proxies, env.evaders, True)[0]
            for p in proxies]


class CapabilityAdapter(CoCapAdapter):
    def __init__(self, seed, arm, *, training=True):
        if arm not in ('N2', 'R2'):
            raise ValueError('Batch02-A permits only N2/R2')
        self.arm = arm
        self.training = bool(training)
        if arm == 'R2':
            initialize_native()
        super().__init__(seed, 'mixed')
        self.base_coverage_ce_speed_weight = float(self.env.reward_cfg.get('coverage_ce_speed_weight', .05))
        self.coverage_ce_speed_weight_schedule = parse_scalar_step_schedule(
            self.env.reward_cfg.get('coverage_ce_speed_weight_schedule', []),
            default_value=self.base_coverage_ce_speed_weight,
            field_name='coverage_ce_speed_weight')
        self.schedule_events = []
        self.reward_sums = {}
        self.reward_residual_max = 0.
        self.capture_event_count = 0
        self.last_episode = None
        self.apply_original_schedule()

    def packed_observations(self):
        if self.arm == 'N2':
            return super().packed_observations()
        return pack_local(terl_observations(self.env))

    def apply_original_schedule(self):
        if not self.training:
            # Successful IQN formal make_env starts a fresh native environment;
            # its evaluator does not invoke the training schedule callback.
            return self.env.reward_cfg['coverage_ce_speed_weight']
        previous = float(self.env.reward_cfg['coverage_ce_speed_weight'])
        view = SimpleNamespace(
            global_step=self.env.total_steps,
            coverage_ce_speed_weight_schedule=self.coverage_ce_speed_weight_schedule,
            base_coverage_ce_speed_weight=self.base_coverage_ce_speed_weight,
            envs={'mixed': self.env})
        value = CoCapTrainer._set_coverage_ce_control_weights(view)
        if value != previous:
            self.schedule_events.append({'pre_action_clock': int(self.env.total_steps),
                                         'previous': previous, 'value': value})
        return value

    def reset(self):
        result = super().reset()
        self.reward_sums = {}
        self.reward_residual_max = 0.
        self.capture_event_count = 0
        self.last_episode = None
        return result

    def step(self, actions):
        self.apply_original_schedule()
        result = super().step(actions)
        _, reward, term, trunc, end, info = result
        self.capture_event_count += len(self.env.last_capture_events)
        for meta in (row['replay_metadata'] for row in info['task_infos']):
            for key in ('reward_capture', 'reward_coverage', 'reward_safety', 'reward_terminal', 'reward_total'):
                self.reward_sums[key] = self.reward_sums.get(key, 0.) + float(meta[key])
            residual = abs(sum(meta[key] for key in ('reward_capture', 'reward_coverage', 'reward_safety', 'reward_terminal')) - meta['reward_total'])
            self.reward_residual_max = max(self.reward_residual_max, residual)
        if end:
            record = self.env.episode_record()
            snapshot = self.env.capture_snapshot
            duration = self.env.pursuers[0].dt * self.env.pursuers[0].N
            states = sorted({str(row['state']) for row in info['task_infos']})
            record.update(
                episode_reward_mean=float(np.mean(self.return_sum)),
                episode_reward_sum=float(np.sum(self.return_sum)),
                capture_completion=bool(record['captured']),
                capture_events=int(self.capture_event_count),
                capture_time_decisions=int(snapshot['step']) if snapshot else None,
                capture_time_seconds=float(snapshot['step'] * duration) if snapshot else None,
                decision_seconds=float(duration),
                task_end_states=states,
                failure_type='none' if record['captured'] else '|'.join(states),
                reward_components=copy.deepcopy(self.reward_sums),
                reward_component_residual_max=self.reward_residual_max,
                terminated=term.tolist(), truncated=trunc.tolist())
            self.last_episode = compact_record(record)
            info['capability_episode'] = self.last_episode
        return result

    def contract(self):
        result = super().contract()
        template = self._template if self.arm == 'N2' else empty_observation()
        result.update(arm=self.arm, training=self.training,
                      learner_observation_shapes={key: list(value.shape) for key, value in template.items()},
                      observation_consumer='successful_IQN_CoCap' if self.arm == 'N2' else 'original_TERL_on_CoCap_physical_state',
                      MAPPO_adaptation='original categorical TERLActor; E1 central critic self9/4P capacity; R2 actor input self4',
                      reward_schedule='original CoCapTrainer callback' if self.training else 'successful IQN fresh-env evaluation')
        return result


def compact_record(record):
    keys = ('length', 'captured', 'fully_capture', 'episode_success', 'episode_success_mode',
            'collision_free', 'collision_event', 'pursuer_collision_event', 'evader_collision_event',
            'collision_type_counts', 'boundary_collision_event', 'active_pursuers',
            'post_capture_episode', 'episode_reward_mean', 'episode_reward_sum', 'capture_completion',
            'capture_events', 'capture_time_decisions', 'capture_time_seconds', 'decision_seconds',
            'task_end_states', 'failure_type', 'reward_components', 'reward_component_residual_max',
            'terminated', 'truncated')
    return {key: record[key] for key in keys}


def build(config, adapter, device):
    # E1's existing dimension binding is reused; no new actor/critic class,
    # topology, layer, head, optimizer or central information is introduced.
    template = adapter._template if adapter.arm == 'N2' else empty_observation()
    return e1_build(config['learner'], SimpleNamespace(_template=template, env=adapter.env), device)


def collect(trainer, adapter, length):
    """E1 rollout equations, with terminal episode metrics and no large trace."""
    rows, episodes = [], []
    for _ in range(length):
        local = adapter.packed_observations()
        central = central_state(adapter.env)
        actions, logp, latent, value = trainer.act(local, {k: v[None] for k, v in central.items()})
        _, reward, term, trunc, end, info = adapter.step(actions)
        following = central_state(adapter.env)
        with torch.no_grad():
            nv = trainer.value(tensor_tree({k: v[None] for k, v in following.items()}, trainer.device)).cpu().numpy()[0]
        rows.append(dict(local_obs=local, global_obs=central, actions=actions, latent=latent,
                         log_prob=logp, values=value[0], next_values=nv, rewards=reward,
                         terminated=term, truncated=trunc, episode_end=np.full(len(reward), end),
                         active_mask=info['active']))
        if end:
            episodes.append(copy.deepcopy(info['capability_episode']))
            adapter.reset()
    batch = {k: {field: np.stack([row[k][field] for row in rows]) for field in rows[0][k]}
             if isinstance(rows[0][k], dict) else np.stack([row[k] for row in rows]) for k in rows[0]}
    return batch, episodes


def sources():
    paths = list((ROOT / 'src/terl_mappo/batch02_a_long').glob('*.py'))
    paths += [ROOT / 'tools/run_batch02_a_long_20261011.py']
    paths += [ROOT / rel for rel in ('vendor/terl/robots/pursuer.py', 'vendor/terl/robots/robot.py',
              'vendor/terl/robots/perception.py',
              'vendor/terl/config/config.yaml')]
    # utils is a file in some TERL distributions; discover the actual module.
    import utils
    paths += [Path(utils.__file__)]
    return {**e1_sources(), **{str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}}


def save_checkpoint(path, trainer, adapter, config, decisions, agent_steps, optimizer_steps):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema='terl.batch02-a.capability.v1', config=copy.deepcopy(config),
                   trainer=trainer.state_dict(), runtime=adapter.state_dict(), rng=rng_state(),
                   decisions=decisions, agent_transitions=agent_steps, optimizer_steps=optimizer_steps,
                   sources=sources(), contract=adapter.contract())
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary); temporary.replace(path)
    return file_hash(path)


def load_checkpoint(path, trainer, adapter, config):
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload['schema'] != 'terl.batch02-a.capability.v1' or payload['config'] != config or payload['sources'] != sources():
        raise ValueError('capability checkpoint config/source mismatch')
    trainer.load_state_dict(payload['trainer'])
    adapter.load_state_dict(payload['runtime'])
    if adapter.contract() != payload['contract']:
        raise ValueError('restored runtime contract mismatch')
    restore_rng(payload['rng'], trainer.device)
    return payload
