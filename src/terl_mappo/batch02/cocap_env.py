"""Lossless learner interface to the successful IQN Z05 CoCap V2 provider.

No reward transform, observation feature transform, or automatic reset. Only
stacking and inactive-row placeholders adapt task-native outputs for MAPPO.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from cocap_voradj.control.apf import ApfAgent
from cocap_voradj.envs.density_sensing import POLICY, runtime_metadata
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.forward_final_v2 import enable_v2
from cocap_voradj.training.runtime_semantics import assert_runtime
from cocap_voradj.training.trainer import deep_update, load_config, set_global_config

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / 'configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage1_4p1e1obs_2m.yaml'
REFERENCE_RESOLVED_HASH = '6955ecf28383f7e146a0ee485ef3e33cb1d1782b197a51ceadd2ef93968d3b5a'


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def reference_config(path=REFERENCE):
    """Same load/enable/scene semantics as successful IQN's resolver."""
    config = load_config(str(path))
    spec = config.get('normsense_v2', {})
    if not spec.get('enabled') or spec.get('policy') != POLICY:
        raise ValueError('successful reference requires explicit NormSense V2')
    return enable_v2(config)


def scene_config(root, scene):
    if scene not in ('mixed', 'capture', 'coverage'):
        raise ValueError('scene must use a successful IQN evaluation profile')
    task = 'voradj_coverage' if scene == 'coverage' else 'voradj'
    config = deep_update(copy.deepcopy(root), root['tasks'][task])
    if scene == 'capture':
        config.setdefault('voradj', {})['capture_episode_ends_on_capture'] = True
        config['voradj']['capture_episode_success_on_capture'] = True
    return config


def pack_observations(observations, template):
    """Copy every live feature unchanged; None rows have matching dimensions.

    No native self4/19-slot fallback: dead placeholders have all-false masks
    and the same type IDs as the task provider. Active loss excludes them.
    """
    empty = {key: np.zeros_like(value) for key, value in template.items()}
    empty['types'] = np.asarray(template['types']).copy()
    rows = [empty if row is None else row for row in observations]
    return {key: np.stack([row[key] for row in rows]) for key in template}


class CoCapAdapter:
    def __init__(self, seed=2056101100, scene='mixed', config_path=REFERENCE):
        self.scene = scene
        self.root_config = reference_config(config_path)
        self.resolved_hash = stable_hash(self.root_config)
        self.config = scene_config(self.root_config, scene)
        set_global_config(self.config)
        self.env = VorAdjEnv(copy.deepcopy(self.config), seed=int(seed))
        self.observations = list(self.env.reset())
        self._template = copy.deepcopy(next(row for row in self.observations if row is not None))
        self.apf = [ApfAgent(evader.a, evader.w) for evader in self.env.evaders]
        self.return_sum = np.zeros(len(self.env.pursuers), dtype=np.float64)
        self.last_evader_actions = []
        self.last_infos = []
        self.assert_contract()

    def assert_contract(self):
        if runtime_metadata(self.env)['policy'] != POLICY:
            raise ValueError('CoCap provider did not consume V2 sensing')
        assert_runtime(self.env)
        if self.env.action_mode != 'unicycle_discrete':
            raise ValueError('E1 uses successful categorical AW9')
        for agent in self.apf:
            if not agent.is_v2 or agent.self_state_mode != 'theta_speed':
                raise ValueError('CoCap evader APF provider mismatch')

    def packed_observations(self):
        return pack_observations(self.observations, self._template)

    def validate_actions(self, actions):
        value = np.asarray(actions)
        if value.shape != (len(self.env.pursuers),) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError('CoCap AW9 needs one integer index per pursuer')
        if np.any((value < 0) | (value > 8)):
            raise ValueError('CoCap AW9 index outside [0,8]')
        return value

    def reset(self):
        set_global_config(self.config)
        self.observations = list(self.env.reset())
        self.apf = [ApfAgent(evader.a, evader.w) for evader in self.env.evaders]
        self.return_sum = np.zeros(len(self.env.pursuers), dtype=np.float64)
        self.last_evader_actions = []
        self.last_infos = []
        return self.packed_observations()

    def step(self, actions):
        actions = self.validate_actions(actions)
        active = np.array([not pursuer.deactivated for pursuer in self.env.pursuers])
        set_global_config(self.config)
        evader_obs = self.env.get_evader_observations_for_apf()
        self.last_evader_actions = [None if evader.deactivated else int(self.apf[i].act(evader_obs[i]))
                                   for i, evader in enumerate(self.env.evaders)]
        result = self.env.step([int(action) if active[i] else None for i, action in enumerate(actions)],
                               self.last_evader_actions)
        self.observations = list(result.observations)
        self.last_infos = copy.deepcopy(result.infos)
        terminated = np.array([row['terminated'] for row in result.infos], dtype=bool)
        truncated = np.array([row['truncated'] for row in result.infos], dtype=bool)
        end = bool(all(result.dones))
        reward = np.asarray(result.rewards).copy()
        self.return_sum += reward * active
        return self.packed_observations(), reward, terminated, truncated, end, {
            'active': active, 'episode_steps': self.env.episode_step,
            'episode_return': self.return_sum.copy(), 'task_infos': self.last_infos,
            'task_dones': list(result.dones), 'evader_actions': list(self.last_evader_actions),
            'post_capture_started': bool(self.env.post_capture_started),
            'post_capture_step': int(self.env.post_capture_step),
        }

    def contract(self):
        return {'reference': 'successful IQN Z05 Stage1 Forward-Final V2',
                'resolved_hash': self.resolved_hash, 'scene': self.scene,
                'counts': [len(self.env.pursuers), len(self.env.evaders), len(self.env.obstacles)],
                'local_shapes': {key: list(value.shape) for key, value in self._template.items()},
                'action': 'categorical_aw9', 'auto_reset': False,
                'reward_transform': None, 'observation_feature_transform': None,
                'sensing': runtime_metadata(self.env),
                'transition_semantics': self.env.transition_semantics}

    def state_dict(self):
        # Preserve environment RNG, collision traces, CE/PBRS/Z/K10/phase/cache,
        # observations and per-evader APF state together, including aliases.
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        if state['resolved_hash'] != self.resolved_hash or state['scene'] != self.scene:
            raise ValueError('CoCap runtime checkpoint profile mismatch')
        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(state))
        set_global_config(self.config)
        self.assert_contract()

    def fingerprint(self):
        e = self.env
        data = {'robots': [[r.x, r.y, r.theta, r.speed, *r.velocity,
                           bool(r.deactivated), bool(r.collision)] for r in e.pursuers + e.evaders],
                'episode_step': e.episode_step, 'total_steps': e.total_steps,
                'z': e.z_state.tolist(), 'z_updates': e._z_update_count,
                'K10': list(e._pursuing_release_counters),
                'phase': [e.post_capture_started, e.post_capture_step, e.distribution_hold_steps],
                'env_rng': [e.rng.get_state()[0], e.rng.get_state()[1].tolist(), *e.rng.get_state()[2:]]}
        return stable_hash(data)
