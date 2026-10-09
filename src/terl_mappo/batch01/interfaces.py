"""Small injectable interfaces, without rewriting TERL or its historical runner."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Callable, Protocol, Any

import numpy as np
import torch

from ..native import NativeStage1, central_state, pack_local
from ..model import TERLActor, make_critic
from ..run import build, collect
from cocap_voradj.models.small_step_ac import CentralValueNetwork
from cocap_voradj.training.small_step_ac import MAPPOConfig, MAPPOTrainer, tensor_tree
from .source_guard import SourceGuard
from .evaluation import isolated_rng


class ContinuousPolicy(Protocol):
    """C0 implements the existing learner's continuous branch, with Jacobians.

    sample -> physical [P,2] AW, transformed log_prob [P], unsquashed latent [P,2].
    evaluate_latent -> log_prob, physical MC entropy, base Gaussian entropy,
    distribution, log_std [P,2]. distribution exposes loc/scale [P,2]; config
    exposes saturation_threshold (or its default). Log densities include tanh
    AND physical scale. Entropy estimation must not perturb the density check.
    """
    config: Any
    def sample(self, obs, deterministic=False): ...
    def evaluate_latent(self, obs, latent): ...


class EnvironmentAdapter(Protocol):
    observations: list
    env: Any
    def reset(self): ...
    def step(self, actions): ...  # obs,reward,terminated,truncated,joint_end,info
    def contract(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
    def fingerprint(self): ...
    def action_capabilities(self): ...  # ActionCapabilities, actual consumer
    def validate_actions(self, actions): ...  # shape/dtype/bounds, no coercion


@dataclass(frozen=True)
class ActionCapabilities:
    family: str
    representation: str  # integer_indices or physical_aw
    bounds: tuple | None = None  # ((min_a,max_a),(min_w,max_w)) for physical AW


def validate_adapter(adapter, backend, guard):
    """Validate the instantiated consumer, including an isolated behavior probe.

    NativeStage1's unchanged step implementation defines its AW9 capability.
    Other adapters declare capabilities and supply a strict action validator.
    Probing a copy preserves the live environment and every training RNG.
    """
    native_step = getattr(adapter.step, '__func__', None) is NativeStage1.step
    if native_step:
        capability = ActionCapabilities('categorical_aw9', 'integer_indices')
    else:
        if not callable(getattr(adapter, 'action_capabilities', None)) or not callable(getattr(adapter, 'validate_actions', None)):
            raise ValueError('adapter must declare actual action capabilities and validation')
        capability = guard.call(adapter.action_capabilities)
    representation = 'integer_indices' if backend.categorical else 'physical_aw'
    if not isinstance(capability, ActionCapabilities) or (capability.family, capability.representation) != (backend.family, representation):
        raise ValueError('physical action/backend incompatible with actual adapter capabilities')
    if native_step: return capability
    with isolated_rng():
        probe = copy.deepcopy(adapter)
        agents = len(pack_local(probe.observations)['self'])
        if backend.categorical:
            valid = np.full(agents, 4, np.int64)
            invalid = np.zeros((agents, 2))
        else:
            bounds = np.asarray(capability.bounds, dtype=float)
            if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ValueError('physical action adapter requires finite AW bounds')
            # Nonzero off-grid AW detects integer coercion in a wrapper consumer.
            valid = np.broadcast_to(bounds[:, 0] + .371 * (bounds[:, 1] - bounds[:, 0]), (agents, 2)).copy()
            invalid = np.zeros(agents, dtype=np.int64)
        guard.call(probe.validate_actions, valid)
        try: guard.call(probe.validate_actions, invalid)
        except (ValueError, TypeError): pass
        else: raise ValueError('adapter validator accepts incompatible action representation')
        try: row = guard.call(probe.step, valid)
        except (ValueError, TypeError) as error:
            raise ValueError('actual adapter rejects physical action behavior probe') from error
        if not isinstance(row, tuple) or len(row) != 6 or np.asarray(row[1]).shape != (agents,) or not np.isfinite(row[1]).all():
            raise ValueError('action adapter reset/step behavior contract mismatch')
    return capability


@dataclass(frozen=True)
class ActionBackend:
    family: str = 'categorical_aw9'
    entropy_measure: str = 'categorical_shannon_nats'
    logprob_measure: str = 'discrete_probability_mass'
    categorical: bool = True

    def validate(self, actions, agents):
        x = np.asarray(actions)
        if self.categorical:
            if x.shape != (agents,) or not np.issubdtype(x.dtype, np.integer) or np.any((x < 0) | (x > 8)):
                raise ValueError('native AW9 needs integer indices [P]; never coerce continuous AW')
        elif x.shape != (agents, 2) or not np.isfinite(x).all():
            raise ValueError('continuous backend needs finite physical AW [P,2]')
        return x


@dataclass
class Hooks:
    """An experiment supplies only its own delta; every factory is testable.

    T1: env_factory + state_encoder (+ critic_factory when shape changes).
    N1: actor_factory. R1: reward_transform. P1: resolved config['ppo'].
    C0: backend + actor_factory + env_factory that accepts physical AW directly.
    reward_transform receives raw native reward/info; may change reward only.
    """
    env_factory: Callable = NativeStage1
    state_encoder: Callable = central_state
    actor_factory: Callable = TERLActor
    critic_factory: Callable = make_critic
    reward_transform: Callable | None = None
    backend: ActionBackend = field(default_factory=ActionBackend)

    def validate_line(self, line):
        changed = {name for name in ('env_factory', 'state_encoder', 'actor_factory', 'critic_factory',
                                     'reward_transform', 'backend')
                   if getattr(self, name) != getattr(Hooks(), name)}
        allowed = {'T0': set(), 'T1': {'env_factory', 'state_encoder', 'critic_factory'},
                   'N1': {'actor_factory'}, 'R1': {'reward_transform'}, 'P1': set(),
                   'C0': {'env_factory', 'actor_factory', 'backend'}}
        if changed - allowed[line]:
            raise ValueError('hooks outside experiment line: ' + ','.join(sorted(changed)))
        if not self.backend.categorical:
            if line != 'C0':
                raise ValueError('continuous C0 requires a physical action environment adapter')
            if self.backend.entropy_measure == 'categorical_shannon_nats':
                raise ValueError('continuous entropy must declare its own measure')


@dataclass
class Runtime:
    trainer: Any
    adapter: EnvironmentAdapter
    hooks: Hooks
    config: dict
    source_guard: SourceGuard

    def collect(self, length):
        with self.source_guard.scope(): return self._collect(length)

    def _collect(self, length):
        if length <= 0:
            raise ValueError('positive on-policy rollout length required')
        if self.hooks == Hooks():
            return collect(self.trainer, self.adapter, length)
        # Preserve pre-reset normalized V, active masks, terminal precedence and
        # raw reward GAE. Dynamic P/E/O shapes are supplied by T1's state encoder.
        rows, episodes = [], []
        for _ in range(length):
            local = pack_local(self.adapter.observations)
            state = self.source_guard.call(self.hooks.state_encoder, self.adapter.env)
            actions, logp, latent, value = self.trainer.act(local, {k: v[None] for k, v in state.items()})
            self.hooks.backend.validate(actions, len(local['self']))
            _, raw, term, trunc, end, info = self.source_guard.call(self.adapter.step, actions)
            reward = np.asarray(raw).copy()
            if self.hooks.reward_transform is not None:
                reward = np.asarray(self.source_guard.call(self.hooks.reward_transform, reward.copy(), copy.deepcopy(info)))
            if reward.shape != np.asarray(raw).shape or not np.isfinite(reward).all():
                raise ValueError('reward adapter must preserve finite per-agent shape')
            following = self.source_guard.call(self.hooks.state_encoder, self.adapter.env)
            with torch.no_grad():
                nv = self.trainer.value(tensor_tree({k: v[None] for k, v in following.items()}, self.trainer.device)).cpu().numpy()[0]
            rows.append({'local_obs': local, 'global_obs': state, 'actions': actions, 'latent': latent,
                         'log_prob': logp, 'values': value[0], 'next_values': nv, 'rewards': reward,
                         'terminated': term, 'truncated': trunc,
                         'episode_end': np.full(len(raw), end), 'active_mask': info['active']})
            if end:
                episodes.append({k: v for k, v in info.items() if k not in ('active', 'native_infos')})
                self.source_guard.call(self.adapter.reset)
        return {k: {j: np.stack([r[k][j] for r in rows]) for j in rows[0][k]}
                if isinstance(rows[0][k], dict) else np.stack([r[k] for r in rows])
                for k in rows[0]}, episodes

    def update(self, batch):
        with self.source_guard.scope(): return self._update(batch)

    def _update(self, batch):
        if not self.hooks.backend.categorical and np.asarray(batch['active_mask']).any():
            # The historical continuous learner lacks the categorical zero-update
            # check. Require the corresponding latent-density check before PPO.
            from cocap_voradj.training.small_step_ac import flatten_local
            local, _ = flatten_local(tensor_tree(batch['local_obs'], self.trainer.device))
            active = torch.as_tensor(batch['active_mask'], device=self.trainer.device).bool().reshape(-1)
            latent = torch.as_tensor(batch['latent'], device=self.trainer.device).reshape(-1, 2)
            with isolated_rng(), torch.no_grad():
                new = self.trainer.actor.evaluate_latent({k: v[active] for k, v in local.items()}, latent[active])[0]
                old = torch.as_tensor(batch['log_prob'], device=self.trainer.device).reshape(-1)[active]
                if not torch.isfinite(new).all() or not torch.allclose(new, old, atol=1e-4, rtol=0):
                    raise ValueError('continuous behavior log-prob mismatch')
        return self.trainer.update(batch, categorical=self.hooks.backend.categorical)


def assemble(config, device='cpu', hooks=None, line='T0', source_guard=None):
    guard = source_guard or SourceGuard.committed()
    with guard.scope():
        return _assemble(config, device, hooks, line, guard)


def _assemble(config, device, hooks, line, guard):
    hooks = hooks or Hooks()
    hooks.validate_line(line)
    guard.preflight([hooks.env_factory, hooks.state_encoder, hooks.actor_factory,
                     hooks.critic_factory, hooks.reward_transform])
    # Native build owns seed/init order. Replacement factories are used only
    # after an explicitly declared line change; T0 remains bit exact.
    if hooks.actor_factory is TERLActor and hooks.critic_factory is make_critic:
        trainer = build(config, device)
    else:
        import random
        random.seed(config['actor_seed']); np.random.seed(config['actor_seed']); torch.manual_seed(config['actor_seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config['actor_seed'])
        actor = hooks.actor_factory(config['hidden_dim'], config['num_heads'], config['num_layers'],
                                    config['actor_seed'], config['masked_pool'])
        value = hooks.critic_factory(config['hidden_dim'], config['num_heads'], config['num_layers'])
        trainer = MAPPOTrainer(actor, value, MAPPOConfig(**config['ppo']), device)
        trainer.value.eval()
    adapter = guard.call(hooks.env_factory, config['seed'])
    validate_adapter(adapter, hooks.backend, guard)
    # Guard before each optimizer mutation too. Keep the learner and numerical
    # operations intact; the hooks neither sample RNG nor alter gradients.
    for optimizer in (trainer.actor_optimizer, trainer.value_optimizer):
        optimizer.register_step_pre_hook(lambda *args: guard.check())
    return Runtime(trainer, adapter, hooks, config, guard)


def multiscale_state(env, *, max_cores, map_scale=120., horizon_scale=3000.):
    """T1 shape seam: preserve physical scales; zero-pad ordered current slots.

    T1 still owns schedule/reset/reward lifecycle. This encoder is NOT a new
    curriculum or a replacement for Stage1's assertion-laden native encoder.
    """
    ps, es, obstacles = env.pursuers, env.evaders, env.obstacles
    if not ps or len(env.cores) > max_cores or min(map_scale, horizon_scale) <= 0:
        raise ValueError('invalid scene capacity/physical scale')
    active = np.array([not p.deactivated for p in ps])
    cv = np.zeros((max_cores, 4))
    for i, c in enumerate(sorted(env.cores, key=lambda c: (c.x, c.y))):
        cv[i] = [c.x / map_scale, c.y / map_scale, 1 if c.clockwise else -1, c.Gamma / (10 * np.pi)]
    focal = np.array([[p.x / map_scale, p.y / map_scale, *(p.velocity / 3.5), np.cos(p.theta),
                      np.sin(p.theta), p.speed / 3, float(p.is_pursuing),
                      env.episode_time_steps / horizon_scale, *cv.reshape(-1)] for p in ps], np.float32)
    pfeat = np.array([[p.x / map_scale, p.y / map_scale, *(p.velocity / 3.5), np.cos(p.theta),
                      np.sin(p.theta), float(p.is_pursuing)] for p in ps], np.float32)
    efeat = np.array([[e.x / map_scale, e.y / map_scale, *(e.velocity / 3.5), np.cos(e.theta),
                      np.sin(e.theta), e.speed / 3.5] for e in es], np.float32).reshape(-1, 7)
    # Critic slot dimensions are scene capacities, including a masked empty slot.
    if not es: efeat = np.zeros((1, 7), np.float32)
    ofeat = np.array([[o.x / map_scale, o.y / map_scale, o.r / map_scale, 0., 0.]
                      for o in obstacles], np.float32).reshape(-1, 5)
    if not obstacles: ofeat = np.zeros((1, 5), np.float32)
    n = len(ps)
    return {'self': focal, 'pursuers': np.broadcast_to(pfeat, (n, n, 7)).copy(),
            'pursuer_mask': np.broadcast_to(active, (n, n)).copy(), 'evaders': efeat,
            'evader_mask': np.array([not e.deactivated for e in es]) if es else np.zeros(1, bool),
            'obstacles': ofeat, 'obstacle_mask': np.ones(len(obstacles), bool) if obstacles else np.zeros(1, bool),
            'active_mask': active}


def multiscale_critic(*, agents, evaders, obstacles, max_cores, hidden_dim=256, num_heads=8, num_layers=4):
    return CentralValueNetwork(hidden_dim, num_heads, num_layers, self_feature_dim=9 + 4 * max_cores,
                               max_agents=agents, max_evaders=max(1, evaders), max_obstacles=max(1, obstacles)).eval()
