"""New experiment wiring; frozen E1 and running N2/R2 sources are untouched."""
import copy
from pathlib import Path

import torch

from cocap_voradj.training.small_step_ac import MAPPOConfig, MAPPOTrainer
from terl_mappo.batch02.cocap_env import ROOT
from terl_mappo.batch02_a_long.runtime import (CapabilityAdapter, build as baseline_build,
    collect, sources as baseline_sources)
from terl_mappo.run import file_hash, rng_state, restore_rng
from .actor import CoCapStyleActor, backbone_config


class NativeAdapter(CapabilityAdapter):
    def __init__(self, seed, arm, *, training=True):
        if arm != 'N3':
            raise ValueError('implemented arm is N3')
        self.experiment_arm = arm
        super().__init__(seed, 'N2', training=training)

    def contract(self):
        contract = super().contract()
        contract.update(arm=self.experiment_arm, backbone='successful IQN VorAdj features + TERL AW9 head',
                        reward_design_delta='none', critic_delta='none')
        return contract


def build(config, adapter, device='cpu'):
    original = baseline_build(config, adapter, device)
    actor = CoCapStyleActor(backbone_config(adapter.root_config, config['learner']), original.actor)
    # Identical existing central V object and PPO/Adam config; actor alone changes.
    return MAPPOTrainer(actor, original.value, MAPPOConfig(**config['learner']['ppo']), device)


def sources():
    paths = list((ROOT / 'src/terl_mappo/batch02_native').glob('*.py'))
    paths += list((ROOT / 'tools').glob('*batch02_native*20261011.py'))
    paths += [ROOT / 'src/cocap_voradj/models/iqn.py']
    return {**baseline_sources(), **{str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}}


def save_checkpoint(path, trainer, adapter, config, decisions, agent_steps, optimizer_steps):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema='terl.batch02-native.v1', config=copy.deepcopy(config),
                   trainer=trainer.state_dict(), runtime=adapter.state_dict(), rng=rng_state(),
                   decisions=decisions, agent_transitions=agent_steps, optimizer_steps=optimizer_steps,
                   sources=sources(), contract=adapter.contract())
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary); temporary.replace(path)
    return file_hash(path)


def load_checkpoint(path, trainer, adapter, config):
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload['schema'] != 'terl.batch02-native.v1' or payload['config'] != config or payload['sources'] != sources():
        raise ValueError('native checkpoint config/source mismatch')
    trainer.load_state_dict(payload['trainer'])
    adapter.load_state_dict(payload['runtime'])
    if adapter.contract() != payload['contract']:
        raise ValueError('native restored runtime contract mismatch')
    restore_rng(payload['rng'], trainer.device)
    return payload
