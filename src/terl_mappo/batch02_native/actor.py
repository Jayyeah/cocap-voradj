"""Successful IQN VorAdj feature path with the existing TERL AW9 policy head.

The feature methods are inherited directly from the successful IQN source.
Quantile and unused task heads are removed, not trained as dormant parameters.
Widths/depth and the policy head remain the registered TERL 256/8/4 contract.
"""
from dataclasses import asdict

import torch
from torch import nn

from cocap_voradj.models.iqn import CoCapIQN, CoCapNetConfig
from terl_mappo.model import TERLActor


def backbone_config(root, learner):
    iqn, perception = root['iqn'], root['perception']
    fields = {key: iqn[key] for key in ('action_size', 'self_feature_dim', 'architecture',
              'pursuer_feature_dim', 'include_is_pursuing', 'include_z_state', 'pursuing_late_fusion',
              'num_quantiles', 'num_cosine_features', 'pursuing_embed_dim')}
    fields.update(hidden_dim=learner['hidden_dim'], num_heads=learner['num_heads'],
                  num_layers=learner['num_layers'], max_pursuers=perception['max_pursuer_num'],
                  max_evaders=perception['max_evader_num'], max_obstacles=perception['max_obstacle_num'])
    config = CoCapNetConfig(**fields)
    if config.architecture != 'voradj_single_head' or config.include_is_pursuing or not config.include_z_state:
        raise ValueError('N3 requires the successful Z05 VorAdj single-head feature contract')
    return config


class CoCapVorAdjBackbone(CoCapIQN):
    def __init__(self, config):
        super().__init__(config)
        for name in ('cos_embedding', 'pis', 'coverage_fusion', 'encirclement_fusion',
                     'coverage_head', 'encirclement_head', 'gate_head', 'single_head'):
            delattr(self, name)

    def forward(self, obs):
        return self.voradj_single_feature(obs, self.features(obs))


class CoCapStyleActor(nn.Module):
    def __init__(self, config, original_head):
        super().__init__()
        self.config = config
        self.backbone = CoCapVorAdjBackbone(config)
        # Reuse the actual N2 head modules and initialization. No new action head.
        self.hidden_layer = original_head.hidden_layer
        self.layer_norm = original_head.layer_norm
        self.output_layer = original_head.output_layer
        self.eval()

    def forward(self, obs):
        # Inactive E1 rows are all-masked; PPO excludes them. Give these rows
        # the same self placeholder safeguard used by TERLActor, to avoid NaNs.
        mask = obs['masks'].bool()
        if not mask.any(dim=1).all():
            mask = mask.clone()
            mask[~mask.any(dim=1), 0] = True
            obs = {**obs, 'masks': mask}
        feature = self.backbone(obs)
        return self.output_layer(self.layer_norm(torch.relu(self.hidden_layer(feature))))

    distribution = TERLActor.distribution
    sample = TERLActor.sample
    evaluate_indices = TERLActor.evaluate_indices

    def architecture_contract(self):
        return dict(source='successful IQN CoCapIQN.features + voradj_single_feature',
                    config=asdict(self.config), actor_decentralized=True,
                    head='unchanged TERL hidden_layer/layer_norm/output_layer; AW9 categorical',
                    dropout='original CoCap0.1 modules, eval mode throughout PPO as existing learner',
                    inactive_rows='self placeholder only; excluded by original active_mask',
                    MAPPO_adaptation='remove quantile/unused IQN heads; retain TERL policy head and 256/8/4')
