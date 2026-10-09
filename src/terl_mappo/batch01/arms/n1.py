"""CoCap decision backbone on unchanged TERL self4/5x7/8x7/5x5."""
import torch
from torch import nn
from torch.distributions import Categorical
from cocap_voradj.models.continuous.local_entity_token_encoder import LegacyVorAdjFeatureBackbone,LegacyVorAdjFeatureBackboneConfig
from terl_mappo.batch01.interfaces import Hooks

class NativeCoCapActor(nn.Module):
    def __init__(self,hidden_dim=256,num_heads=8,num_layers=4,seed=109,masked_pool=False):
        super().__init__()
        if (hidden_dim,num_heads,num_layers)!=(256,8,4):raise ValueError('N1 registered actor architecture256/8/4')
        self.encoder=LegacyVorAdjFeatureBackbone(LegacyVorAdjFeatureBackboneConfig(
            hidden_dim=hidden_dim,num_heads=num_heads,num_layers=num_layers,self_feature_dim=4,
            max_pursuers=5,max_evaders=8,max_obstacles=5,dropout=.1))
        # Native self4 is consumed directly. Legacy pursuing embedding receives
        # its documented constant0 fallback; no role or extra target information.
        self.head=nn.Linear(hidden_dim,9)
        nn.init.orthogonal_(self.head.weight,gain=.01);nn.init.zeros_(self.head.bias)
        self.eval()
    def forward(self,obs):
        if obs['self'].shape[-1]!=4 or obs['pursuers'].shape[-2:]!=(5,7) or obs['evaders'].shape[-2:]!=(8,7) or obs['obstacles'].shape[-2:]!=(5,5):
            raise ValueError('N1 native observation shape drift')
        return self.head(self.encoder(obs))
    def distribution(self,obs):return Categorical(logits=self(obs))
    def sample(self,obs,deterministic=False):
        d=self.distribution(obs);a=d.logits.argmax(-1) if deterministic else d.sample()
        return a,d.log_prob(a),a
    def evaluate_indices(self,obs,index):
        d=self.distribution(obs);return d.log_prob(index.long()),d.entropy()

def factory(parameters):
    if parameters:raise ValueError('N1 has no unlocked parameters')
    return Hooks(actor_factory=NativeCoCapActor)
