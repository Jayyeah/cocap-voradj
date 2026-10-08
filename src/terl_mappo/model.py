"""Native TERL encoders/fusion with categorical backend and explicit NaN guards."""
import torch
from torch import nn
from torch.distributions import Categorical
from .native import initialize_native
from policy.TERL_model import TERLPolicy, TERLConfig
from cocap_voradj.models.small_step_ac import CentralValueNetwork

class TERLActor(TERLPolicy):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=4, seed=109, masked_pool=False):
        super().__init__(TERLConfig(hidden_dim=hidden_dim,num_heads=num_heads,num_layers=num_layers,seed=seed))
        # No quantile/cosine parameters or IQN buffers remain in the actor.
        del self.cos_embedding, self.pis, self.K, self.n
        self.masked_pool = bool(masked_pool)
        nn.init.orthogonal_(self.output_layer.weight, gain=.01)
        nn.init.zeros_(self.output_layer.bias)
        self.eval()

    def encode_entities(self, obs):
        encoded = [enc(obs[k]).unsqueeze(1) if k=='self' else enc(obs[k])
                   for k,enc in self.entity_encoders.items()]
        tokens = torch.cat(encoded,1) + self.type_embedding(obs['types'].long())
        mask = obs['masks'].bool()
        safe = mask.clone()
        safe[~safe.any(1),0] = True
        transformed = self.transformer_encoder(tokens,src_key_padding_mask=~safe)
        pool = transformed.masked_fill(~safe[...,None], -torch.inf) if self.masked_pool else transformed
        feature = torch.cat([transformed[:,0],pool.max(1).values],-1)
        # Native packing uses fixed slots: self, 5P, 8E, 5O. No ragged reshape.
        n_p = obs['pursuers'].shape[1]
        n_e = obs['evaders'].shape[1]
        ef = transformed[:,1+n_p:1+n_p+n_e]
        em = mask[:,1+n_p:1+n_p+n_e]
        ts = self.target_selection
        sf = ts.self_transform(feature)
        if n_e:
            score = ts.query_transform(sf)[:,None] @ ts.key_transform(ef).transpose(1,2) / ts.scale
            score = score.masked_fill(~em[:,None],-torch.inf)
            has = em.any(1)
            score = torch.where(has[:,None,None], score, torch.zeros_like(score))
            weight = torch.softmax(score,-1)*em[:,None].to(score.dtype)
            weighted = (weight @ ts.value_transform(ef)).squeeze(1)
        else:
            weight = sf.new_zeros((sf.shape[0],1,0))
            weighted = torch.zeros_like(sf)
        self._last_target_weights = weight.squeeze(1)
        return ts.layer_norm(sf+weighted)

    def forward(self, obs):
        self._validate_input(obs)
        feature = self.layer_norm(torch.relu(self.hidden_layer(self.encode_entities(obs))))
        return self.output_layer(feature)

    def distribution(self, obs):
        return Categorical(logits=self(obs))

    def sample(self, obs, deterministic=False):
        d = self.distribution(obs)
        index = d.logits.argmax(-1) if deterministic else d.sample()
        return index, d.log_prob(index), index

    def evaluate_indices(self, obs, index):
        d=self.distribution(obs)
        return d.log_prob(index.long()), d.entropy()

def make_critic(hidden_dim=256,num_heads=8,num_layers=4):
    return CentralValueNetwork(hidden_dim,num_heads,num_layers,self_feature_dim=25,
                               max_agents=3,max_evaders=1,max_obstacles=1).eval()
