"""Regression for captured A1/A2 terminal all-masked replay observations."""
import pytest
import torch
from cocap_voradj.models.shared_local_ac import SharedLocalACNetworkConfig, SharedLocalActor, SharedLocalQ
from cocap_voradj.training.discrete_sac import DiscreteSACConfig, DiscreteSACTrainer
from cocap_voradj.training.shared_local_ac import hard_copy


def observations(size=4):
    generator = torch.Generator().manual_seed(9001)
    masks = torch.zeros(size,22,dtype=torch.bool)
    masks[:,0] = True
    masks[1,1:4] = True
    masks[2,9:11] = True
    masks[2,17] = True
    return {
        'self': torch.randn(size,9,generator=generator),
        'pursuers': torch.randn(size,8,7,generator=generator),
        'evaders': torch.randn(size,8,7,generator=generator),
        'obstacles': torch.randn(size,5,5,generator=generator),
        'types': torch.tensor([0]+[1]*8+[2]*8+[3]*5).expand(size,-1).clone(),
        'masks': masks,
    }


@pytest.mark.parametrize('kind',['actor','target_actor','critic'])
def test_zero_filled_terminal_next_obs_has_finite_forward(kind):
    torch.manual_seed(7)
    config=SharedLocalACNetworkConfig(hidden_dim=32,num_heads=4,num_layers=2)
    model=(SharedLocalQ(config) if kind=='critic' else SharedLocalActor(config)).eval()
    if kind=='target_actor': model=hard_copy(model)
    obs=observations()
    for value in obs.values(): value[3].zero_()
    original={k:v.clone() for k,v in obs.items()}
    with torch.no_grad():
        values=model(obs) if kind=='critic' else model.logits(obs)
    assert torch.isfinite(values).all()
    assert all(torch.equal(v,original[k]) for k,v in obs.items()), 'encoder mutated replay input'


def test_valid_empty_neighborhood_self_token_semantics_unchanged():
    torch.manual_seed(11)
    model=SharedLocalActor(SharedLocalACNetworkConfig(hidden_dim=32,num_heads=4,num_layers=2)).eval()
    obs=observations()
    # The same terminal placeholder with its self token explicitly marked
    # valid is the finite reference. Existing self-valid empty neighborhoods
    # remain fully valid observations, including an all-zero self feature.
    for value in obs.values(): value[3].zero_()
    reference={k:v.clone() for k,v in obs.items()}
    reference['masks'][3,0]=True
    with torch.no_grad():
        actual=model.logits(obs)
        expected=model.logits(reference)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.isfinite(actual).all()


def test_sac_update_with_terminal_zero_row_preserves_finite_weights():
    torch.manual_seed(19)
    model=DiscreteSACTrainer(SharedLocalACNetworkConfig(hidden_dim=32,num_heads=4,num_layers=1),DiscreteSACConfig(),device='cpu')
    obs=observations()
    nxt={k:v.clone() for k,v in obs.items()}
    for value in nxt.values(): value[3].zero_()
    batch={'obs':obs,'next_obs':nxt,'actions':torch.arange(4),'rewards':torch.tensor([0.,1.,-1.,-160.]),'terminated':torch.tensor([False,False,False,True])}
    before={k:v.clone() for k,v in nxt.items()}
    stats=model.update(batch)
    assert stats['finite']==1.0
    assert all(torch.isfinite(p).all() for module in [model.actor,model.critic1,model.critic2,model.target_critic1,model.target_critic2] for p in module.parameters())
    assert all(torch.equal(v,before[k]) for k,v in nxt.items())
