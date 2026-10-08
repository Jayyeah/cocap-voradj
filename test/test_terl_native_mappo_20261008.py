import copy
import json
import random
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from terl_mappo.native import (NativeStage1, MarineEnv, ApfAgent, stage1_schedule,
                               central_state, pack_local, VENDOR, TERL_SHA, geometry)
from terl_mappo.model import TERLActor, make_critic
from terl_mappo.run import DEFAULT, build, collect, save_checkpoint, load_checkpoint
from policy.TERL_model import TERLPolicy, TERLConfig
from cocap_voradj.training.small_step_ac import compute_gae, tensor_tree, ValueNorm

torch.set_num_threads(1)

def small_config():
    c=json.loads(DEFAULT.read_text()); c.update(hidden_dim=32,num_heads=4,num_layers=1)
    return c

def obs_tensor(a): return tensor_tree(pack_local(a.observations),'cpu')

def freeze_positions(a, positions, evader=(60,60)):
    e=a.env
    e.cores=[]; e.core_centers=None
    e.assign_robot_position([np.array(x,dtype=float) for x in positions],[np.array(evader,dtype=float)])
    for r in e.pursuers+e.evaders:
        r.speed=0.; r.velocity=np.zeros(2)
    a.observations,_=e.get_pursuers_observations()
    a.evader_obs,_=e.get_evaders_observation()

def test_vendor_byte_identity():
    # Independent upstream Git objects, not a second invocation of the adapter.
    upstream=Path('/home/yjq/rl/CoCap1/TERL')
    if not upstream.exists(): pytest.skip('upstream checkout only available on experiment host')
    for p in VENDOR.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.yaml'):
            original=subprocess.check_output(['git','-C',str(upstream),'show',f'{TERL_SHA}:{p.relative_to(VENDOR)}'])
            assert p.read_bytes()==original

@pytest.mark.parametrize('seed',[9,66,2026100800])
def test_native_transition_reward_done_parity(seed):
    a=NativeStage1(seed); raw=MarineEnv(seed=seed,schedule=stage1_schedule())
    (raw_obs,_),_=raw.reset(); eo,_=raw.get_evaders_observation()
    apf=ApfAgent(raw.evaders[0].a,raw.evaders[0].w)
    np.testing.assert_array_equal(pack_local(a.observations)['self'],pack_local(raw_obs)['self'])
    actions=np.random.RandomState(seed).randint(9,size=(90,3))
    for action in actions:
        expected=raw.step((action.tolist(),[apf.act(eo[0])]))
        eo,_=raw.get_evaders_observation()
        actual=a.step(action)
        np.testing.assert_array_equal(expected[1],actual[1])
        assert expected[2]==actual[5]['native_done']
        assert expected[3]==actual[5]['native_infos']
        for k,v in pack_local(expected[0]).items(): np.testing.assert_array_equal(v,actual[0][k])
        for x,y in zip(a.env.pursuers+a.env.evaders,raw.pursuers+raw.evaders):
            np.testing.assert_array_equal([x.x,x.y,x.theta,x.speed,*x.velocity],[y.x,y.y,y.theta,y.speed,*y.velocity])
        if actual[4]: break

def test_aw9_shapes_information_runtime():
    a=NativeStage1(9); e=a.env
    assert (len(e.pursuers),len(e.evaders),len(e.obstacles),len(e.cores))==(3,1,0,4)
    p=e.pursuers[0]
    assert p.action_list==[(x,y) for x in [-.4,0,.4] for y in [-np.pi/6,0,np.pi/6]]
    assert p.dt==.05 and p.N==10 and e.min_pursuer_evader_init_dis==10
    o=a.observations[0]
    assert {k:v.shape for k,v in o.items()}=={'self':(4,),'pursuers':(5,7),'evaders':(8,7),'obstacles':(5,5),'masks':(19,),'types':(19,)}
    # P0 is far outside sensing range; enemy remains observed in public source.
    freeze_positions(a,[(5,5),(25,5),(5,25)])
    assert a.observations[0]['masks'][6] and np.linalg.norm(a.observations[0]['evaders'][0,:2])>20

@pytest.mark.parametrize('count,expected',[(3,5.),(4,3.5),(5,-8.),(6,-9.5)])
def test_global_reward_unreachable_branch_effective(count,expected):
    a=NativeStage1(9); e=a.env
    base=e.pursuers[0]; e.pursuers=[copy.deepcopy(base) for _ in range(count)]
    for i,p in enumerate(e.pursuers): p.x=60+i*.1; p.y=60; p.deactivated=True
    e.evaders[0].x=e.evaders[0].y=60
    # count includes inactive; keep compatibility, including separate global -10.
    np.testing.assert_allclose(e.global_reward(),expected)

def test_capture_goal_credit_and_joint_terminal():
    a=NativeStage1(9)
    angles=np.arange(3)*2*np.pi/3
    freeze_positions(a,[(60+6*np.cos(x),60+6*np.sin(x)) for x in angles])
    _,r,t,tr,end,info=a.step([4]*3)
    assert end and t.all() and not tr.any() and info['normal_capture']
    assert info['native_done']==[False]*3
    np.testing.assert_allclose(info['components']['goal'],[120*np.pi]*3,rtol=.015)
    assert info['ring3'] and info['strict_geometry']

def test_collision_priority_and_per_agent_done():
    a=NativeStage1(9); freeze_positions(a,[(50,50),(51,50),(80,80)],evader=(100,100))
    a.env.episode_time_steps=3000
    _,_,t,tr,end,info=a.step([4]*3)
    assert end and t.all() and not tr.any() and info['collision']
    # Original source timeout overrides collision penalty at the timeout row.
    assert info['native_done']==[True]*3
    assert info['native_infos'][0]['state']=='too long episode'

def test_native_horizon_off_by_one_and_bootstrap():
    a=NativeStage1(9); freeze_positions(a,[(10,10),(40,10),(10,40)],evader=(80,80))
    a.env.episode_time_steps=2999
    _,_,t,tr,end,_=a.step([4]*3)
    assert not end and not t.any() and not tr.any()
    _,_,t,tr,end,info=a.step([4]*3)
    assert end and not t.any() and tr.all() and info['episode_steps']==3001

def test_native_boundary_is_penalty_only():
    a=NativeStage1(9); freeze_positions(a,[(-5,10),(40,10),(10,40)],evader=(80,80))
    _,_,t,tr,end,info=a.step([4]*3)
    assert not end and not t.any() and not tr.any()
    assert info['components']['global'][0]==-5

def test_actor_original_feature_parity_padding_and_no_iqn():
    a=NativeStage1(9); o=obs_tensor(a)
    actor=TERLActor(32,4,1,109)
    reference=TERLPolicy(TERLConfig(hidden_dim=32,num_heads=4,num_layers=1,seed=109)).eval()
    torch.testing.assert_close(actor.encode_entities(o),reference.encode_entities(o),atol=1e-6,rtol=1e-6)
    assert not any('cos' in k or 'pis' in k for k in actor.state_dict())
    assert actor(o).shape==(3,9)
    padded={k:v.clone() for k,v in o.items()}; padded['obstacles']+=100
    assert not torch.allclose(actor.encode_entities(o),actor.encode_entities(padded))
    fixed=TERLActor(32,4,1,109,masked_pool=True)
    torch.testing.assert_close(fixed.encode_entities(o),fixed.encode_entities(padded),atol=1e-5,rtol=1e-5)

@pytest.mark.parametrize('mode',['no_target','all_masked','zero_slots'])
def test_actor_empty_masks_finite_gradients(mode):
    a=NativeStage1(9); o=obs_tensor(a); actor=TERLActor(32,4,1)
    o['masks'][:,6:14]=False
    if mode=='all_masked': o['masks'][:]=False
    if mode=='zero_slots':
        o['evaders']=o['evaders'][:,:0]; o['types']=torch.cat([o['types'][:,:6],o['types'][:,14:]],1)
        o['masks']=torch.cat([o['masks'][:,:6],o['masks'][:,14:]],1)
    logits=actor(o); assert torch.isfinite(logits).all()
    assert torch.count_nonzero(actor._last_target_weights)==0
    logits.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())

def test_actor_shared_sampling_logprob_entropy_seed_and_gradients():
    o=obs_tensor(NativeStage1(9)); actor=TERLActor(32,4,1)
    i,lp,_=actor.sample(o,True); assert torch.equal(i,actor(o).argmax(-1))
    lp2,en=actor.evaluate_indices(o,i); torch.testing.assert_close(lp,lp2); assert (en>2).all()
    torch.manual_seed(3); i1=actor.sample(o)[0]; torch.manual_seed(3); i2=actor.sample(o)[0]
    assert torch.equal(i1,i2)
    (-lp2.mean()).backward()
    for module in [actor.entity_encoders['self'],actor.transformer_encoder,actor.target_selection,actor.output_layer]:
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0
    duplicate={k:v[:1].expand(3,*v.shape[1:]) for k,v in o.items()}
    torch.testing.assert_close(actor(duplicate)[0],actor(duplicate)[2])

def test_critic_permutation_equivariance_and_no_actor_leakage():
    a=NativeStage1(9); critic=make_critic(32,4,1); actor=TERLActor(32,4,1); local=obs_tensor(a)
    state={k:v[None] for k,v in central_state(a.env).items()}
    original=critic(tensor_tree(state,'cpu')); logits=actor(local).detach().clone()
    perm=[2,0,1]; a.env.pursuers=[a.env.pursuers[i] for i in perm]
    swapped={k:v[None] for k,v in central_state(a.env).items()}
    torch.testing.assert_close(critic(tensor_tree(swapped,'cpu')),original[:,perm],atol=1e-6,rtol=1e-5)
    a.env.cores[0].Gamma*=2
    assert not torch.allclose(critic(tensor_tree({k:v[None] for k,v in central_state(a.env).items()},'cpu')),original[:,perm])
    torch.testing.assert_close(actor(local),logits)

def test_gae_terminal_truncation_reset_cut_and_active():
    r=torch.tensor([[1.,999.],[2.,999.]])
    v=torch.zeros_like(r); nv=torch.tensor([[10.,10.],[20.,20.]])
    active=torch.tensor([[1,0],[1,0]],dtype=torch.bool)
    term=torch.tensor([[0,0],[1,1]],dtype=torch.bool); trunc=torch.tensor([[1,1],[0,0]],dtype=torch.bool)
    adv,ret=compute_gae(r,v,nv,term,active,gamma=.9,gae_lambda=1,truncated=trunc,episode_end=term|trunc)
    torch.testing.assert_close(ret,torch.tensor([[10.,0.],[2.,0.]]))

def test_value_norm_active_update_round_trip():
    vn=ValueNorm(); v=torch.tensor([1.,3.,999.]); vn.update(v,torch.tensor([1,1,0],dtype=torch.bool))
    torch.testing.assert_close(vn.mean,torch.tensor([2.]))
    torch.testing.assert_close(vn.denormalize(vn.normalize(v)),v)

def test_ppo_update_masked_losses_ratio_resume_rng(tmp_path):
    c=small_config(); t=build(c,'cpu'); a=NativeStage1(9)
    batch,_=collect(t,a,12)
    assert t.assert_behavior_log_probs(batch)<1e-6
    old=copy.deepcopy(t.actor.state_dict())
    metric=t.update(batch,True)
    assert all(np.isfinite(x) for x in metric.values()) and metric['minibatch_updates']>0
    assert any(not torch.equal(v,t.actor.state_dict()[k]) for k,v in old.items())
    save_checkpoint(tmp_path/'resume.pt',t,a,c,12,36,int(metric['minibatch_updates']))
    following,_=collect(t,a,8); m1=t.update(following,True)
    t2=build(c,'cpu'); a2=NativeStage1(99); data=load_checkpoint(tmp_path/'resume.pt',t2,a2,c,'cpu')
    restored,_=collect(t2,a2,8); m2=t2.update(restored,True)
    for k in following:
        if isinstance(following[k],dict):
            for j in following[k]: np.testing.assert_array_equal(following[k][j],restored[k][j])
        else: np.testing.assert_array_equal(following[k],restored[k])
    for k in t.actor.state_dict(): torch.testing.assert_close(t.actor.state_dict()[k],t2.actor.state_dict()[k],atol=0,rtol=0)
    assert m1==m2 and data['steps']==12

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_finite_update_and_rng_resume(tmp_path):
    c=small_config(); t=build(c,'cuda:0'); a=NativeStage1(9)
    b,_=collect(t,a,8); m=t.update(b,True); assert all(np.isfinite(x) for x in m.values())
    save_checkpoint(tmp_path/'cuda.pt',t,a,c,8,24,int(m['minibatch_updates']))
    next_batch,_=collect(t,a,4)
    t2=build(c,'cuda:0'); a2=NativeStage1(99); load_checkpoint(tmp_path/'cuda.pt',t2,a2,c,'cuda:0')
    replay,_=collect(t2,a2,4)
    np.testing.assert_array_equal(next_batch['actions'],replay['actions'])
    np.testing.assert_array_equal(next_batch['log_prob'],replay['log_prob'])

def test_ppo_clipping_advantage_normalization_active_weighting():
    c=small_config(); c['ppo'].update(actor_lr=0,critic_lr=0,entropy_coef=0,ppo_epochs=1,minibatches=1)
    t=build(c,'cpu'); a=NativeStage1(9); batch,_=collect(t,a,4)
    # Independent hand calculation of clipped surrogate including both adv signs.
    adv,ret=compute_gae(torch.as_tensor(batch['rewards'],dtype=torch.float32),
        torch.as_tensor(batch['values']),torch.as_tensor(batch['next_values']),
        torch.as_tensor(batch['terminated']),torch.as_tensor(batch['active_mask']),gamma=.99,gae_lambda=.95,
        truncated=torch.as_tensor(batch['truncated']),episode_end=torch.as_tensor(batch['episode_end']))
    normalized=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    altered=copy.deepcopy(batch); ratio=torch.tensor([1.6,.4,1.6,.4,1.6,.4,1.6,.4,1.6,.4,1.6,.4])
    altered['log_prob']-=torch.log(ratio).numpy().reshape(4,3)
    # Likelihood verification is tested above; inject synthetic likelihood solely
    # to exercise non-unit clipping at fixed network parameters.
    t.assert_behavior_log_probs=lambda *args,**kwargs: 0
    expected=-torch.minimum(ratio*normalized.reshape(-1),ratio.clamp(.8,1.2)*normalized.reshape(-1)).mean()
    m=t.update(altered,True)
    assert m['actor_loss']==pytest.approx(float(expected),abs=1e-6)
    assert m['clip_fraction']==1 and m['approx_kl']>0
    # Active-only losses and ValueNorm must ignore arbitrary inactive rewards.
    t1=build(c,'cpu'); t2=build(c,'cpu'); b=copy.deepcopy(batch)
    b['active_mask'][:,1]=False; b2=copy.deepcopy(b); b2['rewards'][:,1]=1e9
    torch.manual_seed(99); m1=t1.update(b,True); torch.manual_seed(99); m2=t2.update(b2,True)
    assert m1==m2
    torch.testing.assert_close(t1.value_norm.mean,t2.value_norm.mean)

def test_checkpoint_selection_and_no_false_strong_positive():
    from terl_mappo.supervise import selection_score,classify
    def point(step,norm=0,collision=1,ring2=0,ring3=0):
        m={'normal_capture_rate':norm,'normal_capture_count':round(norm*20),
           'capture_count':round(norm*20),'collision_rate':collision,'strict_geometry_rate':norm,
           'ring3_rate':ring3,'ring2_rate':ring2}
        return {'steps':step,'seed_domain':'screen','modes':{'argmax':m.copy(),'sample':m.copy()}}
    baseline=point(0); lonely=point(50000,.05)
    assert classify([baseline,lonely],lonely)=='TERL_MAPPO_STAGE1_PARTIAL'
    assert classify([baseline,point(25000)],point(25000))=='TERL_MAPPO_STAGE1_NO_CONVINCING_SIGNAL'
    strong=point(100000,.4,.2,.6,.4)
    assert classify([baseline,point(25000,.2,.3),strong],strong)=='TERL_MAPPO_STAGE1_LEARNABLE'
    assert selection_score(point(25000,.2,.3))>selection_score(point(100000,.1,.1))

@pytest.mark.parametrize('fixture',['capture','collision'])
def test_event_transition_and_reward_parity_direct_native(fixture):
    a=NativeStage1(9)
    positions=[(60+6*np.cos(x),60+6*np.sin(x)) for x in np.arange(3)*2*np.pi/3] if fixture=='capture' else [(50,50),(51,50),(80,80)]
    freeze_positions(a,positions,evader=(60,60) if fixture=='capture' else (100,100))
    raw=copy.deepcopy(a.env); eo=copy.deepcopy(a.evader_obs)
    action=a.apf.act(eo[0]); expected=raw.step(([4]*3,[action])); actual=a.step([4]*3)
    np.testing.assert_array_equal(expected[1],actual[1]); assert expected[2]==actual[5]['native_done']
    assert expected[3]==actual[5]['native_infos']
    if fixture=='collision': assert expected[2]==[True,True,False]


def test_parallel_evaluator_smoke_and_seed_domain(tmp_path):
    from terl_mappo.evaluate import evaluate
    c=small_config(); t=build(c,'cpu'); a=NativeStage1(9)
    # Only this synthetic evaluator test uses 2-step horizon; formal cfg remains3000.
    a.env.episode_max_length=1
    save_checkpoint(tmp_path/'checkpoint.pt',t,a,c,0,0,0)
    # The evaluator intentionally creates fresh native episodes, so run one full
    # seeded episode per mode instead of using checkpoint runtime state.
    r=evaluate(tmp_path/'checkpoint.pt',tmp_path/'screen.json',episodes=1,seed_base=2026100800,workers=2)
    assert r['modes']['argmax']['episodes']==r['modes']['sample']['episodes']==1
    assert r['episodes'][0]['initial_fingerprint']==r['episodes'][1]['initial_fingerprint']
    assert all(np.isfinite(m['action_entropy']) for m in r['modes'].values())

def test_post_update_ratio_telemetry_matches_zero_update():
    from terl_mappo.run import post_update_policy_stats
    c=small_config(); t=build(c,'cpu'); a=NativeStage1(9); batch,_=collect(t,a,4)
    stats=post_update_policy_stats(t,batch)
    assert stats['post_update_ratio_min']==pytest.approx(1,abs=1e-6)
    assert stats['post_update_ratio_max']==pytest.approx(1,abs=1e-6)
    assert stats['post_update_clip_fraction']==0 and stats['post_update_kl']==pytest.approx(0,abs=1e-7)

def test_emergency_friend_distance_is_post_step_not_cached():
    a=NativeStage1(9); freeze_positions(a,[(50,50),(54.2,50),(80,80)],evader=(100,100))
    p=a.env.pursuers[0]; p.speed=1.; p.theta=0.; p.update_velocity(np.zeros(2))
    assert a.env.get_distance_to_other_pursuers()[0]>4
    _,_,_,_,_,info=a.step([4]*3)
    assert a.env.get_distance_to_other_pursuers()[0]<4
    assert info['components']['emergency'][0]==-5
