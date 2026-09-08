import copy
import numpy as np
import pytest
import torch
from test_mappo_v2_contract import _local_obs, _global_obs
from cocap_voradj.models.small_step_ac import CategoricalGridActor, CentralValueNetwork
from cocap_voradj.models.continuous.local_entity_token_encoder import LegacyVorAdjFeatureBackbone, LegacyVorAdjFeatureBackboneConfig
from cocap_voradj.training.small_step_ac import MAPPOTrainer, MAPPOConfig
from cocap_voradj.training.runtime_semantics import assert_config_namespaces, assert_only_changed, assert_paired_records
from cocap_voradj.evaluation.mission_events import MissionEventTracker


def trainer_batch():
    torch.manual_seed(17)
    actor = CategoricalGridActor(LegacyVorAdjFeatureBackbone(LegacyVorAdjFeatureBackboneConfig(hidden_dim=16, num_heads=4, num_layers=1, max_pursuers=2,max_evaders=1,max_obstacles=1,dropout=.5)),np.zeros((9,2)))
    value = CentralValueNetwork(hidden_dim=16,num_heads=4,num_layers=1,max_agents=2,max_evaders=1,max_obstacles=1)
    trainer = MAPPOTrainer(actor,value,MAPPOConfig(ppo_epochs=2,minibatches=2),"cpu")
    obs = _local_obs(8); glob = _global_obs(4)
    with torch.no_grad():
        _, lp, a = actor.sample(obs); v = value(glob)
    batch = {"local_obs":{k:x.reshape(4,2,*x.shape[1:]) for k,x in obs.items()},"global_obs":glob,"actions":torch.zeros(4,2,2),"latent":a.reshape(4,2),"log_prob":lp.reshape(4,2),"values":v,"next_values":v,"active_mask":torch.ones(4,2,dtype=torch.bool),"rewards":torch.arange(8).reshape(4,2).float(),"terminated":torch.zeros(4,2,dtype=torch.bool),"episode_end":torch.zeros(4,2,dtype=torch.bool)}
    return trainer,batch


def test_zero_update_ratio_and_eval_mode_still_allow_learning():
    trainer,batch=trainer_batch()
    assert trainer.assert_behavior_log_probs(batch)<1e-5
    before={k:v.clone() for k,v in trainer.actor.state_dict().items()}
    trainer.actor.train()  # caller cannot accidentally re-enable PPO dropout
    metrics=trainer.update(batch,categorical=True)
    assert not trainer.actor.training
    assert metrics['actor_update_l2']>0
    assert any(not torch.equal(before[k],v) for k,v in trainer.actor.state_dict().items())


def test_stale_likelihood_fails_before_any_update_and_inactive_is_masked():
    trainer,batch=trainer_batch();batch['log_prob'][0,0]+=1
    before={k:v.clone() for k,v in trainer.actor.state_dict().items()}
    with pytest.raises(ValueError,match='zero-update'):trainer.update(batch,categorical=True)
    assert all(torch.equal(before[k],v) for k,v in trainer.actor.state_dict().items())
    batch['active_mask'][0,0]=False
    trainer.assert_behavior_log_probs(batch)


def state(targets, direct=None, friends=None, active=(0,1,2,3)):
    return {'targets':{j:{'participants':set(p),'geometric':g} for j,(p,g) in targets.items()},'active_agents':set(active),'direct':direct or {},'friends':friends or {}}


def test_two_targets_never_create_cross_target_closure():
    t=MissionEventTracker(.5)
    t.observe(state({0:([0,1],False),1:([],False)}),1)
    t.observe(state({0:([0,1],False),1:([1,2,3],True)}),4)
    rows=t.finish(7)['events']
    a=[r for r in rows if r['target_id']==0]
    assert len(a)==2 and all(r['outcome']=='episode_end' for r in a)
    assert all(r['duration_seconds']==3 for r in a)


def test_capture_event_geometry_survives_target_deactivation():
    t=MissionEventTracker(.5);t.observe(state({0:([0,1],False)}),2)
    t.observe(state({}),5,[{'evader_id':0,'participants':[0,1,2],'capture_type':'loose'}])
    assert all(r['outcome']=='completed' and r['duration_steps']==3 for r in t.finish(5)['events'])


def test_support_repeated_events_keep_failed_and_unfinished_requests():
    t=MissionEventTracker(.5)
    s=lambda direct:state({0:([],False)},direct,{0:{1}})
    t.observe(s({1:{0}}),1)
    t.observe(s({0:{0},1:{0}}),3)
    t.observe(s({}),4)
    t.observe(s({1:{0}}),6)
    rows=t.finish(8)['events'];direct=[e for e in rows if e['metric']=='support_to_direct']
    assert [e['outcome'] for e in direct]==['completed','episode_end']
    assert t.finish(8)['summary']['support_to_direct']['completion_rate']==.5


def test_wrong_namespace_and_runtime_noop_are_rejected():
    with pytest.raises(ValueError,match='namespace'):assert_config_namespaces({'tasks':{'capture':{'reward':{'support_reward_capture_weight':1}}}})
    with pytest.raises(ValueError,match='NOOP'):assert_only_changed({'weight':.5},{'weight':.5},['weight'])
    assert assert_only_changed({'weight':.5},{'weight':1},['weight'])==['weight']


def test_pairing_requires_actual_seed_and_initial_state():
    a=[{'seed':1,'initial_state_fingerprint':'a'}]
    with pytest.raises(ValueError,match='seed'):assert_paired_records(a,[{'seed':10001,'initial_state_fingerprint':'a'}])
    with pytest.raises(ValueError,match='fingerprint'):assert_paired_records(a,[{'seed':1}])
    assert_paired_records(a,a)


def test_information_dependencies_and_runtime_expectations():
    from tools.audit_p0_information_20260908 import configs, setup, refresh
    from cocap_voradj.training.runtime_semantics import assert_runtime
    for name,cfg in configs().items():
        env=setup(cfg);before=refresh(env)
        env.evaders[0].x=48.;after=refresh(env)
        identical=all(np.array_equal(a[k],b[k]) for a,b in zip(before,after) for k in a)
        assert identical == (name == 'final')
        if name == 'final':
            env.evaders[0].x=42.;refresh(env)
            env.obstacles[0].x,env.obstacles[0].y=34.,30.
            changed=refresh(env)
            assert not np.array_equal(before[0]['self'][6:8],changed[0]['self'][6:8])
            with pytest.raises(ValueError,match='support_capture_weight'):
                assert_runtime(env,expected={'support_capture_weight':1.})


def test_snapshot_uses_actual_capture_region_and_angle_criterion():
    from tools.audit_p0_information_20260908 import configs, setup, refresh
    from cocap_voradj.evaluation.mission_events import snapshot
    env=setup(configs()['final']);env.evaders[0].x=40.;env.evaders[0].y=40.
    for p,angle,radius in zip(env.pursuers,[0,2*np.pi/3,4*np.pi/3,np.pi],[8,8,8,9]):
        p.x=40+radius*np.cos(angle);p.y=40+radius*np.sin(angle)
    data=snapshot(env,refresh(env))['targets'][0]
    actual=env._loose_capture_events()
    assert data['geometric'] and actual
    assert data['participants']==set(actual[0]['participants'])=={0,1,2}
    env.pursuers[2].x=49.;env.pursuers[2].y=40.
    data=snapshot(env,refresh(env))['targets'][0]
    assert len(data['participants'])==2 and not data['geometric']


def test_legacy_ppo_resume_rejected_before_loading_weights(tmp_path):
    from tools.run_small_step_ac_migration import restore
    trainer,_=trainer_batch();path=tmp_path/'legacy.pt';torch.save({},path)
    with pytest.raises(ValueError,match='incompatible dropout'):
        restore(path,trainer,{},'mappo9_v2')
