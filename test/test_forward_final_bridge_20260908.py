import copy
import numpy as np
import pytest
from cocap_voradj.training.forward_final import preflight,make_env,check_env
from tools.run_forward_final_bridge_20260908 import c0_gate,timing


def test_only_forward_collision_changes_resolved_and_live_contract():
    report=preflight()
    for row in report['scenes'].values():
        assert row['resolved_changed']==['env.collision_semantics']
        assert row['runtime_changed']==['collision_semantics']


def test_wrong_reward_and_information_settings_fail_runtime():
    env,_=make_env('mixed',17)
    env.config['voradj']['support_reward_capture_weight']=1
    with pytest.raises(ValueError,match='support_capture_weight'):check_env(env)
    env.config['voradj']['support_reward_capture_weight']=.5
    env.config['voradj']['perception_topology_version']='legacy_voradj'
    with pytest.raises(ValueError,match='topology'):check_env(env)


def test_full_task_capture_continues_with_ce_and_participant_reward():
    env,_=make_env('mixed',19)
    for p,t in zip(env.pursuers[:3],[0,2*np.pi/3,4*np.pi/3]):
        p.x,p.y=50+7*np.cos(t),50+7*np.sin(t);p.speed=0.;p.velocity=np.zeros(2)
    env.pursuers[3].x,env.pursuers[3].y=10.,10.
    env.evaders[0].x,env.evaders[0].y=50.,50.;env.evaders[0].speed=0.;env.evaders[0].velocity=np.zeros(2)
    env.obstacles[0].x,env.obstacles[0].y=90.,90.
    env._invalidate_voronoi_cache()
    out=env.step([4]*4,[4])
    assert env.last_capture_events and not any(out.dones)
    assert sum(i['replay_metadata']['reward_terminal']>0 for i in out.infos)==3
    after=env.step([4]*4,[None])
    assert any(i['replay_metadata']['reward_coverage']!=0 for i in after.infos)


def test_c0_gate_cannot_pass_on_capture_alone_or_smoke_counts():
    s={name:{'episodes':20,'captured_rate':1.,'normal_capture_rate':1.,'ce_success_rate':1.,'safe_complete_rate':1.,'collision_rate':0.} for name in ('mixed','coverage')}
    assert c0_gate(s)=='PASS'
    s['mixed']['safe_complete_rate']=.5
    assert c0_gate(s)=='STOP_TEACHER_UNQUALIFIED'
    s['mixed']['safe_complete_rate']=.85
    assert c0_gate(s)=='INCONCLUSIVE_EXTEND_TO_50'
    s['mixed']['episodes']=2
    assert c0_gate(s)=='INSUFFICIENT_SAMPLE'
    assert timing([])['mean'] is None


def test_dataset_rows_keep_pre_action_phases_and_index_global_transitions(tmp_path):
    from cocap_voradj.models.iqn import CoCapIQN
    from cocap_voradj.training.forward_final import TEACHER
    from tools.run_forward_final_bridge_20260908 import run_episode
    from tools.collect_forward_final_dataset_20260908 import EpisodeRows,distribution_audit
    teacher=CoCapIQN.load(str(TEACHER),device='cpu').eval()
    storage=EpisodeRows(0)
    run_episode(teacher,'coverage',17,'cpu',max_steps=2,on_transition=storage)
    arrays=storage.arrays(True)
    assert len(arrays['teacher_q'])==8
    assert len(arrays['global_state.active_mask'])==2
    assert arrays['transition_index'].tolist()==[0]*4+[1]*4
    assert (arrays['phase_id']==2).all() and (arrays['target_id']==-1).all()
    assert not arrays['direct'].any()
    np.savez_compressed(tmp_path/'one.npz',**arrays)
    audit=distribution_audit([{'path':'one.npz'}],tmp_path)
    assert audit['status']=='STOP_ADJUST_COLLECTION'
    assert not audit['checks']['has_capture_transitions']


def test_distillation_split_calibration_and_frozen_feature_contract():
    import torch
    from cocap_voradj.models.iqn import CoCapIQN
    from cocap_voradj.training.forward_final import TEACHER
    from cocap_voradj.training.small_step_ac import tensor_tree
    from tools.distill_forward_final_actor_20260908 import split_pairs,calibrate_temperature,make_actor,assert_backbone
    episodes=np.repeat(np.arange(20),3);train,valid=split_pairs(episodes)
    assert set(episodes[valid])=={0,1,10,11}
    rng=np.random.default_rng(12);q=rng.normal(size=(60,9)).astype(np.float32);phase=np.tile(np.arange(3),20)
    t,c=calibrate_temperature(q,phase,train,1)
    changed=q.copy();changed[valid]*=10000
    assert calibrate_temperature(changed,phase,train,1)[0]==t
    assert abs(c['actual_entropy_nats']-.35)<1e-8
    teacher=CoCapIQN.load(str(TEACHER),device='cpu').eval();actor=make_actor(teacher,'cpu')
    env,obs=make_env('mixed',12)
    batch=tensor_tree({k:np.stack([o[k] for o in obs]) for k in obs[0]},torch.device('cpu'))
    with torch.no_grad():
        feature=actor.encoder(batch)
        torch.testing.assert_close(feature,teacher.voradj_single_feature(batch,teacher.features(batch)))
        old=actor.distribution(batch);actions=old.sample();old_logp=old.log_prob(actions)
        assert torch.equal(old_logp,actor.distribution(batch).log_prob(actions))
    loss=-actor.distribution(batch).log_prob(actions).mean()
    optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=3e-4)
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert assert_backbone(actor,teacher)==86


def test_frozen_actor_modes_preserve_initial_states_and_repeat_sampling():
    from cocap_voradj.models.iqn import CoCapIQN
    from cocap_voradj.training.forward_final import TEACHER
    from tools.distill_forward_final_actor_20260908 import make_actor
    from tools.run_forward_final_bridge_20260908 import run_episode
    teacher=CoCapIQN.load(str(TEACHER),device='cpu').eval();actor=make_actor(teacher,'cpu')
    base=run_episode(teacher,'mixed',31,'cpu',max_steps=2)
    argmax=run_episode(teacher,'mixed',31,'cpu',max_steps=2,actor=actor,policy_mode='bc_argmax')
    sample=run_episode(teacher,'mixed',31,'cpu',max_steps=2,actor=actor,policy_mode='bc_sample')
    repeated=run_episode(teacher,'mixed',31,'cpu',max_steps=2,actor=actor,policy_mode='bc_sample')
    assert base['initial_state_fingerprint']==argmax['initial_state_fingerprint']==sample['initial_state_fingerprint']
    assert sample==repeated
    assert sample['policy_diagnostics']['rows']==8
    assert base['policy_diagnostics']['agreement_hits']==8


def test_c3_gate_requires_reliability_and_allows_sample_efficiency_gap():
    from tools.evaluate_forward_final_actor_20260908 import decide,MODES
    scene={'episodes':100,'captured_rate':1.,'safe_complete_rate':1.,'ce_success_rate':1.,'collision_rate':0.,'full_mission_seconds_completed':{'p90':100.}}
    reports={m:{'summary':{s:copy.deepcopy(scene) for s in ('mixed','coverage')}} for m in MODES}
    assert decide(reports)=='PASS'
    reports['bc_sample']['summary']['mixed']['full_mission_seconds_completed']['p90']=180
    assert decide(reports)=='PASS_WITH_STOCHASTIC_EFFICIENCY_GAP'
    reports['bc_sample']['summary']['mixed']['safe_complete_rate']=.8
    assert decide(reports)=='STOP_REPAIR_DISTILLATION_RELIABILITY'
