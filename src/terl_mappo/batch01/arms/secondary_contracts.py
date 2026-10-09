"""Physical/probability and curriculum lifecycle assertions; all sources pinned."""
import copy
import numpy as np
import torch
from terl_mappo.native import NativeStage1,pack_local
from terl_mappo.batch01.evaluation import isolated_rng
from terl_mappo.run import rng_state
from cocap_voradj.training.small_step_ac import tensor_tree

def rejected(fn,*args):
    try:fn(*args)
    except (ValueError,TypeError):return
    raise AssertionError('expected contract rejection')

def physical(adapter):
    e=adapter.env
    return {'robots':[[r.x,r.y,r.theta,r.speed,*r.velocity,float(r.collision),float(r.deactivated)] for r in e.pursuers+e.evaders],
            'trajectory':[r.trajectory for r in e.pursuers+e.evaders], 'rng':e.rd.get_state(),'time':(e.episode_time_steps,e.total_time_steps)}

def verify_c0(runtime,device):
    from .c0 import NativeContinuous,PhysicalActionList,BOUNDS
    from .preflight import same,same_graph
    checks={}
    with isolated_rng():
        for grid in range(9):
            native=NativeStage1(9);continuous=NativeContinuous(9)
            assert native.fingerprint()==continuous.fingerprint()
            values=np.array([p.action_list[grid] for p in native.env.pursuers],float)
            for _ in range(3):
                same_graph(native.step(np.full(3,grid,np.int64)),continuous.step(values))
                same_graph(physical(native),physical(continuous))
        for collision in (False,True):
            native=NativeStage1(9);continuous=NativeContinuous(9)
            if collision:
                for a in (native,continuous):
                    p,q=a.env.pursuers[:2];q.x=p.x+.1;q.y=p.y
            else:
                native.env.episode_time_steps=continuous.env.episode_time_steps=3000
            result=native.step(np.full(3,4,np.int64));actual=continuous.step(np.zeros((3,2)))
            same_graph(result,actual);same_graph(physical(native),physical(continuous))
            assert result[4] and (result[5]['collision'] if collision else result[3].all())
        for speed,command in [(3.,np.array([.4,0.])),(0.,np.array([-.4,0.])),(1.,np.array([.137,.1158]))]:
            base=NativeStage1(9).env.pursuers[0];robot=copy.deepcopy(base);robot.action_list=PhysicalActionList(robot.action_list)
            robot.speed=speed;before=(robot.x,robot.y,robot.theta);expected=speed;theta=before[2];x,y=before[:2];current=np.array([.3,-.2])
            for _ in range(10):
                velocity=expected*np.array([np.cos(theta),np.sin(theta)])+current;x+=velocity[0]*robot.dt;y+=velocity[1]*robot.dt
                expected=np.clip(expected+(command[0]-robot.coefficient_water_resistance*expected)*robot.dt,0,robot.max_speed)
                theta=(theta+command[1]*robot.dt)%(2*np.pi);robot.update_state(command,current)
            np.testing.assert_allclose([robot.x,robot.y,robot.theta,robot.speed],[x,y,theta,expected],rtol=0,atol=1e-12)
            if speed==3.:assert robot.speed==3.
            if speed==0.:assert robot.speed==0.
        continuous=NativeContinuous(9);native=NativeStage1(9)
        continuous.step(np.broadcast_to([.137,.1158],(3,2)).copy());native.step(np.full(3,4,np.int64))
        assert continuous.fingerprint()!=native.fingerprint()
        assert np.all(np.abs(np.asarray(continuous.env.pursuers[0].action_history[-1]))<=BOUNDS)
        rejected(continuous.validate_actions,np.zeros(3,dtype=int));rejected(continuous.validate_actions,np.full((3,2),.9))
        actor=runtime.trainer.actor;obs=tensor_tree(pack_local(runtime.adapter.observations),device)
        with torch.no_grad():
            action,logp,latent=actor.sample(obs);evaluated=actor.evaluate_latent(obs,latent)[0]
            torch.testing.assert_close(torch.exp(evaluated-logp),torch.ones(3,device=device),rtol=0,atol=1e-6)
            assert torch.isfinite(logp).all() and np.all(np.abs(action.cpu().numpy())<=BOUNDS+1e-7)
            d,_=actor.distribution(obs);x=torch.tensor([[.2,-.3],[2.,-3.],[-15.,20.]],device=device,dtype=torch.float64)
            expected=d.log_prob(x).sum(-1)-(torch.log(x.new_tensor([.4,np.pi/6]))+2*(np.log(2)-x-torch.nn.functional.softplus(-2*x))).sum(-1)
            torch.testing.assert_close(actor.density(d,x),expected,rtol=0,atol=1e-12)
            safe=x[:2];sub=torch.distributions.Normal(d.loc[:2],d.scale[:2]);direct=sub.log_prob(safe).sum(-1)-torch.log(safe.new_tensor([.4,np.pi/6])*(1-safe.tanh().square())).sum(-1)
            torch.testing.assert_close(actor.density(sub,safe),direct,rtol=0,atol=1e-10)
            assert torch.isfinite(actor.density(d,x)).all()
            obs['masks'].zero_();assert torch.isfinite(actor.sample(obs)[1]).all()
    checks.update(AW9_nine_point_transition_reward_event_parity=True,native_current_drag_speed_limit_collision_timing=True,
                  offgrid_physical_integration_without_quantization=True,joint_tanh_and_scale_jacobian=True,
                  pre_tanh_latent_authoritative_saturation_finite=True,zero_update_ratio_one=True,no_target_finite=True)
    assert runtime.trainer.update_count==0 and not runtime.trainer.actor_optimizer.state and not runtime.trainer.value_optimizer.state
    checks['scratch_Adam_ValueNorm']=True
    return checks

def verify_t1(runtime,device):
    from .t1 import NativeCurriculum,state_encoder,critic_factory,qualification,ANCHOR
    from .preflight import same,same_graph
    from terl_mappo.batch01.interfaces import Runtime,Hooks,assemble
    from .t1 import env_stage3
    from terl_mappo.batch01.checkpoints import save_bound,load_bound
    from terl_mappo.batch01.provenance import runtime_manifest
    from terl_mappo.batch01.contracts import ROOT
    import json
    from pathlib import Path
    checks={}
    with isolated_rng():
        anchor=torch.load(ANCHOR,map_location='cpu',weights_only=False)
        same(anchor['trainer']['actor'],runtime.trainer.actor.state_dict())
        assert runtime.trainer.update_count==0 and not runtime.trainer.actor_optimizer.state and not runtime.trainer.value_optimizer.state
        assert runtime.adapter.stage==2
        stage1=NativeCurriculum(9,1);native=NativeStage1(9);assert stage1.fingerprint()==native.fingerprint()
        for _ in range(3):
            expected=native.step(np.full(3,4,np.int64));actual=stage1.step(np.full(7,4,np.int64))
            same(expected[0],{k:v[:3] for k,v in actual[0].items()});same(expected[1],actual[1][:3]);same(expected[2],actual[2][:3]);same(expected[3],actual[3][:3]);same(expected[4],actual[4])
            same_graph(physical(native),physical(stage1));assert not actual[1][3:].any() and actual[2][3:].all()
        for stage,counts in [(1,(3,1,0,4)),(2,(4,1,1,6)),(3,(7,2,2,8))]:
            adapter=NativeCurriculum(9,stage);state=state_encoder(adapter.env);n=counts[0]
            assert adapter.contract()['counts']==list(counts) and state['self'].shape==(7,57)
            assert state['active_mask'].sum()==n and state['pursuer_mask'].sum()==7*n
            assert state['evader_mask'].sum()==counts[1] and state['obstacle_mask'].sum()==counts[2]
            currents=state['self'][0,9:].reshape(8,6);assert currents[:,5].sum()==counts[3]
            for i,c in enumerate(sorted(adapter.env.cores,key=lambda c:(c.x,c.y))):
                np.testing.assert_allclose(currents[i],[c.x/120,c.y/120,1 if c.clockwise else -1,c.Gamma/(10*np.pi),adapter.env.vortex_core_radius/120,1],rtol=1e-6)
            if n<7:assert not state['self'][n:].any() and not state['pursuer_mask'][:,n:].any()
            with torch.no_grad():
                output=runtime.trainer.value(tensor_tree({k:v[None] for k,v in state.items()},device));assert torch.isfinite(output).all() and not output[0,n:].any()
            order=np.arange(n)[::-1];env=copy.deepcopy(adapter.env);env.pursuers=[env.pursuers[i] for i in order];permuted=state_encoder(env)
            np.testing.assert_array_equal(permuted['self'][:n],state['self'][order])
            adapter.step(np.full(7,4,np.int64));saved=adapter.state_dict();expected=adapter.step(np.full(7,4,np.int64));expected_state=adapter.state_dict()
            adapter.load_state_dict(saved);same_graph(expected,adapter.step(np.full(7,4,np.int64)));same_graph(expected_state,adapter.state_dict())
        # Actual native collision event deactivates one slot and continues with3.
        adapter=NativeCurriculum(9,2);p=adapter.env.pursuers[0];o=adapter.env.obstacles[0];p.x=o.x;p.y=o.y;p.speed=0;p.update_velocity(np.zeros(2))
        _,_,term,_,end,info=adapter.step(np.full(7,4,np.int64))
        assert term[0] and info['collision'] and not end and adapter.env.pursuers[0].deactivated
        following=adapter.step(np.full(7,4,np.int64));assert not following[5]['active'][0] and following[5]['active'].sum()==3
        assert not state_encoder(adapter.env)['active_mask'][0]
        second=adapter.env.pursuers[1];second.x=o.x;second.y=o.y;second.speed=0
        failed=adapter.step(np.full(7,4,np.int64));assert failed[4] and failed[2].all()
        # Native multi-target capture: one captured E does not end Stage3.
        adapter=NativeCurriculum(9,3);env=adapter.env;e=env.evaders[0]
        for p in env.pursuers[:3]:p.captured_evaderId_list.append(e.id)
        e.deactivated=True;env.update_pursuing_status();adapter.observations,_=env.get_pursuers_observations();adapter.evader_obs,_=env.get_evaders_observation()
        result=adapter.step(np.full(7,4,np.int64));assert not result[4] and not result[5]['capture'] and state_encoder(env)['evader_mask'].sum()==1
        np.testing.assert_allclose(sum(np.asarray(v) for v in result[5]['components'].values()),result[1],atol=1e-6)
        assert len(adapter.apf)==2
        target=env.evaders[1];target.x=60.;target.y=60.;target.speed=0.
        for i,p in enumerate(env.pursuers):
            if i<3:
                angle=i*2*np.pi/3;p.x=60+5*np.cos(angle);p.y=60+5*np.sin(angle)
            else:p.x=15+15*(i-3);p.y=100.
            p.speed=0.;p.collision=False;p.deactivated=False
        for i,o in enumerate(env.obstacles):o.x=5+110*i;o.y=5+110*i
        env.update_pursuing_status();adapter.observations,_=env.get_pursuers_observations();adapter.evader_obs,_=env.get_evaders_observation()
        completed=adapter.step(np.full(7,4,np.int64))
        assert completed[4] and completed[5]['capture'] and not completed[5]['collision']
        assert sum(completed[5]['components']['goal'])>0
        reports=[{'seed_domain':'selection','scene':s,'episodes_per_mode':20,'modes':{m:{'normal_capture_rate':.8,'collision_rate':.2} for m in ('argmax','sample')}} for s in ('4P1E1O6C','3P1E0O4C')]
        assert qualification(*reports,100000);reports[0]['modes']['sample']['normal_capture_rate']=.79;assert not qualification(*reports,100000)
        reports[0]['seed_domain']='final';rejected(qualification,*reports,100000)
        from .t1 import factory
        rejected(factory,{'stage':3,'initialization':'actor775k_critic_adams_valuenorm_fresh'})
        lock=json.loads((ROOT/'configs/experiments/terl_mappo_batch01_20261009/base_lock.json').read_text())
        delta=json.loads((ROOT/'configs/experiments/terl_mappo_batch01_20261009/t1_delta.json').read_text())
        engineering=assemble(runtime.config,device,Hooks(env_factory=env_stage3,state_encoder=state_encoder,critic_factory=critic_factory),'T1',runtime.source_guard)
        manifest=runtime_manifest(lock,delta,runtime.config,engineering,{'device':device,'max_gpu_allocated_mib':2048})
        manifest['execution_mode']='ENGINEERING_STAGE3_CORRECTNESS_ONLY_NOT_TRAINING'
        batch,_=engineering.collect(8);metrics=engineering.update(batch)
        probe=ROOT/'runs/preflight/stage3_resume.pt'
        try:
            save_bound(probe,engineering,manifest,8,int(batch['active_mask'].sum()),int(metrics['minibatch_updates']))
            expected,episodes=engineering.collect(8);expected_metrics=engineering.update(expected)
            state=copy.deepcopy(engineering.trainer.state_dict());environment=engineering.adapter.state_dict();rng=rng_state()
            actual=assemble(runtime.config,device,Hooks(env_factory=env_stage3,state_encoder=state_encoder,critic_factory=critic_factory),'T1',runtime.source_guard)
            load_bound(probe,actual,manifest);restored,restored_episodes=actual.collect(8);restored_metrics=actual.update(restored)
            same(expected,restored);same(episodes,restored_episodes);same(expected_metrics,restored_metrics)
            same(state,actual.trainer.state_dict());same_graph(environment,actual.adapter.state_dict());same(rng,rng_state())
        finally:
            probe.unlink(missing_ok=True);probe.with_suffix('.batch01.json').unlink(missing_ok=True)
    checks.update(selected775k_actor_strict_warm_fresh_critic_Adam_ValueNorm=True,fixed7P2E2O8C_capacity_masks_all_currents=True,
                  native_Stage1_transition_parity=True,multiscale_environment_APF_exact_resume=True,
                  partial_pursuer_collision_lifecycle=True,multiple_evader_reward_capture_lifecycle=True,
                  Stage3_qualification_selection_retention_only=True,Stage3_without_qualified_promotion_rejected=True,
                  cross_scale_migration_not_bit_exact=True,Stage3_full_trainer_environment_rng_resume_exact=True)
    return checks

def verify(runtime,line,device):
    return verify_c0(runtime,device) if line=='C0' else verify_t1(runtime,device)
