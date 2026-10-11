#!/usr/bin/env python3
"""Offline same-state reference fixtures; no MAPPO bridge or training claim.

Run with the verified T1 worktree on PYTHONPATH. T1 provides the actual 4P
native adapter. Fixture assignments do not change any experiment config.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from terl_mappo.batch01.arms.t1 import NativeCurriculum
from terl_mappo.native import pack_local
from tools.iqn_token_matched_20260919 import resolved, make_env
from cocap_voradj.envs.density_sensing import runtime_metadata
from cocap_voradj.training.runtime_semantics import assert_runtime
from cocap_voradj.control.apf import ApfAgent

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/yjq/rl/CoCap1/batch01-t1-long-v3r2-20261010')
OUT = ROOT/'artifacts/2026-10-11_cocap_mappo_batch02'
CONFIG = SOURCE/'configs/experiments/iqn_z_unified_decay_curriculum_20260919/z05_stage1_4p1e1obs_2m.yaml'


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value


def state(env):
    return [[r.x,r.y,r.theta,r.speed,*r.velocity,bool(r.deactivated),bool(r.collision)] for r in env.pursuers+env.evaders]


def set_robot(robot, row):
    robot.x,robot.y,robot.theta,robot.speed = map(float,row)
    robot.velocity = robot.speed*np.array([np.cos(robot.theta),np.sin(robot.theta)])
    robot.deactivated=False;robot.collision=False
    robot.start=(robot.x,robot.y)


def same_obs(first,second):
    return all(a is None and b is None or a is not None and b is not None and
        all(np.array_equal(a[k],b[k]) for k in a) for a,b in zip(first,second))


def fixture(name, root):
    native=NativeCurriculum(2056100900,stage=2)
    target,obs=make_env(root,'mixed',2056100900)
    positions=[[15,15,0,1],[35,20,0,0],[20,45,0,0],[90,90,0,0],[65,65,0,1]]
    obstacle=[80,20,1.05];step=0
    if name=='zero_current_motion':native.env.cores=[]
    if name=='far_enemy_visibility':positions[-1]=[110,110,0,0]
    if name=='boundary':positions[0]=[1.13,15,np.pi,3]
    if name=='swept_contact':
        positions[0]=[40,40,0,3];obstacle=[40.7,42.13,1.05]
        native.env.cores=[]
    if name=='capture_to_recovery':
        positions=[[66,60,0,0],[60,66,0,0],[54,60,0,0],[60,54,0,0],[60,60,0,0]]
        native.env.cores=[]
    if name=='horizon':step=2999
    if name=='active_loss':
        positions[0]=[40,40,0,3];obstacle=[43,40,1.05]
        native.env.cores=[]
    for env in (native.env,target):
        for robot,row in zip(env.pursuers+env.evaders,positions):set_robot(robot,row)
        env.obstacles[0].x,env.obstacles[0].y,env.obstacles[0].r=obstacle
    native.env.episode_time_steps=step
    native.observations=native.env.get_pursuers_observations()[0]+[None]*3
    native.evader_obs=native.env.get_evaders_observation()[0]
    before_native_obs=copy.deepcopy(native.observations[:4])
    before_mappo_obs=copy.deepcopy(pack_local(native.observations))
    target.episode_step=step
    target._invalidate_voronoi_cache();target._pursuing_flags_initialized=False
    target._pursuing_release_counters=[0]*4;target._reset_z_state()
    data=target._capture_voronoi_map()
    target.last_task_labels=target._task_labels_from_map(data,update_effective=True)
    target._advance_z_state(data)
    target_obs=target.get_observations()
    pre_native=state(native.env);pre_target=state(target)
    assert pre_native==pre_target, (name, 'physical common state not identical', pre_native, pre_target)
    old=copy.deepcopy(native.env)
    old_ea=[native.apf[i].act(native.evader_obs[i]) for i in range(1)]
    apf=ApfAgent(target.evaders[0].a,target.evaders[0].w)
    new_ea=[apf.act(target.get_evader_observations_for_apf()[0])]
    # Physics/event fixtures use identical explicit commands to isolate the
    # transition. APF decisions are separately logged at the same pre-state.
    pa=[4]*4;ea=[7]
    if name in ('boundary','swept_contact'):pa[0]=7
    no,reward,done,info=old.step((pa,ea))
    # Native adapter uses the actual APF; for exact native-wrapper parity
    # compare a second native transition with its actual old APF command.
    original_apf=copy.deepcopy(native.env)
    ao,ar,ad,ai=original_apf.step((pa,old_ea))
    mo,mr,mt,mtr,mend,mi=native.step(np.array(pa+[4]*3,dtype=np.int64))
    assert np.array_equal(ar,mr[:4]) and same_obs(ao,native.observations[:4])
    assert np.array_equal(np.array(state(original_apf))[:,:6],np.array(state(native.env))[:,:6])
    tr=target.step(pa,ea)
    next_native=state(old);next_target=state(target)
    position_delta=float(np.max(np.abs(np.array(next_native)[:,:2]-np.array(next_target)[:,:2])))
    passed={
        'same_input_physical_state': True,
        'native_vs_current_mappo_raw_reward_observation_physics': True,
        'native_observation_slots19_target22':len(before_native_obs[0]['masks'])==19 and len(target_obs[0]['masks'])==22,
        'reward_difference_expected':not np.allclose(reward,tr.rewards),
    }
    if name=='zero_current_motion':passed['zero_current_robot_dynamics_equal']=position_delta<1e-12
    if name=='ordinary':passed['native_current_target_zero_current_difference']=position_delta>1e-8
    if name=='far_enemy_visibility':passed['native_enemy_unfiltered_target_local']=bool(native.observations[0]['masks'][6] and not target_obs[0]['masks'][9])
    if name=='boundary':passed['target_boundary_death_native_survives']=bool(target.pursuers[0].deactivated and not old.pursuers[0].deactivated)
    if name=='swept_contact':passed['target_swept_contact_native_endpoint_misses']=bool(target.pursuers[0].collision and not old.pursuers[0].collision)
    if name=='capture_to_recovery':
        passed['native_capture_terminal_target_recovery']=bool(mend and mi['capture'] and not any(tr.dones) and target.evaders[0].deactivated)
    if name=='horizon':passed['native3001_target3000']=bool(not mend and all(tr.dones) and all(i['truncated'] for i in tr.infos))
    if name=='active_loss':passed['native3_survivors_target_requires4']=bool(not mend and all(tr.dones))
    return plain({'name':name,'checks':passed,'input':{'robot_columns':['x','y','theta','speed','vx','vy','deactivated','collision'],
        'robots':pre_native,'obstacle':obstacle,'episode_step':step,'pursuer_actions':pa,'evader_actions':ea,
        'native_cores':[[c.x,c.y,c.Gamma,c.clockwise] for c in old.cores],
        'fixture_only_zero_current':name in ('zero_current_motion','capture_to_recovery','swept_contact','active_loss')},
        'apf_same_state':{'terl_action':old_ea,'cocap_v2_action':new_ea,'transition_commands_explicit':True},
        'before_observation':{'terl':before_native_obs,'current_mappo':before_mappo_obs,'cocap':target_obs},
        'terl':{'observation':no,'reward':reward,'done_raw':done,'infos':info,'next_state':next_native},
        'current_mappo_native_apf':{'observation':mo,'reward':mr,'terminated':mt,'truncated':mtr,'episode_end':mend,'info':mi,
            'next_state':state(native.env)},
        'cocap_reference':{'observation':tr.observations,'reward':tr.rewards,'done':tr.dones,
            'terminated':[i['terminated'] for i in tr.infos],'truncated':[i['truncated'] for i in tr.infos],
            'reward_metadata':[i.get('replay_metadata',{}) for i in tr.infos],'next_state':next_target,
            'post_capture_started':target.post_capture_started,'post_capture_step':target.post_capture_step,
            'runtime_sensing':runtime_metadata(target)},'max_position_delta':position_delta})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    root=resolved(CONFIG)
    rows=[fixture(name,root) for name in ['ordinary','zero_current_motion','far_enemy_visibility','boundary',
                                       'swept_contact','capture_to_recovery','horizon','active_loss']]
    result={'schema':'cocap.batch02.reference-transitions.v1','classification':'COMMANDER_OFFLINE_REFERENCE_DIAGNOSTIC',
        'E1_bridge_implemented':False,'E1_pass':False,'independent_validation':False,
        'source_root':str(SOURCE),'source_config':str(CONFIG),
        'source_files':{str(p):hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in
            ['src/terl_mappo/batch01/arms/t1.py','src/terl_mappo/native.py','src/cocap_voradj/envs/base.py',
             'src/cocap_voradj/envs/voronoi_adjacency.py','src/cocap_voradj/envs/density_sensing.py',
             'src/cocap_voradj/control/apf.py','src/cocap_voradj/dynamics/robot.py',
             'src/cocap_voradj/training/trainer.py','tools/iqn_token_matched_20260919.py',
             'vendor/terl/environment/env.py','vendor/terl/robots/robot.py',
             'vendor/terl/robots/pursuer.py','vendor/terl/thirdparty/APF.py']},
        'checks_passed':all(all(row['checks'].values()) for row in rows),'fixtures':rows}
    (OUT/'reference_transitions.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (OUT/'iqn_z05_reference_resolved.json').write_text(json.dumps(root,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'checks_passed':result['checks_passed'],'fixtures':[{r['name']:r['checks']} for r in rows],
                      'E1_pass':False,'independent_validation':False}))


if __name__=='__main__':main()
