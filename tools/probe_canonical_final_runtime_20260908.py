#!/usr/bin/env python3
"""CPU probe importing only the requested checkout's CoCap runtime."""
import argparse, copy, hashlib, json, subprocess, sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--repo-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.repo_root/'src'))
import numpy as np
import torch
from cocap_voradj.training.trainer import load_config, deep_update, set_global_config
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.models.iqn import CoCapIQN
import cocap_voradj.envs.voronoi_adjacency as implementation

torch.set_num_threads(4)

def refresh(env):
    env._invalidate_voronoi_cache()
    env._update_effective_pursuing_flags(env._raw_task_labels_from_map(env._capture_voronoi_map()))
    return env.get_observations()

def env_for(config):
    set_global_config(config)
    env=VorAdjEnv(copy.deepcopy(config),seed=2026090301);env.reset()
    return env

def setup(config):
    env=env_for(config)
    for p,xy in zip(env.pursuers,[(10,10),(10,70),(70,10),(70,70)]):
        p.x,p.y=xy;p.theta=0.;p.speed=0.;p.velocity=np.zeros(2);p.is_pursuing=False
    e=env.evaders[0];e.x,e.y=42.,42.;e.speed=0.;e.velocity=np.zeros(2)
    env.obstacles[0].x,env.obstacles[0].y,env.obstacles[0].r=32.,32.,3.
    env._pursuing_release_counters=[0]*4
    return env

def digest_obs(obs):
    return hashlib.sha256(b''.join(np.asarray(o[k]).tobytes() for o in obs if o is not None for k in sorted(o))).hexdigest()

def q(model,obs):
    with torch.no_grad():
        batch={k:torch.as_tensor(np.asarray(v))[None] for k,v in obs.items()}
        return model(batch,num_tau=32,mode='voradj',tau=(torch.arange(32)+.5)/32)['q_values'].mean(1).numpy()

result={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo_root,text=True).strip(),'imported_runtime':implementation.__file__,'profiles':{}}
for name,file in [('final4','stage1_4p1e1obs_scratch2m.yaml'),('final8','stage2_8p2e2obs_700k.yaml'),('final12','stage3_12p3e3obs_700k.yaml')]:
    root=load_config(str(args.repo_root/'configs/experiments/cr_ms_support_approach_ce_curriculum_20260802'/file));cfg=deep_update(root,root['tasks']['voradj']);env=env_for(cfg)
    result['profiles'][name]={'resolved':cfg,'actual':{'bounds':env._bounds(),'horizon':env.episode_max_length,'enemy_radius':env._vct_ls_sensing_radius('enemy'),'obstacle_radius':env._vct_ls_sensing_radius('obstacle'),'support_weights':env._vct_ls_support_reward_weights(),'support_enabled':env._vct_ls_support_reward_blend_enabled(),'capture_mode':env._capture_reward_mode(),'ce_enabled':env._ce_coverage_enabled(),'capture_sites':env._capture_voronoi_map()['keys'],'coverage_sites':env._coverage_voronoi_map()['keys'],'action_grid':env.pursuers[0].action_list,'dt':env.pursuers[0].dt,'substeps':env.pursuers[0].N}}
    if name!='final4':continue
    model=CoCapIQN.load(str(args.repo_root/'artifacts/2026-08-04_crms_vctls_ce_final/checkpoints/stage1_4v1_step_2000000.pt'),device='cpu').eval()
    hidden=setup(cfg);before=refresh(hidden);before_q=q(model,before[0]);hidden.evaders[0].x=48.;after=refresh(hidden)
    hidden_result={'token_delta':max(float(np.max(np.abs(np.asarray(a[k],float)-np.asarray(b[k],float)))) for a,b in zip(before,after) for k in a),'iqn_q_delta':float(np.max(np.abs(before_q-q(model,after[0]))))}
    assert hidden_result['token_delta']==0 and hidden_result['iqn_q_delta']==0
    hidden.evaders[0].x,hidden.evaders[0].y=10.,55.;neighbor_seen=refresh(hidden)
    hidden_result['focal_enemy_tokens_when_neighbor_sees']=int(np.asarray(neighbor_seen[0]['masks'])[np.asarray(neighbor_seen[0]['types'])==2].sum())
    hidden_result['focal_friend_role_change']=float(np.max(np.abs(neighbor_seen[0]['pursuers'][:,-1]-before[0]['pursuers'][:,-1])))
    result['information_probe']=hidden_result
    probes={}
    for terminal in (False,True):
        e=setup(cfg);positions=[(40,50),(50,40),(60,50),(10,10)]
        if terminal:positions=[(50+7*np.cos(t),50+7*np.sin(t)) for t in (0,2*np.pi/3,4*np.pi/3)]+[(10,10)]
        for p,xy in zip(e.pursuers,positions):p.x,p.y=map(float,xy);p.speed=0. if terminal else 1.;p.velocity=np.array([p.speed,0.])
        e.evaders[0].x,e.evaders[0].y=50.,50.;refresh(e);o=e.step([4]*4,[4])
        probes['terminal' if terminal else 'dense']={'dones':o.dones,'reward':o.rewards.tolist(),'components':[{k:i['replay_metadata'].get(k) for k in ['reward_capture','reward_coverage','reward_terminal','reward_ce_pbrs','reward_ce_terminal_correction','support_reward_blend_active']} for i in o.infos]}
    result['reward_probes']=probes
    trace=[];rng=np.random.RandomState(17);e=env_for(cfg)
    for t in range(20):
        o=e.step(rng.randint(0,9,size=4).tolist(),[int(rng.randint(0,15))])
        trace.append({'positions':[[p.x,p.y,p.theta,p.speed] for p in e.pursuers+e.evaders],'reward':o.rewards.tolist(),'dones':o.dones,'obs_hash':digest_obs(o.observations)})
        if all(o.dones):break
    result['scripted_trace']=trace
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','commit':result['commit'],'information_probe':result['information_probe']}))
