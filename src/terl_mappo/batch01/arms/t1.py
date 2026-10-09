"""Native accelerated curriculum, fixed7P/2E/2O/8C capacity and masks."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from terl_mappo.native import NativeStage1,MarineEnv,initialize_native,pack_local
from thirdparty.APF import ApfAgent
from terl_mappo.batch01.interfaces import Hooks,ActionCapabilities
from cocap_voradj.models.small_step_ac import CentralValueNetwork
from terl_mappo.run import file_hash

CAPACITY=(7,2,2,8)
STAGES={1:(3,1,0,4),2:(4,1,1,6),3:(7,2,2,8)}
ANCHOR='/home/yjq/rl/CoCap1/terl-backbone-mappo-20261008/runs/terl_mappo_stage1_seed9_1m_continuation/checkpoints/step_000775000.pt'
ANCHOR_HASH='590d486876d4fca1cd75311c2dcc1255f847a1f756d91eed202e737193b58db4'

def geometry(env):
    counts=[];stricts=[]
    for e in env.evaders:
        near=[p for p in env.pursuers if not p.deactivated and np.hypot(p.x-e.x,p.y-e.y)<8]
        count=len(near);strict=False
        if count>=3:
            angles=sorted(np.arctan2(p.y-e.y,p.x-e.x)%(2*np.pi) for p in near)
            gaps=np.diff(angles+[angles[0]+2*np.pi]);strict=bool(gaps.max()<=np.pi and gaps.max()<=3*gaps.min())
        counts.append(count);stricts.append(strict)
    return max(counts,default=0),any(stricts)

def reward_components(env,before_active,rewards,infos):
    n=len(env.pursuers);dense=np.zeros(n)
    for i,p in enumerate(env.pursuers):
        if not before_active[i]:continue
        distances=[np.hypot(p.x-e.x,p.y-e.y) for e in env.evaders]
        if not distances:continue
        dense[i]=sum(env.distance_reward if d<=8 else 5*np.exp(env.decay_factor*(min(distances)-8)) for d in distances)
    global_reward=env.global_reward()
    collision=np.array([env.collision_penalty if x['state']=='collision' else 0 for x in infos])
    goal=np.array([rewards[i]-env.timestep_penalty-dense[i]-global_reward[i]-collision[i] if '✌️capture' in infos[i]['state'] else 0 for i in range(n)])
    return dense,global_reward,collision,goal

class NativeCurriculum(NativeStage1):
    def __init__(self,seed,stage=2):
        if stage not in STAGES:raise ValueError('unregistered native stage')
        self.stage=stage;cfg=initialize_native();schedule={k:[v[stage-1]] for k,v in cfg['training_schedule'].items()}
        schedule['time_steps']=[0]
        self.env=MarineEnv(seed=int(seed),schedule=schedule)
        self.reset();self.apf=[ApfAgent(e.a,e.w) for e in self.env.evaders];self.assert_contract()
    def reset(self):
        (obs,_),_=self.env.reset();self.observations=obs+[None]*(7-len(obs))
        self.evader_obs,_=self.env.get_evaders_observation();self.return_sum=np.zeros(len(self.env.pursuers))
        return pack_local(self.observations)
    def native_contract(self):
        e=self.env; p=e.pursuers[0]; q=e.evaders[0]
        return {
            "counts": [len(e.pursuers),len(e.evaders),len(e.obstacles),len(e.cores)],
            "map": [e.width,e.height],"spawn_min_pe":e.min_pursuer_evader_init_dis,
            "dt":p.dt,"integration_substeps":p.N,"decision_seconds":p.dt*p.N,
            "pursuer_max_speed":p.max_speed,"evader_max_speed":q.max_speed,
            "aw9":p.action_list,"capture_distance":e.capture_distance,"related_distance":e.related_distance,
            "goal_reward":e.goal_reward,"distance_reward":e.distance_reward,"timestep_penalty":e.timestep_penalty,
            "emergency_penalty":e.emergency_penalty,"collision_penalty":e.collision_penalty,
            "boundary_penalty":e.boundary_penalty,"decay_factor":e.decay_factor,
            "evenly_distributed_reward":e.evenly_distributed_reward,"unevenly_distributed_reward":e.unevenly_distributed_reward,
            "perception_range":p.perception.range,"token_count":len(pack_local(self.observations)["masks"][0]),
            "horizon_decisions":e.episode_max_length+1,"episode_reset_min_active":3,
            "collision_semantics":"native_end_of_decision","enemy_distance_filter":False}

    def contract(self):
        c=self.native_contract();c.update(stage=self.stage,counts=list(STAGES[self.stage]),actor_capacity=7,critic_capacity=list(CAPACITY),
             current_features=['x/120','y/120','clockwise_sign','Gamma/(10*pi)','global_vortex_core_radius/120','presence'],
             lifecycle='pursuer collision deactivation; continue while>=3 active; all-target capture or<3 active ends; padding inactive',
             migration='warm transfer across scales; never bit-exact stage transition')
        return c
    def assert_contract(self):
        e=self.env
        if (len(e.pursuers),len(e.evaders),len(e.obstacles),len(e.cores))!=STAGES[self.stage]:raise ValueError('native stage counts not applied')
        if e.min_pursuer_evader_init_dis!=[10.,13.,15.][self.stage-1]:raise ValueError('native spawn separation not applied')
        if any(p.dt!=.05 or p.N!=10 or p.max_speed!=3. for p in e.pursuers):raise ValueError('native AW physics changed')
        if pack_local(self.observations)['self'].shape!=(7,4):raise ValueError('fixed local actor slots violated')
    def action_capabilities(self):return ActionCapabilities('categorical_aw9','integer_indices')
    def validate_actions(self,actions):
        value=np.asarray(actions)
        if value.shape!=(7,) or not np.issubdtype(value.dtype,np.integer) or np.any((value<0)|(value>8)):raise ValueError('T1 fixed7 integer AW9 indices')
        return value
    def pad_transition(self,reward,terminated,truncated,end,info):
        n=len(self.env.pursuers)
        self.observations=self.observations[:n]+[None]*(7-n)
        info['active']=np.pad(info['active'],(0,7-n),constant_values=False)
        info['components']={k:np.pad(v,(0,7-n)).tolist() for k,v in info['components'].items()}
        info['episode_return']=np.pad(info['episode_return'],(0,7-n)).tolist()
        return pack_local(self.observations),np.pad(reward,(0,7-n)),np.pad(terminated,(0,7-n),constant_values=True),np.pad(truncated,(0,7-n),constant_values=False),end,info

    def step(self, indices):
        self.validate_actions(indices)
        indices = np.asarray(indices)[:len(self.env.pursuers)]
        env = self.env
        active = np.array([not p.deactivated for p in env.pursuers])
        # These helpers use cached pre-step perception in the native reward.
        stale_od = env.get_distance_to_obstacles()
        ea = [None if e.deactivated else self.apf[i].act(self.evader_obs[i]) for i,e in enumerate(env.evaders)]
        obs, reward, done, info = env.step(([int(a) if active[i] else None for i,a in enumerate(indices)], ea))
        self.observations = obs
        self.evader_obs, _ = env.get_evaders_observation()
        ring, strict = geometry(env)
        live_pd = env.get_distance_to_other_pursuers()
        dense, glob, collision, goal_residual = reward_components(env, active, reward, info)
        emergency = np.array([env.emergency_penalty if active[i] and
            (min(np.hypot(p.x-e.x,p.y-e.y) for e in env.evaders)<4 or live_pd[i]<4 or stale_od[i]<4)
            else 0 for i,p in enumerate(env.pursuers)])
        goal = goal_residual - np.where(goal_residual!=0, emergency, 0)
        components = {'time': active.astype(float)*env.timestep_penalty, 'distance': dense,
                      'global': glob, 'emergency': emergency, 'collision': collision, 'goal': goal}
        if not np.allclose(sum(components.values()), reward, atol=1e-6):
            raise ValueError('native reward component reconstruction disagrees with effective reward')
        capture = env.check_all_evader_is_captured()
        colliding = np.array([p.collision for p in env.pursuers])
        collision_types = []
        for i,p in enumerate(env.pursuers):
            if not colliding[i] or not active[i]: continue
            if any(np.hypot(p.x-q.x,p.y-q.y)<=p.r+q.r for q in env.pursuers if q is not p and not q.deactivated):
                collision_types.append('pursuer_pursuer')
            if any(np.hypot(p.x-e.x,p.y-e.y)<=p.r+e.r for e in env.evaders):
                collision_types.append('pursuer_evader')
            if any(np.hypot(p.x-o.x,p.y-o.y)<=p.r+o.r for o in env.obstacles):
                collision_types.append('pursuer_obstacle')
            if not collision_types: collision_types.append('cached_collision')
        for p in env.pursuers:
            if p.collision: p.deactivated = True
        failure = sum(not p.deactivated for p in env.pursuers)<3
        terminal = capture or failure
        # Original trainer's ep_length starts at 0 and checks >=3000 AFTER step.
        truncation = env.episode_time_steps>env.episode_max_length and not terminal
        end = terminal or truncation
        terminated = colliding | terminal
        truncated = np.full(len(env.pursuers), truncation) & ~terminated
        self.return_sum += reward * active
        return self.pad_transition(reward, terminated, truncated, end, {
            'capture': bool(capture), 'normal_capture': bool(capture and not colliding.any()),
            'collision': bool(colliding.any()), 'collision_types': collision_types,
            'ring2': ring>=2, 'ring3': ring>=3, 'strict_geometry': strict,
            'components': {k:v.tolist() for k,v in components.items()},
            'native_done': done, 'native_infos': info, 'active': active,
            'episode_steps': env.episode_time_steps, 'episode_return': self.return_sum.tolist()})

def state_encoder(env):
    n=len(env.pursuers)
    if n>7 or len(env.evaders)>2 or len(env.obstacles)>2 or len(env.cores)>8:raise ValueError('T1 fixed capacity overflow')
    active=np.zeros(7,bool);active[:n]=[not p.deactivated for p in env.pursuers]
    currents=np.zeros((8,6),np.float32)
    for i,c in enumerate(sorted(env.cores,key=lambda c:(c.x,c.y))):
        currents[i]=[c.x/120,c.y/120,1 if c.clockwise else -1,c.Gamma/(10*np.pi),env.vortex_core_radius/120,1]
    focal=np.zeros((7,57),np.float32);pfeat=np.zeros((7,7),np.float32)
    for i,p in enumerate(env.pursuers):
        focal[i]=[p.x/120,p.y/120,*(p.velocity/3.5),np.cos(p.theta),np.sin(p.theta),p.speed/3,float(p.is_pursuing),env.episode_time_steps/3000,*currents.reshape(-1)]
        pfeat[i]=[p.x/120,p.y/120,*(p.velocity/3.5),np.cos(p.theta),np.sin(p.theta),float(p.is_pursuing)]
    ef=np.zeros((2,7),np.float32);em=np.zeros(2,bool);of=np.zeros((2,5),np.float32);om=np.zeros(2,bool)
    for i,e in enumerate(env.evaders):
        ef[i]=[e.x/120,e.y/120,*(e.velocity/3.5),np.cos(e.theta),np.sin(e.theta),e.speed/3.5];em[i]=not e.deactivated
    for i,o in enumerate(env.obstacles):of[i]=[o.x/120,o.y/120,o.r/120,0,0];om[i]=True
    return {'self':focal,'pursuers':np.broadcast_to(pfeat,(7,7,7)).copy(),'pursuer_mask':np.broadcast_to(active,(7,7)).copy(),
            'evaders':ef,'evader_mask':em,'obstacles':of,'obstacle_mask':om,'active_mask':active}

def critic_factory(hidden_dim=256,num_heads=8,num_layers=4):
    return CentralValueNetwork(hidden_dim,num_heads,num_layers,self_feature_dim=57,max_agents=7,max_evaders=2,max_obstacles=2).eval()

def qualification(stage_report,retention_report,stage_steps):
    if stage_steps not in {100000,200000}:raise ValueError('Stage2 firstreview100k, preregistered extension200k')
    for report,scene in ((stage_report,'4P1E1O6C'),(retention_report,'3P1E0O4C')):
        if report['seed_domain']!='selection' or report['scene']!=scene or report['episodes_per_mode']!=20:
            raise ValueError('promotion uses20+20 selection and retention only; final must remain blind')
        for mode in ('argmax','sample'):
            if report['modes'][mode]['normal_capture_rate']<.8 or report['modes'][mode]['collision_rate']>.2:return False
    return True

def env_factory(seed):return NativeCurriculum(seed,2)

def promotion_bundle(parameters):
    if set(parameters)!={'stage','initialization','promotion_receipt'} or parameters['stage']!=3 or parameters['initialization']!='qualified_stage2_full_trainer_new_env':
        raise ValueError('Stage3 requires explicit registered promotion bundle')
    pin=parameters['promotion_receipt'];path=Path(pin['path']).resolve()
    if not path.is_relative_to(Path('/home/yjq')) or file_hash(path)!=pin['sha256']:raise ValueError('promotion receipt path/hash mismatch')
    bundle=json.loads(path.read_text())
    from .runner import CENTRAL,read
    state=read(CENTRAL/'batch_state.json')
    if state['base_freeze_status']!='BASE_FROZEN' or state.get('QA_SAME_SHA') not in {'PASS','A3_INDEPENDENT_QA_PASS'}:
        raise ValueError('formal Stage3 requires independent frozen BASE')
    reports=[]
    for key in ('stage_report','retention_report'):
        source=bundle[key];target=Path(source['path']).resolve()
        if not target.is_relative_to(Path('/home/yjq')) or file_hash(target)!=source['sha256']:raise ValueError('qualified report path/hash mismatch')
        report=json.loads(target.read_text())
        if report.get('execution_mode')!='FORMAL' or report.get('status')!='COMPLETE':raise ValueError('pilot/partial report cannot authorize Stage3')
        reports.append(report)
    if not qualification(*reports,bundle['stage_steps']):raise ValueError('Stage2 qualification and Stage1 retention required')
    checkpoint=Path(bundle['checkpoint']).resolve();manifest_path=Path(bundle['manifest']).resolve()
    if not checkpoint.is_relative_to(Path('/home/yjq')) or not manifest_path.is_relative_to(Path('/home/yjq')):raise ValueError('promotion source path must be local')
    digest=file_hash(checkpoint)
    if any(r['checkpoint_sha256']!=digest for r in reports) or file_hash(manifest_path)!=bundle['manifest_sha256']:
        raise ValueError('promotion selection/retention/checkpoint/manifest mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest['line']!='T1' or manifest['execution_mode']!='FORMAL' or manifest['delta_manifest']['extensions']['parameters']['stage']!=2:
        raise ValueError('qualified parent must be formal Stage2')
    if reports[0]['evaluation_protocol']!=manifest['evaluation_protocol'] or reports[1]['evaluation_protocol']!=manifest['evaluation_protocol']:
        raise ValueError('promotion evaluation protocol differs from locked Stage2')
    if state.get('BATCH01_BASE_SHA')!=manifest['base']['candidate_sha']:raise ValueError('promotion parent differs from frozen BASE')
    return bundle,manifest

def env_stage3(seed):return NativeCurriculum(seed,3)

def factory(parameters):
    if parameters=={'stage':2,'initialization':'actor775k_critic_adams_valuenorm_fresh'}:
        return Hooks(env_factory=env_factory,state_encoder=state_encoder,critic_factory=critic_factory)
    promotion_bundle(parameters)
    return Hooks(env_factory=env_stage3,state_encoder=state_encoder,critic_factory=critic_factory)

def warm_start(runtime):
    if file_hash(ANCHOR)!=ANCHOR_HASH:raise ValueError('T1 warm actor anchor hash mismatch')
    checkpoint=torch.load(ANCHOR,map_location='cpu',weights_only=False)
    if checkpoint['steps']!=775000:raise ValueError('T1 requires selected775k actor')
    runtime.trainer.actor.load_state_dict(checkpoint['trainer']['actor'],strict=True)
    if runtime.trainer.actor_optimizer.state or runtime.trainer.value_optimizer.state or runtime.trainer.update_count!=0:raise ValueError('T1 fresh optimizers required')
    return {'anchor':ANCHOR,'anchor_hash':ANCHOR_HASH,'actor_load_strict':True,'transferred':['actor_parameters_only'],
            'fresh':['fixed_capacity_critic','actor_Adam','critic_Adam','ValueNorm','Stage2_env_APF','RNG','stage_counters'],
            'bit_exact_cross_scale_resume':False}

def initialize_stage(runtime,parameters):
    if parameters['stage']==2:return warm_start(runtime)
    bundle,manifest=promotion_bundle(parameters)
    from terl_mappo.batch01.checkpoints import load_bound
    config=runtime.config
    runtime.config=manifest['resolved_config']
    try:load_bound(bundle['checkpoint'],runtime,manifest)
    finally:runtime.config=config
    runtime.adapter=runtime.source_guard.call(env_stage3,config['seed'])
    return {'migration':'strict full trainer transfer; new scale environment/APF and joint counters; not bit-exact resume',
            'stage':3,'transferred':['actor','fixed_capacity_critic','actor_Adam','critic_Adam','ValueNorm','checkpoint_RNG'],
            'fresh':['Stage3_environment','Stage3_APF','stage_joint_counters'],
            'review_windows':[100000,200000,300000],'automatic_extension':False,'promotion_receipt':parameters['promotion_receipt'],
            'checkpoint_sha256':file_hash(bundle['checkpoint']),'bit_exact_cross_scale_resume':False}
