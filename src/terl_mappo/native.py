"""Adapter for byte-identical TERL source plus its original trainer lifecycle."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'vendor' / 'terl'
sys.path.insert(0, str(VENDOR))
from config_manager import ConfigManager
from environment.env import MarineEnv
from thirdparty.APF import ApfAgent
from utils import logger
import wandb

TERL_SHA = '143359b2722d49c29b4fecc0ad1fd8d46326e45a'

def initialize_native():
    cm = ConfigManager()
    cm._config = yaml.safe_load((VENDOR / 'config/config.yaml').read_text())
    # Capture logging has no scientific effect; avoid network and run registration.
    wandb.log = lambda *args, **kwargs: None
    logger.setLevel(40)
    return copy.deepcopy(cm.get_config())

def stage1_schedule():
    cfg = initialize_native()
    return {k: [v[0]] for k, v in cfg['training_schedule'].items()}

def empty_observation():
    return {'self': np.zeros(4, np.float32), 'pursuers': np.zeros((5,7), np.float32),
            'evaders': np.zeros((8,7), np.float32), 'obstacles': np.zeros((5,5), np.float32),
            'masks': np.zeros(19, bool), 'types': np.array([0]+[1]*5+[2]*8+[3]*5)}

def pack_local(observations):
    rows = [empty_observation() if o is None else o for o in observations]
    return {k: np.stack([o[k] for o in rows]) for k in rows[0]}

def central_state(env):
    """Training-only action-free geometry; focal rows equivariant to P order.

    Fixed physical scaling applies only to V, never to native actor tokens/reward.
    Core order is canonical by position; all four Stage1 cores are represented.
    """
    ps, es = env.pursuers, env.evaders
    active = np.array([not p.deactivated for p in ps])
    cores = sorted(env.cores, key=lambda c: (c.x, c.y))
    assert len(cores) == 4 and len(ps) == 3 and len(es) == 1 and not env.obstacles
    cv = np.array([[c.x/120, c.y/120, 1 if c.clockwise else -1,
                    c.Gamma/(10*np.pi)] for c in cores]).reshape(-1)
    focal = np.array([[p.x/120, p.y/120, * (p.velocity/3.5), np.cos(p.theta),
                        np.sin(p.theta), p.speed/3, float(p.is_pursuing),
                        env.episode_time_steps/3000, *cv] for p in ps], np.float32)
    pfeat = np.array([[p.x/120,p.y/120,*(p.velocity/3.5), np.cos(p.theta),
                       np.sin(p.theta), float(p.is_pursuing)] for p in ps], np.float32)
    efeat = np.array([[e.x/120,e.y/120,*(e.velocity/3.5),np.cos(e.theta),
                       np.sin(e.theta),e.speed/3.5] for e in es], np.float32)
    return {'self': focal, 'pursuers': np.broadcast_to(pfeat, (3,3,7)).copy(),
            'pursuer_mask': np.broadcast_to(active, (3,3)).copy(), 'evaders': efeat,
            'evader_mask': np.array([not e.deactivated for e in es]),
            'obstacles': np.zeros((1,5), np.float32), 'obstacle_mask': np.zeros(1, bool),
            'active_mask': active}

def geometry(env):
    e = env.evaders[0]
    near = [p for p in env.pursuers if not p.deactivated and np.hypot(p.x-e.x,p.y-e.y)<8]
    strict = False
    if len(near)>=3:
        a = sorted(np.arctan2(p.y-e.y,p.x-e.x) % (2*np.pi) for p in near)
        gaps = np.diff(a+[a[0]+2*np.pi])
        strict = bool(gaps.max()<=np.pi and gaps.max()<=3*gaps.min())
    return len(near), strict

def reward_components(env, before_active, rewards, infos):
    """Read-only reconstruction of the native step's effective reward terms."""
    # Obstacle emergency uses cached pre-step perception; P distances are live.
    dense = np.zeros(3)
    for i,p in enumerate(env.pursuers):
        if not before_active[i]: continue
        distances = [np.hypot(p.x-e.x,p.y-e.y) for e in env.evaders]
        dense[i] = sum(env.distance_reward if d<=8 else 5*np.exp(env.decay_factor*(min(distances)-8)) for d in distances)
    global_reward = env.global_reward()
    collision = np.array([env.collision_penalty if x['state']=='collision' else 0 for x in infos])
    goal = np.array([rewards[i]-env.timestep_penalty-dense[i]-global_reward[i]-collision[i]
                     if '✌️capture' in infos[i]['state'] else 0 for i in range(3)])
    return dense, global_reward, collision, goal

class NativeStage1:
    def __init__(self, seed):
        schedule = stage1_schedule()
        self.env = MarineEnv(seed=int(seed), schedule=schedule)
        self.apf = ApfAgent(self.env.evaders[0].a, self.env.evaders[0].w)
        self.reset()
        self.assert_contract()

    def contract(self):
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
            "perception_range":p.perception.range,"token_count":len(self.observations[0]["masks"]),
            "horizon_decisions":e.episode_max_length+1,"episode_reset_min_active":3,
            "collision_semantics":"native_end_of_decision","enemy_distance_filter":False}

    def assert_contract(self):
        c=self.contract(); cfg=ConfigManager.get_instance().get_config()
        expected={"counts":[3,1,0,4],"map":[120.,120.],"spawn_min_pe":10.,
                  "dt":.05,"integration_substeps":10,"pursuer_max_speed":3.,"evader_max_speed":3.5,
                  "token_count":19,"horizon_decisions":3001}
        if any(c[k]!=v for k,v in expected.items()): raise ValueError("native Stage1 runtime contract mismatch")
        for k,v in cfg["env"].items():
            if k in c and c[k]!=v: raise ValueError(f"native reward/environment parameter not applied: {k}")
        expected_actions=[(a,w) for a in cfg["pursuer"]["a"] for w in cfg["pursuer"]["w"]]
        if c["aw9"]!=expected_actions: raise ValueError("AW9 mapping mismatch")

    def reset(self):
        (self.observations, _), _ = self.env.reset()
        self.evader_obs, _ = self.env.get_evaders_observation()
        self.return_sum = np.zeros(3)
        return pack_local(self.observations)

    def step(self, indices):
        env = self.env
        active = np.array([not p.deactivated for p in env.pursuers])
        # These helpers use cached pre-step perception in the native reward.
        stale_od = env.get_distance_to_obstacles()
        ea = [None if e.deactivated else self.apf.act(self.evader_obs[i]) for i,e in enumerate(env.evaders)]
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
        truncated = np.full(3, truncation) & ~terminated
        self.return_sum += reward * active
        return pack_local(obs), reward, terminated, truncated, end, {
            'capture': bool(capture), 'normal_capture': bool(capture and not colliding.any()),
            'collision': bool(colliding.any()), 'collision_types': collision_types,
            'ring2': ring>=2, 'ring3': ring>=3, 'strict_geometry': strict,
            'components': {k:v.tolist() for k,v in components.items()},
            'native_done': done, 'native_infos': info, 'active': active,
            'episode_steps': env.episode_time_steps, 'episode_return': self.return_sum.tolist()}

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__.update(copy.deepcopy(state))

    def fingerprint(self):
        e=self.env
        data = [[r.x,r.y,r.theta,r.speed,*r.velocity,float(r.deactivated)] for r in e.pursuers+e.evaders]
        return hashlib.sha256(json.dumps(data).encode()).hexdigest()
