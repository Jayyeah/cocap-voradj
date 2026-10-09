"""Continuous physical AW through unchanged native integration, no quantization."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from types import SimpleNamespace
from terl_mappo.native import NativeStage1,pack_local,geometry,reward_components
from terl_mappo.model import TERLActor
from terl_mappo.batch01.interfaces import Hooks,ActionCapabilities,ActionBackend

BOUNDS=np.array([.4,np.pi/6])

class PhysicalActionList(tuple):
    def __getitem__(self,command):
        if isinstance(command,(int,np.integer,slice)):return super().__getitem__(command)
        value=np.asarray(command,dtype=np.float64)
        if value.shape!=(2,) or not np.isfinite(value).all() or np.any(np.abs(value)>BOUNDS+1e-7):
            raise ValueError('physical AW command outside native bounds')
        # Only canonicalize floating endpoint roundoff; never snap to AW9.
        return tuple(np.clip(value,-BOUNDS,BOUNDS))

class NativeContinuous(NativeStage1):
    def reset(self):
        result=super().reset()
        for p in self.env.pursuers:p.action_list=PhysicalActionList(p.action_list)
        return result
    def contract(self):
        result=super().contract();result['aw9']=list(result['aw9'])
        result.update(action_family='physical_continuous_aw',physical_bounds=[[-.4,.4],[-float(np.pi/6),float(np.pi/6)]],
                      integration_dispatch='native Robot.update_state action_list[physical_command] passthrough',quantization=False)
        return result
    def action_capabilities(self):return ActionCapabilities('tanh_gaussian_aw','physical_aw',((-.4,.4),(-float(np.pi/6),float(np.pi/6))))
    def validate_actions(self,actions):
        value=np.asarray(actions)
        if value.shape!=(3,2) or not np.issubdtype(value.dtype,np.floating) or not np.isfinite(value).all() or np.any(np.abs(value)>BOUNDS+1e-7):
            raise ValueError('continuous adapter needs bounded floating[P,2] AW')
        return value

    def step(self, actions):
        commands = self.validate_actions(actions)
        env = self.env
        active = np.array([not p.deactivated for p in env.pursuers])
        # These helpers use cached pre-step perception in the native reward.
        stale_od = env.get_distance_to_obstacles()
        ea = [None if e.deactivated else self.apf.act(self.evader_obs[i]) for i,e in enumerate(env.evaders)]
        obs, reward, done, info = env.step(([a.copy() if active[i] else None for i,a in enumerate(commands)], ea))
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

class ContinuousTERLActor(TERLActor):
    def __init__(self,hidden_dim=256,num_heads=8,num_layers=4,seed=109,masked_pool=False):
        super().__init__(hidden_dim,num_heads,num_layers,seed,masked_pool)
        del self.output_layer
        self.mean_head=nn.Linear(hidden_dim,2);self.log_std_head=nn.Linear(hidden_dim,2)
        nn.init.orthogonal_(self.mean_head.weight,gain=.01);nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight);nn.init.constant_(self.log_std_head.bias,-.5)
        self.config=SimpleNamespace(saturation_threshold=.99,initial_log_std=-.5,log_std_min=-5.,log_std_max=1.)
        self.eval()
    def distribution(self,obs):
        self._validate_input(obs)
        feature=self.layer_norm(torch.relu(self.hidden_layer(self.encode_entities(obs))))
        log_std=self.log_std_head(feature).clamp(-5.,1.)
        return Normal(self.mean_head(feature),log_std.exp()),log_std
    def squash(self,latent):return latent.tanh()*latent.new_tensor([.4,np.pi/6])
    def density(self,distribution,latent):
        correction=2*(np.log(2.)-latent-nn.functional.softplus(-2*latent))
        jacobian=(torch.log(latent.new_tensor([.4,np.pi/6]))+correction).sum(-1)
        return distribution.log_prob(latent).sum(-1)-jacobian
    def sample(self,obs,deterministic=False):
        d,_=self.distribution(obs);latent=d.loc if deterministic else d.rsample()
        return self.squash(latent),self.density(d,latent),latent
    def evaluate_latent(self,obs,latent):
        d,log_std=self.distribution(obs);logp=self.density(d,latent)
        entropy=-self.density(d,d.rsample());base_entropy=d.entropy().sum(-1)
        return logp,entropy,base_entropy,d,log_std

def factory(parameters):
    if parameters!={'initial_log_std':-.5}:raise ValueError('C0 std initialization is locked')
    return Hooks(env_factory=NativeContinuous,actor_factory=ContinuousTERLActor,
                 backend=ActionBackend('tanh_gaussian_aw','physical_differential_mc_nats','physical_aw_joint_density',False))
