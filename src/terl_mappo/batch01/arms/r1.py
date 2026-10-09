"""Distance-only CoCap reward bridge: fixed pre-step CR-MS geometry target."""
import numpy as np
from terl_mappo.batch01.interfaces import Hooks

PARAMETERS={'coefficient':2.0,'progress_clip':3.0,'inner_surface':4.5,'preferred_center':8.0,
            'outer_center':10.5,'cell':1.5,'radial_sigma':2.0,'phase_width':float(np.pi/6),
            'phase_radial_margin':4.0,'angle_alpha':.25,'velocity_static':.3,'velocity_full':1.0,
            'obstacle_margin':2.0,'boundary_margin':.5}

def angle_delta(a,b):return (np.asarray(a)-np.asarray(b)+np.pi)%(2*np.pi)-np.pi

def ring_target(pre,recipient,target,params=PARAMETERS):
    e=pre['evaders'][target];p=pre['pursuers'][recipient];ep=e[:2];pp=p[:2]
    inner=params['inner_surface']+p[2]+e[4]
    outer=max(params['outer_center'],inner+1.5);preferred=np.clip(params['preferred_center'],inner+1e-6,outer-1e-6)
    xs=np.arange(max(.5,ep[0]-outer),min(pre['map'][0]-.5,ep[0]+outer)+.5*params['cell'],params['cell'])
    ys=np.arange(max(.5,ep[1]-outer),min(pre['map'][1]-.5,ep[1]+outer)+.5*params['cell'],params['cell'])
    if not len(xs) or not len(ys):return None
    candidates=np.stack(np.meshgrid(xs,ys),-1).reshape(-1,2);rel=candidates-ep;radius=np.linalg.norm(rel,axis=1)
    valid=(radius>=inner)&(radius<=outer)
    phases=np.arctan2(rel[:,1],rel[:,0])
    for j,q in enumerate(pre['pursuers']):
        if j==recipient or not q[3]:continue
        qr=q[:2]-ep;distance=np.linalg.norm(qr)
        if abs(distance-preferred)<=params['phase_radial_margin']:
            valid &= np.abs(angle_delta(phases,np.arctan2(qr[1],qr[0])))>params['phase_width']
    for obstacle in pre['obstacles']:
        valid &= np.linalg.norm(candidates-obstacle[:2],axis=1)>obstacle[2]+params['obstacle_margin']
    candidates=candidates[valid];rel=rel[valid];radius=radius[valid]
    if not len(candidates):return None
    speed=np.linalg.norm(e[2:4]);gate=np.clip((speed-params['velocity_static'])/(params['velocity_full']-params['velocity_static']),0,1)
    cosine=(rel@e[2:4])/(np.maximum(radius,1e-9)*max(speed,1e-9))
    weight=np.exp(-(radius-preferred)**2/(2*params['radial_sigma']**2))*np.clip(1+params['angle_alpha']*gate*cosine,.75,1.25)
    raw=np.average(candidates,axis=0,weights=weight);direction=raw-ep;norm=np.linalg.norm(direction)
    if norm<=1e-6:direction=ep-pp;norm=max(np.linalg.norm(direction),1e-6)
    result=ep+preferred*direction/norm
    margin=params['boundary_margin']
    allowed=margin<=result[0]<=pre['map'][0]-margin and margin<=result[1]<=pre['map'][1]-margin
    blocked=any(np.linalg.norm(result-o[:2])<=o[2]+params['obstacle_margin'] for o in pre['obstacles'])
    if not allowed or blocked:result=candidates[np.argmax(weight-.1*np.abs(radius-preferred))]
    return result


def shaping(pre,post,params=PARAMETERS):
    result=np.zeros(len(pre['pursuers']))
    targets=np.flatnonzero(pre['evaders'][:,5].astype(bool)) if len(pre['evaders']) else []
    for i,p in enumerate(pre['pursuers']):
        if not p[3] or not len(targets):continue
        target=int(targets[np.argmin(np.linalg.norm(pre['evaders'][targets,:2]-p[:2],axis=1))])
        goal=ring_target(pre,i,target,params)
        if goal is None:continue
        progress=np.linalg.norm(p[:2]-goal)-np.linalg.norm(post['pursuers'][i,:2]-goal)
        result[i]=params['coefficient']*np.clip(progress,-params['progress_clip'],params['progress_clip'])
    return result

class RewardBridge:
    def __init__(self,parameters):
        if parameters!=PARAMETERS:raise ValueError('R1 fixed preregistered shaping parameters')
        self.parameters=parameters;self.last_components=None
    def bind(self,adapter):self.adapter=adapter
    def snapshot(self,env):
        return {'pursuers':np.array([[p.x,p.y,p.r,float(not p.deactivated)] for p in env.pursuers]),
                'evaders':np.array([[e.x,e.y,*e.velocity,e.r,float(not e.deactivated)] for e in env.evaders]).reshape(-1,6),
                'obstacles':np.array([[o.x,o.y,o.r] for o in env.obstacles]).reshape(-1,3),'map':np.array([env.width,env.height])}
    def __call__(self,raw,info):
        native={k:np.asarray(v,dtype=float) for k,v in info['components'].items()}
        if not np.allclose(sum(native.values()),raw,atol=1e-6):raise ValueError('R1 raw reward reconstruction failed')
        progress=shaping(info['bridge_pre'],info['bridge_post'],self.parameters)
        if not np.array_equal(info['active'],info['bridge_pre']['pursuers'][:,3].astype(bool)):
            raise ValueError('R1 recipient mask is pre-step native active mask')
        shaped=np.asarray(raw)-native['distance']+progress
        self.last_components={**native,'raw':np.asarray(raw).copy(),'native_distance':native['distance'].copy(),
                              'shaping':progress.copy(),'shaped':shaped.copy()}
        if info['episode_steps']==1 or not hasattr(self.adapter,'bridge_returns'):
            self.adapter.bridge_returns={k:np.zeros_like(raw) for k in self.last_components}
        for k,v in self.last_components.items():self.adapter.bridge_returns[k]+=v*info['active']
        return shaped

def factory(parameters):return Hooks(reward_transform=RewardBridge(parameters))
