"""Default entry for NEW Forward-Final experiments. Legacy modules reproduce R20."""
from __future__ import annotations
import copy
from cocap_voradj.envs.density_sensing import POLICY,enable_v2,runtime_metadata
from cocap_voradj.envs.voronoi_adjacency import VorAdjEnv
from cocap_voradj.training.forward_final import scene_config


def new_scene_config(scene='mixed', *, alpha_capture=1.):
    return scene_config(scene,sensing_policy=POLICY,alpha_capture=alpha_capture)


def new_pure_capture_config():
    from tools.forward_final_single_task_20260915 import task_config
    return enable_v2(task_config('capture'))


def make_pure_capture_env(seed):
    from tools.forward_final_single_task_20260915 import p
    cfg=new_pure_capture_config();p.set_global_config(cfg)
    env=VorAdjEnv(copy.deepcopy(cfg),seed=int(seed));obs=env.reset()
    check_pure_capture_env(env)
    return env,obs


def check_pure_capture_env(env):
    from tools.forward_final_single_task_20260915 import check_env
    assert env.config==new_pure_capture_config()
    metadata=runtime_metadata(env)
    assert env._vct_ls_sensing_radius('enemy')==env._vct_ls_sensing_radius('obstacle')==metadata['resolved_onboard_radius']
    check_env(env,'capture')
    return metadata


from tools import forward_final_single_task_20260915 as st


class NormSensePureCaptureStream(st.SingleTaskStream):
    def __init__(self,seed):
        self.task_name='capture';self.multiplier=1.;self.task='capture'
        env,self.observations=make_pure_capture_env(seed);self.envs={'capture':env}
        self.apf_agents={'capture':[st.p.ApfAgent(e.a,e.w) for e in env.evaders]}
        self.episode=0;self.telemetry=st.Telemetry(env,'capture');self.episode_metrics=[];self.env_steps=0

    def finish(self):
        row=dict(episode=self.episode,**self.telemetry.finish(self.env))
        self.episode_metrics.append(row);self.episode+=1
        st.p.set_global_config(self.env.config);self.observations=self.env.reset()
        self.apf_agents[self.task]=[st.p.ApfAgent(e.a,e.w) for e in self.env.evaders]
        check_pure_capture_env(self.env);self.telemetry=st.Telemetry(self.env,self.task_name)
        return row


def make_pure_capture_stream(seed):
    return NormSensePureCaptureStream(seed)


def make_fullmix_stream(seed,run_dir,*,alpha_capture=1.):
    from tools.train_forward_final_ppo_20260909 import FinalMissionStream
    return FinalMissionStream(seed,run_dir,sensing_policy=POLICY,alpha_capture=alpha_capture)
