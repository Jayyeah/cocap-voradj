"""Diagnostic only: tiny optimizer settings are never pilot settings."""
import copy,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools.z_guided_support_20261016 import config,PARENT,PARENT_SHA,PilotTrainer,write,sha256_file
from tools import iqn_z_unified_decay_curriculum_20260919 as source
from tools import iqn_token_matched_20260919 as matched

def main():
    torch.set_num_threads(1)
    root=Path('/home/yjq/rl/CoCap1/iqn-z-guided-support-20261016-runtime/optimizer_smoke_v2')
    if root.exists():raise FileExistsError(root)
    results={}
    for arm in ('control','treatment'):
        cfg=config(arm);cfg.update(output_root=str(root),run_name=arm,device='cpu',total_timesteps=16,train_mode='voradj')
        cfg['pretrained']={'path':str(PARENT),'compatibility_mode':'shape_compatible'}
        cfg['iqn'].update(batch_size=8,min_replay_size=8,replay_capacity=256,train_freq=1,target_update_freq=1,checkpoint_freq=16,log_freq_steps=1)
        t=PilotTrainer(cfg)
        assert t.global_step==0 and not t.optimizer.state and all(len(b)==0 for b in t.replays.values())
        assert matched.state_hash(t.model)==matched.state_hash(t.target_model)
        fresh=dict(global_step=0,optimizer_entries=0,replay_sizes={k:len(b) for k,b in t.replays.items()},parent_sha256=PARENT_SHA)
        before=matched.state_hash(t.model);checkpoint=t.train()
        assert t.update_steps>=10 and math.isfinite(t.loss_ema)
        assert before!=matched.state_hash(t.model)
        resume=root/arm/'checkpoints/resume_latest.pt'
        rcfg=copy.deepcopy(cfg);rcfg['run_name']=arm+'_restored';rcfg['checkpointing']['resume_path']=str(resume)
        rt=PilotTrainer(rcfg)
        assert rt.global_step==t.global_step==16
        assert matched.state_hash(rt.model)==matched.state_hash(t.model)
        env=t.envs[t.current_task];renv=rt.envs[rt.current_task]
        assert env._support_policy_snapshots.keys()==renv._support_policy_snapshots.keys()
        for i in env._support_policy_snapshots:
            x=env._support_policy_snapshots[i];y=renv._support_policy_snapshots[i]
            assert x['friend_ids']==y['friend_ids'] and x['self_z']==y['self_z']
            np.testing.assert_array_equal(x['tokens'],y['tokens'])
        result=renv.step([4]*len(renv.pursuers),[4]*len(renv.evaders))
        assert np.isfinite(result.rewards).all()
        for log in (rt.episode_log,rt.metric_log):log.close()
        results[arm]=dict(fresh_gate=fresh,updates=t.update_steps,target_syncs=t.target_update_count,loss_ema=t.loss_ema,checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),full_resume=str(resume),restored_policy_mapping=True,next_step_finite=True)
    write(ROOT/'artifacts/2026-10-16_z_guided_support/optimizer_smoke.json',dict(status='PASS',diagnostic_only=True,results=results))
if __name__=='__main__':main()
