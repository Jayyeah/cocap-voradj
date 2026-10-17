#!/usr/bin/env python3
"""Repeat archived coverage seeds under exchanged reward switches."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from tools import z_guided_support_20261016 as p
from tools import iqn_token_matched_20260919 as f
from tools.rollout_voradj_visual import act_evaders
from cocap_voradj.control.apf import ApfAgent

def main():
    torch.set_num_threads(1);rows=[];device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
    for arm,step in (('control',50000),('control',100000),('treatment',100000)):
        path=p.RUNTIME/arm/'checkpoints'/f'final_step_{step}.pt';model=p.model_load(path,device)
        for ep in range(3):
            seed=p.EVAL_SEED+2000+ep
            variants=[]
            for env_arm in ('control','treatment'):
                env,obs=p.env_make(p.config(env_arm),'coverage',seed)
                fingerprint=p.initial_state_fingerprint(env);apf=[];collision=False
                for st in range(1,env.episode_max_length+1):
                    result=env.step(p.actions(model,obs,device,'argmax'),act_evaders(env,apf));p.ledger_check(result)
                    collision|=bool(env.last_collision_events);obs=result.observations
                    if all(result.dones):break
                record=dict(seed=seed,steps=st,ce_success=env.post_capture_coverage_success,ce_rms=env.episode_record()['coverage_ce_center_rms'],collision=collision,initial_state_fingerprint=fingerprint)
                variants.append(record)
            assert variants[0]==variants[1],(arm,step,seed)
            directories=[d for d in (p.RUNTIME/arm/'evaluations').glob('step_*') if int(d.name.split('_')[1])==step]
            archived=json.loads((directories[0]/f'argmax_coverage_{ep:03d}.json').read_text())
            assert variants[0]['steps']==archived['steps']
            assert variants[0]['ce_success']==archived['post_capture_ce']
            assert np.isclose(variants[0]['ce_rms'],archived['ce_rms']),(arm,step,seed,variants[0],archived)
            rows.append(dict(arm=arm,step=step,checkpoint=str(path),sha256=p.sha256_file(path),**variants[0],exchanged_reward_switch_identical=True,archive_reproduced=True))
    model_hashes={}
    for arm,step in (('control',50000),('control',100000),('treatment',100000)):
        model_hashes[f'{arm}/{step}']=f.state_hash(p.model_load(p.RUNTIME/arm/'checkpoints'/f'final_step_{step}.pt','cpu'))
    assert model_hashes['control/50000']!=model_hashes['control/100000']
    p.write(ROOT/'artifacts/2026-10-17_z_support_full3stage/ce_model_attribution.json',dict(device=device,status='PASS_MODEL_CHANGE_NOT_EVALUATION_SWITCH',rows=rows,model_hashes=model_hashes,interpretation='Archived seeds reproduce exactly; swapping the Support switch in pure coverage does not change outcome. The Control policy itself changed. Three repeated seeds diagnose configuration consistency, not the cause of optimizer forgetting or population performance.',completed_at=p.now()))
    print('PASS: 18 full-horizon episodes; 9 archived outcomes reproduced; exchanged env switches identical.')
if __name__=='__main__':main()
