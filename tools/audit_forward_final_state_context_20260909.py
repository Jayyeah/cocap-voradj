#!/usr/bin/env python3
"""Runtime counterexamples using the native Final CE-hold and pursuit-memory consumers."""
import copy, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
from tools.audit_forward_final_root_cause_20260909 import same
from cocap_voradj.training.forward_final import make_env
from tools.run_forward_final_bridge_20260908 import atomic_json


def audit():
    env,_=make_env('coverage',2026099902)
    # Construct a stable, converged physical state using this environment's own CE map.
    for _ in range(60):
        data=env._coverage_voronoi_map()
        centers=[data['centroids'][('pursuer',i)] for i in range(4)]
        for p,c in zip(env.pursuers,centers):p.x,p.y=map(float,c);p.speed=0.;p.velocity=np.zeros(2)
        env._invalidate_voronoi_cache()
    assert env._voradj_coverage_geometry(env._coverage_voronoi_map(),strict=True)['converged_now']
    env.distribution_hold_steps=0;other=copy.deepcopy(env);other.distribution_hold_steps=29
    assert same(env,other)
    a=env.step([4]*4,[]);b=other.step([4]*4,[])
    assert not all(a.dones) and all(b.dones)
    results={'classification':'RUNTIME FACT','ce_hold':{'central_input_bit_identical':True,'same_physical_action':True,'hold_before':[0,29],'done_after':[all(a.dones),all(b.dones)],'rewards_after':[list(map(float,a.rewards)),list(map(float,b.rewards))]},'limits':'Constructed reachable-geometry/history counterexamples, not measured visitation frequency; proves state insufficiency, not PPO degradation causation.'}
    env,_=make_env('mixed',2026099903)
    env._pursuing_flags_initialized=True;env.pursuers[0].is_pursuing=True;env._pursuing_release_counters[0]=1
    other=copy.deepcopy(env);other._pursuing_release_counters[0]=9
    assert same(env,other)
    env._update_effective_pursuing_flags(['coverage']*4);other._update_effective_pursuing_flags(['coverage']*4)
    assert env.pursuers[0].is_pursuing!=other.pursuers[0].is_pursuing
    results['pursuing_memory']={'central_input_bit_identical_before':True,'same_raw_labels':['coverage']*4,'release_counter_before':[1,9],'effective_flag_after':[bool(env.pursuers[0].is_pursuing),bool(other.pursuers[0].is_pursuing)]}
    results['resolved_context']={'reward':{k:v for k,v in env.reward_cfg.items() if any(x in k for x in ('schedule','pbrs','hold','stationary','window','timestep','weight'))},'zone_enabled':env._zone_enabled(),'ce_settle_enabled':env._coverage_settle_enabled(),'release_delay':env._pursuing_release_delay_steps}
    return results

if __name__=='__main__':
    atomic_json(ROOT/'artifacts/2026-09-09_root_cause/p1/context_extended.json',audit())
