import numpy as np
import torch
from tools.summarize_forward_final_root_cause_20260909 import returns, calibration
from tools.audit_forward_final_root_cause_20260909 import context_probe
from tools.bridge_forward_final_continuous_aw_20260909 import continuous_env, physical_probe
from tools.run_forward_final_bridge_20260908 import run_episode
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import TEACHER


def test_phase_returns_restart_discount_at_capture():
    got=returns(np.array([100.,-3.,-2.]),np.array([0,1,1]))
    assert np.isclose(got['total_discounted'],100-.99*3-.99**2*2)
    assert np.isclose(got['post_capture_discounted'],-3-.99*2)
    # Calibration orientation is target = slope * prediction + intercept.
    c=calibration(np.array([3.,5.,7.]),np.array([1.,2.,3.]))
    assert np.isclose(c['slope'],2) and np.isclose(c['intercept'],1)
    assert calibration([],[])['rows']==0


def test_actual_critic_input_aliases_native_deadline(tmp_path):
    context_probe(tmp_path)


def test_continuous_grid_center_rollout_preserves_transition_engine():
    physical_probe()
    teacher=CoCapIQN.load(str(TEACHER),device='cpu').eval()
    grid=np.asarray([(a,w) for a in (-.4,0,.4) for w in (-np.pi/6,0,np.pi/6)])
    for scene in ('mixed','coverage'):
        original=run_episode(teacher,scene,2026099911,'cpu',max_steps=8)
        def policy(local,q):
            index=q.argmax(1);return grid[index],index
        continuous=run_episode(teacher,scene,2026099911,'cpu',max_steps=8,env_factory=continuous_env,physical_policy=policy)
        for key in ('initial_state_fingerprint','length','captured','collision','ce_rms','policy_diagnostics'):
            assert original[key]==continuous[key],key


def test_critic_context_distinguishes_timers_without_changing_actor_observation():
    import copy
    from cocap_voradj.training.forward_final import make_env
    from cocap_voradj.training.forward_final_value_context import value_context,augment,expand_value_input,NAMES
    from cocap_voradj.models.small_step_ac import CentralValueNetwork
    from tools.audit_forward_final_root_cause_20260909 import central,same
    env,_=make_env('mixed',2026099909)
    for e in env.evaders:e.deactivated=True
    other=copy.deepcopy(env);other.post_capture_step=499
    assert same(env,other)
    assert not np.array_equal(value_context(env),value_context(other))
    raw=central(env);batch=augment(raw,value_context(env))
    net=CentralValueNetwork(hidden_dim=32,num_heads=4,num_layers=1,max_agents=4).eval()
    extended=expand_value_input(net).eval()
    with torch.no_grad():
        original=net({k:torch.as_tensor(v[None]) for k,v in raw.items()})
        new=extended({k:torch.as_tensor(v[None]) for k,v in batch.items()})
    torch.testing.assert_close(original,new,rtol=0,atol=1e-6)
    assert extended.self_feature_dim==9+len(NAMES)


def test_capture_source_recording_survives_native_position_repair(tmp_path):
    import random
    from tools.fit_forward_final_context_critic_20260909 import RecordedSourceStream,content_hash
    from tools.train_forward_final_ppo_20260909 import FinalMissionStream
    from cocap_voradj.training.runtime_semantics import initial_state_fingerprint
    # Deliberately invalid snapshot: native reset must repair it. The source identity
    # remains exact and instrumentation must preserve both reset output and RNG.
    snapshot={'step':7,'positions':[[0.,0.]]*4,'active_mask':[True]*4}
    results=[]
    for cls,name in [(FinalMissionStream,'plain'),(RecordedSourceStream,'recorded')]:
        random.seed(2026099991)
        stream=cls(2026099991,tmp_path/name)
        stream.recovery_init_pool.append(snapshot);stream.recovery_from_capture_ratio=1.
        stream.reset()
        assert stream.last_reset_source[stream.task]=='capture_snapshot'
        results.append((initial_state_fingerprint(stream.env),random.getstate()))
        if name=='recorded':
            assert stream.selected_capture_source_hash==content_hash(snapshot)
            assert stream.capture_positions_preserved is False
    assert results[0]==results[1]
