import copy,random
import numpy as np
import torch
from cocap_voradj.training.small_step_ac import tensor_tree
from tools.train_forward_final_ppo_20260909 import FinalMissionStream,make_trainer,collect_transition,empty_rollout,stack_rollout,tensor_hash,save_checkpoint


def test_native_final_recovery_and_teacher_age_schedule(tmp_path):
    random.seed(71);stream=FinalMissionStream(71,tmp_path)
    assert stream.current_coverage_ce_speed_weight==.0005
    # Force the native capture-pool branch and verify the consumer uses positions.
    positions=[[25.,25.],[45.,25.],[45.,45.],[25.,45.]]
    stream.recovery_init_pool.append({'step':2,'positions':positions,'active_mask':[True]*4})
    stream.recovery_from_capture_ratio=1.
    stream.reset()
    assert stream.last_reset_source['voradj_coverage']=='capture_snapshot'
    np.testing.assert_allclose([[p.x,p.y] for p in stream.env.pursuers],positions)
    assert stream.env.reward_cfg['coverage_ce_speed_weight']==.0005


def test_timeout_bootstrap_state_precedes_reset(tmp_path):
    trainer=make_trainer('cpu',71);stream=FinalMissionStream(71,tmp_path)
    stream.env.episode_max_length=1
    row,episode=collect_transition(trainer,stream)
    assert episode is not None and stream.task=='voradj_coverage'
    assert row['truncated'].all() and not row['terminated'].any()
    # next_values refer to the just-ended mixed world, not freshly reset coverage.
    from cocap_voradj.training.continuous.central_schema import build_central_global_obs
    original=build_central_global_obs(stream.envs['voradj'],max_agents=4,max_evaders=8,max_obstacles=5,self_feature_dim=9)
    with torch.no_grad():expected=trainer.value(tensor_tree({k:v[None] for k,v in original.items()},trainer.device))[0].numpy()
    np.testing.assert_allclose(row['next_values'],expected)


def test_warmup_preserves_actor_and_checkpoint_is_restorable(tmp_path):
    random.seed(71);np.random.seed(71);trainer=make_trainer('cpu',71);stream=FinalMissionStream(71,tmp_path)
    before=tensor_hash(trainer.actor.state_dict());critic_before=tensor_hash(trainer.value.state_dict());rollout=empty_rollout()
    for _ in range(4):
        row,_=collect_transition(trainer,stream)
        for k,v in row.items():rollout[k].append(v)
    batch=stack_rollout(rollout);assert trainer.assert_behavior_log_probs(batch)<1e-4
    metrics=trainer.update_critic_only(batch)
    assert tensor_hash(trainer.actor.state_dict())==before
    assert tensor_hash(trainer.value.state_dict())!=critic_before
    saved=save_checkpoint(tmp_path,4,trainer,stream,{},empty_rollout(),metrics)
    payload=torch.load(saved['checkpoint'],map_location='cpu',weights_only=False)
    restored=FinalMissionStream.__new__(FinalMissionStream);restored.__dict__.update(payload['stream_state'])
    clone=make_trainer('cpu',71);clone.load_state_dict(payload['trainer'])
    assert tensor_hash(clone.actor.state_dict())==before
    assert restored.episode==stream.episode
    with torch.no_grad():
        local=batch['local_obs'];obs={k:torch.as_tensor(v[0],dtype=torch.long if k=='types' else torch.bool if 'mask' in k else torch.float32) for k,v in local.items()}
        torch.testing.assert_close(clone.actor.distribution(obs).logits,trainer.actor.distribution(obs).logits)
    result=trainer.update(batch,categorical=True)
    assert result['actor_update_l2']>0 and tensor_hash(trainer.actor.state_dict())!=before
