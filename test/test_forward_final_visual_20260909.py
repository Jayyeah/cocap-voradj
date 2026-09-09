import numpy as np
import torch
from tools.run_forward_final_rollout_gifs_20260909 import capture_frame, assert_reference
from tools.run_forward_final_bridge_20260908 import run_episode
from tools.evaluate_forward_final_ppo_20260909 import load_frozen_actor
from tools.rollout_voradj_visual import frame_ids
from cocap_voradj.models.iqn import CoCapIQN
from cocap_voradj.training.forward_final import TEACHER


def test_visual_snapshot_preserves_policy_rollout_and_rng():
    actor, _, _ = load_frozen_actor(None, 'cpu')
    teacher = CoCapIQN.load(str(TEACHER), device='cpu').eval()
    for scene in ('mixed', 'coverage'):
        baseline = run_episode(teacher, scene, 2026098101, 'cpu', actor=actor,
                               policy_mode='bc_sample', max_steps=4)
        frames = []
        def hook(env, step):
            rng = torch.get_rng_state().clone()
            labels = list(env.last_raw_task_labels)
            frames.append(capture_frame(env, scene, step))
            assert labels == env.last_raw_task_labels
            assert torch.equal(rng, torch.get_rng_state())
        visual = run_episode(teacher, scene, 2026098101, 'cpu', actor=actor,
                             policy_mode='bc_sample', max_steps=4, on_snapshot=hook)
        assert_reference(visual, baseline)
        assert len(frames) == 5 and frames[0]['global_step'] == 0 and frames[-1]['global_step'] == 4
        if scene == 'coverage': assert frames[-1]['ce_targets']
    assert frame_ids(3001, 1000)[0] == 0 and frame_ids(3001, 1000)[-1] == 3000


def test_mc_score_excludes_inactive_padding_and_keeps_phase_episode_counts():
    from tools.fit_forward_final_mc_critic_20260909 import score
    data = dict(target=np.array([[1., 999.], [2., 999.]]), active=np.array([[True, False]]*2),
                phase=np.array([1, 1]), episode=np.array([0, 1]))
    result = score(data, np.array([[1., -999.], [2., -999.]]))
    assert result['post_capture']['rmse'] == 0
    assert result['post_capture']['episodes'] == 2 and result['post_capture']['rows'] == 2
    assert result['pure_coverage']['rows'] == 0


def test_reference_allows_only_event_order_not_changed_event_content():
    import copy, pytest
    row = dict(initial_state_fingerprint='f', captured=True, normal_capture=True, stationary_capture=False,
               collision=False, boundary=False, ce_success=True, safe_complete=True, length=4,
               capture_seconds=1., recovery_seconds=1., mission_seconds=2.,
               mission_events={'summary': {}, 'events': [{'metric':'a','end_step':2}, {'metric':'b','end_step':2}]},
               policy_diagnostics={'action_histogram':[1]*9}, scene='mixed', seed=1)
    ref = copy.deepcopy(row); ref['mission_events']['events'].reverse()
    assert_reference(row, ref)
    ref['mission_events']['events'][0]['end_step'] = 3
    with pytest.raises(AssertionError): assert_reference(row, ref)
