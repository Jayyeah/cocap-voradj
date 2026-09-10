import numpy as np
import torch
from cocap_voradj.models.small_step_ac import CentralValueNetwork
from cocap_voradj.models.forward_final_two_head_value import TwoHeadCentralValue
from cocap_voradj.training.small_step_ac import tensor_tree
from tools.audit_forward_final_critic_gradients_20260909 import BASE


def setup():
    torch.manual_seed(7)
    single = CentralValueNetwork(hidden_dim=16, num_heads=2, num_layers=1, max_agents=4, max_evaders=8, max_obstacles=5).eval()
    data = np.load(BASE / 'train/fixed_bank.npz')
    b = tensor_tree({k[7:]: data[k][:6] for k in data.files if k.startswith('global_')}, 'cpu')
    b['phase'] = torch.tensor([0, 1, 2, 0, 1, 2])
    return single, b


def test_two_head_initial_function_and_shared_gradient_match_single():
    single, b = setup()
    two = TwoHeadCentralValue(single)
    torch.testing.assert_close(single(b), two(b), rtol=0, atol=0)
    single(b).square().mean().backward()
    two(b).square().mean().backward()
    for name, p in single.named_parameters():
        q = dict(two.shared.named_parameters())[name]
        expected = q.grad
        if name.startswith('value_head.'):
            expected = expected + dict(two.recovery_head.named_parameters())[name.removeprefix('value_head.')].grad
        torch.testing.assert_close(p.grad, expected, atol=1e-6, rtol=1e-5)
    assert all(torch.equal(v, two.shared.state_dict()[k]) for k, v in single.state_dict().items())


def test_two_head_routes_true_phase_and_zeroes_other_head_gradient():
    single, b = setup()
    two = TwoHeadCentralValue(single)
    with torch.no_grad():
        two.recovery_head[-1].bias.add_(2.)
    expected = single(b) + (b['phase'] != 0)[:, None].float() * 2
    torch.testing.assert_close(two(b), expected)
    for phase in [0, 1, 2]:
        two.zero_grad(set_to_none=True)
        two(b)[b['phase'] == phase].sum().backward()
        unused = two.recovery_head if phase == 0 else two.shared.value_head
        used = two.shared.value_head if phase == 0 else two.recovery_head
        assert all(p.grad is not None and not torch.count_nonzero(p.grad) for p in unused.parameters())
        assert any(torch.count_nonzero(p.grad) for p in used.parameters())
    restored = TwoHeadCentralValue(single)
    restored.load_state_dict(two.state_dict(), strict=True)
    torch.testing.assert_close(restored(b), two(b), rtol=0, atol=0)
