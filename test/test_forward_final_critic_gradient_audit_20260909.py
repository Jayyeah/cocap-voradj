"""Check masked gradient decomposition independently of the full transformer."""
import numpy as np
import torch
from tools.audit_forward_final_critic_gradients_20260909 import (
    phase_gradients, direct_gradient, gradient_summary, baseline_tables, baseline_predictions,
)
from cocap_voradj.training.small_step_ac import ValueNorm


class TinyValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(2, 3)
        self.value_head = torch.nn.Linear(3, 1)

    def forward(self, data):
        return self.value_head(torch.tanh(self.trunk(data['self']))).squeeze(-1)


def test_masked_phase_gradients_sum_to_direct_and_ignore_inactive_targets():
    torch.manual_seed(3)
    model = TinyValue()
    rng = np.random.default_rng(3)
    central = {'self': rng.normal(size=(6, 2, 2)).astype(np.float32)}
    data = {'phase': np.array([0, 1, 2, 0, 1, 2]),
            'active': np.array([[1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1]], dtype=bool),
            'target': rng.normal(size=(6, 2)).astype(np.float32)}
    norm = ValueNorm()
    norm.update(torch.as_tensor(data['target']), torch.as_tensor(data['active']))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    gradients, losses, _ = phase_gradients(model, central, data, norm, 'cpu', chunk=1)
    direct = direct_gradient(model, central, data, norm, 'cpu')
    for i, expected in enumerate(direct):
        torch.testing.assert_close(sum(g[i] for g in gradients).float(), expected, atol=1e-6, rtol=1e-5)
    altered = dict(data, target=data['target'].copy())
    altered['target'][~data['active']] = 1e8
    other, other_losses, _ = phase_gradients(model, central, altered, norm, 'cpu', chunk=3)
    np.testing.assert_allclose(losses, other_losses, rtol=1e-5)
    for g, h in zip(gradients, other):
        for a, b in zip(g, h):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    summary = gradient_summary(gradients, [k for k, _ in model.named_parameters()])
    for group in summary.values():
        assert abs(sum(group['signed_projection_on_global'].values()) - 1) < 1e-10
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_phase_time_baseline_uses_train_targets_and_unseen_bin_fallback():
    train = {'phase': np.repeat(np.arange(3), 50), 'episode': np.repeat(np.arange(3), 50),
             'active': np.ones((150, 1), bool), 'target': np.repeat([1., 2., 3.], 50)[:, None]}
    tables = baseline_tables(train)
    heldout = {'phase': np.ones(100, int), 'episode': np.zeros(100, int),
               'active': np.ones((100, 1), bool), 'target': np.full((100, 1), 1e9)}
    predictions = baseline_predictions(heldout, tables)
    np.testing.assert_array_equal(predictions['phase_time'], np.full((100, 1), 2.))
