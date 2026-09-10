"""Experimental pre/recovery value routing; original central trunk is unmodified."""
import copy
import torch
from torch import nn
from torch.func import functional_call


class TwoHeadCentralValue(nn.Module):
    """Keep one shared parameter set, duplicate only the original value head.

    Functional substitution reuses CentralValueNetwork.forward verbatim. It
    evaluates the same dropout-free trunk twice, with shared parameters, so
    no copied trunk implementation can drift. Only the selected output enters
    the loss. This costs extra compute, not extra trunk parameters or updates.
    """
    def __init__(self, single):
        super().__init__()
        assert all(not isinstance(m, nn.Dropout) or m.p == 0 for m in single.modules())
        self.shared = copy.deepcopy(single)  # shared.value_head is V_pre
        self.recovery_head = copy.deepcopy(single.value_head)
        self.eval()

    def forward(self, observation):
        phase = observation['phase']
        if phase.ndim != 1 or phase.shape[0] != observation['active_mask'].shape[0]:
            raise ValueError('phase must be one true phase id per joint state')
        if not bool(((phase >= 0) & (phase <= 2)).all()):
            raise ValueError('phase must be pre=0, post=1 or pure=2')
        pre = self.shared(observation)
        parameters = {'value_head.' + name: p for name, p in self.recovery_head.named_parameters()}
        recovery = functional_call(self.shared, parameters, (observation,), strict=False)
        return torch.where((phase == 0)[:, None], pre, recovery)

    def audit_parameter_names(self):
        names = []
        for name, _ in self.named_parameters():
            if name.startswith('shared.value_head.'):
                name = 'value_head.pre.' + name.removeprefix('shared.value_head.')
            elif name.startswith('recovery_head.'):
                name = 'value_head.recovery.' + name.removeprefix('recovery_head.')
            else:
                name = name.removeprefix('shared.')
            names.append(name)
        return names
