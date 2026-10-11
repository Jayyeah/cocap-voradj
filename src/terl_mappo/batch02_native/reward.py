"""Lossless view of successful CoCap rewards, never a reward replacement.

Categories are accounting views. In particular support is identified by the
provider's pre-action reward_role, not a newly invented role or reward weight.
Terminal stays separate because native capture and mix completion share it.
"""
import math

CATEGORIES = ('capture', 'support', 'safety', 'geometry', 'auxiliary_shaping', 'terminal')
AUXILIARY_KEYS = ('reward_ce_control', 'reward_ce_pbrs', 'reward_speed_repeat',
                  'reward_early_speed', 'reward_deceleration', 'reward_motion_penalty')


def decompose(meta, actual_reward):
    auxiliary = sum(float(meta[key]) for key in AUXILIARY_KEYS)
    # reward_ce_terminal_correction is already included in reward_ce_pbrs.
    # reward_settled_terminal is already included in reward_terminal.
    # support blend fields likewise are subsets of capture/coverage, not addends.
    capture, coverage = float(meta['reward_capture']), float(meta['reward_coverage'])
    support_role = meta['reward_role'] == 'support'
    values = dict(capture=0. if support_role else capture,
                  support=capture + coverage - auxiliary if support_role else 0.,
                  safety=float(meta['reward_safety']),
                  geometry=0. if support_role else coverage - auxiliary,
                  auxiliary_shaping=auxiliary, terminal=float(meta['reward_terminal']))
    reconstructed = math.fsum(values.values())
    actual = float(actual_reward)
    residual = reconstructed - actual
    if not all(math.isfinite(value) for value in (*values.values(), actual)):
        raise FloatingPointError('nonfinite successful CoCap reward ledger')
    if abs(residual) > 1e-10 * (1. + abs(actual)) or actual != float(meta['reward_total']):
        raise ValueError('successful CoCap reward reconstruction mismatch')
    return dict(**values, reconstructed=reconstructed, actual_reward=actual, residual=residual,
                reward_role=meta['reward_role'], phase=meta['phase'],
                native_components={key:float(value) for key,value in meta.items()
                    if key.startswith('reward_') and isinstance(value,(int,float))},
                semantics='accounting only; PPO consumes unchanged provider reward')
