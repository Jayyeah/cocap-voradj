"""Independent continuous-action models.

The modules in this namespace do not alter the legacy CoCapIQN execution path.
"""

from cocap_voradj.models.continuous.local_entity_token_encoder import (
    LocalEntityTokenEncoder,
    LocalEntityTokenEncoderConfig,
)
from cocap_voradj.models.continuous.radial_actor import (
    AccelerationActorConfig,
    RadialSquashedGaussianAccelerationActor,
    RadialActorConfig,
    RadialSquashedGaussianActor,
)
from cocap_voradj.models.continuous.central_attention_critic import (
    CentralAttentionCritic,
    CentralCriticConfig,
    CentralTwinCritics,
)

__all__ = [
    "LocalEntityTokenEncoder",
    "LocalEntityTokenEncoderConfig",
    "AccelerationActorConfig",
    "RadialSquashedGaussianAccelerationActor",
    "RadialActorConfig",
    "RadialSquashedGaussianActor",
    "CentralAttentionCritic",
    "CentralCriticConfig",
    "CentralTwinCritics",
]
