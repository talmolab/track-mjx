"""Score-Matching Motion Prior utilities for rodent MJX tasks."""

from track_mjx.agent.smp.features import (
    DEFAULT_KEY_BODY_NAMES,
    SMPFeatureSpec,
    compute_smp_obs,
    sample_reference_smp_obs,
)
from track_mjx.agent.smp.reward import SMPRewardConfig, compute_smp_reward

__all__ = [
    "DEFAULT_KEY_BODY_NAMES",
    "SMPFeatureSpec",
    "SMPRewardConfig",
    "compute_smp_obs",
    "compute_smp_reward",
    "sample_reference_smp_obs",
]
