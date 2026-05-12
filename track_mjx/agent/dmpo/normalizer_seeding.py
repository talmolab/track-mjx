"""Seed DMPO's running-statistics normalizer with imit checkpoint stats.

The DMPO kl-anchor pipeline warm-starts prior + decoder weights from a
SCAMPER imitation checkpoint. Those weights expect proprio that is
normalized by the imit checkpoint's per-dimension running-statistics
mean/std. DMPO's own normalizer is initialised from scratch (mean=0,
std=1, count=0); without seeding, the warm-started weights see raw
proprio and produce garbage at step 0, breaking the warm-start
invariant.

This module provides a single function, ``seed_proprio_from_imit``, that
returns a new ``DictRunningStatisticsState`` where the ``proprioception``
field is copied from the imit normalizer and the ``imitation_target``
field is left as-is. The ``imitation_target`` field stays fresh because
the gap-jump task obs has a different size and distribution from the
imit-task obs.
"""
from __future__ import annotations

from track_mjx.agent.observation_utils import DictRunningStatisticsState


def seed_proprio_from_imit(
    dmpo_norm: DictRunningStatisticsState,
    imit_norm: DictRunningStatisticsState,
) -> DictRunningStatisticsState:
    """Return a normalizer with proprio fields copied from imit_norm.

    Args:
        dmpo_norm: DMPO's freshly-initialized normalizer (from
            ``init_training_state``). Provides the ``imitation_target``
            field unchanged in the output.
        imit_norm: The imit checkpoint's normalizer (loaded via
            ``load_prior_checkpoint``). Provides the ``proprioception``
            field in the output.

    Returns:
        A new ``DictRunningStatisticsState`` with imit proprio stats and
        DMPO imit-target stats.

    Raises:
        ValueError: if the proprio dim of dmpo_norm and imit_norm differ.
    """
    if dmpo_norm.proprioception.mean.shape != imit_norm.proprioception.mean.shape:
        raise ValueError(
            "proprio shape mismatch: dmpo_norm.proprioception.mean.shape="
            f"{dmpo_norm.proprioception.mean.shape}, "
            f"imit_norm.proprioception.mean.shape={imit_norm.proprioception.mean.shape}. "
            "The DMPO env's proprio dim must match the imit checkpoint's."
        )
    return DictRunningStatisticsState(
        imitation_target=dmpo_norm.imitation_target,
        proprioception=imit_norm.proprioception,
    )
