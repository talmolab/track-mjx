"""Utilities for handling dictionary observations.

This module provides utilities for working with dictionary observations
where each key maps to a flat array. Key components:

- DictRunningStatisticsState: Holds separate running stats for each observation key
- Normalizer functions: init, update, and normalize for dict observations
"""

from typing import Mapping

import flax
import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs


@flax.struct.dataclass
class DictRunningStatisticsState:
    """Running statistics state for dictionary observations.

    Holds separate RunningStatisticsState for each observation key,
    enabling independent normalization of different observation components.
    """

    imitation_target: running_statistics.RunningStatisticsState
    proprioception: running_statistics.RunningStatisticsState


def init_dict_normalizer(
    obs: Mapping[str, jnp.ndarray],
) -> DictRunningStatisticsState:
    """Initialize running statistics state from an example observation dict.

    Args:
        obs: Example observation dict with flat arrays at each key.

    Returns:
        Initialized DictRunningStatisticsState with proper shapes.
    """
    return DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(obs["imitation_target"].shape[-1:], jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(obs["proprioception"].shape[-1:], jnp.dtype("float32"))
        ),
    )


def update_dict_normalizer(
    state: DictRunningStatisticsState,
    obs: Mapping[str, jnp.ndarray],
    pmap_axis_name: str | None = None,
) -> DictRunningStatisticsState:
    """Update running statistics from an observation dict.

    Args:
        state: Current running statistics state.
        obs: Observation dict with flat arrays at each key.
        pmap_axis_name: Axis name for pmap aggregation (optional).

    Returns:
        Updated DictRunningStatisticsState.
    """
    return DictRunningStatisticsState(
        imitation_target=running_statistics.update(
            state.imitation_target,
            obs["imitation_target"],
            pmap_axis_name=pmap_axis_name,
        ),
        proprioception=running_statistics.update(
            state.proprioception,
            obs["proprioception"],
            pmap_axis_name=pmap_axis_name,
        ),
    )


def normalize_dict_obs(
    obs: Mapping[str, jnp.ndarray],
    state: DictRunningStatisticsState,
) -> dict[str, jnp.ndarray]:
    """Normalize observation dict using running statistics.

    Args:
        obs: Observation dict with flat arrays at each key.
        state: Running statistics state for normalization.

    Returns:
        Dict with normalized observation arrays.
    """
    return {
        "imitation_target": running_statistics.normalize(
            obs["imitation_target"], state.imitation_target
        ),
        "proprioception": running_statistics.normalize(
            obs["proprioception"], state.proprioception
        ),
    }


def flatten_dict_obs(obs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Flatten observation dict to single array.

    Concatenates imitation_target and proprioception in that order.
    Useful for value network or legacy compatibility.

    Args:
        obs: Observation dict with flat arrays at each key.

    Returns:
        Single flat array with all observations concatenated.
    """
    return jnp.concatenate(
        [obs["imitation_target"], obs["proprioception"]], axis=-1
    )


def get_obs_sizes(obs: Mapping[str, jnp.ndarray]) -> dict[str, int]:
    """Extract observation sizes from an example observation dict.

    Args:
        obs: Example observation dict.

    Returns:
        Dict mapping observation keys to their sizes.
    """
    return {
        "imitation_target": obs["imitation_target"].shape[-1],
        "proprioception": obs["proprioception"].shape[-1],
    }
