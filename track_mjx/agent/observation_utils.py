"""Utilities for handling environment observations.

This module supports two observation layouts:
    1. Nested dict observations:
        {
            'state': {'task_obs': ..., 'proprioception': ...},
            'privileged_state': {'task_obs': ..., 'proprioception': ...}
        }
    2. Flat top-level observations:
        {
            'state': ...,
            'privileged_state': ...,
        }

Each leaf value is a flat 1D array (unbatched) or 2D array (batched).
"""

from typing import Any, Mapping

import jax.numpy as jnp
from brax.training.acme import running_statistics, specs


def normalizer_select(
    processor_params: running_statistics.RunningStatisticsState,
    obs_key: str,
) -> running_statistics.RunningStatisticsState:
    """Extract per-key running statistics from a pytree-structured normalizer.

    When running_statistics.init_state is called with a pytree observation shape,
    the resulting RunningStatisticsState has pytree-structured mean, std, etc.
    This function extracts the statistics for a specific top-level key.

    Args:
        processor_params: RunningStatisticsState with pytree-structured fields
            (mean, std, summed_variance are dicts keyed by observation keys).
        obs_key: Top-level key to extract (e.g., 'state' or 'privileged_state').

    Returns:
        RunningStatisticsState with statistics for just the specified key.
    """
    return running_statistics.RunningStatisticsState(
        count=processor_params.count,
        mean=processor_params.mean[obs_key],
        summed_variance=processor_params.summed_variance[obs_key],
        std=processor_params.std[obs_key],
        std_eps=processor_params.std_eps,
        mode=processor_params.mode,
    )


def get_obs_sizes(obs: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Extract observation sizes from an example observation dict.

    Args:
        obs: Example observation dict. Supports either nested dict observations
            or flat top-level arrays.

    Returns:
        Dict mapping feature keys to their trailing dimension sizes.
    """
    if "state" not in obs:
        raise KeyError(
            f"Expected 'state' key in observation dict, got keys: {list(obs.keys())}. "
            "Ensure the environment produces observations with a top-level "
            "'state' key."
        )
    state_obs = obs["state"]
    if isinstance(state_obs, Mapping):
        return {key: val.shape[-1] for key, val in state_obs.items()}
    return {key: val.shape[-1] for key, val in obs.items()}


def get_obs_shape(
    obs: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, specs.Array] | specs.Array]:
    """Extract observation shapes as a pytree for running_statistics.init_state.

    Preserves the container types (e.g. OrderedDict) of the input so that the
    resulting normalizer pytree is compatible with environment observations in
    ``jax.tree_util.tree_map`` calls.

    Args:
        obs: Example observation dict with nested or flat leaves.

    Returns:
        Nested mapping matching obs structure, with specs.Array for each leaf.
    """
    state_obs = obs["state"]
    if isinstance(state_obs, Mapping):

        def get_specs(inner_obs: Mapping[str, Any]) -> Mapping[str, specs.Array]:
            specs_dict = {
                key: specs.Array((val.shape[-1],), jnp.dtype("float32"))
                for key, val in inner_obs.items()
            }
            return type(inner_obs)(specs_dict)

        return type(obs)({key: get_specs(inner) for key, inner in obs.items()})

    return type(obs)(
        {
            key: specs.Array((val.shape[-1],), jnp.dtype("float32"))
            for key, val in obs.items()
        }
    )
