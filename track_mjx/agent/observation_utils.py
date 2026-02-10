"""Utilities for handling dictionary observations.

This module provides utilities for working with nested dictionary observations
where the structure is:
    {
        'state': {'task_obs': ..., 'proprioception': ...},
        'privileged_state': {'task_obs': ..., 'proprioception': ...}
    }

Each leaf value is a flat 1D array (unbatched) or 2D array (batched).

Key components:
- normalizer_select: Extracts per-key running statistics from a pytree-structured normalizer
- get_obs_sizes / get_obs_shape: Extract observation metadata from example observations
- make_dict_value_network: Creates a value network accepting nested dict observations
"""

from collections.abc import Sequence
from typing import Mapping, Any

import jax.numpy as jnp
from brax.training import networks, types
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
        obs: Example observation dict with structure:
            {'state': {'task_obs': array, 'proprioception': array}, ...}
            Each leaf array should be flat (1D unbatched or 2D batched).

    Returns:
        Dict mapping each inner key to its feature dimension size.
    """
    state_obs = obs.get("state", next(iter(obs.values())))
    return {key: val.shape[-1] for key, val in state_obs.items()}


def get_obs_shape(
    obs: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, specs.Array]]:
    """Extract observation shapes as a pytree for running_statistics.init_state.

    Preserves the container types (e.g. OrderedDict) of the input so that the
    resulting normalizer pytree is compatible with environment observations in
    ``jax.tree_util.tree_map`` calls.

    Args:
        obs: Example observation dict with flat leaf arrays.

    Returns:
        Nested mapping matching obs structure, with specs.Array for each leaf.
    """

    def get_specs(inner_obs: Mapping[str, Any]) -> Mapping[str, specs.Array]:
        specs_dict = {
            key: specs.Array((val.shape[-1],), jnp.dtype("float32"))
            for key, val in inner_obs.items()
        }
        return type(inner_obs)(specs_dict)

    return type(obs)({key: get_specs(inner) for key, inner in obs.items()})


def make_dict_value_network(
    obs_sizes: Mapping[str, int],
    hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_obs_key: str = "privileged_state",
) -> networks.FeedForwardNetwork:
    """Create a value network that accepts nested dictionary observations.

    The value network uses the specified observation key (default: 'privileged_state')
    which contains both task_obs and proprioception.

    Args:
        obs_sizes: Dict with 'task_obs' and 'proprioception' sizes.
        hidden_layer_sizes: MLP layer sizes for value network.
        value_obs_key: Top-level observation key for value network (default: 'privileged_state').

    Returns:
        FeedForwardNetwork that accepts nested dict observations.
    """
    total_obs_size = obs_sizes["task_obs"] + obs_sizes["proprioception"]

    # Create underlying value network with flat observations
    base_value_network = networks.make_value_network(
        total_obs_size,
        preprocess_observations_fn=types.identity_observation_preprocessor,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        value_params,
        obs: Mapping[str, Mapping[str, jnp.ndarray]],
    ):
        """Apply value network with nested observation normalization."""
        value_normalizer = normalizer_select(processor_params, value_obs_key)
        normalized_inner = running_statistics.normalize(
            obs[value_obs_key], value_normalizer
        )
        # Concatenate task_obs and proprioception
        flat_obs = jnp.concatenate(
            [normalized_inner["task_obs"], normalized_inner["proprioception"]],
            axis=-1,
        )
        return base_value_network.apply((), value_params, flat_obs)

    return networks.FeedForwardNetwork(
        init=lambda key: base_value_network.init(key),
        apply=apply,
    )
