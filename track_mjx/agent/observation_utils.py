"""Utilities for handling dictionary observations.

This module provides utilities for working with nested dictionary observations
where the structure is:
    {
        'state': {'imitation_target': ..., 'proprioception': ...},
        'privileged_state': {'imitation_target': ..., 'proprioception': ...}
    }

Key components:
- normalizer_select: Extracts per-key running statistics from a pytree-structured normalizer
- Flattening utilities for nested observation structures
"""

from typing import Mapping, Any

import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs
from jax import flatten_util


def _flatten_nested_obs(nested: Any) -> jnp.ndarray:
    """Flatten a potentially nested observation, preserving batch dimensions.

    Handles both flat arrays and nested dicts/pytrees. Preserves the first
    dimension (batch) and flattens all trailing dimensions.

    Args:
        nested: Either a flat array or a nested dict of arrays.
            - 1D array (obs_size,): returned as-is (unbatched)
            - 2D array (batch, obs_size): returned as-is (already flat)
            - 3D+ array (batch, d1, d2, ...): flattened to (batch, d1*d2*...)
            - Nested dict: leaves are concatenated along last axis

    Returns:
        Flattened array with shape (obs_size,) or (batch, obs_size).
    """
    if isinstance(nested, jnp.ndarray):
        if nested.ndim <= 2:
            # 1D (unbatched) or 2D (batched, already flat) - return as-is
            return nested
        else:
            # 3D+ array: preserve batch dim (first), flatten the rest
            # (batch, d1, d2, ...) -> (batch, d1*d2*...)
            return nested.reshape(nested.shape[0], -1)

    # For nested dicts/pytrees, flatten the observation structure
    leaves = jax.tree_util.tree_leaves(nested)
    if not leaves:
        return jnp.array([])

    # Find batch dims: leading dimensions that are identical across all leaves
    # E.g., (1,64,5,3) and (1,64,18,5,3) share prefix (1,64)
    ref = min(leaves, key=lambda x: x.ndim)
    n_batch = 0
    for i in range(ref.ndim - 1):
        if len({leaf.shape[i] for leaf in leaves}) == 1:
            n_batch += 1
        else:
            break

    if n_batch == 0:
        # Unbatched: flatten everything to 1D
        flat, _ = flatten_util.ravel_pytree(nested)
        return flat

    # Batched: reshape each leaf to (*batch_shape, -1) and concatenate
    batch_shape = ref.shape[:n_batch]
    return jnp.concatenate([leaf.reshape(*batch_shape, -1) for leaf in leaves], axis=-1)


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


def flatten_obs_dict(obs: Mapping[str, Any]) -> jnp.ndarray:
    """Flatten an observation dict to a single array.

    Concatenates imitation_target and proprioception in that order.

    Args:
        obs: Observation dict with 'imitation_target' and 'proprioception' keys.

    Returns:
        Single flat array with all observations concatenated.
    """
    imitation_target = _flatten_nested_obs(obs["imitation_target"])
    proprioception = _flatten_nested_obs(obs["proprioception"])
    return jnp.concatenate([imitation_target, proprioception], axis=-1)


def flatten_to_dict(obs: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
    """Flatten each leaf of an observation dict while preserving dict structure.

    Used after normalization to prepare observations for networks that expect
    flat arrays for each observation key.

    Args:
        obs: Observation dict with potentially nested arrays.

    Returns:
        Dict with same keys but flattened array values.
    """
    return {k: _flatten_nested_obs(v) for k, v in obs.items()}


def flatten_full_obs(
    obs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, jnp.ndarray]]:
    """Flatten the full nested observation dict for normalizer update.

    Flattens observations at both the 'state' and 'privileged_state' levels.

    Args:
        obs: Full observation dict with structure:
            {'state': {'imitation_target': ..., 'proprioception': ...}, ...}

    Returns:
        Flattened dict with same structure but flat array values.
    """
    return {key: flatten_to_dict(inner) for key, inner in obs.items()}


def get_obs_sizes(obs: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Extract observation sizes from an example nested observation dict.

    Args:
        obs: Example nested observation dict with structure:
            {'state': {'imitation_target': ..., 'proprioception': ...}, ...}

    Returns:
        Dict with 'imitation_target' and 'proprioception' sizes.
    """
    # Use 'state' (or first available key) to determine observation shapes
    state_obs = obs.get("state", next(iter(obs.values())))

    imitation_target_flat = _flatten_nested_obs(state_obs["imitation_target"])
    proprioception_flat = _flatten_nested_obs(state_obs["proprioception"])

    return {
        "imitation_target": imitation_target_flat.shape[-1],
        "proprioception": proprioception_flat.shape[-1],
    }


def get_obs_shape(
    obs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, specs.Array]]:
    """Extract flattened observation shapes as a pytree for running_statistics.init_state.

    Returns specs for FLATTENED observations. The normalizer stores flat stats,
    and observations should be flattened before update/normalize calls.

    Args:
        obs: Example nested observation dict.

    Returns:
        Nested dict with same structure as top two levels, containing specs.Array
        objects with flattened shapes (e.g., {'state': {'imitation_target': Array(640,), ...}}).
    """

    def flatten_and_get_spec(inner_obs: Mapping[str, Any]) -> dict[str, specs.Array]:
        return {
            key: specs.Array(
                (_flatten_nested_obs(val).shape[-1],), jnp.dtype("float32")
            )
            for key, val in inner_obs.items()
        }

    return {key: flatten_and_get_spec(inner) for key, inner in obs.items()}
