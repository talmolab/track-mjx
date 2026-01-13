"""Generic observation utilities for bowl escape task.

These utilities handle arbitrary observation dict keys, unlike the imitation-specific
observation_utils in the parent directory.
"""

from typing import Any, Mapping

import jax
import jax.numpy as jnp
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
            return nested
        else:
            return nested.reshape(nested.shape[0], -1)

    leaves = jax.tree_util.tree_leaves(nested)
    if not leaves:
        return jnp.array([])

    ref = min(leaves, key=lambda x: x.ndim)
    n_batch = 0
    for i in range(ref.ndim - 1):
        if len({leaf.shape[i] for leaf in leaves}) == 1:
            n_batch += 1
        else:
            break

    if n_batch == 0:
        flat, _ = flatten_util.ravel_pytree(nested)
        return flat

    batch_shape = ref.shape[:n_batch]
    return jnp.concatenate([leaf.reshape(*batch_shape, -1) for leaf in leaves], axis=-1)


def flatten_obs_dict(obs: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
    """Flatten each top-level key in an observation dict, preserving batch dim.

    Generic version that handles ANY keys in the observation dict, unlike the
    imitation-specific version that hardcodes imitation_target and proprioception.

    Args:
        obs: Observation dict where values may be nested dicts or flat arrays.

    Returns:
        Dict with the same keys but flattened array values.
    """
    return {key: _flatten_nested_obs(value) for key, value in obs.items()}


def concat_flat_dict_obs(obs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Concatenate flat observation dict to single array.

    Concatenates all keys in sorted order for deterministic results.

    Args:
        obs: Observation dict with flat arrays at each key.

    Returns:
        Single flat array with all observations concatenated.
    """
    sorted_keys = sorted(obs.keys())
    return jnp.concatenate([obs[k] for k in sorted_keys], axis=-1)
