"""Utilities for handling dictionary observations.

This module provides utilities for working with nested dictionary observations
where the structure is:
    {
        'state': {'imitation_target': ..., 'proprioception': ...},
        'privileged_state': {'imitation_target': ..., 'proprioception': ...}
    }

Key components:
- DictRunningStatisticsState: Holds running stats for imitation_target and proprioception
- Normalizer functions: init, update, and normalize for nested dict observations
- Flattening utilities for nested observation structures
"""

from typing import Mapping, Any

import flax
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


@flax.struct.dataclass
class DictRunningStatisticsState:
    """Running statistics state for nested dictionary observations.

    Holds separate RunningStatisticsState for imitation_target and proprioception.
    These stats are shared across 'state' and 'privileged_state' top-level keys.
    """

    imitation_target: running_statistics.RunningStatisticsState
    proprioception: running_statistics.RunningStatisticsState


def init_dict_normalizer(
    obs: Mapping[str, Mapping[str, Any]],
) -> DictRunningStatisticsState:
    """Initialize running statistics state from an example nested observation dict.

    Args:
        obs: Example nested observation dict with structure:
            {'state': {'imitation_target': ..., 'proprioception': ...}, ...}

    Returns:
        Initialized DictRunningStatisticsState with proper shapes.
    """
    # Use 'state' (or first available key) to determine observation shapes
    state_obs = obs.get("state", next(iter(obs.values())))

    imitation_target_flat = _flatten_nested_obs(state_obs["imitation_target"])
    proprioception_flat = _flatten_nested_obs(state_obs["proprioception"])

    return DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(imitation_target_flat.shape[-1:], jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(proprioception_flat.shape[-1:], jnp.dtype("float32"))
        ),
    )


def update_dict_normalizer(
    state: DictRunningStatisticsState,
    obs: Mapping[str, Mapping[str, Any]],
    pmap_axis_name: str | None = None,
) -> DictRunningStatisticsState:
    """Update running statistics from a nested observation dict.

    Uses 'state' observations to update statistics (shared with 'privileged_state').

    Args:
        state: Current running statistics state.
        obs: Nested observation dict with structure:
            {'state': {'imitation_target': ..., 'proprioception': ...}, ...}
        pmap_axis_name: Axis name for pmap aggregation (optional).

    Returns:
        Updated DictRunningStatisticsState.
    """
    # Use 'state' to update (could also use 'privileged_state', they should be same)
    state_obs = obs.get("state", next(iter(obs.values())))

    imitation_target_flat = _flatten_nested_obs(state_obs["imitation_target"])
    proprioception_flat = _flatten_nested_obs(state_obs["proprioception"])

    return DictRunningStatisticsState(
        imitation_target=running_statistics.update(
            state.imitation_target,
            imitation_target_flat,
            pmap_axis_name=pmap_axis_name,
        ),
        proprioception=running_statistics.update(
            state.proprioception,
            proprioception_flat,
            pmap_axis_name=pmap_axis_name,
        ),
    )


def normalize_dict_obs(
    obs: Mapping[str, Mapping[str, Any]],
    state: DictRunningStatisticsState,
) -> dict[str, dict[str, jnp.ndarray]]:
    """Normalize nested observation dict using running statistics.

    Normalizes imitation_target and proprioception within each top-level key
    (state, privileged_state).

    Args:
        obs: Nested observation dict with structure:
            {'state': {'imitation_target': ..., 'proprioception': ...}, ...}
        state: Running statistics state for normalization.

    Returns:
        Normalized nested dict with same structure but flat normalized arrays.
    """
    result = {}
    for top_key, inner_obs in obs.items():
        imitation_target_flat = _flatten_nested_obs(inner_obs["imitation_target"])
        proprioception_flat = _flatten_nested_obs(inner_obs["proprioception"])

        result[top_key] = {
            "imitation_target": running_statistics.normalize(
                imitation_target_flat, state.imitation_target
            ),
            "proprioception": running_statistics.normalize(
                proprioception_flat, state.proprioception
            ),
        }
    return result


def concat_inner_obs(inner_obs: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Concatenate inner observation dict (imitation_target + proprioception) to single array.

    Concatenates in consistent order: imitation_target first, then proprioception.

    Args:
        inner_obs: Inner observation dict with 'imitation_target' and 'proprioception' keys.

    Returns:
        Single flat array with all observations concatenated.
    """
    return jnp.concatenate(
        [inner_obs["imitation_target"], inner_obs["proprioception"]], axis=-1
    )


def get_obs_sizes(obs: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Extract observation sizes from an example nested observation dict.

    Args:
        obs: Example nested observation dict with structure:
            {'state': {'imitation_target': ..., 'proprioception': ...}, ...}

    Returns:
        Dict with 'imitation_target' and 'proprioception' sizes.
    """
    state_obs = obs.get("state", next(iter(obs.values())))

    imitation_target_flat = _flatten_nested_obs(state_obs["imitation_target"])
    proprioception_flat = _flatten_nested_obs(state_obs["proprioception"])

    return {
        "imitation_target": imitation_target_flat.shape[-1],
        "proprioception": proprioception_flat.shape[-1],
    }
